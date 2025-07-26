# 数据集预处理器
# 将 (输入: 文本, 输出: [数字;2]) -> (输入: 隐藏层, 输出: [数字;2])

import torch
import json
from util import device, dtype
from transformers import AutoModelForCausalLM, Qwen3ForCausalLM, AutoTokenizer


def preprocess(model, layer_idx, dataloader):
    layer_out_storage = []  # 函数内似乎无法直接赋值，故用一个 list 封一下

    def hook_fn(_, input, output):
        layer_out_storage.clear()
        layer_out_storage.append(output[0].detach())

    model.layers[layer_idx].register_forward_hook(hook_fn)

    results = []
    with torch.no_grad():
        for i, (input, target) in enumerate(dataloader):
            print(f"Preprocessing {i+1}/{len(dataloader)}")
            model(**input)
            layer_out = layer_out_storage[0]
            for j in range(layer_out.size(0)):  # batch 内元素单独保存
                results.append(
                    {
                        "layer_out": layer_out[j].to(device="cpu", dtype=dtype),
                        "target": torch.tensor(target[j]).to(
                            device="cpu", dtype=torch.long
                        ),
                    }
                )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("preprocessed_dataset_path")
    args = parser.parse_args()

    def load(path, local):
        lm: Qwen3ForCausalLM = (
            AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=False,
                local_files_only=local,
                torch_dtype=torch.float16,  # float32 对显存要求太高了！！！
            )
            .to(device=device)
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=False, local_files_only=local
        )
        return lm, tokenizer

    try:
        lm, tokenizer = load("qwen3-1.7b", True)
    except FileNotFoundError:
        lm, tokenizer = load("Qwen/Qwen3-1.7B", False)
    model = lm.model

    with open(args.dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f.readlines()]

    def collate_fn(batch):
        inputs = [
            tokenizer(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": item["input"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
                padding=False,
                truncation=True,
            )
            for item in batch
        ]
        targets = [item["target"] - 1 for item in batch]
        inputs_padded = tokenizer.pad(inputs, padding=True, return_tensors="pt").to(
            device=device
        )
        return inputs_padded, targets

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,  # 要与 train 那里的一致 (?)
        shuffle=True,
        collate_fn=collate_fn,
    )

    torch.save(
        preprocess(
            model,
            layer_idx=int(len(model.layers) * 0.4),
            dataloader=dataloader,
        ),
        args.preprocessed_dataset_path,
    )

__all__ = [preprocess]
