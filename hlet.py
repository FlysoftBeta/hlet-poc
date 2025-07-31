from typing import Generator, Iterable, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Model

from transformers import AutoModelForCausalLM, Qwen3ForCausalLM, AutoTokenizer

type RawData = list[tuple[str, Optional[int]]]
type PreprocessedData = Generator[tuple[torch.Tensor, torch.LongTensor]]


class HLET:
    def __init__(self, training: bool):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.layer_idx = 11  # = int(len(self.lm.model.layers) * 0.4)
        self.dtype = torch.float32
        self.training = training
        self.lm = None
        self.model = Model().to(dtype=self.dtype, device=self.device).train(training)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_lm(self):  # 按需加载
        if self.lm is not None:
            return

        def load(path, local):
            lm: Qwen3ForCausalLM = (
                AutoModelForCausalLM.from_pretrained(
                    path,
                    trust_remote_code=False,
                    local_files_only=local,
                    torch_dtype=torch.float16,  # float16 对显存要求比较低
                )
                .to(device=self.device)
                .eval()
            )
            tokenizer = AutoTokenizer.from_pretrained(
                path, trust_remote_code=False, local_files_only=local
            )
            return lm, tokenizer

        try:
            self.lm, self.tokenizer = load("qwen3-1.7b", True)  # 本地加载
        except FileNotFoundError:
            self.lm, self.tokenizer = load("Qwen/Qwen3-1.7B", False)  # Huggingface 下载

    def free_lm(self):  # 只在显存紧张时使用此方法
        if self.lm is not None:
            del self.lm
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def infer(self, pp_data: PreprocessedData) -> tuple[torch.LongTensor, torch.LongTensor]:
        targets = []
        outs = []
        with torch.no_grad():
            for layer_out, target in pp_data:
                out = torch.argmax(self.model(layer_out), dim=-1)
                targets.append(target)
                outs.append(out)

        return (torch.cat(outs, dim=0), torch.cat(targets, dim=0))

    def train(
        self,
        epochs: int,
        pp_data: PreprocessedData,
        pp_val_data: PreprocessedData,
    ):
        assert self.training is not None
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, mode="min"
        )

        for epoch in range(epochs):
            total_loss = 0
            for layer_out, target in pp_data:
                optimizer.zero_grad()
                loss = F.cross_entropy(
                    self.model(layer_out),
                    target,  # target 不需要 one-hot 编码，cross_entropy 会自动处理
                )
                loss.backward()
                optimizer.step()
                assert not torch.isnan(loss).any()
                total_loss += loss.item()
            avg_loss = total_loss / len(pp_data)

            total_val_loss = 0
            with torch.no_grad():
                for layer_out, target in pp_val_data:
                    loss = F.cross_entropy(self.model(layer_out), target)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(pp_val_data)

            scheduler.step(avg_val_loss)
            print(
                f"Epoch {epoch+1}/{epochs}: train loss: {avg_loss}, val loss: {avg_val_loss}"
            )

    def preprocess(
        self, raw_data: RawData, shuffle: bool = False
    ) -> Generator[PreprocessedData]:
        self.load_lm()

        def collate_fn(batch):
            inputs = [
                self.tokenizer(
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": input}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    ),
                    padding=False,
                    truncation=True,
                )
                for input, _target in batch
            ]
            targets = torch.tensor(
                [target - 1 if target is not None else 0 for _input, target in batch],
                dtype=torch.long,
                device=self.device,
            )
            inputs_padded = self.tokenizer.pad(
                inputs, padding=True, return_tensors="pt"
            ).to(device=self.device)
            return inputs_padded, targets

        dataloader = torch.utils.data.DataLoader(
            raw_data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

        layer_out_storage = []

        def hook_fn(_, _input, output):
            layer_out_storage.clear()
            layer_out_storage.append(output[0].to(dtype=self.dtype).detach())

        hook = self.lm.model.layers[self.layer_idx].register_forward_hook(hook_fn)

        with torch.no_grad():
            for input, target in dataloader:
                self.lm.model(**input)
                layer_out = layer_out_storage[0]
                yield (layer_out, target)

        hook.remove()


__all__ = [HLET]
