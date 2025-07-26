import torch
from model import Model
from util import device, dtype, max_value

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_file")
    parser.add_argument("preprocessed_dataset_path")
    args = parser.parse_args()

    model = Model().to(device=device, dtype=dtype).eval()
    model.load_state_dict(torch.load(args.checkpoint_file))

    dataset = [
        (
            item["layer_out"].to(device=device, dtype=dtype),
            item["target"].to(device=device, dtype=dtype),
        )
        for item in torch.load(args.preprocessed_dataset_path)
    ]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    total_hit = 0
    with torch.no_grad():
        for i, (layer_out, target) in enumerate(dataloader):
            out = torch.argmax(model(layer_out), dim=-1)
            batch_hit = (out == target).sum().item()
            total_hit += batch_hit
            print(
                f"Batch {i+1}/{len(dataloader)}: expected: {target} actual: {out} acc: {batch_hit / target.size(0)}"
            )

    print(f"Acc: {total_hit / len(dataloader.dataset)} Yay!")
