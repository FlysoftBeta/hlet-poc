import torch
import torch.optim as optim
import torch.nn.functional as F
from model import Model
from util import device, dtype

def train(model, dataloader, val_dataloader, epochs, lr):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, mode="min")

    for epoch in range(epochs):
        total_loss = 0
        for layer_out, target in dataloader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(layer_out), target) # 这种 loss 不需要 one hot编码
            loss.backward()
            optimizer.step()
            assert not torch.isnan(loss).any()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)

        total_val_loss = 0
        with torch.no_grad():
            for layer_out, target in val_dataloader:
                loss = F.cross_entropy(model(layer_out), target)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_dataloader)

        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs}: train loss: {avg_loss}, val loss: {avg_val_loss}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs')
    parser.add_argument('checkpoint_file')
    parser.add_argument('preprocessed_dataset_path')
    parser.add_argument('preprocessed_val_dataset_path')
    args = parser.parse_args()

    model = Model().to(device=device, dtype=dtype).train()
    try:
        model.load_state_dict(torch.load(args.checkpoint_file))
        print("成功加载 checkpoint")
    except FileNotFoundError:
        pass
    def make_dataloader(file):
        dataset = [(item["layer_out"].to(device=device), item["target"].to(device=device)) for item in torch.load(file)]
        return torch.utils.data.DataLoader(dataset, batch_size=16)
    dataloader = make_dataloader(args.preprocessed_dataset_path)
    val_dataloader = make_dataloader(args.preprocessed_val_dataset_path)
    train(model, dataloader, val_dataloader, epochs=int(args.epochs), lr=0.0001)
    torch.save(model.state_dict(), args.checkpoint_file)
