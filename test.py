import torch
from ldr import load_dataset
from hlet import HLET

inst = HLET(True)
inst.load("./checkpoint.pt")

val_dataset = load_dataset("./val_dataset.jsonl")
outs, targets = inst.infer(inst.preprocess(val_dataset))
num_samples = targets.size(0)
acc = (outs == targets).sum().item() / num_samples
print(f"Tested {num_samples}, accuracy: {acc}")