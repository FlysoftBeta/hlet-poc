import torch
from ldr import load_dataset
from hlet import HLET

inst = HLET(True)
inst.load("./checkpoint.pt")

for plan in [
    "chinese",
    "chinese_roman",
    "chinese_trad",
    "spanish",
    "spanish_arabic",
    "english",
    "english_roman",
    "english_arabic"
]:
    try:
        val_dataset = load_dataset(f"./{plan}_val_dataset.jsonl")
    except FileNotFoundError:
        print(f"Skip plan {plan} testing")
        continue
    outs, targets = inst.infer(inst.preprocess(val_dataset))
    num_samples = targets.size(0)
    acc = (outs == targets).sum().item() / num_samples
    print(f"{plan}({num_samples}): accuracy: {acc}")