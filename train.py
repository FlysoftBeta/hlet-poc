from ldr import load_dataset
from hlet import HLET

inst = HLET(True)
try:
    inst.load("./checkpoint.pt")
except FileNotFoundError:
    print("Train from scratch")

dataset, val_dataset = load_dataset("./dataset.jsonl"), load_dataset("./val_dataset.jsonl")
inst.train(80, inst.preprocess(dataset, True), inst.preprocess(val_dataset, True))
inst.save("./checkpoint.pt")
