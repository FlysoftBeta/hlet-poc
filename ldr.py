import json


def load_dataset(path):
    with open(path, "r") as f:
        return list(map(lambda x: (x["input"], x["target"]), map(lambda x: json.loads(x), f.readlines())))

__all__ = [load_dataset]