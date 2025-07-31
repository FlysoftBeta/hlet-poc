from hlet import HLET
from time import perf_counter as timer

inst = HLET(False)
inst.load("./checkpoint.pt")
while True:
    input_ = input("Input: ")
    start = timer()
    outs, _targets = inst.infer(inst.preprocess([(input_, None)]))  # 输入的第二个参数被忽略
    end = timer()
    print(f"Time elapsed: {(end - start):.2f}s")
    print(f"Prediction: {outs.item() + 1}")