from hlet import HLET

inst = HLET(False)
inst.load("./checkpoint.pt")
while True:
    input_ = input("输入字符串:")
    outs, _targets = inst.infer(inst.preprocess(([input_], None)))  # 第二个参数被忽略
    print("猜测结果:", outs + 1)