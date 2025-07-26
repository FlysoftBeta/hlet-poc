# “隐藏层专家训练”（Hidden Layer Expert Training, HLET）演示

这是一个简单的数字提取模型，能够从设问中提取数字，演示了 HLET 的训练方式和推理效果。

例子: 输入: 听说英才计划特训营会有10个老师，是吗？ --> 输出: 10

## 下载

Checkpoints/训练数据集/验证数据集可在 [Releases](https://github.com/FlysoftBeta/hlet-poc/releases) 中下载，下载后放入项目目录即可。

## Linux 上运行

```shell
$ # 配置环境
$ python -m venv .venv
$ . .venv/bin/activate
$ pip install torch --index-url https://download.pytorch.org/whl/cu118
$ pip install -r ./requirements.txt

$ # 只测试(使用现有checkpoint)，不训练，需要下载 checkpoint
$ ./test.sh
...
Acc: 0.99 Yay!

# 训练并测试
$ rm -rf ./checkpoint && ./train_and_test.sh # 重新训练
```