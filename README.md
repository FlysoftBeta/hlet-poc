# “隐藏层专家训练”（Hidden Layer Expert Training, HLET）演示

这是一个简单的数字提取模型，能够从设问中提取数字，演示了 HLET 的训练方式和推理效果。

例子: 输入: 听说英才计划特训营会有10个老师，是吗？ --> 输出: 10

## 下载

模型检查点/训练数据集/验证数据集可在 [Releases](https://github.com/FlysoftBeta/hlet-poc/releases) 中下载，下载后放入项目目录即可。

## 运行

演示环境:
- 操作系统: Manjaro Linux
- 内存: 16G
- CPU: AMD Ryzen 7 7435H (16) @ 4.553GHz 
- GPU: NVIDIA GeForce RTX 4060 Mobile 8GB

```
# 配置环境
$ python3 -m venv .venv
$ . .venv/bin/activate
$ python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu118
$ python3 -m pip install -r ./requirements.txt

# 交互式环境 (请先训练或下载预训练的模型)
$ python3 ./repl.py
Input: hello the second era
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.91it/s]
You're using a Qwen2TokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
Time elapsed: 2.08s
Prediction: 2
Input: 他们经历了九九八十一难
Time elapsed: 0.05s
Prediction: 81
Input: La vingt-septième pomme est une Fuji ou une pomme de Yantai?
Time elapsed: 0.06s
Prediction: 27

# 使用测试数据集测试准确率
$ python3 ./test.py


# 重新生成所需数据集
$ python3 ./gen.py ./dataset.jsonl n 4000
$ python3 ./gen.py ./val_dataset.jsonl y 1000

# 重新训练 (epochs=80)
$ rm -rf ./checkpoint.pt && python3 ./train.py
```