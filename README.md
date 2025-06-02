# C-MCTS项目官方开源代码指南
[English Document](README_EN.md)
## 项目概述

本项目实现了论文[《Leveraging Constrained Monte Carlo Tree Search to Generate Reliable Long Chain-of-Thought for Mathematical Reasoning》](https://arxiv.org/abs/2502.11169 )中提出的C-MCTS算法。该项目旨在通过约束蒙特卡洛树搜索提高大型语言模型的数学推理能力，特别是生成可靠的长思维链。

## 代码结构
### COT测试方法
COT的baseline请使用[Qwen的仓库](https://github.com/QwenLM/Qwen2.5-Math )运行脚本以达到最好的效果

### 数据集
所有使用的数据集位于`data`目录下，涵盖多个数学推理基准测试，包括但不限于GSM8K、Math500、AquA等。

### 测试脚本
测试脚本位于`examples`目录下，每个数据集对应一个子目录，包含针对该数据集的评估脚本。

## 以AquA数据集为例的文件说明

### `examples/AquA/eval_aqua_cot.py`
- **用途**：评估COT（Chain-of-Thought）基线模型。
- **细节**：在某些情况下，我们的答案提取方法可能比Qwen的效果更好。此时，我们会使用我们的方法作为最终答案。

### `examples/AquA/run_chatmcts_aqua_PRM.py`
- **用途**：运行[RAP-MCTS基线](https://arxiv.org/abs/2305.14992 )。
- **细节**：我们对原始的prompt进行了修改，并改成了对话形式。这比使用原始方法的效果要好得多。

### `examples/AquA/run_chatmcts_aqua_PRM.py`（带PRM版本）
- **用途**：与上述脚本类似，但加入了PRM（Process Reward Model）。

### `examples/AquA/run_cot_aqua.py`
- **用途**：使用与我们方法相同的参数生成多个COT，然后通过投票法选择答案。

### `examples/AquA/run_deepmcts_aqua.py`
- **用途**：我们的C-MCTS方法。
- **细节**：大部分实现与论文中介绍的一致。在测试过程中可能会输出各种各样的准确率，但只有`max_reward_path_acc`会被作为实验结果。

## 参数调整

以下是需要调整的关键参数：

```python
partial_order = [
    True,
    True,
    False,
    True,
    False,
]  # 启动哪条偏序规则，不使用就全部选False
native_rewards_mode = True  # 是否使用自身作为PRM
# 没有PRM的情况下，需要更多规则限制，所以全开。当然你也可以全关
if native_rewards_mode:
    partial_order = [True, True, True, True, True]
```

### 参数说明
- `partial_order`：控制是否启用特定的偏序规则。如果不需要，可以全部设置为`False`。
- `native_rewards_mode`：决定是否使用模型自身作为PRM（过程奖励模型）。启用此模式时，在大部分脚本里会自动开启所有偏序规则，少部分脚本里会保持和使用PRM模型时保持一致
