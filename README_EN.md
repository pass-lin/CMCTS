# C-MCTS Project Official Open-Source Code Guide

## Project Overview

This project implements the C-MCTS algorithm proposed in the paper [“Leveraging Constrained Monte Carlo Tree Search to Generate Reliable Long Chain-of-Thought for Mathematical Reasoning”](https://arxiv.org/abs/2502.11169). The project aims to enhance the mathematical reasoning capabilities of large language models, particularly in generating reliable long chains of thought, through constrained Monte Carlo tree search.

## Code Structure

### COT Testing Method

For the COT (Chain-of-Thought) baseline, please use the [Qwen repository](https://github.com/QwenLM/Qwen2.5-Math) to run the scripts for optimal performance.

### Datasets

All datasets used are located in the `data` directory, covering multiple mathematical reasoning benchmarks, including GSM8K, Math500, AquA, etc.

### Testing Scripts

Testing scripts are located in the `examples` directory, with each dataset corresponding to a subdirectory containing evaluation scripts for that dataset.

## File Description for AquA Dataset as an Example

### `examples/AquA/eval_aqua_cot.py`

- **Purpose**: Evaluate the COT (Chain-of-Thought) baseline model.
- **Details**: In some cases, our answer extraction method may outperform Qwen. In such cases, we use our method as the final answer.

### `examples/AquA/run_chatmcts_aqua_PRM.py`

- **Purpose**: Run the RAP-MCTS baseline.
- **Details**: We modified the original prompt and changed it to a conversational format, which significantly improves performance compared to the original method.

### `examples/AquA/run_chatmcts_aqua_PRM.py` (with PRM version)

- **Purpose**: Similar to the above script, but with the addition of PRM (Process Reward Model).

### `examples/AquA/run_cot_aqua.py`

- **Purpose**: Generate multiple COTs using the same parameters as our method and select answers through voting.

### `examples/AquA/run_deepmcts_aqua.py`

- **Purpose**: Our C-MCTS method.
- **Details**: The implementation is mostly consistent with the paper. During testing, various accuracy rates may be output, but only `max_reward_path_acc` is considered as the experimental result.

## Parameter Adjustment

Here are the key parameters that need adjustment:

```python
partial_order = [
    True,
    True,
    False,
    True,
    False,
]  # Enable specific partial order rules; set all to False if not used
native_rewards_mode = True  # Whether to use the model itself as PRM
# Without PRM, more rules are needed, so all are enabled. Of course, you can also disable all.
if native_rewards_mode:
    partial_order = [True, True, True, True, True]
```

### Parameter Description

- `partial_order`: Controls whether specific partial order rules are enabled. Set all to `False` if not needed.
- `native_rewards_mode`: Determines whether to use the model itself as PRM (Process Reward Model). When enabled, all partial order rules are automatically turned on in most scripts. In a few scripts, it remains consistent with using the PRM model.

## Experimental Results

Note that the code in this repository is slightly different from the code in the paper. After debugging and modifications, we found that the plan set did not contribute significantly to the performance, so it was removed in this repository. Additionally, some minor adjustments were made. In the original paper, all scripts enabled all partial order rules, whereas here they are selectively enabled.

### Main Experiment Table

| Dataset               | qwen25-it-cot-7B | qwen25-it-cot-72B | qwen25-it-maj-7B | C-MCTS | C-MCTS+RULE | C-MCTS+RULE-wo-PRM |
|:----------------------|:-----------------|:------------------|:-----------------|:-------|:------------|:-------------------|
| gaokao2023en          | 66.0             | 73.2              | 69.0             | 75.8   | 76.6        | 71.1               |
| MATH-500              | 77.0             | 83.4              | 79.6             | 84.6   | 85.4        | 79.2               |
| AauA                  | 74.4             | 79.2              | 81.1             | 86.2   | 87.7        | 85.4               |
| svamp                 | 93.9             | 95.4              | 93.5             | 95.9   | 96.4        | 95.3               |
| GSM8K                 | 92.4             | 95.8              | 93.0             | 95.1   | 95.4        | 92.7               |
| cmath                 | 89.7             | 93.0              | 93.1             | 94.3   | 95.0        | 92.5               |
| cn_middle_school      | 70.2             | 83.1              | 82.1             | 87.1   | 87.1        | 83.1               |
| gaokao_math_qa        | 60.9             | 74.3              | 68.6             | 78.9   | 80.3        | 72.6               |
| weak12k               | 85.6             | 91.3              | 90.0             | 92.5   | 93.1        | 88.5               |
| **Average**           | 78.9             | 85.4              | 83.3             | 87.8   | 88.5        | 84.7               |

### Comparison with RAP-MCTS

| Dataset               | gaokao2023en | MATH-500 | AauA | svamp | GSM8K | cmath | cn_middle_school | gaokao_math_qa | weak12k | avg  |
|:----------------------|:-------------|:---------|:-----|:------|:------|:------|:-----------------|:---------------|:--------|:-----|
| C-MCTS                | 75.8         | 84.6     | 86.2 | 95.9  | 95.1  | 94.3  | 87.1            | 78.9           | 92.5    | 87.8 |
| C-MCTS-wo-PRM         | 68.8         | 77.0     | 80.7 | 95.8  | 92.7  | 93.0  | 75.2            | 70.3           | 88.2    | 82.4 |
| RAP-mcts              | 67.5         | 75.0     | 84.7 | 93.0  | 92.5  | 93.0  | 83.1            | 72.0           | 89.5    | 83.3 |
| RAP-mcts+PRM          | 69.8         | 77.2     | 83.7 | 93.5  | 92.5  | 93.3  | 83.1            | 72.6           | 89.3    | 83.8 |

With PRM, our method significantly outperforms RAP-MCTS, but without PRM, it falls short. This indicates that PRM is crucial for our current work. Moreover, RAP-MCTS performs similarly with or without PRM, suggesting that the different actions generated by the original RAP do not make a significant difference.

### Ablation Study

The datasets are divided into two parts: those with accuracy above 90% are considered easy datasets, while the others are considered difficult.

#### Results on Difficult Datasets

| Method                        | gaokao2023en | MATH-500 | AauA | cn_middle_school | gaokao_math_qa | avg  |
|:------------------------------|:-------------|:---------|:-----|:-----------------|:---------------|:-----|
| C-MCTS-wo-PRM                 | 68.8         | 77.0     | 80.7 | 75.2            | 70.3           | 74.4 |
| C-MCTS-wo-Reflect-code        | 74.0         | 82.4     | 85.0 | 86.1            | 76.9           | 80.8 |
| C-MCTS-wo-Reflect             | 74.8         | 83.8     | 84.6 | 86.1            | 78.3           | 81.5 |
| C-MCTS                        | 75.8         | 84.6     | 86.2 | 87.1            | 78.9           | 82.5 |

#### Results on Easy Datasets

| Method                        | svamp | GSM8K | cmath | weak12k | avg  |
|:------------------------------|:------|:------|:------|:--------|:-----|
| C-MCTS-wo-PRM                 | 95.8  | 92.7  | 93.0  | 88.2    | 92.4 |
| C-MCTS-wo-Reflect-code        | 96.0  | 95.2  | 94.5  | 92.6    | 94.5 |
| C-MCTS-wo-Reflect             | 96.2  | 95.1  | 94.6  | 92.6    | 94.6 |
| C-MCTS                        | 95.9  | 95.1  | 94.3  | 92.5    | 94.4 |

We can observe that on the easy datasets, the absence of certain action sets does not significantly impact performance. However, on the difficult datasets, each added action set contributes to performance improvement. Moreover, even with only the understand set, the model can achieve good results.

Additionally, on both easy and difficult datasets, the absence of PRM results in significant performance loss.