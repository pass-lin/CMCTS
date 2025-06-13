
[中文文档](README_CN.md)

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

It should be noted that the current version only supports models from the qwen2.5 series, and other models need to be modified by yourself.
