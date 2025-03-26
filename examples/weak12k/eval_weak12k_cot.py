# 运行采用qwen-math官方仓库，这里只用eval
filename = "outputs/Qwen/Qwen2.5-72B-Instruct/math_eval/weak12k/test_qwen25-math-cot_-1_seed0_t0.0_s0_e-1.jsonl"
import jsonlines
from utils import gsm8k_utils


def eval_acc(output, answer):
    try:
        answer = eval(answer)
        output = abs(
            eval(output)
        )  # 这个数据集没有负数的答案，但是题目可能让题目回答负数，比如水底下50米表达是-50，但答案是50
        flags = []
        for t in [
            output / 100,
            output,
            output * 100,
        ]:  # 考虑百分数的不同回答情况，当然，baseline也是有做出相应修正的
            # 这里的精度取1e-2，因为大模型经常会四舍五入到小数点三位，baseline也会做出对应的修正
            flags.append(abs(t - answer) <= 1e-2)
        return any(flags)
    except:
        pass
    return False


total = 0
right = 0
with jsonlines.open(filename, mode="r") as reader:
    # 遍历文件中的每一行，将其作为字典对象读取
    for obj in reader:
        if obj["question"].count("?") > 2 or "提出" in obj["question"]:
            continue
        total += 1
        try:
            pred = gsm8k_utils.retrieve_chat_weak12k_answer(obj["code"][0])
            answer = obj["gt"]
            right += eval_acc(str(pred), str(answer))
        except:
            pass
print(right / total)
