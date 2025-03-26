# 运行采用qwen-math官方仓库，这里只用eval
# math系列的就直接用他们文章里报道的结果好了
filename = "outputs/Qwen/Qwen2.5-Math-7B-Instruct/math_eval/cn_middle_school/test_qwen25-math-cot_-1_seed0_t0.0_s0_e-1.jsonl"
import jsonlines
from utils import gsm8k_utils
from latex2sympy2 import latex2sympy
from reasoners.benchmark import MathEvaluator

evaluator = MathEvaluator(
    output_extractor=gsm8k_utils.retrieve_chat_middleschool_answer,
    answer_extractor=lambda x: [
        gsm8k_utils.fullwidth_to_halfwidth(x["answer"])
        .replace(" ", "")
        .replace("%", "/100"),
        x["choice_answer"].replace(" ", ""),
    ],
    filename="data/cn_middle_school/test.jsonl",
    sample_prompt_type="rap",
)


def eval_acc(output, answer):
    pred, flag = output
    answer_str, choice = answer
    try:
        if flag == "sympy":
            answer_str = latex2sympy(answer_str)
            return (answer_str == pred) or (answer_str.simplify() == pred.simplify())
        elif flag == "option":
            return choice == pred
        else:
            return pred == answer_str
    except:
        return False


dataset = list(evaluator.full_dataset)
total = 0
right = 0
with jsonlines.open(filename, mode="r") as reader:
    # 遍历文件中的每一行，将其作为字典对象读取
    for i, obj in enumerate(reader):
        total += 1
        try:
            pred = evaluator.output_extractor(obj["code"][0])
            answer = evaluator.answer_extractor(dataset[i])
            right += eval_acc(pred, answer)

        except:
            pass
print(right / total)
