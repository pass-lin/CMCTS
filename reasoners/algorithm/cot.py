import random


def build_cot_promt(prompt, question_prefix, example, num_shot, language="en"):
    if language == "en":
        cot_prompt = "Please think step by step to solve the following problem.\n"
    else:
        cot_prompt = "请一步步思考并解决下面的问题.\n"
    prefix = ""
    for idx, t in enumerate(random.sample(prompt["cot_pool"], num_shot)):
        prefix += cot_prompt + t.format(idx=idx + 1) + "\n\n"
    question = cot_prompt + question_prefix + example + " \n"
    return prefix + question
