import io
from typing import NamedTuple, TypedDict
from collections import defaultdict
from reasoners import WorldModel, LanguageModel
from reasoners.base import Example
import numpy as np


class SubResult(NamedTuple):
    sub_question: str
    sub_answer: str
    confidence: float


GSM8kState = list[SubResult]
GSM8kAction = str
GSM8kExample = str


class GSM8kPromptDict(TypedDict):
    instruction: str
    interactive_examples: list[str]
    useful_examples: list[str]
    question_prefix: str
    subquestion_prefix: str
    overall_question_prefix: str
    answer_prefix: str


class GSM8kWorldModel(WorldModel[GSM8kState, GSM8kAction, GSM8kExample]):
    """
    GSM8k World Model
    State: [[sub_question_1, sub_answer_1, confidence_1], [sub_question_2, sub_answer_2, confidence_2], ...]
    Action: sub_question
    """

    def __init__(
        self,
        base_model: LanguageModel,
        n_confidence=8,
        batch_size=2,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        early_stop_base=None,
        early_stop_threshold=1.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.batch_size = batch_size
        self.n_confidence = n_confidence
        self.temperature = temperature
        self.early_stop_base = (
            early_stop_base if early_stop_base is not None else n_confidence
        )
        self.early_stop_threshold = early_stop_threshold
        self.prompt_examples = ""
        self.n_shots = 0
        self.top_k = top_k
        self.top_p = top_p
        from utils import gsm8k_utils

        self.utils = gsm8k_utils

    def update_example(self, example: Example, prompt) -> None:
        super().update_example(example, prompt)
        assert prompt is not None
        self.prompt = prompt
        with io.StringIO() as f:
            f.write(self.prompt["instruction"] + "\n\n")
            for idx, example in enumerate(self.prompt["interactive_examples"]):
                f.write(example.format(idx=idx + 1) + "\n\n")
            self.n_shots = len(self.prompt["interactive_examples"])
            self.prompt_examples = f.getvalue()

    def init_state(self) -> list:
        return []

    def get_step_inputs(
        self, state: GSM8kState, action: GSM8kAction
    ) -> tuple[GSM8kState, dict]:
        state = state.copy()

        with io.StringIO() as f:
            f.write(self.prompt_examples)
            f.write(
                self.prompt["question_prefix"].format(
                    idx=self.n_shots + 1, question=self.example
                )
                + "\n"
            )
            for idx, t in enumerate(state):
                q, a = t[:2]
                f.write(
                    self.prompt["subquestion_prefix"].format(
                        idx=self.n_shots + 1, sub_idx=idx + 1
                    )
                    + " "
                    + q
                    + "\n"
                )
                f.write(
                    self.prompt["answer_prefix"].format(
                        idx=self.n_shots + 1, sub_idx=idx + 1
                    )
                    + " "
                    + a
                    + "\n"
                )
            f.write(
                self.prompt["subquestion_prefix"].format(
                    idx=self.n_shots + 1, sub_idx=len(state) + 1
                )
                + " "
                + action
                + "\n"
            )
            f.write(
                self.prompt["answer_prefix"].format(
                    idx=self.n_shots + 1, sub_idx=len(state) + 1
                )
            )
            model_input = f.getvalue()
        num = 0
        for start1 in range(0, self.n_confidence, self.early_stop_base):
            stop1 = min(start1 + self.early_stop_base, self.n_confidence)

            for start in range(start1, stop1, self.batch_size):
                stop = min(start + self.batch_size, stop1)
                num += stop - start
        return num, state, model_input

    def get_step_outputs(self, outputs, state, action):
        answer_dict = defaultdict(list)  # map from answer to list of thoughts
        result = ""
        for output in outputs:
            result = output.strip()
            answer = self.utils.retrieve_answer(result)
            answer_dict[answer].append(result)
        sorted_answer_dict = sorted(
            answer_dict.items(), key=lambda p: len(p[1]), reverse=True
        )
        if len(answer_dict) == 0:
            print("Warning: no answer found")
            confidence, answer = (
                0,
                result,
            )  # No reasonable answer found. Fall back to choose the last response
        else:
            sorted_answer_dict = sorted(
                answer_dict.items(), key=lambda p: len(p[1]), reverse=True
            )
            max_answer = sorted_answer_dict[0]
            max_answer_output_list = max_answer[1]
            max_len = len(max_answer_output_list)
            answer = max_answer_output_list[
                0
            ]  # Here we simply choose the first appearance of the answer
            confidence = max_len / sum(len(v) for v in answer_dict.values())

        state.append(SubResult(action, answer, confidence))
        aux = {"confidence": confidence}
        return state, aux

    def step(self, state: GSM8kState, action: GSM8kAction) -> tuple[GSM8kState, dict]:
        state = state.copy()

        with io.StringIO() as f:
            f.write(self.prompt_examples)
            f.write(
                self.prompt["question_prefix"].format(
                    idx=self.n_shots + 1, question=self.example
                )
                + "\n"
            )
            for idx, (q, a, _) in enumerate(state):
                f.write(
                    self.prompt["subquestion_prefix"].format(
                        idx=self.n_shots + 1, sub_idx=idx + 1
                    )
                    + " "
                    + q
                    + "\n"
                )
                f.write(
                    self.prompt["answer_prefix"].format(
                        idx=self.n_shots + 1, sub_idx=idx + 1
                    )
                    + " "
                    + a
                    + "\n"
                )
            f.write(
                self.prompt["subquestion_prefix"].format(
                    idx=self.n_shots + 1, sub_idx=len(state) + 1
                )
                + " "
                + action
                + "\n"
            )
            f.write(
                self.prompt["answer_prefix"].format(
                    idx=self.n_shots + 1, sub_idx=len(state) + 1
                )
            )
            model_input = f.getvalue()

        answer_dict = defaultdict(list)  # map from answer to list of thoughts
        result = ""
        for start1 in range(0, self.n_confidence, self.early_stop_base):
            stop1 = min(start1 + self.early_stop_base, self.n_confidence)

            for start in range(start1, stop1, self.batch_size):
                stop = min(start + self.batch_size, stop1)
                num = stop - start

                outputs = self.base_model.generate(
                    [model_input] * num,
                    hide_input=True,
                    do_sample=True,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    eos_token_id="\n",
                ).text
                for output in outputs:
                    result = output.strip()
                    answer = self.utils.retrieve_answer(result)
                    answer_dict[answer].append(result)

            # Early stop if confidence is high enough
            if len(answer_dict) == 0:  # no answer yet
                continue
            sorted_answer_dict = sorted(
                answer_dict.items(), key=lambda p: len(p[1]), reverse=True
            )
            max_len = len(sorted_answer_dict[0][1])
            if max_len / stop1 >= self.early_stop_threshold:
                if len(sorted_answer_dict) >= 2 and max_len == len(
                    sorted_answer_dict[1][1]
                ):
                    pass  # Tie with the second best answer
                else:
                    break

        if len(answer_dict) == 0:
            print("Warning: no answer found")
            confidence, answer = (
                0,
                result,
            )  # No reasonable answer found. Fall back to choose the last response
        else:
            sorted_answer_dict = sorted(
                answer_dict.items(), key=lambda p: len(p[1]), reverse=True
            )
            max_answer = sorted_answer_dict[0]
            max_answer_output_list = max_answer[1]
            max_len = len(max_answer_output_list)
            answer = max_answer_output_list[
                0
            ]  # Here we simply choose the first appearance of the answer
            confidence = max_len / sum(len(v) for v in answer_dict.values())

        state.append(SubResult(action, answer, confidence))
        aux = {"confidence": confidence}
        return state, aux

    def is_terminal(self, state: GSM8kState) -> bool:
        if len(state) > 0 and "Now we can answer" in state[-1].sub_question:
            return True
        else:
            return False


class Math23KWorldModel(GSM8kWorldModel):
    def is_terminal(self, state: GSM8kState) -> bool:
        if len(state) > 0 and "现在我们可以回答这个问题" in state[-1].sub_question:
            return True
        else:
            return False


class SubResult(NamedTuple):
    sub_question: str
    sub_answer: str
    confidence: float
    answer_list: list[str] = None
    answer_values: list[str] = None


MATHState = list[SubResult]
MATHAction = str


class MATHPromptDict(TypedDict):
    instruction: str
    interactive_examples: list[str]
    useful_examples: list[str]
    question_prefix: str
    subquestion_prefix: str
    overall_question_prefix: str
    answer_prefix: str


import json


class MATHWorldModel(GSM8kWorldModel):
    """
    MATH World Model
    State: [[sub_question_1, sub_answer_1, confidence_1], [sub_question_2, sub_answer_2, confidence_2], ...]
    Action: sub_question
    """

    def __init__(
        self,
        base_model: LanguageModel,
        n_confidence=8,
        batch_size=2,
        temperature=0.8,
        early_stop_base=None,
        early_stop_threshold=1.0,
        utils=None,
        score_prompts="/home/xinyuan/workspace/llm-reasoners/examples/AQuA_rap/prompts/score_examples.json",
    ) -> None:
        super(GSM8kWorldModel, self).__init__()
        self.base_model = base_model
        self.batch_size = batch_size
        self.n_confidence = n_confidence
        self.temperature = temperature
        self.early_stop_base = (
            early_stop_base if early_stop_base is not None else n_confidence
        )
        self.early_stop_threshold = early_stop_threshold
        self.prompt_examples = ""
        self.utils = utils
        self.n_shots = 0
        with open(score_prompts) as f:
            self.score_prompts = json.load(f)

    def update_example(self, example: Example, prompt: MATHPromptDict = None) -> None:
        super().update_example(example, prompt)
        assert prompt is not None
        self.prompt = prompt
        with io.StringIO() as f:
            f.write(self.prompt["instruction"] + "\n\n")
            for idx, example in enumerate(self.prompt["interactive_examples"]):
                f.write(example.format(idx=idx + 1) + "\n\n")
            self.n_shots = len(self.prompt["interactive_examples"])
            self.prompt_examples = f.getvalue()

    def init_state(self) -> list:
        return []

    def get_step_inputs(
        self, state: GSM8kState, action: GSM8kAction
    ) -> tuple[GSM8kState, dict]:
        state = state.copy()

        with io.StringIO() as f:
            f.write(self.prompt_examples)
            f.write(
                self.prompt["question_prefix"].format(
                    idx=self.n_shots + 1, question=self.example
                )
                + "\n"
            )
            for idx, (q, a, *_) in enumerate(state):
                f.write(
                    self.prompt["subquestion_prefix"].format(
                        idx=self.n_shots + 1, sub_idx=idx + 1
                    )
                    + " "
                    + q
                    + "\n"
                )
                f.write(
                    self.prompt["answer_prefix"].format(
                        idx=self.n_shots + 1, sub_idx=idx + 1
                    )
                    + " "
                    + a
                    + "\n"
                )
            f.write(
                self.prompt["subquestion_prefix"].format(
                    idx=self.n_shots + 1, sub_idx=len(state) + 1
                )
                + " "
                + action
                + "\n"
            )
            f.write(
                self.prompt["answer_prefix"].format(
                    idx=self.n_shots + 1, sub_idx=len(state) + 1
                )
            )
            model_input = f.getvalue()
        num = 0
        for start1 in range(0, self.n_confidence, self.early_stop_base):
            stop1 = min(start1 + self.early_stop_base, self.n_confidence)

            for start in range(start1, stop1, self.batch_size):
                stop = min(start + self.batch_size, stop1)
                num += stop - start
        return num, state, model_input

    def step(self, state: MATHState, action: MATHAction) -> tuple[MATHState, dict]:
        print("********* world model step *******")
        state = state.copy()

        with io.StringIO() as f:
            f.write(self.prompt_examples)
            f.write(
                self.prompt["question_prefix"].format(
                    idx=self.n_shots + 1, question=self.example
                )
                + "\n"
            )
            for idx, (q, a, *_) in enumerate(state):
                f.write(
                    self.prompt["subquestion_prefix"].format(
                        idx=self.n_shots + 1, sub_idx=idx + 1
                    )
                    + " "
                    + q
                    + "\n"
                )
                f.write(
                    self.prompt["answer_prefix"].format(
                        idx=self.n_shots + 1, sub_idx=idx + 1
                    )
                    + " "
                    + a
                    + "\n"
                )
            f.write(
                self.prompt["subquestion_prefix"].format(
                    idx=self.n_shots + 1, sub_idx=len(state) + 1
                )
                + " "
                + action
                + "\n"
            )
            f.write(
                self.prompt["answer_prefix"].format(
                    idx=self.n_shots + 1, sub_idx=len(state) + 1
                )
            )
            model_input = f.getvalue()

        answer_dict = defaultdict(list)  # map from answer to list of thoughts
        score_dict = defaultdict(list)
        result = ""
        result_count = 0
        answer_count = 0
        none_count = 0

        for start1 in range(0, self.n_confidence, self.early_stop_base):
            stop1 = min(start1 + self.early_stop_base, self.n_confidence)

            for start in range(start1, stop1, self.batch_size):
                stop = min(start + self.batch_size, stop1)
                num = stop - start

                outputs = self.base_model.generate(
                    [model_input] * num,
                    hide_input=True,
                    do_sample=True,
                    temperature=self.temperature,
                    eos_token_id="\n",
                ).text
                for output in outputs:
                    result = output.strip()
                    result_count += 1
                    if "Now we can" in action:
                        answer = self.utils.retrieve_answer(result)
                    else:
                        answer = self.utils.retrieve_answer_not_option(result)

                    if answer is not None:
                        if len(score_dict[(answer, result)]) < len(self.score_prompts):
                            for score_prompt_index in range(len(self.score_prompts)):
                                with io.StringIO() as f:
                                    f.write(
                                        self.score_prompts[score_prompt_index]["input"]
                                        + "\n\n"
                                    )
                                    f.write(
                                        self.score_prompts[score_prompt_index][
                                            "question_prefix"
                                        ]
                                        + self.example
                                        + "\n"
                                    )
                                    for idx, (q, a, *_) in enumerate(state):
                                        f.write(
                                            self.score_prompts[score_prompt_index][
                                                "subquestion_prefix"
                                            ].format(idx + 1)
                                            + " "
                                            + q
                                            + "\n"
                                        )
                                        f.write(
                                            self.score_prompts[score_prompt_index][
                                                "subanswer_prefix"
                                            ].format(idx + 1)
                                            + " "
                                            + a
                                            + "\n"
                                        )
                                    f.write(
                                        self.score_prompts[score_prompt_index][
                                            "subquestion_prefix"
                                        ].format(len(state) + 1)
                                        + " "
                                        + action
                                        + "\n"
                                    )
                                    f.write(
                                        self.score_prompts[score_prompt_index][
                                            "new_subanswer_prefix"
                                        ].format(len(state) + 1)
                                        + " "
                                        + result
                                        + "\n"
                                    )
                                    f.write(
                                        self.score_prompts[score_prompt_index][
                                            "score_prefix"
                                        ]
                                    )
                                    score_input = f.getvalue()

                                print(f"score_input: {score_input}")

                                logits = self.base_model.get_next_token_logits(
                                    score_input, ["Yes", "No"]
                                )[0]
                                probs = np.exp(logits) / np.sum(np.exp(logits))
                                score = probs[0]
                                score_dict[(answer, result)].append(score)
                                print(f"score:\n{score}\n")

                        print(f"model output: \n{result}")
                        print(f"retrieved answer: \n{answer}")
                        answer_dict[answer].append(result)
                        print(f"answer {answer_count}: \n{answer}")
                        answer_count += 1
                    else:
                        none_count += 1

                    print("------------------------------")

            # Early stop if confidence is high enough
            """if len(answer_dict) == 0:  # no answer yet
                continue
            sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            max_len = len(sorted_answer_dict[0][1])
            if max_len / stop1 >= self.early_stop_threshold:
                if len(sorted_answer_dict) >= 4 and max_len == len(sorted_answer_dict[1][1]): # change from 2 to 4
                    pass  # Tie with the second best answer
                else:
                    break"""

        if len(answer_dict) == 0:
            print("Warning: no answer found")
            print("Output:", result)
            confidence, answer = (
                0,
                result,
            )  # No reasonable answer found. Fall back to choose the last response
        else:
            result_dict = defaultdict(float)

            for answer_tuple in score_dict:
                result_dict[answer_tuple] = (
                    np.mean(score_dict[answer_tuple])
                    + len(answer_dict[answer_tuple[0]]) / answer_count
                ) / 2  # test only divide confidence

            sorted_answer_dict = sorted(
                result_dict.items(), key=lambda p: p[1], reverse=True
            )
            for answer_tuple in sorted_answer_dict:
                print(answer_tuple)
            max_answer = sorted_answer_dict[0]
            answer = max_answer[0][
                1
            ]  # Here we simply choose the first appearance of the answer
            confidence = max_answer[1]

        state.append(
            SubResult(
                action,
                answer,
                confidence,
                list(answer_dict.keys()),
                list(answer_dict.values()),
            )
        )
        print(f"action: \n{action}\nanswer:\n{answer}\nconfidence:{confidence}\n")
        aux = {"confidence": confidence}
        print("********************************")
        return state, aux

    def is_terminal(self, state: MATHState) -> bool:
        if len(state) > 0 and "Now we can answer" in state[-1].sub_question:
            return True
        else:
            return False

    def get_step_outputs(self, outputs, state, action):
        score_inputs = []
        answer_dict = defaultdict(list)  # map from answer to list of thoughts
        score_dict = defaultdict(list)
        result = ""
        result_count = 0
        answer_count = 0
        none_count = 0
        new_output = []
        for output in outputs:
            result = output.strip()
            if "Now we can" in action:
                answer = self.utils.retrieve_answer(result)
            else:
                answer = self.utils.retrieve_answer_not_option(result)

            if answer is not None:
                if len(score_dict[(answer, result)]) < len(self.score_prompts):
                    for score_prompt_index in range(len(self.score_prompts)):
                        with io.StringIO() as f:
                            f.write(
                                self.score_prompts[score_prompt_index]["input"] + "\n\n"
                            )
                            f.write(
                                self.score_prompts[score_prompt_index][
                                    "question_prefix"
                                ]
                                + self.example
                                + "\n"
                            )
                            for idx, (q, a, *_) in enumerate(state):
                                f.write(
                                    self.score_prompts[score_prompt_index][
                                        "subquestion_prefix"
                                    ].format(idx + 1)
                                    + " "
                                    + q
                                    + "\n"
                                )
                                f.write(
                                    self.score_prompts[score_prompt_index][
                                        "subanswer_prefix"
                                    ].format(idx + 1)
                                    + " "
                                    + a
                                    + "\n"
                                )
                            f.write(
                                self.score_prompts[score_prompt_index][
                                    "subquestion_prefix"
                                ].format(len(state) + 1)
                                + " "
                                + action
                                + "\n"
                            )
                            f.write(
                                self.score_prompts[score_prompt_index][
                                    "new_subanswer_prefix"
                                ].format(len(state) + 1)
                                + " "
                                + result
                                + "\n"
                            )
                            f.write(
                                self.score_prompts[score_prompt_index]["score_prefix"]
                            )
                            score_input = f.getvalue()
                            score_inputs.append(score_input)
                            new_output.append(output)
        return score_inputs, "fast_reward", new_output

    def get_step_outputs_finnal(self, outputs, state, action, score_logits):
        answer_dict = defaultdict(list)  # map from answer to list of thoughts
        score_dict = defaultdict(list)
        result = ""
        result_count = 0
        answer_count = 0
        none_count = 0
        for i, output in enumerate(outputs):
            result = output.strip()
            result_count += 1
            if "Now we can" in action:
                answer = self.utils.retrieve_answer(result)
            else:
                answer = self.utils.retrieve_answer_not_option(result)
            if answer == None:
                continue
            if answer[-1] == ".":
                answer = answer[:-1]
            if answer is not None:
                if len(score_dict[(answer, result)]) < len(self.score_prompts):
                    for score_prompt_index in range(len(self.score_prompts)):
                        logits = score_logits[i]
                        probs = np.exp(logits) / np.sum(np.exp(logits))
                        score = probs[0]
                        score_dict[(answer, result)].append(score)
                        # print(f"score:\n{score}\n")

                # print(f"model output: \n{result}")
                # print(f"retrieved answer: \n{answer}")
                answer_dict[answer].append(result)
                # print(f"answer {answer_count}: \n{answer}")
                answer_count += 1
            else:
                none_count += 1

            # print("------------------------------")
        if len(answer_dict) == 0:
            print("Warning: no answer found")
            # print("Output:", result)
            confidence, answer = (
                0,
                result,
            )  # No reasonable answer found. Fall back to choose the last response
        else:
            result_dict = defaultdict(float)

            for answer_tuple in score_dict:
                result_dict[answer_tuple] = (
                    np.mean(score_dict[answer_tuple])
                    + len(answer_dict[answer_tuple[0]]) / answer_count
                ) / 2  # test only divide confidence

            sorted_answer_dict = sorted(
                result_dict.items(), key=lambda p: p[1], reverse=True
            )
            # for answer_tuple in sorted_answer_dict:
            #    print(answer_tuple)
            max_answer = sorted_answer_dict[0]
            answer = max_answer[0][
                1
            ]  # Here we simply choose the first appearance of the answer
            confidence = max_answer[1]

        state.append(
            SubResult(
                action,
                answer,
                confidence,
                list(answer_dict.keys()),
                list(answer_dict.values()),
            )
        )
        # print(f"action: \n{action}\nanswer:\n{answer}\nconfidence:{confidence}\n")
        aux = {"confidence": confidence}
        # print("********************************")
        return state, aux


from examples.RAP.prontoqa.dataset import ProntoQAExample
import sys


class ProntoQAState:
    def __init__(self, last_state, last_action, body):
        self.body = body
        self.last_state = last_state
        self.last_action = last_action

    def __str__(self) -> str:
        return self.body


ProntoQAAction = str

from examples.RAP.prontoqa import prompts
from examples.RAP.prontoqa.prompts import output, transition

prompts.output = output
prompts.transition = transition


class ProntoQAWorldModel(WorldModel[ProntoQAState, ProntoQAAction, ProntoQAExample]):
    def __init__(self, base_model: LanguageModel) -> None:
        super().__init__()
        self.base_model = base_model
        self.example: ProntoQAExample = self.example

    def init_state(self) -> ProntoQAState:
        *base_facts, init_state = self.example.test_example.question.split(". ")

        return ProntoQAState(body=init_state, last_state=None, last_action=None)

    def step(
        self, state: ProntoQAState, action: ProntoQAAction
    ) -> tuple[ProntoQAState, dict]:
        input_prompt = ""

        match action:
            case "Finish.":  # transition to terminal state
                input_prompt += prompts.output.EXAMPLES
                input_prompt += prompts.output.QUERY_FORMAT.format(
                    self.example.test_example.query
                )
                input_prompt += prompts.output.CLAIM_FORMAT.format(state)
                input_prompt += prompts.output.OUTPUT_PREFIX
                print("Reached terminal state.")

            case _:  # transition to non-terminal state
                input_prompt += prompts.transition.EXAMPLES
                input_prompt += prompts.transition.FACTS_FORMAT.format(state, action)
                input_prompt += prompts.transition.NEXT_CLAIM_PREFIX
                print("Reached non-terminal state.")

        output = (
            self.base_model.generate(
                [input_prompt], eos_token_id="\n", hide_input=True, temperature=0
            )
            .text[0]
            .strip()
        )

        print(input_prompt, file=sys.stderr, flush=True)
        print(f"S[{state}] A[{action}] -> S'[{output}]", flush=True)
        return ProntoQAState(body=output, last_state=state, last_action=action), {}

    def is_terminal(self, state: ProntoQAState) -> bool:
        return state.last_action == "Finish."

    def get_step_inputs(self, state, action):
        input_prompt = ""
        match action:
            case "Finish.":  # transition to terminal state
                input_prompt += prompts.output.EXAMPLES
                input_prompt += prompts.output.QUERY_FORMAT.format(
                    self.example.test_example.query
                )
                input_prompt += prompts.output.CLAIM_FORMAT.format(state)
                input_prompt += prompts.output.OUTPUT_PREFIX
                # print("Reached terminal state.")

            case _:  # transition to non-terminal state
                input_prompt += prompts.transition.EXAMPLES
                input_prompt += prompts.transition.FACTS_FORMAT.format(state, action)
                input_prompt += prompts.transition.NEXT_CLAIM_PREFIX
                # print("Reached non-terminal state.")
        return 1, state, input_prompt

    def get_step_outputs(self, outputs, state, action):
        # print(outputs)
        output = outputs[0].strip()
        # print(output)
        # print(input_prompt, file=sys.stderr, flush=True)
        # print(f"S[{state}] A[{action}] -> S'[{output}]", flush=True)
        return ProntoQAState(body=output, last_state=state, last_action=action), {}


StrategyQAState = list[SubResult]
StrategyQAAction = str


class StrategyQAPrompt(TypedDict):
    input: str
    question_prefix: str
    subquestion_prefix: str
    answer_prefix: str
    overall_question_prefix: str


# class StrategyQAWorldModel(WorldModel[StrategyQAState, StrategyQAAction]):
class StrategyQAWorldModel(GSM8kWorldModel):
    """
    strategyQA World Model
    State: [[sub_question_1, sub_answer_1, confidence_1], [sub_question_2, sub_answer_2, confidence_2], ...]
    Action: sub_question
    """

    def __init__(
        self,
        base_model: LanguageModel,
        prompt: dict,
        n_confidence=8,
        batch_size=2,
        temperature=0.8,
        eos_token_id="\n",
        early_stop_base=None,
        early_stop_threshold=1.0,
    ) -> None:
        super(GSM8kWorldModel, self).__init__()
        self.base_model = base_model
        self.prompt: StrategyQAPrompt = prompt
        self.batch_size = batch_size
        self.n_confidence = n_confidence
        self.temperature = temperature
        self.early_stop_base = (
            early_stop_base if early_stop_base is not None else n_confidence
        )
        self.early_stop_threshold = early_stop_threshold
        self.eos_token_id = eos_token_id
        from utils import strategyQA_utils

        self.utils = strategyQA_utils

    def init_state(self) -> list:
        return []

    def step(
        self, state: StrategyQAState, action: StrategyQAAction
    ) -> tuple[StrategyQAState, dict]:
        state = state.copy()

        with io.StringIO() as f:
            f.write(self.prompt["input"])
            f.write(self.prompt["question_prefix"] + self.example + "\n")
            for idx, (q, a, _) in enumerate(state):
                f.write(
                    self.prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n"
                )
                f.write(self.prompt["answer_prefix"].format(idx + 1) + " " + a + "\n")
            f.write(
                self.prompt["subquestion_prefix"].format(len(state) + 1)
                + " "
                + action
                + "\n"
            )
            f.write(self.prompt["answer_prefix"].format(len(state) + 1))
            model_input = f.getvalue()

        answer_dict = defaultdict(list)  # map from answer to list of thoughts
        result = ""
        # print(f'====\nsubanswer prompt: {model_input}\n====')
        for start1 in range(0, self.n_confidence, self.early_stop_base):
            stop1 = min(start1 + self.early_stop_base, self.n_confidence)

            for start in range(start1, stop1, self.batch_size):
                stop = min(start + self.batch_size, stop1)
                num = stop - start

                outputs = self.base_model.generate(
                    [model_input] * num,
                    hide_input=True,
                    do_sample=True,
                    temperature=self.temperature,
                    eos_token_id=self.eos_token_id,
                ).text
                for output in outputs:
                    result = output.strip()
                    # print(f"subanswer output: {result}")
                    answer = self.utils.retrieve_answer(result)
                    if answer is not None:
                        answer_dict[answer].append(result)
                    # print(f"subanswer output (extracted): {answer}")

            # Early stop if confidence is high enough
            if len(answer_dict) == 0:  # no answer yet
                continue
            sorted_answer_dict = sorted(
                answer_dict.items(), key=lambda p: len(p[1]), reverse=True
            )
            max_len = len(sorted_answer_dict[0][1])
            if max_len / stop1 >= self.early_stop_threshold:
                if len(sorted_answer_dict) >= 2 and max_len == len(
                    sorted_answer_dict[1][1]
                ):
                    pass  # Tie with the second best answer
                else:
                    break

        if len(answer_dict) == 0:
            confidence, answer = (
                0,
                result,
            )  # No reasonable answer found. Fall back to choose the last response
        else:
            sorted_answer_dict = sorted(
                answer_dict.items(), key=lambda p: len(p[1]), reverse=True
            )
            max_answer = sorted_answer_dict[0]
            max_answer_output_list = max_answer[1]
            max_len = len(max_answer_output_list)
            answer = max_answer_output_list[
                0
            ]  # Here we simply choose the first appearance of the answer
            confidence = max_len / sum(len(v) for v in answer_dict.values())

        state.append(SubResult(action, answer, confidence))
        # print(confidence)
        aux = {"confidence": confidence}
        return state, aux

    def get_step_inputs(
        self, state: GSM8kState, action: GSM8kAction
    ) -> tuple[GSM8kState, dict]:
        state = state.copy()

        with io.StringIO() as f:
            f.write(self.prompt["input"])
            f.write(self.prompt["question_prefix"] + self.example + "\n")
            for idx, t in enumerate(state):
                q, a = t[:2]
                f.write(
                    self.prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n"
                )
                f.write(self.prompt["answer_prefix"].format(idx + 1) + " " + a + "\n")
            f.write(
                self.prompt["subquestion_prefix"].format(len(state) + 1)
                + " "
                + action
                + "\n"
            )
            f.write(self.prompt["answer_prefix"].format(len(state) + 1))
            model_input = f.getvalue()
        num = 0
        for start1 in range(0, self.n_confidence, self.early_stop_base):
            stop1 = min(start1 + self.early_stop_base, self.n_confidence)

            for start in range(start1, stop1, self.batch_size):
                stop = min(start + self.batch_size, stop1)
                num += stop - start
        return num, state, model_input

    def is_terminal(self, state) -> bool:
        if len(state) > 0 and "Now we can answer" in state[-1].sub_question:
            return True
        else:
            ## try word match
            last_sub_words = set(state[-1].sub_question.lower().split(" "))
            overall_ques_words = set(self.example.lower().split(" "))
            new_words = last_sub_words - overall_ques_words
            if len(new_words) <= 1:
                return True
        return False

    def update_example(self, example: Example, prompt) -> None:
        return super(GSM8kWorldModel, self).update_example(example, prompt)


BWAction = str


class BWState(NamedTuple):
    """The state of the Blocksworld.

    See the docstring of BlocksWorldModel for more details.
    """

    step_idx: int
    last_blocks_state: str
    blocks_state: str
    buffered_action: BWAction


from copy import deepcopy


class BlocksWorldModel(WorldModel):
    """Blocks World World Model
    State: (step_idx, last_blocks_state, blocks_state, buffered_action)
    Action: e.g. "put the red block on the green block"
    Additional notes about the state:
        the block state is updated every two actions. When there is a block in hand,
        the block state is not updated, but the action is buffered. With the next action,
        the block state is updated and the buffer is cleared.
    """

    def __init__(
        self, base_model: LanguageModel, prompt: dict, max_steps: int = 6, batch_size=2
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size
        import reasoners.benchmark.bw_utils as utils

        self.utils = utils

    def init_state(self) -> BWState:
        """Initialize the world model.

        :return: the initial state
        """
        return BWState(
            step_idx=0,
            last_blocks_state="",
            blocks_state=self.utils.extract_init_state(self.example),
            buffered_action="",
        )

    def get_step_inputs(
        self, state: BWState, action: BWAction
    ) -> tuple[GSM8kState, dict]:
        state = deepcopy(state)
        buffered_action = state.buffered_action
        blocks_state = state.blocks_state
        step_idx = state.step_idx
        if "pick" in action:
            key = "world_update_pickup"
        elif "unstack" in action:
            key = "world_update_unstack"
        elif "put" in action:
            key = "world_update_putdown"
        elif "stack" in action:
            key = "world_update_stack"
        else:
            raise ValueError("Invalid action")
        world_update_prompt = self.prompt[key].format(
            blocks_state, action.capitalize() + "."
        )
        return 1, state, world_update_prompt

    def get_step_outputs(self, world_output, state, action):
        step_idx = state.step_idx
        buffered_action = state.buffered_action
        world_output = world_output[0]
        blocks_state = self.utils.apply_change(world_output, state.blocks_state)
        if state.buffered_action == "":
            # if no action buffered, buffer the action
            new_buffered_action = action
        else:
            # if action buffered, clear the buffer
            new_buffered_action = ""

        state = BWState(
            step_idx=step_idx + 1,
            last_blocks_state=state.blocks_state,
            blocks_state=blocks_state,
            buffered_action=new_buffered_action,
        )
        return state, {
            "goal_reached": self.utils.goal_check(
                self.utils.extract_goals(self.example), blocks_state
            )
        }

    def is_terminal(self, state: BWState) -> bool:
        if self.utils.goal_check(
            self.utils.extract_goals(self.example), state.blocks_state
        )[0]:
            return True
        elif state.step_idx == self.max_steps:
            return True
        return False

    def step(self, state: BWState, action: BWAction) -> tuple[BWState, dict]:
        pass
