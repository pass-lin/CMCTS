import math
from copy import deepcopy
from typing import Optional, NamedTuple, Callable, Hashable
import random
import numpy as np
from .. import State, Trace


class MiddleResult:
    def __init__(self):
        self.step_outputs = None
        self.action_outputs = None
        self.logits = None
        self.prompt = None
        self.questions = None

    def reset(self):
        self.step_outputs = None
        self.action_outputs = None
        self.logits = None
        self.prompt = None
        self.questions = None


class TOTNode:
    def __init__(
        self,
        state=None,
        action=None,
        parent=None,
        cum_prompt="",
        fast_reward: float = 0.0,
        fast_reward_details=None,
        calc_q=np.mean,
        prompt: str = "",
    ):
        """
        A node in the MCTS search tree

        :param state: the current state
        :param action: the action of the last step, i.e., the action from parent node to current node
        :param parent: the parent node, None if root of the tree
        :param fast_reward: an estimation of the reward of the last step
        :param is_terminal: whether the current state is a terminal state
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        """
        if fast_reward_details is None:
            fast_reward_details = {}
        self.cum_rewards: list[float] = []
        self.fast_reward = self.reward = fast_reward
        self.fast_reward_details = fast_reward_details
        self.action = action
        self.state = state
        self.parent = parent
        self.children: "Optional[list[TOTNode]]" = None
        self.calc_q = calc_q
        self.prompt = deepcopy(prompt)
        self.cum_prompt = deepcopy(cum_prompt)
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    def Q(self) -> float:
        if self.state is None:
            return self.fast_reward
        else:
            return self.calc_q(self.cum_rewards)

    def reward(self):
        return self.fast_reward

    def is_terminal(self):
        if "Now we can answer" in self.action:
            return True
        return False


def is_terminal(action):
    if "Now we can answer" in action:
        return True
    return False


class TOTResult(NamedTuple):
    terminal_state: State
    cum_reward: float
    trace: Trace
    trace_of_nodes: list[TOTNode]
    tree_state: TOTNode
    trace_in_each_iter: list[list[TOTNode]] = None
    tree_state_after_each_iter: list[TOTNode] = None
    aggregated_result: Optional[Hashable] = None


class TOT_BFS:
    def __init__(
        self,
        prompt: dict = None,
        useful_prompt: dict = None,
        n_action=2,
        output_trace_in_each_iter: bool = True,
        w_exp: float = 1.0,
        depth_limit: int = 3,
        n_iters: int = 3,
        cum_reward=np.sum,
        calc_q=np.mean,
        simulate_strategy="max",
        output_strategy: str = "max_reward",
        uct_with_fast_reward: bool = True,
        disable_tqdm: bool = True,
        max_child=4,
        num_shot=4,
        is_terminal_function=is_terminal,
        auto_generate_leaf_node=True,
    ):
        """
        MCTS algorithm

        :param output_trace_in_each_iter: whether to output the trace of the chosen trajectory in each iteration ; the trace is *deepcopy*-ed
                                          will also output *tree_state_after_each_iter*, which is the *deepcopy*-ed root
        :param w_exp: the weight of exploration in UCT
        :param cum_reward: the way to calculate the cumulative reward from each step. Defaults: sum
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        :param simulate_strategy: simulate strategy. Options: 'max', 'sample', 'random', or use a custom function
        :param output_strategy: the way to output the result. The nodes are not *deepcopy*-ed, so the information is after all iterations
                                Options: 'max_reward': dfs on the final tree to find a trajectory with max reward using :param cum_reward:
                                         'follow_max': starting from root, choose the maximum reward child at each step. May output a non-terminal node if dead end
                                         'max_visit': the terminal node with maximum number of visits
                                         'max_iter': the trajectory with a terminal node and max reward among those in each iteration
                                         'last_iter': the last trajectory. May output a non-terminal node if the last iteration leads to a dead end
                                         'last_terminal_iter': the last trajectory with a terminal node
                                Outputs *None* if no trajectory with terminal node but required
        :param uct_with_fast_reward: if True, use fast_reward instead of reward for unvisited children in UCT
                                     Otherwise, visit the *unvisited* children with maximum fast_reward first
        """
        super().__init__()
        self.world_model = None
        self.search_config = None
        self.output_trace_in_each_iter = output_trace_in_each_iter
        self.w_exp = w_exp
        self.is_terminal_function = is_terminal_function
        self.depth_limit = depth_limit
        self.n_iters = n_iters
        self.cum_reward = cum_reward
        self.calc_q = calc_q
        self.max_child = max_child
        self.num_shot = num_shot
        self.auto_generate_leaf_node = auto_generate_leaf_node
        default_simulate_strategies: dict[str, Callable[[list[float]], int]] = {
            "max": lambda x: np.argmax(x),
            "sample": lambda x: np.random.choice(len(x), p=x),
            "random": lambda x: np.random.choice(len(x)),
        }
        self.simulate_choice: Callable[[list[float]], int] = (
            default_simulate_strategies.get(simulate_strategy, simulate_strategy)
        )
        assert output_strategy in [
            "max_reward",
            "follow_max",
            "max_visit",
            "max_iter",
            "last_iter",
            "last_terminal_iter",
        ]
        self.output_strategy = output_strategy
        self.uct_with_fast_reward = uct_with_fast_reward
        self._output_iter: list[TOT_BFS] = None
        self._output_cum_reward = -math.inf
        self.trace_in_each_iter: list[list[TOT_BFS]] = None
        self.root: Optional[TOT_BFS] = None
        self.disable_tqdm = disable_tqdm
        self.n_action = n_action
        self.useful_prompt = deepcopy(useful_prompt)
        if "prefix" not in prompt.keys():
            prefix = ""
            for idx, t in enumerate(
                random.sample(prompt["interactive_examples"], num_shot)
            ):
                prefix += t.format(idx=idx + 1) + "\n\n"
            prompt["prefix"] = prefix
        self.prompt = deepcopy(prompt)
        self.depth = 0

    def initial(self, world_model, search_config):
        self.world_model = world_model
        self.search_config = search_config
        self._output_cum_reward = -math.inf
        self._output_iter = None
        self.trace_in_each_iter = []

        self.question = (
            self.search_config.question_prefix
            + " %d:" % (self.num_shot + 1)
            + world_model.example
            + " \n"
        )
        self.root = TOTNode(
            state=self.world_model.init_state(),
            action=None,
            parent=None,
            calc_q=self.calc_q,
            prompt=self.question
            + self.search_config.question_prefix
            + " %d.%d:" % (self.num_shot + 1, 1),
        )

    def get_question_inputs(self, nodes):
        model_inputs = []
        for node in nodes:
            model_inputs.extend(
                [self.prompt["prefix"] + node.cum_prompt + node.prompt] * self.n_action
            )
        return model_inputs

    def get_answer_inputs(self, nodes, TempResult, answer_id=1):
        model_inputs = []
        for node in nodes:
            for i in range(self.n_action):
                question = (
                    self.prompt["prefix"]
                    + node.cum_prompt
                    + node.prompt
                    + TempResult.questions[i]
                )
                answer_prefix = self.search_config.answer_prefix + " %d.%d:" % (
                    self.num_shot + 1,
                    self.depth + answer_id,
                )
                model_inputs.append(question + answer_prefix)
        return model_inputs

    def get_rewards_inputs(self, childs, cum_reward=True):
        model_inputs = []
        for i, child in enumerate(childs):
            if cum_reward:
                inputs = (
                    self.useful_prompt["input"]
                    + child.cum_prompt
                    + self.useful_prompt["useful_prefix"]
                )
            else:
                inputs = (
                    self.useful_prompt["input"]
                    + child.state
                    + self.useful_prompt["useful_prefix"]
                )
            model_inputs.append(inputs)
        return model_inputs

    def iterate(self, TempResult: MiddleResult = None, cum_reward=True, answer_id=1):
        self.results = []
        self.nodes = [self.root]
        for depth in range(self.depth_limit + 1):
            if len(self.nodes) == 0:
                break
            questions_prompt = self.get_question_inputs(self.nodes)
            if depth == self.depth_limit and self.auto_generate_leaf_node:
                if len(self.results) != 0:
                    break
                TempResult.questions = []
                question = self.search_config.get_finnal_question(self.question)
                for i in range(len(questions_prompt)):
                    TempResult.questions.append(question)
            else:
                yield questions_prompt, "get_question"
            answer_prompt = self.get_answer_inputs(self.nodes, TempResult, answer_id)
            parents, answer_prompt_set, actions = [], [], []
            for i, node in enumerate(self.nodes):
                for j in range(self.n_action):
                    if answer_prompt[i * self.n_action + j] not in answer_prompt_set:
                        parents.append(node)
                        actions.append(TempResult.questions[i * self.n_action + j])
                        answer_prompt_set.append(answer_prompt[i * self.n_action + j])
            yield answer_prompt_set, "step"
            self.depth += 1
            childs = []
            for i, prompt in enumerate(answer_prompt_set):
                child = TOTNode(
                    parent=parents[i],
                    cum_prompt=prompt.replace(self.prompt["prefix"], "")
                    + TempResult.step_outputs[i],
                    prompt=self.search_config.question_prefix
                    + " %d.%d:" % (self.num_shot + 1, self.depth + 1),
                    state=TempResult.step_outputs[i],
                    action=actions[i],
                )
                if self.is_terminal_function(child.action) or depth == self.depth_limit:
                    self.results.append(child)
                else:
                    childs.append(child)
            if len(childs) == 0:
                break
            yield self.get_rewards_inputs(childs, cum_reward), "fast_reward"
            rewards = np.exp(TempResult.logits) / np.sum(
                np.exp(TempResult.logits), axis=-1, keepdims=1
            )
            rewards = rewards[:, 0]
            for i, child in enumerate(childs):
                child.fast_reward = rewards[i]
            self.nodes = sorted(childs, key=TOTNode.reward, reverse=True)[
                : self.max_child
            ]
        yield None, "Search_End"

    def initial_prontoqa(self, world_model, search_config):
        self.world_model = world_model
        self.search_config = search_config
        self._output_cum_reward = 0
        self._output_iter = None
        self.trace_in_each_iter = []

        input_prompt = self.prompt["prefix"] + "\n\n"
        *base_facts, init_state = search_config.example.test_example.question.split(
            ". "
        )
        from examples.RAP.prontoqa.prompts import next_step

        input_prompt += next_step.FACTS_FORMAT.format(
            self.num_shot + 1, ". ".join(base_facts)
        )
        input_prompt += next_step.QUERY_FORMAT.format(
            self.num_shot + 1, search_config.example.test_example.query
        )
        self.name = world_model.example.test_example.query.split(" ")[3]
        self.prompt["prefix"] = input_prompt
        self.useful_prompt["input"] += (
            "Query: %s \nClaim: "
            % (world_model.example.test_example.query.split(":")[1])
        )
        self.question = "Claim %d.1: " % (self.num_shot + 1) + init_state + " \n"
        self.root = TOTNode(
            state=self.world_model.init_state(),
            action=None,
            parent=None,
            calc_q=self.calc_q,
            prompt=self.question
            + self.search_config.question_prefix
            + " %d.%d:" % (self.num_shot + 1, 1),
        )
