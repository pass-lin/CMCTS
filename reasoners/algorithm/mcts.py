import math
from copy import deepcopy
from typing import Generic, Optional, NamedTuple, Callable, Hashable
import itertools
from abc import ABC
from collections import defaultdict

import numpy as np
from tqdm import trange

from .. import SearchAlgorithm, WorldModel, SearchConfig, State, Action, Example, Trace


class MCTSNode(Generic[State, Action, Example]):
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(
        self,
        state: Optional[State],
        action: Optional[Action],
        parent: "Optional[MCTSNode]" = None,
        fast_reward: float = 0.0,
        fast_reward_details=None,
        is_terminal: bool = False,
        calc_q: Callable[[list[float]], float] = np.mean,
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
        self.id = next(MCTSNode.id_iter)
        if fast_reward_details is None:
            fast_reward_details = {}
        self.cum_rewards: list[float] = []
        self.fast_reward = self.reward = fast_reward
        self.fast_reward_details = fast_reward_details
        self.is_terminal = is_terminal
        self.action = action
        self.state = state
        self.parent = parent
        self.children: "Optional[list[MCTSNode]]" = None
        self.calc_q = calc_q
        self.exec_code = None
        self.para_input = None
        self.para_output = None
        self.revise_flag = "initial"
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    # noinspection PyPep8Naming
    @property
    def Q(self) -> float:
        if self.state is None:
            return self.fast_reward
        else:
            return self.calc_q(self.cum_rewards)


class MCTSResult(NamedTuple):
    terminal_state: State
    cum_reward: float
    trace: Trace
    trace_of_nodes: list[MCTSNode]
    tree_state: MCTSNode
    trace_in_each_iter: list[list[MCTSNode]] = None
    tree_state_after_each_iter: list[MCTSNode] = None
    aggregated_result: Optional[Hashable] = None


class MCTSAggregation(Generic[State, Action, Example], ABC):
    def __init__(
        self, retrieve_answer: Callable[[State], Hashable], weight_policy: str = "edge"
    ):
        assert weight_policy in ["edge", "edge_inverse_depth", "uniform"]
        self.retrieve_answer = retrieve_answer
        self.weight_policy = weight_policy

    def __call__(
        self, tree_state: MCTSNode[State, Action, Example]
    ) -> Optional[Hashable]:
        answer_dict = defaultdict(lambda: 0)

        def visit(cur: MCTSNode[State, Action, Example]):
            if cur.state is None:
                return []
            if cur.is_terminal:
                answer = self.retrieve_answer(cur.state)
                if answer is None:
                    print("MCTSAggregation: no answer retrieved.")
                    return []
                if self.weight_policy == "edge":
                    answer_dict[answer] += cur.reward
                elif self.weight_policy == "edge_inverse_depth":
                    answer_dict[answer] += cur.reward / cur.depth
                elif self.weight_policy == "uniform":
                    answer_dict[answer] += 1.0
                return [(answer, cur.depth)]
            depth_list = defaultdict(list)
            cur_list = []
            for child in cur.children:
                cur_list.extend(child_info := visit(child))
                for answer, depth in child_info:
                    depth_list[answer].append(depth)
            for answer, depths in depth_list.items():
                if self.weight_policy == "edge":
                    answer_dict[answer] += cur.reward
                elif self.weight_policy == "edge_inverse_depth":
                    answer_dict[answer] += cur.reward / np.mean(depths)
            return cur_list

        visit(tree_state)

        if len(answer_dict) == 0:
            return None
        return max(answer_dict, key=lambda answer: answer_dict[answer])


class MiddleResult:
    def __init__(self):
        self.step_outputs = None
        self.action_outputs = None
        self.logits = None
        self.action_prompt = None
        self.reward_prompt = None
        self.step_prompt = None
        self.questions = None
        self.exec_code = None
        self.para_input = None
        self.para_output = None
        self.prompt = None
        self.revise_result = []

    def reset(self):
        super().__init__()


class MCTS(SearchAlgorithm, Generic[State, Action, Example]):
    def __init__(
        self,
        output_trace_in_each_iter: bool = True,
        w_exp: float = 1.0,
        depth_limit: int = 5,
        n_iters: int = 3,
        cum_reward=np.sum,
        calc_q: Callable[[list[float]], float] = np.mean,
        simulate_strategy: str | Callable[[list[float]], int] = "max",
        output_strategy: str = "max_reward",
        uct_with_fast_reward: bool = False,
        revise_function=None,
        question: str = "",
        aggregator: Optional[MCTSAggregation] = None,
        action_mode=1,
        disable_tqdm: bool = True,
        auto_generate_leaf_node=False,
        rule_action=False,
        prompt_reward=True,
        node_visualizer: Callable[[MCTSNode], dict] = lambda x: x.__dict__,
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
        self.auto_generate_leaf_node = auto_generate_leaf_node
        self.search_config = None
        self.output_trace_in_each_iter = output_trace_in_each_iter
        self.w_exp = w_exp
        self.action_mode = action_mode
        self.depth_limit = depth_limit
        self.n_iters = n_iters
        self.prompt_reward = prompt_reward
        self.cum_reward = cum_reward
        self.calc_q = calc_q
        self.rule_action = rule_action
        self.revise_results = []
        self.question = question
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
        self._output_iter: list[MCTSNode] = None
        self._output_cum_reward = -math.inf
        self.trace_in_each_iter: list[list[MCTSNode]] = None
        self.root: Optional[MCTSNode] = None
        self.disable_tqdm = disable_tqdm
        self.node_visualizer = node_visualizer
        self.aggregator = aggregator
        self.node_visualizer = node_visualizer
        self.aggregator = aggregator
        self.revise_function = revise_function

    def iterate(self, node: MCTSNode) -> list[MCTSNode]:
        path = self._select(node)
        if not self._is_terminal_with_depth_limit(path[-1]):
            self._expand(path[-1])
            self._simulate(path)
        cum_reward = self._back_propagate(path)
        if (
            self.output_strategy == "max_iter"
            and path[-1].is_terminal
            and cum_reward > self._output_cum_reward
        ):
            self._output_cum_reward = cum_reward
            self._output_iter = path
        if self.output_strategy == "last_iter":
            self._output_cum_reward = cum_reward
            self._output_iter = path
        if self.output_strategy == "last_terminal_iter" and path[-1].is_terminal:
            self._output_cum_reward = cum_reward
            self._output_iter = path
        return path

    def is_no_search_space(self):
        path = self._select(self.root)
        return self._is_terminal_with_depth_limit(path[-1])

    def parallel_iterate(self, TempResult: MiddleResult = None):
        while True:
            TempResult.reset()
            path = self._select(self.root)
            if not self._is_terminal_with_depth_limit(path[-1]):
                expand = self.parallel_expand(path[-1], TempResult)
                while True:
                    model_inputs, state = next(expand)
                    if state == "END":
                        break
                    yield model_inputs, state
                simulate = self.parallel_simulate(path, TempResult)
                while True:
                    model_inputs, state = next(simulate)
                    if state == "END_simulate":
                        break
                    yield model_inputs, state
            if self.revise_function != None:
                if len(model_inputs[-1].state) > 1:
                    new_model_inputs, state = self.revise_function(
                        model_inputs[-1].state
                    )
                    yield new_model_inputs, state
                yield [model_inputs, TempResult.revise_result], "Search_End"
            else:
                yield model_inputs, "Search_End"

    def get_cum_reward(self, path, alpha=1):
        cum_reward = self._back_propagate(path, alpha=alpha)
        if (
            self.output_strategy == "max_iter"
            and path[-1].is_terminal
            and cum_reward > self._output_cum_reward
        ):
            self._output_cum_reward = cum_reward
            self._output_iter = path
        if self.output_strategy == "last_iter":
            self._output_cum_reward = cum_reward
            self._output_iter = path
        if self.output_strategy == "last_terminal_iter" and path[-1].is_terminal:
            self._output_cum_reward = cum_reward
            self._output_iter = path

    def _is_terminal_with_depth_limit(self, node: MCTSNode):
        return node.is_terminal or node.depth >= self.depth_limit

    def _select(self, node: MCTSNode) -> list[MCTSNode]:
        path = []

        while True:
            path.append(node)
            if (
                node.children is None
                or len(node.children) == 0
                or self._is_terminal_with_depth_limit(node)
            ):
                return path
            parent = node
            node = self._uct_select(parent)
            if parent.revise_flag == "activate":
                parent.revise_flag = "used"
                path.append(node)
                return path

    def _uct(self, node: MCTSNode) -> float:
        return node.Q + self.w_exp * np.sqrt(
            np.log(len(node.parent.cum_rewards)) / max(1, len(node.cum_rewards))
        )

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        if self.uct_with_fast_reward or all(x.state is not None for x in node.children):
            return max(node.children, key=self._uct)
        else:
            unvisited_children = filter(lambda x: x.state is None, node.children)
            return max(unvisited_children, key=lambda x: x.fast_reward)

    def search_state(self, node, action, TempResult: MiddleResult):
        num, state, model_input = self.world_model.get_step_inputs(
            node.parent.state, action
        )
        yield [model_input] * num, "step"
        outputs = self.world_model.get_step_outputs(
            TempResult.step_outputs, state, node.action, TempResult
        )
        if outputs[1] == "fast_reward":
            yield outputs[:2]
            TempResult.step_outputs = outputs[-1]
            outputs = self.world_model.get_step_outputs_finnal(
                TempResult.step_outputs,
                state,
                node.action,
                TempResult.logits,
                TempResult,
            )
        node.state, aux = outputs
        node.reward, node.reward_details = self.search_config.reward(
            node.parent.state, node.action, **node.fast_reward_details, **aux
        )
        node.is_terminal = self.world_model.is_terminal(node.state)
        yield None, "END"

    def parallel_expand(
        self, node: MCTSNode, TempResult: MiddleResult = None, action_inputs=None
    ):
        if node.state is None:
            search_state_func = self.search_state(node, node.action, TempResult)
            modeld_input, state = next(search_state_func)
            while state != "END":
                yield modeld_input, state
                modeld_input, state = next(search_state_func)
        if node.is_terminal and action_inputs is None:
            yield None, "END"

        children = []
        if self.rule_action:
            actions = self.search_config.get_actions(node.state)
        else:
            if self.action_mode == 1:
                n_samples, model_input, at_depth_limit = (
                    self.search_config.get_actions_inputs(node.state)
                )
            elif self.action_mode == 2:
                n_samples, model_input, at_depth_limit = (
                    self.search_config.get_actions_inputs(node.state, node.action)
                )
            else:
                raise ("not support this mode")
            yield model_input if action_inputs is None else action_inputs, "get_action"
            actions = self.search_config.get_actions_output(
                TempResult.action_outputs, at_depth_limit
            )

        reward_inputs = []
        flag = False
        for action in actions:
            reward_input = self.search_config.get_fast_reward_input(node.state, action)
            if isinstance(reward_input, list) and not isinstance(reward_input[0], dict):
                reward_inputs.extend(reward_input)
                flag = True
            else:
                reward_inputs.append(reward_input)
        yield reward_inputs, "fast_reward"

        for i in range(len(actions)):
            if flag:
                num = len(TempResult.logits) // len(actions)
                fast_reward, fast_reward_details = (
                    self.search_config.get_fast_reward_output(
                        TempResult.logits[i * num : (i + 1) * num]
                    )
                )
            else:
                fast_reward, fast_reward_details = (
                    self.search_config.get_fast_reward_output(TempResult.logits[i])
                )
            child = MCTSNode(
                state=None,
                action=actions[i],
                parent=node,
                fast_reward=fast_reward,
                fast_reward_details=fast_reward_details,
                calc_q=self.calc_q,
            )
            children.append(child)
        node.children = children
        yield None, "END"

    def _expand(self, node: MCTSNode):
        if node.state is None:
            node.state, aux = self.world_model.step(node.parent.state, node.action)
            # reward is calculated after the state is updated, so that the
            # information can be cached and passed from the world model
            # to the reward function with **aux without repetitive computation
            node.reward, node.reward_details = self.search_config.reward(
                node.parent.state, node.action, **node.fast_reward_details, **aux
            )
            node.is_terminal = self.world_model.is_terminal(node.state)

        if node.is_terminal:
            return

        children = []
        actions = self.search_config.get_actions(node.state)
        for action in actions:
            fast_reward, fast_reward_details = self.search_config.fast_reward(
                node.state, action
            )
            child = MCTSNode(
                state=None,
                action=action,
                parent=node,
                fast_reward=fast_reward,
                fast_reward_details=fast_reward_details,
                calc_q=self.calc_q,
            )
            children.append(child)

        node.children = children

    def parallel_simulate(self, path: list[MCTSNode], TempResult: MiddleResult = None):
        node = path[-1]
        flag = False
        while True:
            if node.state is None:
                expand = self.parallel_expand(node, TempResult)
                while True:
                    model_inputs, state = next(expand)
                    if state == "END":
                        break
                    yield model_inputs, state
            if self._is_terminal_with_depth_limit(node) or len(node.children) == 0:
                action_inputs = None
                if self.auto_generate_leaf_node:
                    if not node.is_terminal and self._is_terminal_with_depth_limit(
                        node
                    ):
                        node.state = None
                        node.action = self.search_config.get_finnal_question()

                if node.state is None:
                    search_state_func = self.search_state(node, node.action, TempResult)
                    modeld_input, state = next(search_state_func)
                    while state != "END":
                        yield modeld_input, state
                        modeld_input, state = next(search_state_func)

                yield path, "END_simulate"
            fast_rewards = [child.fast_reward for child in node.children]
            node = node.children[self.simulate_choice(fast_rewards)]
            path.append(node)

    def _simulate(self, path: list[MCTSNode]):
        node = path[-1]
        while True:
            if node.state is None:
                self._expand(node)
            if self._is_terminal_with_depth_limit(node) or len(node.children) == 0:
                return
            fast_rewards = [child.fast_reward for child in node.children]
            node = node.children[self.simulate_choice(fast_rewards)]
            path.append(node)

    def _back_propagate(self, path: list[MCTSNode], alpha=1):
        rewards = []
        cum_reward = -math.inf
        for node in reversed(path):
            rewards.append(node.reward)
            cum_reward = self.cum_reward(rewards[::-1])
            node.cum_rewards.append(cum_reward * alpha)
        return cum_reward

    def _dfs_max_reward(self, path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
        cur = path[-1]
        if cur.is_terminal:
            return self.cum_reward([node.reward for node in path[1:]]), path
        if cur.children is None:
            return -math.inf, path
        visited_children = [x for x in cur.children if x.state is not None]
        if len(visited_children) == 0:
            return -math.inf, path
        return max(
            (self._dfs_max_reward(path + [child]) for child in visited_children),
            key=lambda x: x[0],
        )

    def initial(self, world_model, search_config):
        self.world_model = world_model
        self.search_config = search_config
        self._output_cum_reward = -math.inf
        self._output_iter = None
        self.trace_in_each_iter = []
        self.root = MCTSNode(
            state=self.world_model.init_state(),
            action=None,
            parent=None,
            calc_q=self.calc_q,
        )

    def get_output(self):
        if self.output_strategy == "follow_max":
            self._output_iter = []
            cur = self.root
            while True:
                self._output_iter.append(cur)
                if cur.is_terminal:
                    break
                visited_children = [x for x in cur.children if x.state is not None]
                if len(visited_children) == 0:
                    break
                cur = max(visited_children, key=lambda x: x.reward)
            self._output_cum_reward = self.cum_reward(
                [node.reward for node in self._output_iter[1::-1]]
            )
        if self.output_strategy == "max_reward":
            self._output_cum_reward, self._output_iter = self._dfs_max_reward(
                [self.root]
            )
            if self._output_cum_reward == -math.inf:
                self._output_iter = None
        if self._output_iter is None:
            terminal_state = trace = None
        else:
            terminal_state = self._output_iter[-1].state
            trace = (
                [node.state for node in self._output_iter],
                [node.action for node in self._output_iter[1:]],
            )
        if self.output_trace_in_each_iter:
            trace_in_each_iter = self.trace_in_each_iter
            tree_state_after_each_iter = [trace[0] for trace in trace_in_each_iter]
        else:
            trace_in_each_iter = tree_state_after_each_iter = None
        result = MCTSResult(
            terminal_state=terminal_state,
            cum_reward=self._output_cum_reward,
            trace=trace,
            trace_of_nodes=self._output_iter,
            tree_state=self.root,
            trace_in_each_iter=trace_in_each_iter,
            tree_state_after_each_iter=tree_state_after_each_iter,
        )
        if self.aggregator is not None:
            result = MCTSResult(
                terminal_state=result.terminal_state,
                cum_reward=result.cum_reward,
                trace=result.trace,
                trace_of_nodes=result.trace_of_nodes,
                tree_state=result.tree_state,
                trace_in_each_iter=result.trace_in_each_iter,
                tree_state_after_each_iter=result.tree_state_after_each_iter,
                aggregated_result=self.aggregator(result.tree_state),
            )
        return result

    def search(self):
        self._output_cum_reward = -math.inf
        self._output_iter = None
        self.root = MCTSNode(
            state=self.world_model.init_state(),
            action=None,
            parent=None,
            calc_q=self.calc_q,
        )
        if self.output_trace_in_each_iter:
            self.trace_in_each_iter = []

        for _ in trange(
            self.n_iters, disable=self.disable_tqdm, desc="MCTS iteration", leave=False
        ):
            path = self.iterate(self.root)
            if self.output_trace_in_each_iter:
                self.trace_in_each_iter.append(deepcopy(path))

        if self.output_strategy == "follow_max":
            self._output_iter = []
            cur = self.root
            while True:
                self._output_iter.append(cur)
                if cur.is_terminal:
                    break
                visited_children = [x for x in cur.children if x.state is not None]
                if len(visited_children) == 0:
                    break
                cur = max(visited_children, key=lambda x: x.reward)
            self._output_cum_reward = self.cum_reward(
                [node.reward for node in self._output_iter[1::-1]]
            )
        if self.output_strategy == "max_reward":
            self._output_cum_reward, self._output_iter = self._dfs_max_reward(
                [self.root]
            )
            if self._output_cum_reward == -math.inf:
                self._output_iter = None

    def __call__(
        self,
        world_model: WorldModel[State, Action, Example],
        search_config: SearchConfig[State, Action, Example],
        log_file: Optional[str] = None,
        **kwargs,
    ) -> MCTSResult:
        MCTSNode.reset_id()
        self.world_model = world_model
        self.search_config = search_config

        self.search()

        if self._output_iter is None:
            terminal_state = trace = None
        else:
            terminal_state = self._output_iter[-1].state
            trace = (
                [node.state for node in self._output_iter],
                [node.action for node in self._output_iter[1:]],
            )
        if self.output_trace_in_each_iter:
            trace_in_each_iter = self.trace_in_each_iter
            tree_state_after_each_iter = [trace[0] for trace in trace_in_each_iter]
        else:
            trace_in_each_iter = tree_state_after_each_iter = None
        result = MCTSResult(
            terminal_state=terminal_state,
            cum_reward=self._output_cum_reward,
            trace=trace,
            trace_of_nodes=self._output_iter,
            tree_state=self.root,
            trace_in_each_iter=trace_in_each_iter,
            tree_state_after_each_iter=tree_state_after_each_iter,
        )
        if self.aggregator is not None:
            result = MCTSResult(
                terminal_state=result.terminal_state,
                cum_reward=result.cum_reward,
                trace=result.trace,
                trace_of_nodes=result.trace_of_nodes,
                tree_state=result.tree_state,
                trace_in_each_iter=result.trace_in_each_iter,
                tree_state_after_each_iter=result.tree_state_after_each_iter,
                aggregated_result=self.aggregator(result.tree_state),
            )
        return result
