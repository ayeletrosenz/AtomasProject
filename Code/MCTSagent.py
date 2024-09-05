from game import Agent
import random
import math


class MCTSAgent(Agent):
    def __init__(self, simulations=1000):
        super(MCTSAgent, self).__init__()
        self.simulations = simulations

    def get_action(self, game_state):
        root = Node(game_state)

        for _ in range(self.simulations):
            # print(f"simulation {_}\n")
            node = self._select(root)
            # print("selected\n")
            reward = self._simulate(node.state)
            # print("simulated\n")
            self._backpropagate(node, reward)

        return self._best_child(root, exploration_constant=0).action

    def _select(self, node):
        while not node.state.done:
            if not node.is_fully_expanded():
                return self._expand(node)
            else:
                node = self._best_child(node)
        return node

    def _expand(self, node):
        # print("expanding...\n")
        actions = node.state.get_legal_actions(agent_index=0)
        for action in actions:
            # print("new action\n")
            if action not in [child.action for child in node.children]:
                next_state = node.state.generate_successor(agent_index=0, action=action)
                child_node = Node(next_state, parent=node, action=action)
                # print("create new child node\n")
                node.children.append(child_node)
                return child_node
        raise Exception("Should never reach here")

    def _simulate(self, state):
        current_state = state
        # print("simulating...\n")
        # print(f"current state: {current_state}\n")
        for _ in range(10):
            legal_actions = current_state.get_legal_actions(agent_index=0)
            # print(f"legal actions: {legal_actions}\n")
            action = random.choice(legal_actions)
            # print(f"action: {action}\n")
            current_state = current_state.generate_successor(agent_index=0, action=action)
        return current_state.score

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def _best_child(self, node, exploration_constant=1.414):
        best_value = float('-inf')
        best_nodes = []
        for child in node.children:
            uct_value = (child.reward / child.visits) + exploration_constant * math.sqrt(math.log(node.visits) / child.visits)
            if uct_value > best_value:
                best_value = uct_value
                best_nodes = [child]
            elif uct_value == best_value:
                best_nodes.append(child)
        return random.choice(best_nodes)


class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0
        self.action = action
        # print("Node created\n")

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions(agent_index=0))

