import numpy as np


class ExpectimaxAgent():
    """
    Your expectimax agent
    """
    def __init__(self, evaluation_function=None, depth=2):
        self.evaluation_function = evaluation_function if evaluation_function else self.default_evaluation_function
        self.depth = depth

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluation_function

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        best_value = -np.inf
        best_action = None

        # Get all legal actions for the agent (agent_index=0)
        legal_actions = game_state.get_legal_actions(agent_index=0)

        for action in legal_actions:
            # Generate the successor game state for each action
            successor = game_state.generate_successor(agent_index=0, action=action)

            # Evaluate the successor state using the expectimax function
            value = self.expectiMax(successor, self.depth, maximizing_player=False)

            # Update the best action if a better value is found
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def expectiMax(self, state, depth, maximizing_player):
        if depth == 0 or state.done:
            return self.evaluation_function(state)
        if maximizing_player:
            value = -np.inf
            legal_moves = state.get_agent_legal_actions()
            for action in legal_moves:
                successor = state.generate_successor(agent_index=0, action=action)
                value = max(value, self.expectiMax(successor, depth - 1, False))
            return value
        else:
            legal_moves = state.get_opponent_legal_actions()
            scores = [self.expectiMax(state.generate_successor(agent_index=1, action=action), depth - 1, True) for action in legal_moves]
            if scores:
                return np.mean(scores)
            return -np.inf

    def default_evaluation_function(self, state):
        """
        Default evaluation function that returns a heuristic score of the game state.
        This could be based on the current score, the highest atom, and other factors.
        """
        score = state.score
        highest_atom = state.highest_atom

        # A simple heuristic: higher score and higher atom are better
        return score + highest_atom * 100

