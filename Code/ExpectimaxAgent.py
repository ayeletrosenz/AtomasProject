import numpy as np
from game import Agent, Action
import main_no_shiny


class ExpectimaxAgent(Agent):
    """
    Your expectimax agent
    """

    def __init__(self, evaluation_function=None, depth=2):
        super().__init__()
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
            legal_moves = state.get_legal_actions(agent_index=0)  # Assuming legal actions for agent
            for action in legal_moves:
                successor = state.generate_successor(agent_index=0, action=action)
                value = max(value, self.expectiMax(successor, depth - 1, False))
            return value
        else:
            legal_moves = state.get_legal_actions(agent_index=1)  # Assuming legal actions for opponent
            scores = [self.expectiMax(state.generate_successor(agent_index=1, action=action), depth - 1, True) for action in legal_moves]
            if scores:
                return np.mean(scores)
            return -np.inf

    def default_evaluation_function(self, state):
        """
        Default evaluation function that returns a heuristic score of the game state.
        This could be based on the current score, the highest atom, and other factors.
        """
        # Prioritize the state's score heavily
        score = state._score * 10000

        # Extract information from the ring
        atoms = state._ring.atoms  # Get the list of atoms in the ring
        atom_weights = [atom.atom_number for atom in atoms]  # Get their weights
        chains = self.get_chains(atoms)  # Get chains of consecutive similar atoms

        # Calculate the lengths of all chains
        chain_lengths = [len(chain) for chain in chains]

        # Determine the longest and second longest chain lengths
        if len(chain_lengths) > 0:
            chain_lengths.sort(reverse=True)
            longest_chain_length = chain_lengths[0]
            # Use 0 if there's no second chain to avoid indexing errors
            second_longest_chain_length = chain_lengths[1] if len(chain_lengths) > 1 else 0

            # Score for the longest chain (e.g., give it a weight of 1000)
            score += longest_chain_length * 1000

            # Score for the second longest chain (e.g., 10% of the longest chain's score)
            score += second_longest_chain_length * 100

        # 2. Favor states where atoms of similar weights are adjacent, excluding special atoms
        for i in range(len(atoms) - 1):
            atom_current = atoms[i]
            atom_next = atoms[i + 1]
            if not atom_current.special and not atom_next.special:  # Only consider non-special atoms
                if abs(atom_current.atom_number - atom_next.atom_number) <= 1:
                    score += 200  # Bonus for adjacent similar atoms
                elif abs(atom_current.atom_number - atom_next.atom_number) <= 2:
                    score += 100  # Bonus for maintaining light-to-heavy order
                else:
                    score -= 50  # Penalize bad arrangements

        # 3. Penalty for the number of atoms in the ring
        atom_count = len(atoms)
        penalty_for_atom_count = atom_count * 10000  # Apply a penalty per atom in the ring
        score -= penalty_for_atom_count

        return score

    def get_chains(self, atoms):
        """
        Finds all symmetric chains in the list of atoms, considering the circular nature of the ring.

        Args:
        atoms (list): List of Atom objects.

        Returns:
        list: A list of symmetric chains, where each chain is a list of Atom objects.
        """
        chains = []
        n = len(atoms)

        if n < 2:
            return chains  # No chains possible if fewer than 2 atoms

        # Extend the list to handle the circular nature
        extended_atoms = atoms + atoms

        # Check for symmetric chains
        for center in range(n):

            # Even-length symmetric chains centered between two atoms
            left, right = center, center + 1
            while left >= 0 and \
                    right < n + center and extended_atoms[left].atom_number == extended_atoms[right].atom_number:
                if extended_atoms[left].special or extended_atoms[right].special:  # No special atoms allowed in the chain
                    break
                if (right - left + 1) >= 2:  # At least two atoms form a chain
                    if left >= center and right < center + n:
                        chains.append(extended_atoms[left:right + 1])
                left -= 1
                right += 1

        return chains

