import numpy as np
from game import Agent, Action
import main_no_shiny

NO_SELECTION = -1
class ExpectimaxAgent(Agent):
    """
    Your expectimax agent
    """

    def __init__(self, evaluation_function=None, prioritize_score = True ,depth=2):
        super().__init__()
        self.evaluation_function = evaluation_function if evaluation_function else score_evaluation_function
        self.depth = depth
        self.prioritize_score = prioritize_score

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluation_function.
        Handles the minus atom only when prioritizing score.
        """
        best_value = -np.inf
        best_action = None

        # Get all legal actions for the agent (agent_index=0)
        legal_actions = game_state.get_legal_actions(agent_index=0)

        # Check if prioritizing score and the center atom is a minus atom
        if self.prioritize_score and game_state.ring.center_atom.symbol == "-":
            if game_state.pending_minus_action is None:
                # Step 1: Select the highest atom
                highest_atom_index = self.find_highest_atom_index(game_state)
                best_action = (Action.SELECT_ATOM_MINUS, highest_atom_index, NO_SELECTION)
                return best_action
            else:
                # Step 2: Convert the selected atom to plus and place it
                selected_atom = game_state.pending_minus_action
                for midway_index in range(len(game_state.ring.atoms)):
                    best_action = (Action.CONVERT_TO_PLUS, selected_atom, midway_index)
                return best_action

        # Standard expectimax for all other cases or when not prioritizing score
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
            legal_moves = state.get_legal_actions(agent_index=0)
            for action in legal_moves:
                successor = state.generate_successor(agent_index=0, action=action)
                value = max(value, self.expectiMax(successor, depth - 1, False))
            return value
        else:
            # Expectimax for the opponent's turn
            legal_moves = state.get_legal_actions(agent_index=1)
            scores = [self.expectiMax(state.generate_successor(agent_index=1, action=action), depth - 1, True) for
                      action in legal_moves]
            if scores:
                return np.mean(scores)
            return -np.inf

    def find_highest_atom_index(self, state):
        """
        Find the index of the highest atom in the ring.
        """
        atoms = state.ring.atoms
        highest_atom_index = None
        highest_atom_number = -np.inf

        for i, atom in enumerate(atoms):
            if atom.atom_number > highest_atom_number:
                highest_atom_number = atom.atom_number
                highest_atom_index = i

        return highest_atom_index



def score_evaluation_function(state):
    """
    Default evaluation function that returns a heuristic score of the game state.
    This could be based on the current score, the longest symmetric chain, and other factors.
    """
    # Prioritize the state's score heavily
    score = state._score * 10000

    # Extract information from the ring
    atoms = state.ring.atoms  # Get the list of atoms in the ring
    atom_weights = [atom.atom_number for atom in atoms]  # Get their weights

    # Assuming get_chains(atoms) returns a list of chains, where each chain is a list of consecutive similar atoms
    chains = get_chains(atoms)  # Get chains of consecutive similar atoms

    # Calculate the lengths of all chains
    chain_lengths = [len(chain) for chain in chains]

    # Determine the longest and second longest chain lengths
    if len(chain_lengths) > 0:
        chain_lengths.sort(reverse=True)
        longest_chain_length = chain_lengths[0]
        second_longest_chain_length = chain_lengths[1] if len(chain_lengths) > 1 else 0

        # Score for the longest chain (e.g., give it a weight of 1000)
        score += longest_chain_length * 1000

        # Score for the second longest chain (e.g., 10% of the longest chain's score)
        # score += second_longest_chain_length * 100

    # Apply step 2 logic only to the longest chain
    if len(chains) > 0:
        longest_chain = chains[chain_lengths.index(longest_chain_length)]  # Find the longest chain
        if len(longest_chain) > 1:  # Only consider chains with more than one atom
            mid = len(longest_chain) // 2
            left_side = [atom.atom_number for atom in longest_chain[:mid]]
            right_side = [atom.atom_number for atom in reversed(longest_chain[mid:])]

            # Check if left side is increasing and right side is decreasing
            if all(left_side[i] < left_side[i + 1] for i in range(len(left_side) - 1)) and \
               all(right_side[i] < right_side[i + 1] for i in range(len(right_side) - 1)):
                # Prioritize symmetric chains (increasing-decreasing pattern)
                # score += 500 * len(longest_chain)  # Higher score for longer symmetric chains

                # Bonus for smaller distances between adjacent atoms
                for i in range(len(left_side) - 1):
                    distance = abs(left_side[i + 1] - left_side[i])
                    score += max(0, 200 - (distance * 20))  # Bonus for smaller distances

                for i in range(len(right_side) - 1):
                    distance = abs(right_side[i + 1] - right_side[i])
                    score += max(0, 200 - (distance * 20))  # Bonus for smaller distances
            else:
                score -= 200 * len(longest_chain)  # Penalize for breaking the symmetric pattern

    # 3. Favor states where atoms of similar weights are adjacent, excluding special atoms
    # Only relevant if no chain can be formed; otherwise chains take priority
    if len(chains) == 0:
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

    # 4. Penalty for the number of atoms in the ring
    atom_count = len(atoms)
    penalty_for_atom_count = atom_count * 10000  # Apply a penalty per atom in the ring
    score -= penalty_for_atom_count

    return score




def highest_atom_evaluation_function(state):
    score = state.highest_atom * 10000

    # Extract information from the ring
    ring = state._ring
    atoms = state._ring.atoms  # Get the list of atoms in the ring
    atom_weights = [atom.atom_number for atom in atoms]  # Get their weights
    chains = get_chains(ring)  # Get chains of consecutive similar atoms


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


def get_chains(atoms):
    """
    Finds all even-length symmetric chains in a circular list of Atom objects.

    Args:
    atoms (list): List of Atom objects.

    Returns:
    list: A list of unique symmetric chains, where each chain is a list of Atom objects.
    """
    n = len(atoms)
    chains = []
    seen_chains = set()  # Track unique chains based on their atom numbers.

    # Loop over every possible center point for even-length chains
    for center in range(n):
        for half_length in range(1, n // 2 + 1):  # Only consider even-length chains
            is_symmetric = True
            chain = []

            # Check symmetry on both sides of the center
            for i in range(half_length):
                left_index = (center - i - 1) % n  # Wrap around left
                right_index = (center + i) % n  # Wrap around right

                if atoms[left_index].atom_number != atoms[right_index].atom_number:
                    is_symmetric = False
                    break

                # Insert atoms symmetrically
                chain.insert(0, atoms[left_index])
                chain.append(atoms[right_index])

            # If a symmetric chain is found and it's unique, collect it
            if is_symmetric:
                chain_key = tuple(atom.atom_number for atom in chain)

                # Ensure no subchains are added if a longer chain exists
                if chain_key not in seen_chains:
                    chains.append(chain)
                    seen_chains.add(chain_key)

    # Special case: Check if the entire ring is a symmetric chain
    if n % 2 == 0:  # Only consider even-length rings
        is_symmetric = True
        full_chain = []

        # Check symmetry across the entire ring
        for i in range(n // 2):
            if atoms[i].atom_number != atoms[(n - 1 - i) % n].atom_number:
                is_symmetric = False
                break
            full_chain.insert(0, atoms[i])
            full_chain.append(atoms[(n - 1 - i) % n])

        if is_symmetric:
            full_chain_key = tuple(atom.atom_number for atom in atoms)
            if full_chain_key not in seen_chains:
                chains.append(atoms[:])  # Add the full chain (entire ring) only once
                seen_chains.add(full_chain_key)

    # Check for two full-length chains and keep the one with the heavier center
    full_length_chains = [chain for chain in chains if len(chain) == n]
    if len(full_length_chains) == 2:
        # Get center atoms for both chains
        mid_idx1 = n // 2
        mid_idx2 = n // 2 - 1

        chain1_center = full_length_chains[0][mid_idx1].atom_number
        chain2_center = full_length_chains[1][mid_idx2].atom_number

        # Remove the chain with the lighter center atom
        if chain1_center < chain2_center:
            chains.remove(full_length_chains[0])
        else:
            chains.remove(full_length_chains[1])

    return chains

