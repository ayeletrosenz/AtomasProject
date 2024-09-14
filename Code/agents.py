import math
import random
from typing import Tuple, List
import numpy as np
import abc
from game import Agent, Action
from config import PLUS, MINUS, NO_SELECTION


def choose_random(ring) -> int:
    return random.randint(0, ring.atom_count - 1)


def can_switch_to_plus(game_state):
    return game_state.pending_minus_action is not None


def lowest_atom(ring) -> int:
    lowest_atom_number = float('inf')
    lowest_index = -1
    for index, atom in enumerate(ring.atoms):
        if atom.atom_number != PLUS and atom.atom_number < lowest_atom_number:
            lowest_atom_number = atom.atom_number
            lowest_index = index
    return lowest_atom_number, lowest_index


def second_highest_atom(ring) -> Tuple[int, int]:
    atom_vals = [atom.atom_number for atom in ring.atoms]
    max_val = max(atom_vals)
    second_highest_val = max([val for val in atom_vals if val != max_val])
    second_highest_index = atom_vals.index(second_highest_val)
    return second_highest_val, second_highest_index


def find_symmetry_indices(atoms, pivot) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    atom_list = [atom.atom_number for atom in atoms]
    atom_indices = [i for i in range(len(atom_list))]
    n = len(atom_list)

    atom_list.insert(pivot, "p")
    atom_indices.insert(pivot, "p")
    n = len(atom_list)

    roll = np.roll(atom_list, n // 2 - pivot)
    roll = [str(i) if i.isalpha() else int(i) for i in roll]
    roll_indices = np.roll(atom_indices, n // 2 - pivot)
    roll_indices = [str(i) if i.isalpha() else int(i) for i in roll_indices]

    sym_numbers = []
    sym_indices = []

    for i in range(1, n // 2):
        if (roll[n // 2 - i] == roll[n // 2 + i]) and (roll[n // 2 - i] >= 0) and (roll[n // 2 + i] >= 0):
            sym_numbers.append((roll[n // 2 - i], roll[n // 2 + i]))
            sym_indices.append((roll_indices[n // 2 - i], roll_indices[n // 2 + i]))
        else:
            break
    return sym_numbers, sym_indices


def calculate_chain_length(ring, i: int) -> int:
    atoms = ring.atoms
    sym_numbers, sym_indices = find_symmetry_indices(atoms, i)
    chain_length = len(sym_indices) * 2 + 1 if sym_indices else 1
    return chain_length


def find_longest_chain(ring) -> Tuple[int, int]:
    longest_chain_length = 0
    longest_chain_midway = None
    atom_count = ring.atom_count

    for i in range(atom_count):
        chain_length = calculate_chain_length(ring, i)
        if chain_length > longest_chain_length:
            longest_chain_length = chain_length
            longest_chain_midway = i

    return longest_chain_midway - 1, longest_chain_length


def find_midway_next_to_plus(ring) -> int:
    atom_count = ring.atom_count
    center_atom = ring.center_atom.atom_number

    for i in range(atom_count):
        if ring.atoms[i].atom_number == PLUS:
            if ring.atoms[i - 1].atom_number == center_atom:
                return i
            elif ring.atoms[(i + 1) % atom_count].atom_number == center_atom:
                return i - 1
    return None


def find_bigger_chain_midway(ring) -> int:
    atom_count = ring.atom_count

    for midway_index in range(atom_count):
        simulated_ring = ring.copy()
        simulated_ring.place_atom(NO_SELECTION, midway_index, False)
        if find_longest_chain(simulated_ring)[1] > find_longest_chain(ring)[1]:
            return midway_index


def find_best_placement(ring) -> int:
    closest_index = None
    closest_difference = float('inf')
    atom_count = ring.atom_count
    center_atom = ring.center_atom.atom_number

    for i in range(atom_count):
        cur_atom = ring.atoms[i].atom_number
        if cur_atom > 0 and cur_atom <= center_atom and (
                center_atom - cur_atom) <= closest_difference:
            closest_difference = center_atom - cur_atom
            closest_index = i

    return closest_index if closest_index is not None else atom_count - 1


def exists_chain_of_length(ring, length: int) -> bool:
    atom_count = ring.atom_count
    for i in range(atom_count):
        # Calculate the chain length starting at index `i`
        chain_length = calculate_chain_length(ring, i)
        if chain_length >= length:
            return True
    return False

class RandomAgent(Agent):
    def get_action(self, game_state) -> Tuple[Action, int, int]:
        ring = game_state.ring
        center_atom = ring.center_atom.atom_number

        if center_atom == MINUS:
            return Action.SELECT_ATOM_MINUS, choose_random(ring), NO_SELECTION
        else:
            return Action.PLACE_ATOM, NO_SELECTION, choose_random(ring)
        
class AyeletAgent(Agent):
    def get_action(self, game_state) -> Tuple[Action, int, int]:
        ring = game_state.ring
        atom_count = ring.atom_count
        center_atom = ring.center_atom.atom_number
        total_turns = ring.total_turns
        min_spawning_atom = 1 + total_turns // 40

        if center_atom == MINUS:
            if atom_count == 1:
                return Action.SELECT_ATOM_MINUS, 0, NO_SELECTION
            return self.handle_minus_atom(ring, min_spawning_atom)
        
        elif atom_count == 1 or atom_count == 0:
            return Action.PLACE_ATOM, NO_SELECTION, 0

        elif center_atom == PLUS:
            return self.handle_plus_atom(ring, atom_count)

        else:
            return self.handle_regular_atom(ring, atom_count, game_state)

    def handle_minus_atom(self, ring, min_spawning_atom) -> Tuple[Action, int, int]:
        lowest_atom_value, lowest_atom_index = lowest_atom(ring)

        if lowest_atom_value < min_spawning_atom:
            chosen_atom_index = lowest_atom_index
        else:
            chosen_atom_index = second_highest_atom(ring)[1]

        return (Action.SELECT_ATOM_MINUS, chosen_atom_index, NO_SELECTION)

    def handle_plus_atom(self, ring, atom_count) -> Tuple[Action, int, int]:
        chosen_midway_index, longest_chain_length = find_longest_chain(ring)
        # Check if there is a plus next to the chosen midway
        if longest_chain_length < 2 or self.is_plus_nearby(ring, chosen_midway_index, atom_count):
            return self.place_random_near_spawning_atoms(ring, atom_count)
        return Action.PLACE_ATOM, NO_SELECTION, chosen_midway_index

    def handle_regular_atom(self, ring, atom_count,game_state) -> Tuple[Action, int, int]:
        if can_switch_to_plus(game_state) and (
                (atom_count > 10 and exists_chain_of_length(ring, 4)) or
                (atom_count > 14 and exists_chain_of_length(ring, 2))):
            return Action.CONVERT_TO_PLUS, NO_SELECTION, NO_SELECTION

        midway_next_to_plus = find_midway_next_to_plus(ring)
        if midway_next_to_plus is not None:
            return Action.PLACE_ATOM, NO_SELECTION, midway_next_to_plus

        bigger_chain_midway = find_bigger_chain_midway(ring)
        if bigger_chain_midway is not None:
            return (Action.PLACE_ATOM, NO_SELECTION, bigger_chain_midway)

        best_placement_index = find_best_placement(ring)
        return Action.PLACE_ATOM, NO_SELECTION, best_placement_index

    def is_plus_nearby(self, ring, chosen_midway_index, atom_count) -> bool:
        """Checks if there is a PLUS atom near the chosen midway."""
        return (ring.atoms[(chosen_midway_index + 1) % atom_count].atom_number == PLUS or
                ring.atoms[chosen_midway_index].atom_number == PLUS)

    def is_spawing(self, ring, atom_number):
        total_turns = ring.total_turns
        min_spawning_atom = 1 + total_turns // 40
        max_spawning_atom = 3 + total_turns // 40
        return min_spawning_atom <= atom_number <= max_spawning_atom

    def place_random_near_spawning_atoms(self, ring, atom_count) -> Tuple[Action, int, int]:
        """Finds a random placement midway between two spawning atoms, or chooses randomly if none found."""
        i = choose_random(ring)
        # First look for a midway where both neighbors are below spawning limit
        for _ in range(atom_count):
            if (self.is_spawing(ring, ring.atoms[i].atom_number) and
                    self.is_spawing(ring, ring.atoms[(i + 1) % atom_count].atom_number)):
                return Action.PLACE_ATOM, NO_SELECTION, i
            i = (i + 1) % atom_count
        # Then look for a midway where at least one neighbor is below spawning limit
        i = choose_random(ring)
        for _ in range(atom_count):
            if (self.is_spawing(ring, ring.atoms[i].atom_number) or
                    self.is_spawing(ring, ring.atoms[(i + 1) % atom_count].atom_number)):
                return Action.PLACE_ATOM, NO_SELECTION, i
            i = (i + 1) % atom_count

        # If no suitable midway found, choose randomly
        return Action.PLACE_ATOM, NO_SELECTION, choose_random(ring)


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        The evaluation function takes in the current GameState and a proposed action.
        It returns the score of the successor state after applying the action.
        """

        # Generate the successor game state based on the current action
        successor_game_state = current_game_state.generate_successor(action=action)

        # Return the score of the successor game state
        return successor_game_state.score


class MCTSAgent(Agent):
    def __init__(self, simulations=20, prioritize_score=False):
        super(MCTSAgent, self).__init__()
        self.simulations = simulations
        self.prioritize_score = prioritize_score

    def get_action(self, game_state):
        root = Node(game_state)

        for _ in range(self.simulations):
            node = self._select(root)
            reward = self._simulate(node.state, self.prioritize_score)
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
        actions = node.state.get_legal_actions(agent_index=0)
        for action in actions:
            if action not in [child.action for child in node.children]:
                next_state = node.state.generate_successor(action=action)
                child_node = Node(next_state, parent=node, action=action)
                node.children.append(child_node)
                return child_node
        raise Exception("Should never reach here")

    def _simulate(self, state, prioritize_score):
        current_state = state
        for _ in range(5):  # Simulate for a fixed number of steps
            legal_actions = current_state.get_legal_actions(agent_index=0)

            if prioritize_score:
                best_action = max(legal_actions, key=lambda a: ReflexAgent.get_action(ReflexAgent(), current_state.generate_successor(agent_index=0, action=a)))
            else:
                best_action = max(legal_actions, key=lambda a: current_state.generate_successor(agent_index=0, action=a).ring.get_highest_atom())

            current_state = current_state.generate_successor(agent_index=0, action=best_action)

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

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions(agent_index=0))

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


def get_chains(atoms):
    """
    Finds all even-length symmetric chains in a circular list of Atom objects.

    Args:
    atoms (list): List of Atom objects.

    Returns:
    list: A list of unique symmetric chains, where each chain is a list of Atom objects.
    """
    n = len(atoms)

    # Early return for cases where n is less than 2, as no symmetric chain can exist
    if n < 2:
        return []

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

