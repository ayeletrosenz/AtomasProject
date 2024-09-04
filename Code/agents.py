
import math
import random
from typing import Tuple, List
import numpy as np
import abc
from game import Agent, Action

PLUS = -1
MINUS = -2
NO_SELECTION = None

def choose_random(ring) -> int:
    return random.randint(0, ring.atom_count - 1)

def can_switch_to_plus(ring) -> Tuple[int, int]:
    return ring.turns_since_last_minus == 0 and ring.total_turns >= 1

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
            if ring.atoms[i-1].atom_number == center_atom:
                return i
            elif ring.atoms[(i+1) % atom_count].atom_number == center_atom:
                return i - 1
    return None

# TODO it should find a longer chain, not the longest necessarily
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
    """
    Check if there exists a chain of atoms in the ring that matches or exceeds a given length.
    """
    atom_count = ring.atom_count
    for i in range(atom_count):
        # Calculate the chain length starting at index `i`
        chain_length = calculate_chain_length(ring, i)
        if chain_length >= length:
            return True
    return False

class AyeletAgent(Agent):
    def get_action(self, game_state) -> Tuple[Action, int, int]:
        ring = game_state.ring
        atom_count = ring.atom_count
        center_atom = ring.center_atom.atom_number
        total_turns = ring.total_turns
        min_spawning_atom = 1 + total_turns // 40

        if center_atom == MINUS:
            return self.handle_minus_atom(ring, min_spawning_atom)

        elif center_atom == PLUS:
            return self.handle_plus_atom(ring, atom_count)

        else:
            return self.handle_regular_atom(ring, atom_count)


    def handle_minus_atom(self, ring, min_spawning_atom) -> Tuple[Action, int, int]:
        """Handles logic when center atom is MINUS."""
        lowest_atom_value, lowest_atom_index = lowest_atom(ring)
        
        if lowest_atom_value < min_spawning_atom:
            chosen_atom_index = lowest_atom_index
        else:
            chosen_atom_index = second_highest_atom(ring)[1]
        
        return (Action.PLACE_ATOM, chosen_atom_index, NO_SELECTION)


    def handle_plus_atom(self, ring, atom_count) -> Tuple[Action, int, int]:
        """Handles logic when center atom is PLUS."""
        chosen_midway_index, longest_chain_length = find_longest_chain(ring)
        # Check if there is a plus next to the chosen midway
        if longest_chain_length < 2 or self.is_plus_nearby(ring, chosen_midway_index, atom_count):
            return self.place_random_near_spawning_atoms(ring, atom_count)
        return (Action.PLACE_ATOM, NO_SELECTION, chosen_midway_index)


    def handle_regular_atom(self, ring, atom_count) -> Tuple[Action, int, int]:
        """Handles logic when the center atom is neither PLUS nor MINUS."""
        if can_switch_to_plus(ring) and (
            (atom_count > 10 and exists_chain_of_length(ring, 4)) or
            (atom_count > 14 and exists_chain_of_length(ring, 2))):
            return (Action.CONVERT_TO_PLUS, NO_SELECTION, NO_SELECTION)

        midway_next_to_plus = find_midway_next_to_plus(ring)
        if midway_next_to_plus is not None:
            return (Action.PLACE_ATOM, NO_SELECTION, midway_next_to_plus)

        bigger_chain_midway = find_bigger_chain_midway(ring)
        if bigger_chain_midway is not None:
            return (Action.PLACE_ATOM, NO_SELECTION, bigger_chain_midway)

        best_placement_index = find_best_placement(ring)
        return (Action.PLACE_ATOM, NO_SELECTION, best_placement_index)


    def is_plus_nearby(self, ring, chosen_midway_index, atom_count) -> bool:
        """Checks if there is a PLUS atom near the chosen midway."""
        return (ring.atoms[(chosen_midway_index + 1) % atom_count].atom_number == PLUS or
                ring.atoms[chosen_midway_index].atom_number == PLUS)
    
    def is_spawing(self, ring, atom_number):
        total_turns = ring.total_turns
        min_spawning_atom = 1 + total_turns // 40
        max_spawning_atom = 3 + total_turns // 40
        return atom_number >= min_spawning_atom and atom_number <= max_spawning_atom


    def place_random_near_spawning_atoms(self, ring, atom_count) -> Tuple[Action, int, int]:
        """Finds a random placement midway between two spawning atoms, or chooses randomly if none found."""
        print("Placing random near spawning atoms")
        i = choose_random(ring)
        # First look for a midway where both neighbors are below spawning limit
        for _ in range(atom_count):
            if (self.is_spawing(ring, ring.atoms[i].atom_number) and 
                    self.is_spawing(ring, ring.atoms[(i + 1) % atom_count].atom_number)):
                return (Action.PLACE_ATOM, NO_SELECTION, i)
            i = (i + 1) % atom_count
        # Then look for a midway where at least one neighbor is below spawning limit
        i = choose_random(ring)
        for _ in range(atom_count):
            if (self.is_spawing(ring, ring.atoms[i].atom_number) or 
                    self.is_spawing(ring, ring.atoms[(i + 1) % atom_count].atom_number)):
                return (Action.PLACE_ATOM, NO_SELECTION, i)
            i = (i + 1) % atom_count

        # If no suitable midway found, choose randomly
        return (Action.PLACE_ATOM, NO_SELECTION, choose_random(ring))


class SmartRandomAgent(Agent):
    def get_action(self, game_state) -> Tuple[Action, int, int]:
        ring = game_state.ring
        center_atom = ring.center_atom.atom_number
        atom_count = ring.atom_count

        if center_atom == MINUS:
            return (Action.PLACE_ATOM, choose_random(ring), NO_SELECTION)
        
        elif center_atom == PLUS:
            i = choose_random(ring)
            for _ in range(atom_count):
                if ring.atoms[i].atom_number == ring.atoms[(i+1)%atom_count].atom_number:
                    return (Action.PLACE_ATOM, NO_SELECTION, i)
                i = (i+1)%atom_count
            return (Action.PLACE_ATOM, NO_SELECTION, choose_random(ring))
        
        else:
            if can_switch_to_plus(ring) and random.random() < 0.5:
                return (Action.CONVERT_TO_PLUS, NO_SELECTION, NO_SELECTION)
            else:
                return (Action.PLACE_ATOM, NO_SELECTION, choose_random(ring))
            


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
    def __init__(self, simulations=1000):
        super(MCTSAgent, self).__init__()
        self.simulations = simulations

    def get_action(self, game_state):
        root = Node(game_state)

        for _ in range(self.simulations):
            node = self._select(root)
            reward = self._simulate(node.state)
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

    def _simulate(self, state):
        current_state = state
        while not current_state.done:
            legal_actions = current_state.get_legal_actions(agent_index=0)
            action = random.choice(legal_actions)
            current_state = current_state.generate_successor(action=action)
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

