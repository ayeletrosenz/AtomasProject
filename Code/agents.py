import random
from typing import Tuple, List, Dict
import numpy as np
import abc
from game import Agent, Action
import math

PLUS = -1
MINUS = -2
NO_SELECTION = None
SWITCH_TO_PLUS = (NO_SELECTION, NO_SELECTION, True)

# class GameState:
#     def __init__(self, ring, current_turn: int):
#         self.atoms: List[int] = [atom.atom_number for atom in ring.atoms]
#         self.atom_count: int = len(ring.atoms)
#         self.center_atom: int = ring.center_atom.atom_number
#         self.highest_atom: int = ring.highest_atom
#         self.total_turns: int = current_turn
#         self.score: int = ring.score.score
#         self.turns_since_last_plus: int = ring.turns_since_last_plus
#         self.turns_since_last_minus: int = ring.turns_since_last_minus

#     def as_dict(self) -> Dict[str, int]:
#         return {
#             'atoms': self.atoms,
#             'atom_count': self.atom_count,
#             'center_atom': self.center_atom,
#             'highest_atom': self.highest_atom,
#             'total_turns': self.total_turns,
#             'score': self.score,
#             'turns_since_last_plus': self.turns_since_last_plus,
#             'turns_since_last_minus': self.turns_since_last_minus
#         }
    
def choose_random(game_state: 'GameState') -> int:
    # choose a random atom index or midway index
    return random.randint(0, game_state.ring.atom_count - 1)

def can_switch_to_plus(game_state: 'GameState') -> bool:
    # return true if before this atom the center atom was a minus, meaning you can click it to switch to plus
    return game_state.ring.turns_since_last_minus == 0 and game_state.ring.total_turns >= 1

def lowest_atom_index(game_state: 'GameState') -> int:
    # return the index of the lowest atom in the ring that is not a plus
    lowest_atom_number = float('inf')
    lowest_index = -1
    for index, atom in enumerate(game_state.ring.atoms):
        if atom.atom_number != PLUS and atom.atom_number < lowest_atom_number:
            lowest_atom_number = atom.atom_number
            lowest_index = index
    return lowest_index


# class RandomAgent(Agent):

#     def get_action(self, game_state):
#         legal_moves = game_state.get_agent_legal_actions()
#         if legal_moves:
#             return random.choice(legal_moves)
#         # else:
#         #     return None
#     def choose_action(self, game_state: GameState) -> Tuple[int, int, bool]:
#         if game_state.center_atom == MINUS: 
#             return choose_random(game_state), None, False
#         else:
#             if can_switch_to_plus(game_state):
#                 if random.random() < 0.5:
#                     return SWITCH_TO_PLUS
#             return None, choose_random(game_state), False

# class SmartRandomAgent:
#     def choose_action(self, game_state: GameState) -> Tuple[int, int, bool]:
#         if game_state.center_atom == MINUS: # there's a minus in the center
#             return choose_random(game_state), None, False # choose a random atom to remove
#         elif game_state.center_atom == PLUS: # there's a plus in the center
#             for i in range(game_state.atom_count): # check if there's a pair of atoms you can combine
#                 if game_state.ring.atoms[i] == game_state.ring.atoms[(i+1)%game_state.atom_count]: # combine the first pair you find
#                     return None, i, False # combine the pair
#             return None, choose_random(game_state), False # if there's no pair, choose a random modway
#         else: # there's a regular atom in the center
#             if can_switch_to_plus(game_state): # if it was removed by a minus and you can switch it to a plus
#                 if random.random() < 0.5: # 50% chance to switch it to a plus
#                     return SWITCH_TO_PLUS # switch it to a plus
#             return None, choose_random(game_state), False # choose a random midway to throw the atom to
        
class AyeletAgent:
    def get_action(self, game_state: 'GameState') -> Tuple[Action, int, int]:
        def find_symmetry_indices(atoms, pivot):
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

        def calculate_chain_length(i: int) -> int:
            """
            Calculate the length of the chain starting at index `i`.
            """
            atoms = game_state.ring.atoms  # Assuming you have access to the game state's ring
            sym_numbers, sym_indices = find_symmetry_indices(atoms, i)

            # The chain length is twice the number of symmetric pairs plus one for the pivot
            chain_length = len(sym_indices) * 2 + 1 if sym_indices else 1
            return chain_length

        def find_longest_chain_midway() -> int:
            """
            Find the midway index that leads to the longest chain of atoms.
            """
            longest_chain_length = 0
            longest_chain_midway = None
            for i in range(game_state.ring.atom_count):
                chain_length = calculate_chain_length(i)
                if chain_length > longest_chain_length:
                    longest_chain_length = chain_length
                    longest_chain_midway = i
            return longest_chain_midway
        
        def find_midway_next_to_plus() -> int:
            """Find a midway next to a plus atom that would trigger a chain reaction."""
            for i in range(game_state.ring.atom_count):
                if game_state.ring.atoms[i] == PLUS:
                    if game_state.ring.atoms[i-1] == game_state.ring.center_atom:
                        return i
                    elif game_state.ring.atoms[(i+1)%game_state.ring.atom_count] == game_state.ring.center_atom:
                        return i-1
            return None

        # def find_bigger_chain_midway() -> int:
        #     """Find a midway to place the atom that would create a larger chain."""
        #     for i in range(game_state.ring.atom_count):
        #         if self.would_increase_chain_size(i):
        #             return i
        #     return None

        def find_best_placement() -> int:
            """Find the best place to put the atom by comparing to the closest lower atom value."""
            closest_index = None
            closest_difference = float('inf')
            for i in range(game_state.ring.atom_count):
                if game_state.ring.atoms[i].atom_number < game_state.ring.center_atom.atom_number and (game_state.ring.center_atom.atom_number - game_state.ring.atoms[i].atom_number) < closest_difference:
                    closest_difference = game_state.ring.center_atom.atom_number - game_state.ring.atoms[i].atom_number
                    closest_index = i
            return closest_index

        if game_state.ring.center_atom == MINUS:
            # Choose the lowest atom
            chosen_atom_index = lowest_atom_index(game_state)
            return (Action.PLACE_ATOM, chosen_atom_index, NO_SELECTION)

        elif game_state.ring.center_atom == PLUS:
            # Choose the midway of the longest chain
            chosen_midway_index = find_longest_chain_midway()
            return (Action.PLACE_ATOM, NO_SELECTION, chosen_midway_index)

        else:
            if can_switch_to_plus(game_state) and (
                (game_state.ring.atom_count > 10 and self.exists_chain_of_length(4)) or 
                (game_state.ring.atom_count > 14 and self.exists_chain_of_length(2))):
                # Switch to plus if conditions are met
                return (Action.CONVERT_TO_PLUS, NO_SELECTION, NO_SELECTION)

            # Check for a midway next to a plus atom that would cause a chain reaction
            midway_next_to_plus = find_midway_next_to_plus()
            if midway_next_to_plus is not None:
                return (Action.PLACE_ATOM, NO_SELECTION, midway_next_to_plus)

            # TODO
            # Check for a midway to place the atom that would create a larger chain
            # bigger_chain_midway = find_bigger_chain_midway()
            # if bigger_chain_midway is not None:
            #     return (Action.PLACE_ATOM, NO_SELECTION, bigger_chain_midway)

            # Default to placing the atom after the closest lower value atom
            best_placement_index = find_best_placement()
            return (Action.PLACE_ATOM, NO_SELECTION, best_placement_index)

    # The helper methods below need to be implemented to complete the logic
    def calculate_chain_length(self, index: int) -> int:
        pass

    def would_cause_chain_reaction(self, index: int) -> bool:
        pass

    def would_increase_chain_size(self, index: int) -> bool:
        pass

    def exists_chain_of_length(self, length: int) -> bool:
        pass



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

