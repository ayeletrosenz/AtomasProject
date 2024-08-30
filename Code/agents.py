import random
from typing import Tuple, List, Dict
import numpy as np
import abc
from game import Agent, Action

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