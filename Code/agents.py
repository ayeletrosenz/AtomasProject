import random
from typing import Tuple, List, Dict

PLUS = -1
MINUS = -2
SWITCH_TO_PLUS = (-1, -1, True)

class GameState:
    def __init__(self, ring, current_turn: int):
        self.atoms: List[int] = [atom.atom_number for atom in ring.atoms]
        self.num_atoms: int = len(ring.atoms)
        self.center_atom: int = ring.center_atom.atom_number
        self.highest_atom: int = ring.highest_atom
        self.total_turns: int = current_turn
        self.score: int = ring.score.score
        self.turns_since_last_plus: int = ring.turns_since_last_plus
        self.turns_since_last_minus: int = ring.turns_since_last_minus

    def as_dict(self) -> Dict[str, int]:
        return {
            'atoms': self.atoms,
            'num_atoms': self.num_atoms,
            'center_atom': self.center_atom,
            'highest_atom': self.highest_atom,
            'total_turns': self.total_turns,
            'score': self.score,
            'turns_since_last_plus': self.turns_since_last_plus,
            'turns_since_last_minus': self.turns_since_last_minus
        }
    
def choose_random(game_state: GameState) -> int:
    # choose a random atom index or midway index
    return random.randint(0, game_state.num_atoms - 1)

def can_switch_to_plus(game_state: GameState) -> bool:
    # return true if before this atom the center atom was a minus, meaning you can click it to switch to plus
    return game_state.turns_since_last_minus == 0 and game_state.total_turns >= 1

def lowest_atom_index(game_state: GameState) -> int:
    # return the index of the lowest atom in the ring that is not a plus
    lowest_atom_number = float('inf')
    lowest_index = -1
    for index, atom in enumerate(game_state.atoms):
        if atom.atom_number != PLUS and atom.atom_number < lowest_atom_number:
            lowest_atom_number = atom.atom_number
            lowest_index = index
    return lowest_index


"""
HOW TO WRITE AN AGENT:
It has to have a method "choose_action" that takes a GameState object and returns a tuple:
(chosen_atom_index: int, chosen_atom_index: int, switch_to_plus: bool)
actually only one of them is needed for each action
look at the SmartRandomAgent for an example
"""
class RandomAgent:
    def choose_action(self, game_state: GameState) -> Tuple[int, int, bool]:
        if game_state.center_atom == MINUS: 
            return choose_random(game_state), None, False
        else:
            if can_switch_to_plus(game_state):
                if random.random() < 0.5:
                    return SWITCH_TO_PLUS
            return None, choose_random(game_state), False

class SmartRandomAgent:
    def choose_action(self, game_state: GameState) -> Tuple[int, int, bool]:
        if game_state.center_atom == MINUS: # there's a minus in the center
            return choose_random(game_state), None, False # choose a random atom to remove
        elif game_state.center_atom == PLUS: # there's a plus in the center
            for i in range(game_state.num_atoms): # check if there's a pair of atoms you can combine
                if game_state.atoms[i] == game_state.atoms[(i+1)%game_state.num_atoms]: # combine the first pair you find
                    return None, i, False # combine the pair
            return None, choose_random(game_state), False # if there's no pair, choose a random modway
        else: # there's a regular atom in the center
            if can_switch_to_plus(game_state): # if it was removed by a minus and you can switch it to a plus
                if random.random() < 0.5: # 50% chance to switch it to a plus
                    return SWITCH_TO_PLUS # switch it to a plus
            return None, choose_random(game_state), False # choose a random midway to throw the atom to
        
class AyeletAgent:
    def choose_action(self, game_state: GameState) -> Tuple[int, int, bool]:
        if game_state.center_atom == MINUS:
            return lowest_atom_index(game_state), None, False
        elif game_state.center_atom == PLUS:
            # TODO how do i access functions of Ring and Score?
            pass
        else:
            pass

