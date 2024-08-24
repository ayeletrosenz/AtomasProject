import random
from typing import Tuple
from main_no_shiny import GameState
from abc import ABC, abstractmethod

PLUS = -1
MINUS = -2

class AbstractAgent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def choose_action(self, game_state):
        """Choose an action based on the current game state.
        
        Args:
            game_state: The current state of the game, which can include information about the ring, atoms, score, etc.

        Returns:
            action: The chosen action, which could be a placement of an atom or a special move.
        """
        pass

    def learn(self, state, action, reward, next_state):
        """Optional: Implement learning behavior if you want to create an agent that learns over time.
        
        Args:
            state: The previous game state before the action was taken.
            action: The action that was taken.
            reward: The reward received after taking the action.
            next_state: The game state after the action was taken.
        """
        pass




