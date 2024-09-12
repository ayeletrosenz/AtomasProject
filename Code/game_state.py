import pygame
import random
import math
import json
import numpy as np
from ring import Ring
import atom

pi = math.pi
from enum import Enum


class Action(Enum):
    PLACE_ATOM = 1
    CONVERT_TO_PLUS = 2
    SELECT_ATOM_MINUS = 3
    STOP = 4

NO_SELECTION = -1  # Indicates no atom was selected from the ring.
OpponentAction = "generate inner"


class GameState(object):
    def __init__(self, ring=None):
        super(GameState, self).__init__()
        if ring is None:
            # ring = main_no_shiny.Ring()
            ring = Ring()
            ring.start_game()
        self._ring = ring
        self._done = False
        self._score = self._ring.get_score()
        self.pending_minus_action = None  # Track pending minus action

    # When using minus, we select an atom and wait for the next action
    def select_atom_for_minus(self):
        self.pending_minus_action = True

    # Reset pending action once the sequence is complete
    def clear_pending_action(self):
        self.pending_minus_action = None

    @property
    def score(self):
        return self._ring.get_score()

    @property
    def done(self):
        return self._done

    @property
    def highest_atom(self):
        return self._ring.get_highest_atom()

    @property
    def ring(self):
        return self._ring

    def get_legal_actions(self, agent_index):
        if agent_index == 0:
            return self.get_agent_legal_actions()
        elif agent_index == 1:
            return self.get_opponent_legal_actions()
        else:
            raise Exception("illegal agent index.")

    def get_opponent_legal_actions(self):
        return [OpponentAction]

    def get_agent_legal_actions(self):
        legal_moves = []

        # If a minus action is pending, offer convert or place options
        if self.pending_minus_action is not None:
            chosen_atom_index = self.pending_minus_action
            for midway_index in range(len(self._ring.atoms)):
                legal_moves.append((Action.PLACE_ATOM, chosen_atom_index, midway_index))
                legal_moves.append((Action.CONVERT_TO_PLUS, chosen_atom_index, midway_index))
        else:
            # Standard moves when no pending action
            if self._ring.center_atom.special == False or self._ring.center_atom.symbol == "+":
                for midway_index in range(len(self._ring.atoms)):
                    legal_moves.append((Action.PLACE_ATOM, NO_SELECTION, midway_index))

            if self._ring.center_atom.symbol == "-":
                # When minus atom is in play, the first step is selecting an atom
                for chosen_atom_index in range(len(self._ring.atoms)):
                    legal_moves.append((Action.SELECT_ATOM_MINUS, chosen_atom_index, NO_SELECTION))

        return legal_moves

    def apply_opponent_action(self, action):
        if action == OpponentAction:
            self._ring.generate_inner()

    def apply_action(self, action):
        action_type, chosen_atom_index, midway_index = action

        if action_type == Action.PLACE_ATOM:
            clicked_mid = False
            self._ring.place_atom(NO_SELECTION, midway_index, clicked_mid)
            self.clear_pending_action()

        elif action_type == Action.CONVERT_TO_PLUS:
            clicked_mid = True
            self._ring.place_atom(NO_SELECTION, NO_SELECTION, clicked_mid)

        elif action_type == Action.SELECT_ATOM_MINUS:
            clicked_mid = False
            # Store selected atom for the next step
            self.select_atom_for_minus()
            self._ring.place_atom(chosen_atom_index, NO_SELECTION, clicked_mid)

    def generate_successor(self, agent_index=0, action=Action.STOP):
        successor = GameState(ring=self._ring.copy())
        if agent_index == 0:
            successor.apply_action(action)
        elif agent_index == 1:
            successor.apply_opponent_action(action)
        else:
            raise Exception("illegal agent index.")
        return successor
