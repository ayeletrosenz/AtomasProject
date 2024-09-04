import pygame
import random
import math
import json
import numpy as np
import main_no_shiny
pi = math.pi
from enum import Enum

class Action(Enum):
    PLACE_ATOM = 1
    CONVERT_TO_PLUS = 2
    STOP = 3
NO_SELECTION = -1  # Indicates no atom was selected from the ring.
OpponentAction = "generate inner"

class GameState(object):
    def __init__(self, ring=None):
        super(GameState, self).__init__()
        if ring is None:
            ring = main_no_shiny.Ring()
            ring.start_game()
        self._ring = ring
        self._done = False  # You need to define this attribute
        self._score = self._ring.get_score()

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

        if self._ring.center_atom.special == False or self._ring.center_atom.symbol == "+":
            for midway_index in range(len(self._ring.atoms)):
                legal_moves.append((Action.PLACE_ATOM, NO_SELECTION, midway_index))

        if self._ring.center_atom.symbol == "-":
            for chosen_atom_index in range(len(self._ring.atoms)):
                for midway_index in range(len(self._ring.atoms)):
                    legal_moves.append((Action.PLACE_ATOM, chosen_atom_index, midway_index))
                    legal_moves.append((Action.CONVERT_TO_PLUS, chosen_atom_index, midway_index))

        return legal_moves

    def apply_opponent_action(self, action):
        if action == OpponentAction:
            self._ring.generate_inner()

    def apply_action(self, action):
        action_type, chosen_atom_index, midway_index = action

        if action_type == Action.PLACE_ATOM:
            clicked_mid = False
            self._ring.place_atom(chosen_atom_index, midway_index, clicked_mid)

        elif action_type == Action.CONVERT_TO_PLUS:
            clicked_mid = True
            self._ring.place_atom(chosen_atom_index, midway_index, clicked_mid)

    def generate_successor(self, agent_index=0, action=Action.STOP):
        successor = GameState(ring=self._ring.copy())
        if agent_index == 0:
            successor.apply_action(action)
        elif agent_index == 1:
            successor.apply_opponent_action(action)
        else:
            raise Exception("illegal agent index.")
        return successor