import abc
from collections import namedtuple
from game_state import Action, OpponentAction
import pygame
import json
import numpy as np
import time
import main_no_shiny


SCREEN_WIDTH = 400
SCREEN_HEIGHT = 700

PLUS = -1
MINUS = -2

class Agent(object):
    def __init__(self):
        super(Agent, self).__init__()

    @abc.abstractmethod
    def get_action(self, game_state):
        return

    def stop_running(self):
        pass


class RandomOpponentAgent(Agent):

    def get_action(self, game_state):
        return OpponentAction


class Game(object):
    def __init__(self, agent, opponent_agent, display=None, sleep_between_actions=False):
        super(Game, self).__init__()
        self.sleep_between_actions = sleep_between_actions
        self.agent = agent
        self.opponent_agent = opponent_agent
        self._state = None
        self._should_quit = False

    def run(self, initial_state):
        self._should_quit = False
        self._state = initial_state
        return self._game_loop()

    def quit(self):
        self._should_quit = True
        self.agent.stop_running()
        self.opponent_agent.stop_running()

    def _game_loop(self):
        # pygame.init()
        # pygame.display.set_caption('Atomas')
        # screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        # clock = pygame.time.Clock()
        # background = main_no_shiny.Background()

        while not self._should_quit:
            # background.draw()
            if self.sleep_between_actions:
                time.sleep(1)
            action = self.agent.get_action(self._state)
            if action == Action.STOP:
                return
            main_no_shiny.print_move(self._state,action)
            self._state.apply_action(action)
            opponent_action = self.opponent_agent.get_action(self._state)
            self._state.apply_opponent_action(opponent_action)
            self._state.ring.total_turns += 1
            self._state.ring.update_highest()
            # self._state.ring.score.draw(self._state.ring.highest_atom)
            self._state.ring.update_atom_count()
            #
            # self._state.ring.draw_outer()
            # self._state.ring.draw_inner()

            # pygame.display.flip()
            # clock.tick(5)

        return self._state.score, self._state.highest_atom  # Access the score and highest atom correctly


