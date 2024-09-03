import argparse
import numpy
import os
import pygame
from game import Game, RandomOpponentAgent
from game_state import GameState
import main_no_shiny
import agents


class GameRunner(object):
    def __init__(self, agent=None, sleep_between_actions=False):
        super(GameRunner, self).__init__()
        self.sleep_between_actions = sleep_between_actions
        self.human_agent = agent is None
        # if agent is None:
        #     agent = KeyboardAgent(display)

        self._agent = agent
        self.current_game = None

    def new_game(self, initial_state=None, *args, **kw):
        self.quit_game()
        if initial_state is None:
            initial_state = GameState()
        opponent_agent = RandomOpponentAgent()
        game = Game(self._agent, opponent_agent, sleep_between_actions=self.sleep_between_actions)
        self.current_game = game
        return game.run(initial_state)

    def quit_game(self):
        if self.current_game is not None:
            self.current_game.quit()


# def create_agent(args):
#     if args.agent == 'ReflexAgent':
#         from multi_agents import ReflexAgent
#         agent = ReflexAgent()
#     else:
#         agent = util.lookup('multi_agents.' + args.agent, globals())(depth=args.depth,
#                                                                      evaluation_function=args.evaluation_function)
#     return agent


def main():
    initial_state = GameState()
    agent = agents.ReflexAgent()
    game_runner = GameRunner(agent=agent, sleep_between_actions= True)
    game_runner.new_game(initial_state)
    # pygame.quit()

if __name__ == '__main__':
    main()
    input("Press Enter to continue...")
