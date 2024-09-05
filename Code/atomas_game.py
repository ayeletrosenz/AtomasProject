import argparse
import numpy
import os
import pygame
from game import Game, RandomOpponentAgent
from game_state import GameState
import ExpectimaxAgent
import main_no_shiny
import agents


class GameRunner(object):
    def __init__(self, agent=None, sleep_between_actions=False , print_move = True):
        super(GameRunner, self).__init__()
        self.sleep_between_actions = sleep_between_actions
        self.human_agent = agent is None
        # if agent is None:
        #     agent = KeyboardAgent(display)
        self.print_move = print_move
        self._agent = agent
        self.current_game = None

    def new_game(self, initial_state=None, *args, **kw):
        self.quit_game()
        if initial_state is None:
            initial_state = GameState()
        opponent_agent = RandomOpponentAgent()
        game = Game(self._agent, opponent_agent, sleep_between_actions=self.sleep_between_actions,print_move = self.print_move)
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
    num_of_games = 1
    # agent = ExpectimaxAgent.ExpectimaxAgent()
    agent = agents.ReflexAgent()

    total_score = 0
    highest_score = 0
    highest_atom_achieved = 0
    initial_state = None
    for i in range(num_of_games):

        game_runner = GameRunner(agent=agent, sleep_between_actions=True,print_move = False)
        score, highest_atom = game_runner.new_game(initial_state)

        # Track the total score for average calculation
        total_score += score

        # Check if this is the highest score so far
        if score > highest_score:
            highest_score = score

        # Check if this is the highest atom achieved
        if highest_atom > highest_atom_achieved:
            highest_atom_achieved = highest_atom

    # Calculate average score
    average_score = total_score / num_of_games if num_of_games > 0 else 0

    # Print the results
    print(f"\nHighest Score: {highest_score}")
    print(f"Highest Atom Achieved: {highest_atom_achieved}")
    print(f"Average Score: {average_score}")



if __name__ == '__main__':
    main()
    input("Press Enter to continue...")
