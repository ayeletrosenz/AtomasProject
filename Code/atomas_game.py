import argparse
import pygame
from game import Game, RandomOpponentAgent
from game_state import GameState
from agents import AyeletAgent, ReflexAgent, ExpectimaxAgent, SmartRandomAgent
from agents import score_evaluation_function, highest_atom_evaluation_function
from MCTSagent import MCTSAgent

class GameRunner(object):
    def __init__(self, agent=None, sleep_between_actions=False, print_move=True, display=None):
        super(GameRunner, self).__init__()
        self.sleep_between_actions = sleep_between_actions
        self.human_agent = agent is None
        self.print_move = print_move
        self._agent = agent
        self.current_game = None
        self.display = display

    def new_game(self, initial_state=None, *args, **kw):
        self.quit_game()
        if initial_state is None:
            initial_state = GameState()
        opponent_agent = RandomOpponentAgent()
        # Pass the display surface to the Game class
        game = Game(self._agent, opponent_agent, sleep_between_actions=self.sleep_between_actions,
                    print_move=self.print_move, display=self.display)
        self.current_game = game
        return game.run(initial_state)

    def quit_game(self):
        if self.current_game is not None:
            self.current_game.quit()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Atomas game with different options.')
    parser.add_argument('--num_of_games', type=int, default=1, help='Number of games to run')
    parser.add_argument('--display', type=bool, default=True, help='Whether to display the game window (True/False)')
    parser.add_argument('--sleep_between_actions', help='Should sleep between actions.', default=True, type=bool)
    parser.add_argument('--print_move', type=bool, default=True, help='Whether to print each move (True/False)')
    parser.add_argument('--agent', type=str, choices=['random', 'ayelet', 'reflex', 'mcts', 'expectimax'], default='expectimax', help='Type of agent to use [random, ayelet, reflex, mcts, expectimax]')
    parser.add_argument('--depth', type=int, default=2, help='Depth of the Expectimax search.')
    parser.add_argument('--simulations', type=int, default=200, help='Number of simulations of the MCTS agent.')
    parser.add_argument('--priority', type=str, choices=['score', 'highest_atom'], default='score', help='Priority for the Expectimax agent: "score" or "highest_atom"')
    return parser.parse_args()

def agent_builder(agent_type, depth, simulations, priority):
    evaluation_function = None
    if agent_type == 'ayelet':
        agent = AyeletAgent()
    elif agent_type == 'reflex':
        agent = ReflexAgent()
    elif agent_type == 'mcts':
        agent = MCTSAgent(simulations=simulations)
    elif agent_type == 'expectimax':
        # Set the evaluation function based on the priority argument
        if priority == 'score':
            evaluation_function = score_evaluation_function
        elif priority == 'highest_atom':
            evaluation_function = highest_atom_evaluation_function
        agent = ExpectimaxAgent(depth=depth, evaluation_function=evaluation_function)
    elif agent_type == 'random':
        agent = SmartRandomAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return agent


def main():
    args = parse_arguments()

    if args.display:
        pygame.init()

    num_of_games = args.num_of_games
    agent = agent_builder(agent_type=args.agent, depth=args.depth, simulations=args.simulations, priority=args.priority)

    # Informative printings about agent type and number of games
    print(f"Running {num_of_games} game(s) with agent type: {args.agent}")
    if args.agent == 'expectimax':
        print(f"Expectimax agent with depth: {args.depth} and prioritize {args.priority}")
    elif args.agent == 'mcts':
        print(f"MCTS agent with {args.simulations} simulations")

    total_score = 0
    highest_score = 0
    highest_atom_achieved = 0
    initial_state = None

    for i in range(num_of_games):
        # Informative print about the current game
        print(f"\nStarting game {i + 1} of {num_of_games}...")

        # Pass the display surface to GameRunner
        game_runner = GameRunner(agent=agent, sleep_between_actions=args.sleep_between_actions,
                                 print_move=args.print_move, display=args.display)
        score, highest_atom = game_runner.new_game(initial_state)
        score_value = score.get_value()

        # Track the total score for average calculation
        total_score += score_value

        # Check if this is the highest score so far
        if score_value > highest_score:
            highest_score = score_value

        # Check if this is the highest atom achieved
        if highest_atom > highest_atom_achieved:
            highest_atom_achieved = highest_atom

    # Calculate average score
    average_score = total_score / num_of_games if num_of_games > 0 else 0

    # Print the results
    print(f"\nHighest Score: {highest_score}")
    print(f"Highest Atom Achieved: {highest_atom_achieved}")
    print(f"Average Score: {average_score}")

    if args.display:
        pygame.quit()

if __name__ == '__main__':
    main()
    input("Press Enter to continue...")
