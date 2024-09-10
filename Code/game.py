import abc

from game_state import Action, OpponentAction
import pygame
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
    def __init__(self, agent, opponent_agent, display= False, sleep_between_actions=False, print_move=False):
        super(Game, self).__init__()
        self.sleep_between_actions = sleep_between_actions
        self.agent = agent
        self.opponent_agent = opponent_agent
        self._state = None
        self._should_quit = False
        self.print_move = print_move
        self.display = display  # Add display argument to control whether Pygame runs or not

        # Initialize pygame only if display is True
        if self.display:
            pygame.init()
            pygame.display.set_caption('Atomas')
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.background = main_no_shiny.Background()

    def run(self, initial_state):
        self._should_quit = False
        self._state = initial_state
        return self._game_loop()

    def quit(self):
        self._should_quit = True
        self.agent.stop_running()
        self.opponent_agent.stop_running()

    def _game_loop(self):
        while not self._should_quit:
            # Handle Pygame events (e.g., quitting the game) only if display is enabled
            if self.display:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.quit()

                # Draw the background
                self.screen.fill((0, 0, 0))  # Optional if background is not fully covering
                self.background.draw(self.screen)

            if self.sleep_between_actions:
                time.sleep(1)  # Slow down for visual clarity

            # Get the action from the agent and apply it to the game state
            action = self.agent.get_action(self._state)
            if action == Action.STOP:
                self.quit()
                return

            if self.print_move:
                main_no_shiny.print_move(self._state, action)

            # Check if the game has ended
            if self._state._ring.check_game_end():
                self.quit()
                return self._state._ring.score, self._state._ring.highest_atom  # Access score and highest atom

            # Apply the player's action and opponent's action
            self._state.apply_action(action)
            opponent_action = self.opponent_agent.get_action(self._state)
            self._state.apply_opponent_action(opponent_action)

            # Update game state details (score, highest atom, etc.)
            self._state._ring.total_turns += 1
            self._state._ring.update_highest()
            self._state._ring.update_atom_count()

            # Draw the updated game state only if display is enabled
            if self.display:
                self._state._ring.draw_outer(self.screen)
                self._state._ring.draw_inner(self.screen)
                self._state._ring.score.draw(self._state._ring.highest_atom, self.screen)

                # Update the display and control the frame rate
                pygame.display.flip()
                self.clock.tick(30)  # Set the frame rate (adjust as needed)

        return self._state._ring.score , self._state._ring.highest_atom  # Return final score and highest atom
