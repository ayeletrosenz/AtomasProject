import abc
from game_state import Action, OpponentAction
import pygame
import time
import main_no_shiny
import json

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 700
BACKGROUND_COLOR = (82, 42, 50)
PLUS = -1
MINUS = -2
with open(r"atom_data.json", "r") as f:
    atom_data = json.load(f)
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
    def __init__(self, agent, opponent_agent, display=False, sleep_between_actions=False, print_move=False):
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
            if self.display:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.quit()
                self.screen.fill((0, 0, 0))
                self.background.draw(self.screen)

            if self.sleep_between_actions:
                time.sleep(1)

            # Check if the game has ended
            if self._state.ring.check_game_end():
                print(self._state.done)
                final_score = self._state._ring.score
                highest_atom = self._state._ring.highest_atom
                self.quit()
                self._show_end_screen(final_score, highest_atom)
                return final_score, highest_atom

            # Get the action from the agent
            action = self.agent.get_action(self._state)
            if action == Action.STOP:
                self.quit()
                return

            if self.print_move:
                main_no_shiny.print_move(self._state, action)

            # Apply the player's action
            self._state.apply_action(action)

            # If there are no pending actions, proceed with the opponent's turn
            if self._state.pending_minus_action is None:
                opponent_action = self.opponent_agent.get_action(self._state)
                self._state.apply_opponent_action(opponent_action)

            # Update game state details
            self._state._ring.total_turns += 1
            self._state._ring.update_highest()
            self._state._ring.update_atom_count()

            if self.display:
                self._state._ring.draw_outer(self.screen)
                self._state._ring.draw_inner(self.screen)
                self._state._ring.score.draw(self._state._ring.highest_atom, self.screen)

                pygame.display.flip()
                self.clock.tick(30)

        return self._state._ring.score, self._state._ring.highest_atom  # Return final score and highest atom

    def _show_end_screen(self, score, highest_atom):
        """
        Displays the end game screen with the final score and highest atom, and waits for the user to press Enter.
        """
        if not self.display:
            return

        font = pygame.font.SysFont(None, 36)
        small_font = pygame.font.SysFont(None, 24)

        # End game message
        end_message = f"Game Over!"
        score_message = f"Score: {score.get_value()}"
        continue_message = "Press Enter to continue..."

        def draw_atom_name(atom_nb):
            atom_nb = str(atom_nb)
            atom_name = atom_data['Name'][str(int(atom_nb) - 1)]
            atom_text = f"{atom_name} ({atom_nb})"
            atom_colour = atom_data['Color'][str(int(atom_nb) - 1)]
            highest_atom_text = font.render(atom_text, True, atom_colour)
            text_rect = highest_atom_text.get_rect(
                center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3 + 100))  # Adjust vertical position
            self.screen.blit(highest_atom_text, text_rect)

        # Create a loop to display the end screen
        waiting_for_enter = True
        while waiting_for_enter:
            self.screen.fill(BACKGROUND_COLOR)

            # Render the text and blit it to the screen
            end_text = font.render(end_message, True, (255, 255, 255))
            score_text = font.render(score_message, True, (255, 255, 255))
            continue_text = small_font.render(continue_message, True, (255, 255, 255))

            # Center the texts on the screen
            self.screen.blit(end_text, (SCREEN_WIDTH // 2 - end_text.get_width() // 2, SCREEN_HEIGHT // 3))
            self.screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, SCREEN_HEIGHT // 3 + 50))

            # Draw the highest atom name and color just below the score
            draw_atom_name(highest_atom)

            self.screen.blit(continue_text,
                             (SCREEN_WIDTH // 2 - continue_text.get_width() // 2, SCREEN_HEIGHT // 3 + 200))

            # Update the display
            pygame.display.flip()

            # Check for key press events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting_for_enter = False
                    self.quit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    waiting_for_enter = False

            self.clock.tick(30)  # Set the frame rate for the end screen


