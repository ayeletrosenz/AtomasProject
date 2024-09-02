import pygame
import math

import config
from agents import GameState, RandomAgent, SmartRandomAgent
from ring import Ring
from background import Background
from config import SCREEN_WIDTH, SCREEN_HEIGHT, clock, IS_HUMAN_PLAYER
pi = math.pi

if __name__ == "__main__":
    background = Background()
    ring = Ring()
    ring.start_game()

    # Decide if the game will be played by a human or an AI agent
    config.set_human_player(True)  # Set to False for AI
    agent = SmartRandomAgent() if not config.IS_HUMAN_PLAYER else None

    run = True
    while run:
        background.draw()

        # Handle events (only relevant if human is playing)
        if IS_HUMAN_PLAYER:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = config.mouse_pos
                    chosen_atom_index = ring.closest_atom(mouse_pos)[1]
                    chosen_midway_index = ring.closest_midway(mouse_pos)[1]
                    clicked_mid = config.clicked_mid
                    # Human player makes a move
                    ring.place_atom(chosen_atom_index, chosen_midway_index, clicked_mid, IS_HUMAN_PLAYER)
                    ring.total_turns += 1

        else:
            # AI agent's turn (continuous, not tied to events)
            game_state = GameState(ring, ring.total_turns)
            chosen_atom_index, chosen_midway_index, clicked_mid = agent.choose_action(game_state)
            ring.print_move(game_state, chosen_atom_index, chosen_midway_index, clicked_mid)
            ring.place_atom(chosen_atom_index, chosen_midway_index, clicked_mid, IS_HUMAN_PLAYER)
            ring.total_turns += 1

        # Update game state and draw
        ring.update_highest()
        ring.score.draw(ring.highest_atom)
        ring.update_atom_count()

        ring.draw_ring()

        pygame.display.flip()
        clock.tick(5)

    pygame.quit()
