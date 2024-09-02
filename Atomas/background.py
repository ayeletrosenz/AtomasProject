from config import SCREEN_WIDTH, SCREEN_HEIGHT, screen
import pygame


class Background:
    def __init__(self):
        self.BACKGROUND_COLOR = (82, 42, 50)
        self.RING_COLOUR = (133, 94, 97)
        self.ATOM_RING_CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30)

    def draw(self):
        '''Draws the background as well as the ring surrouding the atoms.'''
        screen.fill(self.BACKGROUND_COLOR)
        pygame.draw.circle(screen, self.RING_COLOUR,
                           self.ATOM_RING_CENTER, 190)
        pygame.draw.circle(screen, self.BACKGROUND_COLOR,
                           self.ATOM_RING_CENTER, 187)
