import pygame
import json
import math
from enum import Enum

class Action(Enum):
    PLACE_ATOM = 1
    CONVERT_TO_PLUS = 2
    STOP = 3

NO_SELECTION = None  # Indicates no atom was selected from the ring.
OpponentAction = "generate inner"


SCREEN_WIDTH = 400
SCREEN_HEIGHT = 700
BACKGROUND_COLOR = (82, 42, 50)

PLUS = -1
MINUS = -2

IS_HUMAN_PLAYER = True

pygame.init()
mouse_pos = pygame.mouse.get_pos()
clicked_mid = abs(math.dist((SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30), mouse_pos)) < 40

def set_human_player(is_human):
    global IS_HUMAN_PLAYER
    IS_HUMAN_PLAYER = is_human

pygame.display.set_caption('Atomas')
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
with open("atom_data.json", "r") as f:
    atom_data = json.load(f)
