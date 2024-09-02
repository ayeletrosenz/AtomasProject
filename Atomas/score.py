import pygame
import math
from config import SCREEN_WIDTH, SCREEN_HEIGHT, screen, atom_data


class Score:
    def __init__(self):
        self.score = 0
        self.base_font = pygame.font.Font(None, 32)
        self.score_location = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 6)
        self.score_name_location = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 6 + 30)

    def update_score(self, score):
        self.score += score

    def reset(self):
        self.score = 0

    def calc_chain_score(self):
        '''Calculates the score of a chain of atoms.'''
        # print("Calculating chain score...")

        score_increase = 0
        reaction_step = 1

        def reaction_multiplier(reaction_step):
            return 1 + (0.5 * reaction_step)

        def simple_reaction(atom_nb):
            '''Helper function to calculate the score of a simple reaction.'''
            multiplier = reaction_multiplier(reaction_step)
            score = math.floor(multiplier * (atom_nb + 1))
            # print(f"math.floor({multiplier} * ({atom_nb} + 1)) = {math.floor(multiplier * (atom_nb + 1))}")
            return score, atom_nb+1

        def complex_reaction(atom_nb_inner, atom_nb_outer):
            multiplier = reaction_multiplier(reaction_step)
            score, _ = simple_reaction(atom_nb_inner)

            score += int(2 * multiplier * (atom_nb_outer - atom_nb_inner + 1))
            return score, atom_nb_outer + 2

    def draw(self, atom_nb):
        '''Draws the score and the name of the highest scoring atom.
        The text colour of the name of the atom is based on the atom itself.'''
        def draw_num_score():
            text_surface = self.base_font.render(
                str(self.score), True, (255, 255, 255))
            text_rect = text_surface.get_rect(
                center=self.score_location)
            screen.blit(text_surface, text_rect)

        def draw_atom_name(atom_nb):
            atom_nb = str(atom_nb)
            atom_name = atom_data['Name'][str(int(atom_nb)-1)]
            atom_colour = atom_data['Color'][str(int(atom_nb)-1)]
            text_surface = self.base_font.render(
                atom_name, True, atom_colour)
            text_rect = text_surface.get_rect(
                center=self.score_name_location)
            screen.blit(text_surface, text_rect)

        draw_num_score()
        draw_atom_name(atom_nb)


