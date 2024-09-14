from score import Score
from atom import Atom
import pygame
import random
import math
import numpy as np
import copy
from config import SCREEN_WIDTH, SCREEN_HEIGHT, PLUS, MINUS
from math import pi

class Ring:
    def __init__(self):
        self.atom_count = 0
        self.max_atoms = 18
        self.highest_atom = 1
        self.atoms = []
        self.score = Score()
        self.center_atom = ""
        self.font = pygame.font.Font(None, 25)
        self.locations = []

        self.total_turns = 1
        self.turns_since_last_plus = 0
        self.turns_since_last_minus = 0

    def copy(self):
        # Create a new instance of Ring
        new_ring = Ring()

        # Deep copy all attributes
        new_ring.atom_count = self.atom_count
        new_ring.max_atoms = self.max_atoms
        new_ring.highest_atom = self.highest_atom
        new_ring.atoms = copy.deepcopy(self.atoms)  # Deep copy the list of atoms
        new_ring.score = self.score.copy()  # Assuming Score class has a copy method
        new_ring.center_atom = copy.deepcopy(self.center_atom)  # Deep copy the center atom
        new_ring.font = self.font  # Font can be shallow copied (immutable)
        new_ring.locations = copy.deepcopy(self.locations)  # Deep copy the list of locations
        new_ring.total_turns = self.total_turns
        new_ring.turns_since_last_plus = self.turns_since_last_plus
        new_ring.turns_since_last_minus = self.turns_since_last_minus

        return new_ring

    def update_highest(self):
        '''Helper function to update the highest scoring atom in the ring.'''
        to_check = [i.atom_number for i in self.atoms if i.atom_number > 0]

        current_highest = int(sorted(to_check, key=int)[-1])
        if current_highest > self.highest_atom:
            self.highest_atom = current_highest

    def get_score(self):
        return self.score.get_value()

    def max_atom(self):
        return np.max(self.atoms)

    def get_highest_atom(self):
        return self.highest_atom

    def start_game(self):
        '''Function to start the game.
        This will randomly pick 6 atoms ranging from 1 to 3 and a center atom, the last of which can include a "+".'''

        self.atoms = [Atom(random.randint(1, 3)) for _ in range(6)]
        self.update_atom_count()
        self.update_highest()
        self.generate_inner()

    def update_atom_count(self):
        '''Helper function to update the atom count.'''
        self.atom_count = len(self.atoms)
        return

    def generate_inner(self):
        '''Function to generate the atom in the center of the ring.
        The one that the player can place next.'''

        # spawn a minus atom if there hasn't been one in the last 20 turns
        if self.turns_since_last_minus >= 20:
            self.center_atom = Atom()
            self.center_atom.create_minus()
            self.turns_since_last_minus = 0
            return

        # spawn a plus atom if there hasn't been one in the last 5 turns
        if self.turns_since_last_plus >= 5:
            self.center_atom = Atom()
            self.center_atom.create_plus()
            self.turns_since_last_plus = 0
            self.turns_since_last_minus += 1
            return

        # check current range of which atoms can spawn
        # every 40 turns the highest atom which can spawn is increased by 1
        range_max = 3 + self.total_turns // 40
        range_min = 1 + self.total_turns // 40

        special_atoms = ["+"]
        outlier_atoms = []
        for atom in self.atoms:
            if (atom.atom_number > 0) and (atom.atom_number < range_min) and (atom.atom_number not in outlier_atoms):
                outlier_atoms.append(atom.atom_number)

        possible_atoms = outlier_atoms + special_atoms + [x for x in range(range_min, range_max + 1)]
        outlier_weight = 1/self.atom_count
        possible_atoms_weights = [outlier_weight for _ in range(len(outlier_atoms))] + [(1-outlier_weight*len(outlier_atoms)) / 4 for _ in range(4)]

        chosen_atom = random.choices(possible_atoms, possible_atoms_weights, k=1)[0]
        if chosen_atom == "+":
            self.center_atom = Atom()
            self.center_atom.create_plus()
            self.turns_since_last_plus = 0
            return
        else:
            self.center_atom = Atom()
            self.center_atom.create(chosen_atom)

        self.turns_since_last_plus += 1
        self.turns_since_last_minus += 1

    def draw_inner(self, screen):
        '''Function to draw the atom in the center of the ring.
        The one that the player can place next.'''

        match self.center_atom.symbol:
            case "":
                return
            case "+":
                self.special_atoms("+", (218, 77, 57), (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30), screen)
            case "-":
                self.special_atoms("-", (68, 119, 194), (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30), screen)
            case _:
                self.normal_atoms(self.center_atom, (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30), screen)

    def check_game_over(self):
        return self.atom_count > self.max_atoms

    def check_game_end(self):
        '''Function to check if the player loses the game.
        '''
        if self.check_game_over():
            for atom in self.atoms:
                if atom.atom_number >= 1:
                    self.score.update(atom.atom_number)

            print("\nGame over")
            print("Score:", self.score.score)
            print("Highest atom:", self.highest_atom)
            return True
        return False

    def place_atom(self, chosen_atom_index, chosen_midway_index, clicked_mid):

        def place_normal(self):
            '''Function to place a non-special atom.'''
            closest_midway = chosen_midway_index

            self.atoms.insert(closest_midway+1, self.center_atom)
            self.update_atom_count()

        def find_symmetry_indices(atoms, pivot):
            atom_list = [atom.atom_number for atom in atoms]
            atom_indices = [i for i in range(len(atom_list))]
            n = len(atom_list)

            if n == 1:
                return [], []
            
            # Special case: If there are only 2 atoms and they are the same, fuse them
            if n == 2:
                if atom_list[0] == atom_list[1]:  # If both atoms are the same
                    return [(atom_list[0], atom_list[1])], [(0, 1)]  # Return fusion for both atoms
                else:
                    return [], []  # No fusion possible if they're not the same

            # General case for more than 2 atoms
            atom_list.insert(pivot, "p")
            atom_indices.insert(pivot, "p")
            n = len(atom_list)

            roll = np.roll(atom_list, n//2 - pivot)
            roll = [str(i) if i.isalpha() else int(i) for i in roll]
            roll_indices = np.roll(atom_indices, n//2 - pivot)
            roll_indices = [str(i) if i.isalpha() else int(i) for i in roll_indices]

            sym_numbers = []
            sym_indices = []

            for i in range(1, n//2):
                if (roll[n//2 - i] == roll[n//2 + i]) and (roll[n//2 - i] >= 0) and (roll[n//2 + i] >= 0):
                    sym_numbers.append((roll[n//2 - i], roll[n//2 + i]))
                    sym_indices.append((roll_indices[n//2 - i], roll_indices[n//2 + i]))
                else:
                    break
            return sym_numbers, sym_indices

        def use_plus(self, indice=-1):
            closest_midway = chosen_midway_index
            atom_list = self.atoms.copy()

            if indice != -1:
                sym, sym_indices = find_symmetry_indices(atom_list, indice)
            else:
                sym, sym_indices = find_symmetry_indices(atom_list, closest_midway+1)

            if len(sym) == 0:
                place_normal(self)
                self.update_atom_count()
                return
            elif len(sym) >= 1:
                self.score.calc_chain_score(self,sym, sym_indices, atom_list)
            self.update_atom_count()

        def use_minus(self):
            closest_atom_index = chosen_atom_index
            self.center_atom = self.atoms.pop(closest_atom_index)
            self.update_atom_count()

        def convert_to_plus(self):
            self.center_atom = Atom()
            self.center_atom.create_plus()
            self.turns_since_last_minus = 1

        def check_new_fusions(self):
            '''Function to check if there are any new fusions after a turn due to played atoms.
            Example 1: [2, +, 1] > [2, +, 2, 1] => [3, 1]
            Example 2: [2, +, 1, 1] > [2, +, 1, +, 1] > [2, +, 2] => [3]'''
            while True:
                atoms = self.atoms
                atom_list = [atom.atom_number for atom in atoms]
                atom_indices = [i for i in range(len(atom_list))]

                # Break if there are only two atoms and one is a PLUS
                if len(atom_list) == 2 and PLUS in atom_list:
                    break

                new_fusions = []
                for number, indice in zip(range(len(atom_list)), atom_indices):
                    if atom_list[number] == PLUS:
                        left_index = number-1
                        right_index = (number+1)%len(atom_list)
                        if left_index != right_index and atom_list[left_index] == atom_list[right_index] and atom_list[left_index] > 0:
                            new_fusions.append(indice)
                if len(new_fusions) == 0:
                    break
                del self.atoms[new_fusions[0]]
                use_plus(self, new_fusions[0])

        turn_played = False
        if self.center_atom.special == False:
            if (self.turns_since_last_minus == 0 and self.total_turns >= 1 and clicked_mid): # there was just a minus
                convert_to_plus(self)
                turn_played = True
                return
            place_normal(self)
            turn_played = True

        if turn_played == False:
            if self.center_atom.symbol == "+" and self.atom_count == 1:
                place_normal(self)
            elif self.center_atom.symbol == "+" and self.atom_count > 1:
                use_plus(self)
            elif self.center_atom.symbol == "-":
                use_minus(self)

        check_new_fusions(self)

    def draw_outer(self, screen, clicked_mid = False, is_human_player=False):
        if self.atom_count == 0:
            return

        def atom_ring_locations(r, n=100):
            return [(SCREEN_WIDTH // 2 + math.cos(2*pi/n*x)*r, SCREEN_HEIGHT // 2 + 30 + math.sin(2*pi/n*x)*r) for x in range(0, n+1)]

        self.locations = atom_ring_locations(170, self.atom_count)

        if is_human_player and pygame.mouse.get_focused():
            if self.turns_since_last_minus == 0 and self.total_turns >= 1 and self.center_atom.symbol != "-":
                if not clicked_mid:
                    self.generate_placement_line()
            else:
                self.generate_placement_line()

        for atom, location in zip(self.atoms, self.locations):
            if atom.symbol == "+":
                self.special_atoms("+", (218, 77, 57), location,screen)
            elif atom.symbol == "-":
                self.special_atoms("-", (68, 119, 194), location,screen)
            else:
                self.normal_atoms(atom, location, screen)

    def normal_atoms(self, atom, location, screen):
        '''Helper function to draw normal atoms.'''
        atom_nb = atom.atom_number
        atom_symbol = atom.symbol
        atom_colour = atom.colour

        pygame.draw.circle(screen, pygame.Color(atom_colour),
                           (location[0], location[1]), 23)

        text_atom_symbol = self.font.render(
            atom_symbol, True, (255, 255, 255))
        text_rect_symbol = text_atom_symbol.get_rect(
            center=(location[0], location[1] - 7))
        screen.blit(text_atom_symbol, text_rect_symbol)

        text_atom_number = self.font.render(
            str(atom_nb), True, (255, 255, 255))
        text_rect_number = text_atom_number.get_rect(
            center=(location[0], location[1] + 10))
        screen.blit(text_atom_number, text_rect_number)

    def special_atoms(self, symbol, colour, location,screen):
        '''Helper function to draw special ("+") atoms.'''

        pygame.draw.circle(screen, colour,
                           (location[0],location[1]), 23)

        text_atom_symbol = self.font.render(
            symbol, True, (255, 255, 255))
        text_rect_symbol = text_atom_symbol.get_rect(
            center=(location[0], location[1]))
        screen.blit(text_atom_symbol, text_rect_symbol)

    def midway_points(self):
        '''Helper function to get the midway points between each atom.'''
        return [[[(self.locations[i][0] + self.locations[i+1][0]) / 2, (self.locations[i][1] + self.locations[i+1][1]) / 2], i]
                for i in range(len(self.locations) - 1)]

    def closest_midway(self, mouse_pos):
        distances = [[[i[0], abs(math.dist(i[0], mouse_pos))], i[1]] for i in self.midway_points()]
        return min(distances, key=lambda x: x[0][1])

    def closest_atom(self, mouse_pos):
        distances = [[[self.locations[i], abs(math.dist(self.locations[i], mouse_pos))], i] for i in range(len(self.locations))]
        return min(distances, key=lambda x: x[0][1])

    def generate_placement_line(self, screen):
        mouse_pos = pygame.mouse.get_pos()

        if self.center_atom.symbol == "-":
            closest_point = self.closest_atom(mouse_pos)[0][0]
        else:
            closest_point = self.closest_midway(mouse_pos)[0][0]

        pygame.draw.line(screen, (0, 0, 0), (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30), closest_point)
