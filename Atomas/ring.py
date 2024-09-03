import numpy as np
import pygame
import random
import math
from math import pi
from atom import Atom
from score import Score
from config import SCREEN_WIDTH, SCREEN_HEIGHT, screen, clicked_mid, IS_HUMAN_PLAYER


class Ring:
    def __init__(self):
        self.atom_count = 0
        self.max_atoms = 18
        self.atoms = []
        self.center_atom = ""
        self.highest_atom = 1
        self.score = Score()
        self.locations = []
        self.font = pygame.font.Font(None, 25)

        self.total_turns = 1
        self.turns_since_last_plus = 0
        self.turns_since_last_minus = 0

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

    def update_highest(self):
        '''Helper function to update the highest scoring atom in the ring.'''
        to_check = [i.atom_number for i in self.atoms if i.atom_number > 0]

        current_highest = int(sorted(to_check, key=int)[-1])
        if current_highest > self.highest_atom:
            self.highest_atom = current_highest

    def generate_inner(self):
        '''Function to generate the atom in the center of the ring.
        The one that the player can place next.'''

        # spawn a minus atom if there hasn't been one in the last 20 turns
        if self.turns_since_last_minus >= 20:
            self.center_atom = Atom()
            self.center_atom.create_minus()
            self.turns_since_last_minus = 0
            self.turns_since_last_plus += 1
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

    def check_game_end(self):
        '''Function to check if the game has ended.
        This will return True if the game has ended, False otherwise.'''
        if self.atom_count >= self.max_atoms:
            return True
        return False

    def place_atom(self, atom_index, midway_index, switch_to_plus, is_human_player=True):
        '''Function to place an atom in the ring.
        This will place the atom in the ring and update the score.'''
        def place_normal(self):
            '''Helper function to place a normal atom in the ring.'''
            self.atoms.insert(midway_index+1, self.center_atom)
            self.update_atom_count()
            self.update_highest()
            self.generate_inner()

        def find_symmetry_atoms(atoms, pivot):
            atom_list = [atom.atom_number for atom in atoms]
            atom_indices = [i for i in range(len(atom_list))]
            n = len(atom_list)

            atom_list.insert(pivot, "p")
            atom_indices.insert(pivot, "p")
            n = len(atom_list)

            roll = np.roll(atom_list, n//2 - pivot)
            roll = [str(i) if i.isalpha() else int(i) for i in roll]
            # print(roll)
            roll_indices = np.roll(atom_indices, n//2 - pivot)
            roll_indices = [str(i) if i.isalpha() else int(i) for i in roll_indices]

            sym_numbers = []
            sym_indices = []

            for i in range(1, n//2):
                # print(roll[n//2 - i], roll[n//2 + i])
                if (roll[n//2 - i] == roll[n//2 + i]) and (roll[n//2 - i] >= 0) and (roll[n//2 + i] >= 0):
                    sym_numbers.append((roll[n//2 - i], roll[n//2 + i]))
                    sym_indices.append((roll_indices[n//2 - i], roll_indices[n//2 + i]))
                else:
                    break
            return sym_numbers, sym_indices

        def place_plus(self, indice = -1):
            atom_list = self.atoms.copy()
            score_increase = 0
            if indice != -1:
                sym, sym_indices = find_symmetry_atoms(atom_list, indice)
            else:
                sym, sym_indices = find_symmetry_atoms(atom_list, midway_index+1)

            if len(sym) == 0:
                place_normal(self)
                self.update_atom_count()
                return
            elif len(sym) == 1:
                new_score, new_atom_nb = Score.calc_chain_score(self).simple_reaction(sym[0][0])
                score_increase += new_score

                self.atoms.insert(sym_indices[0][0]+1, Atom(new_atom_nb))
                del self.atoms[sym_indices[0][0]]
                del self.atoms[sym_indices[0][1]]
            else:
                reaction_step = 1
                for atom_nb, atom_indices in zip(sym, sym_indices):
                    if reaction_step == 1:
                        new_score, atom_nb_inner = Score.calc_chain_score(self).simple_reaction(atom_nb[0])
                        score_increase += new_score
                        reaction_step += 1
                        continue
                    new_score = 0
                    outer_number = atom_nb[0]
                    if outer_number > atom_nb_inner:
                        new_score, atom_nb_inner = Score.calc_chain_score(self).complex_reaction(atom_nb_inner, outer_number)
                    else:
                        new_score, atom_nb_inner = Score.calc_chain_score(self).simple_reaction(atom_nb_inner)
                    score_increase += new_score
                    reaction_step += 1

            indices = sorted([item for sublist in sym_indices for item in sublist], reverse=True)
            for index in indices:
                del self.atoms[index]

            self.atoms.insert(sym_indices[0][0]+1, Atom(atom_nb_inner))

            self.update_atom_count()
            self.update_highest()
            self.score.update_score(score_increase)

        def place_minus(self):
            '''Helper function to place a minus atom in the ring.'''
            self.center_atom = self.atoms.pop(atom_index)

        def convert_to_plus(self):
            '''Helper function to convert the center atom to a plus atom.'''
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

                new_fusions = []
                for number, indice in zip(range(len(atom_list)), atom_indices):
                    if atom_list[number] == -1:
                        if atom_list[number-1] == atom_list[(number+1)%len(atom_list)]:
                            new_fusions.append(indice)
                            # print(atom_list[number-1], atom_list[number], atom_list[(number+1)%len(atom_list)])
                # print(new_fusions)
                if len(new_fusions) == 0:
                    break
                del self.atoms[new_fusions[0]]
                place_plus(self, new_fusions[0])

        # check if player loses when they play an atom
        self.check_game_end()

        turn_played = False
        # print("--------------")
        # print("Current middle atom: ", self.center_atom.atom_number, self.center_atom.symbol)

        # check if player has clicked on an atom grabbed by a minus atom to convert it to a plus atom
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
                # print("here", self.center_atom.symbol)
                place_plus(self)
            elif self.center_atom.symbol == "-":
                place_minus(self)

        check_new_fusions(self)

    def draw_ring(self):
        def draw_inner(self):
            '''Function to draw the atom in the center of the ring.
            The one that the player can place next.'''

        match self.center_atom.symbol:
            case "":
                return
            case "+":
                self.special_atoms("+", (218, 77, 57), (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30))
            case "-":
                self.special_atoms("-", (68, 119, 194), (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30))
            case _:
                self.normal_atoms(self.center_atom, (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30))

        def draw_outer(self):
            if self.atom_count == 0:
                return

            def atom_ring_locations(r, n=100):
                return [(SCREEN_WIDTH // 2 + math.cos(2*pi/n*x)*r, SCREEN_HEIGHT // 2 + 30 + math.sin(2*pi/n*x)*r) for x in range(0, n+1)]

            self.locations = atom_ring_locations(170, self.atom_count)

            if IS_HUMAN_PLAYER and pygame.mouse.get_focused():
                if self.turns_since_last_minus == 0 and self.total_turns >= 1 and self.center_atom.symbol != "-":
                    if not clicked_mid:
                        self.generate_placement_line()
                else:
                    self.generate_placement_line()

            for atom, location in zip(self.atoms, self.locations):
                if atom.symbol == "+":
                    self.special_atoms("+", (218, 77, 57), location)
                elif atom.symbol == "-":
                    self.special_atoms("-", (68, 119, 194), location)
                else:
                    self.normal_atoms(atom, location)
        draw_outer(self)
        draw_inner(self)


    def normal_atoms(self, atom, location):
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

    def special_atoms(self, symbol, colour, location):
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

    def generate_placement_line(self):
        mouse_pos = pygame.mouse.get_pos()

        if self.center_atom.symbol == "-":
            closest_point = self.closest_atom(mouse_pos)[0][0]
        else:
            closest_point = self.closest_midway(mouse_pos)[0][0]

        pygame.draw.line(screen, (0, 0, 0), (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30), closest_point)

    def print_move(self, game_state, chosen_atom_index, chosen_midway_index, clicked_mid):
        print("\n----------", game_state.total_turns, "----------")
        print("Center atom: ", game_state.center_atom)
        print("Atoms: ", game_state.atoms)
        if clicked_mid:
            print("Switched to plus")
        else:
            if chosen_atom_index != -1:
                print("Chosen atom index: ", chosen_atom_index)
            if chosen_midway_index != -1:
                print("Chosen midway index: ", chosen_midway_index)
