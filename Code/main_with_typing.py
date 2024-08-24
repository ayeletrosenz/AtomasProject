import pygame
import random
import math
import json
import numpy as np
from typing import List, Tuple, Union
import agents

pi = math.pi

pygame.init()

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 700

pygame.display.set_caption('Atomas')
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
with open("Code/atom_data.json", "r") as f:
    atom_data = json.load(f)


class Score:
    def __init__(self):
        self.score: int = 0
        self.base_font: pygame.font.Font = pygame.font.Font(None, 32)
        self.score_location: Tuple[int, int] = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 6)
        self.score_name_location: Tuple[int, int] = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 6 + 30)

    def reset(self) -> None:
        self.score = 0

    def update(self, amount: int) -> None:
        self.score += amount

    def draw(self, atom_nb: int) -> None:
        '''Draws the score and the name of the highest scoring atom. 
        The text colour of the name of the atom is based on the atom itself.'''
        def draw_num_score() -> None:
            text_surface = self.base_font.render(
                str(self.score), True, (255, 255, 255))
            text_rect = text_surface.get_rect(
                center=self.score_location)
            screen.blit(text_surface, text_rect)

        def draw_atom_name(atom_nb: int) -> None:
            atom_nb_str = str(atom_nb)
            atom_name = atom_data['Name'][str(int(atom_nb_str)-1)]
            atom_colour = atom_data['Color'][str(int(atom_nb_str)-1)]
            text_surface = self.base_font.render(
                atom_name, True, pygame.Color(atom_colour))
            text_rect = text_surface.get_rect(
                center=self.score_name_location)
            screen.blit(text_surface, text_rect)

        draw_num_score()
        draw_atom_name(atom_nb)

    def calc_chain_score(self, sym: List[Tuple[int, int]], sym_indices: List[List[int]], atoms: List['Atom']) -> None:
        '''Calculates the score of a chain of atoms.'''
        print("Calculating chain score...")

        score_increase: int = 0
        reaction_step: int = 1
        shiny: bool = False

        def reaction_multiplier(reaction_step: int) -> float:
            return 1 + (0.5 * reaction_step)

        def simple_reaction(atom_nb: int) -> Tuple[int, int]:
            '''Helper function to calculate the score of a simple reaction.'''
            multiplier = reaction_multiplier(reaction_step)
            score = math.floor(multiplier * (atom_nb + 1))
            print(f"math.floor({multiplier} * ({atom_nb} + 1)) = {score}")
            return score, atom_nb + 1

        def complex_reaction(atom_nb_inner: int, atom_nb_outer: int) -> Tuple[int, int]:
            multiplier = reaction_multiplier(reaction_step)
            score, _ = simple_reaction(atom_nb_inner)

            score += int(2 * multiplier * (atom_nb_outer - atom_nb_inner + 1))
            print(f"2 * {multiplier} * ({atom_nb_outer} - {atom_nb_inner} + 1) = {int(2 * multiplier * (atom_nb_outer - atom_nb_inner + 1))}")

            return score, atom_nb_outer + 2

        if len(sym) == 1:
            print("Simple reaction")
            score, new_atom_nb = simple_reaction(sym[0][0])
            score_increase += score

            for atom_indices in sym_indices:
                for indice in atom_indices:
                    if atoms[indice].shiny:
                        score_increase += atoms[indice].score

            ring.atoms.insert(sym_indices[0][0] + 1, Atom(new_atom_nb))
            del ring.atoms[sym_indices[0][0]]
            del ring.atoms[sym_indices[0][1]]

        else:
            for atom_nb, atom_indices in zip(sym, sym_indices):

                if reaction_step == 1:
                    score, atom_nb_inner = simple_reaction(atom_nb[0])
                    score_increase += score
                    reaction_step += 1
                    continue
                score = 0

                outer_number = atom_nb[0]
                if outer_number > atom_nb_inner:
                    print('complex reaction')
                    score, atom_nb_inner = complex_reaction(atom_nb_inner, outer_number)
                else:
                    print("Simple reaction")
                    score, atom_nb_inner = simple_reaction(atom_nb_inner)

                for indice in atom_indices:
                    if atoms[indice].shiny:
                        score_increase += atoms[indice].score

                if reaction_step >= 4:
                    shiny = True

                score_increase += score
                reaction_step += 1

            indices = sorted([item for sublist in sym_indices for item in sublist], reverse=True)
            for index in indices:
                del ring.atoms[index]

            ring.atoms.insert(sym_indices[0][0] + 1, Atom(atom_nb_inner, shiny))
        self.update(score_increase)
        print("Score:", score_increase)


class Background:
    def __init__(self):
        self.BACKGROUND_COLOR: Tuple[int, int, int] = (82, 42, 50)
        self.RING_COLOUR: Tuple[int, int, int] = (133, 94, 97)
        self.ATOM_RING_CENTER: Tuple[int, int] = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30)

    def draw(self) -> None:
        '''Draws the background as well as the ring surrounding the atoms.'''
        screen.fill(self.BACKGROUND_COLOR)
        pygame.draw.circle(screen, self.RING_COLOUR, self.ATOM_RING_CENTER, 190)
        pygame.draw.circle(screen, self.BACKGROUND_COLOR, self.ATOM_RING_CENTER, 187)


class Atom:
    def __init__(self, atom_nb: int = 0, shiny: bool = False):
        self.shiny: bool = shiny
        self.score: int = 0
        self.special: bool = False
        if atom_nb != 0:
            self.create(atom_nb)

    def create(self, atom_nb: int = 1) -> None:
        self.atom_number: int = atom_nb
        self.colour: str = atom_data['Color'][str(self.atom_number - 1)]
        self.symbol: str = atom_data['Sym'][str(self.atom_number - 1)]

    def create_random(self) -> None:
        self.atom_number = random.randint(1, 125)
        self.colour = atom_data['Color'][str(self.atom_number - 1)]
        self.symbol = atom_data['Sym'][str(self.atom_number - 1)]

    def create_plus(self) -> None:
        self.atom_number = -1
        self.colour = (218, 77, 57)
        self.symbol = "+"
        self.special = True

    def create_minus(self) -> None:
        self.atom_number = -2
        self.colour = (68, 119, 194)
        self.symbol = "-"
        self.special = True


class Ring:
    def __init__(self):
        self.atom_count: int = 0
        self.max_atoms: int = 18
        self.highest_atom: int = 1
        self.atoms: List[Atom] = []
        self.score: Score = Score()
        self.center_atom: Union[str, Atom] = ""
        self.font: pygame.font.Font = pygame.font.Font(None, 25)
        self.locations: List[Tuple[int, int]] = []

        self.total_turns: int = 1
        self.turns_since_last_plus: int = 0
        self.turns_since_last_minus: int = 0

    def update_highest(self) -> None:
        '''Helper function to update the highest scoring atom in the ring.'''
        to_check = [i.atom_number for i in self.atoms if i.atom_number > 0]

        current_highest = int(sorted(to_check, key=int)[-1])
        if current_highest > self.highest_atom:
            self.highest_atom = current_highest

    def start_game(self) -> None:
        '''Function to start the game.
        This will randomly pick 6 atoms ranging from 1 to 3 and a center atom, the last of which can include a "+".'''

        self.atoms = [Atom(random.randint(1, 3)) for _ in range(6)]
        self.update_atom_count()
        self.update_highest()
        self.generate_inner()

    def update_atom_count(self) -> None:
        '''Helper function to update the atom count.'''
        self.atom_count = len(self.atoms)
        return

    def generate_inner(self) -> None:
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
        # every 40 turns the highest atom which can spawn is is increased by 1
        range_max = 3 + self.total_turns // 40
        range_min = 1 + self.total_turns // 40

        special_atoms: List[str] = ["+"]
        outlier_atoms: List[int] = []
        for atom in self.atoms:
            if (atom.atom_number > 0) and (atom.atom_number < range_min) and (atom.atom_number not in outlier_atoms):
                outlier_atoms.append(atom.atom_number)

        possible_atoms: List[Union[int, str]] = outlier_atoms + special_atoms + [x for x in range(range_min, range_max + 1)]
        outlier_weight: float = 1 / self.atom_count
        possible_atoms_weights: List[float] = [outlier_weight for _ in range(len(outlier_atoms))] + [(1 - outlier_weight * len(outlier_atoms)) / 4 for _ in range(4)]

        chosen_atom: Union[int, str] = random.choices(possible_atoms, possible_atoms_weights, k=1)[0]
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

    def draw_inner(self) -> None:
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

    def place_atom(self) -> None:
        def check_game_end(self) -> None:
            '''Function to check if the player loses the game.
            '''
            if self.atom_count >= self.max_atoms:
                for atom in self.atoms:
                    if atom.atom_number >= 1:
                        self.score.update(atom.atom_number)

                print("Game over")
                print("Score:", self.score.score)
                print("Highest atom:", self.highest_atom)
                pygame.quit()
                exit()

        def place_normal(self) -> None:
            '''Function to place a non-special atom.'''
            mouse_pos = pygame.mouse.get_pos()
            closest_midway = self.closest_midway(mouse_pos)[1]
            print("Place normal: ", self.center_atom.symbol, self.center_atom.atom_number)

            self.atoms.insert(closest_midway + 1, self.center_atom)
            self.update_atom_count()
            self.generate_inner()

        def find_symmetry_indices(atoms: List['Atom'], pivot: int) -> Tuple[List[Tuple[int, int]], List[List[int]]]:
            atom_list = [atom.atom_number for atom in atoms]
            atom_indices = [i for i in range(len(atom_list))]
            n = len(atom_list)

            atom_list.insert(pivot, "p")
            atom_indices.insert(pivot, "p")
            n = len(atom_list)

            roll = np.roll(atom_list, n // 2 - pivot)
            roll = [str(i) if isinstance(i, str) else int(i) for i in roll]
            print(roll)
            roll_indices = np.roll(atom_indices, n // 2 - pivot)
            roll_indices = [str(i) if isinstance(i, str) else int(i) for i in roll_indices]

            sym_numbers: List[Tuple[int, int]] = []
            sym_indices: List[List[int]] = []

            for i in range(1, n // 2):
                print(roll[n // 2 - i], roll[n // 2 + i])
                if (roll[n // 2 - i] == roll[n // 2 + i]) and (roll[n // 2 - i] >= 0) and (roll[n // 2 + i] >= 0):
                    sym_numbers.append((roll[n // 2 - i], roll[n // 2 + i]))
                    sym_indices.append((roll_indices[n // 2 - i], roll_indices[n // 2 + i]))
                else:
                    break
            return sym_numbers, sym_indices

        def use_plus(self, indice: int = -1) -> None:
            mouse_pos = pygame.mouse.get_pos()
            closest_midway = self.closest_midway(mouse_pos)[1]
            atom_list = self.atoms.copy()
            print("using plus", indice)

            if indice != -1:
                sym, sym_indices = find_symmetry_indices(atom_list, indice)
            else:
                sym, sym_indices = find_symmetry_indices(atom_list, closest_midway + 1)

            if len(sym) == 0:
                place_normal(self)
                self.update_atom_count()
                return
            elif len(sym) >= 1:
                self.score.calc_chain_score(sym, sym_indices, atom_list)
            self.update_atom_count()

        def use_minus(self) -> None:
            closest_atom_index = self.closest_atom(pygame.mouse.get_pos())[1]
            self.center_atom = self.atoms.pop(closest_atom_index)

        def convert_to_plus(self) -> None:
            self.center_atom = Atom()
            self.center_atom.create_plus()
            self.turns_since_last_minus = 1

        def check_new_fusions(self) -> None:
            '''Function to check if there are any new fusions after a turn due to played atoms.
            Example 1: [2, +, 1] > [2, +, 2, 1] => [3, 1]
            Example 2: [2, +, 1, 1] > [2, +, 1, +, 1] > [2, +, 2] => [3]'''
            while True:
                atoms = self.atoms
                atom_list = [atom.atom_number for atom in atoms]
                atom_indices = [i for i in range(len(atom_list))]

                new_fusions: List[int] = []
                for number, indice in zip(range(len(atom_list)), atom_indices):
                    if atom_list[number] == -1:
                        if atom_list[number - 1] == atom_list[(number + 1) % len(atom_list)]:
                            new_fusions.append(indice)
                            print(atom_list[number - 1], atom_list[number], atom_list[(number + 1) % len(atom_list)])
                print(new_fusions)
                if len(new_fusions) == 0:
                    break
                del self.atoms[new_fusions[0]]
                use_plus(self, new_fusions[0])

        # check if player loses when they play an atom
        check_game_end(self)

        turn_played: bool = False
        print("--------------")
        print("Current middle atom: ", self.center_atom.atom_number, self.center_atom.symbol)

        # check if player has clicked on an atom grabbed by a minus atom to convert it to a plus atom
        if not self.center_atom.special:
            mouse_pos = pygame.mouse.get_pos()
            if self.turns_since_last_minus == 0 and self.total_turns >= 1 and abs(
                    math.dist((SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30), mouse_pos)) < 40:
                convert_to_plus(self)
                turn_played = True
                return
            place_normal(self)
            turn_played = True

        if not turn_played:
            if self.center_atom.symbol == "+" and self.atom_count == 1:
                place_normal(self)
            elif self.center_atom.symbol == "+" and self.atom_count > 1:
                print("here", self.center_atom.symbol)
                use_plus(self)
            elif self.center_atom.symbol == "-":
                use_minus(self)

        check_new_fusions(self)

    def draw_outer(self) -> None:
        if self.atom_count == 0:
            return

        def atom_ring_locations(r: int, n: int = 100) -> List[Tuple[int, int]]:
            return [(SCREEN_WIDTH // 2 + math.cos(2 * pi / n * x) * r,
                     SCREEN_HEIGHT // 2 + 30 + math.sin(2 * pi / n * x) * r) for x in range(0, n + 1)]

        self.locations = atom_ring_locations(170, self.atom_count)

        if pygame.mouse.get_focused():
            if self.turns_since_last_minus == 0 and self.total_turns >= 1 and self.center_atom.symbol != "-":
                mouse_pos = pygame.mouse.get_pos()
                if abs(math.dist((SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30), mouse_pos)) > 40:
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

    def normal_atoms(self, atom: Atom, location: Tuple[int, int]) -> None:
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

    def special_atoms(self, symbol: str, colour: Tuple[int, int, int], location: Tuple[int, int]) -> None:
        '''Helper function to draw special ("+") atoms.'''

        pygame.draw.circle(screen, colour,
                           (location[0], location[1]), 23)

        text_atom_symbol = self.font.render(
            symbol, True, (255, 255, 255))
        text_rect_symbol = text_atom_symbol.get_rect(
            center=(location[0], location[1]))
        screen.blit(text_atom_symbol, text_rect_symbol)

    def midway_points(self) -> List[List[Union[Tuple[float, float], int]]]:
        '''Helper function to get the midway points between each atom.'''
        return [[[(self.locations[i][0] + self.locations[i + 1][0]) / 2,
                  (self.locations[i][1] + self.locations[i + 1][1]) / 2], i]
                for i in range(len(self.locations) - 1)]

    def closest_midway(self, mouse_pos: Tuple[int, int]) -> List[Union[Tuple[float, float], int]]:
        distances = [[[i[0], abs(math.dist(i[0], mouse_pos))], i[1]] for i in self.midway_points()]
        return min(distances, key=lambda x: x[0][1])

    def closest_atom(self, mouse_pos: Tuple[int, int]) -> List[Union[Tuple[float, float], int]]:
        distances = [[[self.locations[i], abs(math.dist(self.locations[i], mouse_pos))], i] for i in range(len(self.locations))]
        return min(distances, key=lambda x: x[0][1])

    def generate_placement_line(self) -> None:
        mouse_pos = pygame.mouse.get_pos()

        if self.center_atom.symbol == "-":
            closest_point = self.closest_atom(mouse_pos)[0][0]
        else:
            closest_point = self.closest_midway(mouse_pos)[0][0]

        pygame.draw.line(screen, (0, 0, 0), (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30), closest_point)


if __name__ == "__main__":
    background = Background()
    ring = Ring()
    ring.start_game()
    agent = agents.SimpleAgent()

    run: bool = True
    while run:
        background.draw()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                ring.place_atom()
                ring.total_turns += 1

        ring.update_highest()
        ring.score.draw(ring.highest_atom)
        ring.update_atom_count()

        ring.draw_outer()
        ring.draw_inner()

        pygame.display.flip()
        clock.tick(5)

    pygame.quit()
