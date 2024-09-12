from config import PLUS, MINUS, atom_data
import random


class Atom:
    def __init__(self, atom_nb=0):
        self.score = 0
        self.special = False
        if atom_nb != 0:
            self.create(atom_nb)

    def create(self, atom_nb=1):
        self.atom_number = atom_nb
        self.colour = atom_data['Color'][str(self.atom_number-1)]
        self.symbol = atom_data['Sym'][str(self.atom_number-1)]

    def create_random(self):
        self.atom_number = random.randint(1, 125)
        self.colour = atom_data['Color'][str(self.atom_number-1)]
        self.symbol = atom_data['Sym'][str(self.atom_number-1)]

    def create_plus(self):
        self.atom_number = PLUS
        self.colour = (218, 77, 57)
        self.symbol = "+"
        self.special = True

    def create_minus(self):
        self.atom_number = MINUS
        self.colour = (68, 119, 194)
        self.symbol = "-"
        self.special = True