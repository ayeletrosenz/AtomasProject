from config import Action, NO_SELECTION


def print_move(game_state, action):
    action_type, chosen_atom_index, midway_index = action
    clicked_mid = False
    if action_type == Action.CONVERT_TO_PLUS:
        clicked_mid = True
    print("----------", game_state.ring.total_turns, "----------")
    print("Center atom: ", game_state.ring.center_atom.atom_number)
    print("Atoms: [", ", ".join(str(atom.atom_number) for atom in game_state.ring.atoms), "]")
    if clicked_mid:
        print("Switched to plus")
    else:
        if chosen_atom_index and chosen_atom_index != -1:
            print("Chosen atom index: ", chosen_atom_index)
        if midway_index and midway_index != -1:
            print("Chosen midway index: ", midway_index)
