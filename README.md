
# Atomas AI Bot - Command-Line Agent

This repository provides an AI bot that can play the Atomas game with multiple agent configurations. It includes various agents like Expectimax, MCTS, Ayelet, and Reflex agents. The game is run from the command line, and the user can specify various options, including which agent to use and other game settings.

## Setup

1. Install dependencies:
   ```bash
   pip install pygame pandas numpy
   ```

2. Clone the repository and navigate to the project directory.

3. The implementation is based on and modifies the Atomas game code from the `atomas-python` repository found on GitHub [here](https://github.com/Mjnstag/atomas-python). 

## Running the Game

The main entry point of the AI bot is through the `main()` function, which processes command-line arguments to run the game. You can configure various aspects of the game, including which AI agent to use, whether to display the game visually, and how many games to run.

### Example Command

```bash
python atomas_game.py --num_of_games 5 --agent expectimax --depth 3 --priority highest_atom --display True
```

### Command-Line Arguments

| Argument                | Type    | Default | Description                                                                 |
|-------------------------|---------|---------|-----------------------------------------------------------------------------|
| `--num_of_games`         | `int`   | `1`     | Number of games to run.                                                     |
| `--display`              | `bool`  | `True`  | Whether to display the game window (True/False).                            |
| `--sleep_between_actions`| `bool`  | `True`  | Should the game pause briefly between each action (True/False).             |
| `--print_move`           | `bool`  | `True`  | Whether to print each move to the console (True/False).                     |
| `--agent`                | `str`   | `expectimax` | Type of agent to use (`random`, `ayelet`, `reflex`, `mcts`, `expectimax`). |
| `--depth`                | `int`   | `2`     | Depth of the Expectimax search (used when agent is `expectimax`).           |
| `--simulations`          | `int`   | `20`    | Number of simulations for the MCTS agent (used when agent is `mcts`).       |
| `--priority`             | `str`   | `score` | Priority for the Expectimax and MCTS agents: `score` or `highest_atom`.     |

### Agents

- **ExpectimaxAgent**: A depth-based agent that evaluates moves by calculating the maximum possible outcomes using either a scoring or highest atom priority.
- **MCTSAgent**: A Monte Carlo Tree Search agent that simulates multiple random games to evaluate the best move.
- **AyeletAgent**: A custom rule-based agent with predefined strategies for playing the game.
- **ReflexAgent**: An agent that uses simple rules to make decisions based on the current game state.
- **SmartRandomAgent**: A more intelligent random agent that makes decisions with basic heuristics.

### Running the Game

Once you've set your options, the game runs the number of games specified in the `--num_of_games` argument. Results, including the final score and the highest atom achieved, are printed after each game.

### Output Example
```bash
Starting game 1 of 5...
Highest Score: 12345
Highest Atom Achieved: 54
Average Score: 9876
```


