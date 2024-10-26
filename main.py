from generator import *
from utils import *

generator = SudokuGenerator()
sudoku = SudokuSolver()
sudoku.load_grid(generator.next())

if sudoku.is_valid():
    sudoku.print()
    print('==============================')
    if sudoku.solve_backtracking():
        sudoku.print()
else:
    print('Invalid grid')
