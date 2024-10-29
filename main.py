from generator import *

solver = SudokuSolver()
generator = SudokuGenerator()
grid1 = generator.next()
solver.load_grid(grid1)
# solver.load_file('input.txt')
solver.print()
print('===========================================')

if solver.solve_constraint_propagation_heuristic():
    solver.print()
    # print(solver.test_from_file('test.txt'))
else:
    print('Can\'t solve')
