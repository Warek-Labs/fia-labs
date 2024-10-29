import random

from sudoku import *

GRID_SIZE = 9
SUBGRID_SIZE = 3
MIN_CLUES = 17


class SudokuGenerator:
    def __init__(self):
        pass

    def next(self) -> Grid:
        full_board = self._create_full_sudoku()
        return self._remove_cells(full_board, GRID_SIZE * GRID_SIZE - MIN_CLUES)

    def _is_valid(self, grid: Grid, row: int, col: int, value: int) -> bool:
        for x in range(GRID_SIZE):
            if grid[row][x] == value or grid[x][col] == value:
                return False

        start_row, start_col = row - row % SUBGRID_SIZE, col - col % SUBGRID_SIZE
        for i in range(SUBGRID_SIZE):
            for j in range(SUBGRID_SIZE):
                if grid[i + start_row][j + start_col] == value:
                    return False
        return True

    def _fill_sudoku(self, grid: Grid) -> bool:
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if grid[row][col] is None:
                    nums = list(range(1, GRID_SIZE + 1))
                    random.shuffle(nums)
                    for num in nums:
                        if self._is_valid(grid, row, col, num):
                            grid[row][col] = num
                            if self._fill_sudoku(grid):
                                return True
                            grid[row][col] = None
                    return False
        return True

    def _create_full_sudoku(self) -> Grid:
        grid = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self._fill_sudoku(grid)
        return grid

    def _remove_cells(self, grid: Grid, num_cells_to_remove: int) -> Grid:
        cells_removed = 0
        attempts = 0
        while cells_removed < num_cells_to_remove and attempts < 100:
            row, col = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
            if grid[row][col] is not None:
                original_value = grid[row][col]
                grid[row][col] = None
                if not self._has_unique_solution(grid):
                    grid[row][col] = original_value
                else:
                    cells_removed += 1
            attempts += 1
        return grid

    def _count_solutions(self, board: Grid) -> int:
        empty_cell = self._find_empty(board)
        if not empty_cell:
            return 1
        row, col = empty_cell
        count = 0
        for num in range(1, GRID_SIZE + 1):
            if self._is_valid(board, row, col, num):
                board[row][col] = num
                count += self._count_solutions(board)
                board[row][col] = None
        return count

    def _find_empty(self, grid: Grid) -> Optional[tuple[int, int]]:
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r][c] is None:
                    return (r, c)
        return None

    def _has_unique_solution(self, grid: Grid) -> bool:
        solution_count = self._count_solutions([row[:] for row in grid])
        return solution_count == 1
