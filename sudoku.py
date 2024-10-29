from collections import deque
from collections.abc import Callable
from typing import Tuple, Optional
import copy

GRID_SIZE = 9
MIN_VALUE = 1
MAX_VALUE = GRID_SIZE
SUBGRID_SIZE = int(GRID_SIZE ** 0.5)
MIN_CLUES = 17
Grid = list[list[Optional[int]]]
DomainsGrid = list[list[set[int]]]


class SudokuSolver:
    grid: Grid
    _domains: DomainsGrid

    def __init__(self):
        self.grid = []
        self._domains = []

    def load_file(self, filename: str):
        """Load grid from a file."""
        self.grid = self.parse_file(filename)
        self.init()

    def load_grid(self, ls: Grid):
        """Load grid from an existing array."""
        self.grid = ls
        self.init()

    def init(self):
        """Tasks to do when loading grid."""
        self._domains = []
        self._compute_all_domains()

    @staticmethod
    def parse_file(filename: str) -> Grid:
        """Load the unsolved grid from a file."""
        grid: Grid = []

        with open(filename, 'r') as file:
            # omit empty lines and trim whitespace
            total_lines = [line.strip() for line in file.readlines() if line.strip()]

        if len(total_lines) != GRID_SIZE:
            raise Exception(f'Invalid input, number of lines ({len(total_lines)})')

        for index, line in enumerate(total_lines):
            if len(line) != GRID_SIZE:
                raise Exception(f'Invalid input, line length ({len(line)}) at {index}')

            row = [int(c) if c.isnumeric() else None for c in line]
            grid.append(row)

        return grid

    def test_from_file(self, filename: str) -> bool:
        """Check if the grid is equal as the one from file"""
        with open(filename, 'r') as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.grid[y][x] != int(lines[y][x]):
                    return False

        return True

    def print(self) -> None:
        """Print the grid"""
        for i, row in enumerate(self.grid):
            for j, value in enumerate(row):
                print(value if value is not None else '*', end=' ')

                # vertical subgrid spacing
                if j % SUBGRID_SIZE == SUBGRID_SIZE - 1 and j != GRID_SIZE - 1:
                    print(end=' ')

            # horizontal subgrid spacing
            if i % SUBGRID_SIZE == SUBGRID_SIZE - 1 and i != GRID_SIZE - 1:
                print()

            print()

    def is_valid(self) -> bool:
        """Is the grid valid"""
        for row in range(GRID_SIZE):
            if not self._is_valid_row(row) or not self._is_valid_col(row):
                return False

        for subgrid_row in range(SUBGRID_SIZE):
            for subgrid_col in range(SUBGRID_SIZE):
                if not self._is_valid_subgrid(subgrid_row, subgrid_col):
                    return False

        return True

    def solve_backtracking(self) -> bool:
        """Try to solve the grid using backtracking and return the success"""
        return self._solve_backtracking(0, 0)

    def solve_ac3_with_backtracking(self) -> bool:
        """Solve the grid using AC-3 for initial constraint propagation, followed by backtracking."""
        if not self._ac3():
            return False

        return self._solve_backtracking_with_constraint_propagation(0, 0)

    def solve_constraint_propagation(self) -> bool:
        """Try to solve the grid using constraint propagation and return the success"""
        return self._solve_constraint_propagation()

    def solve_constraint_propagation_heuristic(self) -> bool:
        """Try to solve the grid using constraint propagation and return the success"""
        return self._solve_backtracking_with_constraint_propagation_heuristic()

    @staticmethod
    def _get_subgrid_offsets(row: int, col: int) -> Tuple[int, int]:
        """Gets the offset of grid row and columns of cells in a certain subgrid, ex: for (1, 1) on 9x9 grid
        will return (3, 3)"""
        return row - row % SUBGRID_SIZE, col - col % SUBGRID_SIZE

    @staticmethod
    def _get_subgrid_coords(cell_row: int, cell_col: int) -> Tuple[int, int]:
        """Get subgrid row and column"""
        return cell_row // SUBGRID_SIZE, cell_col // SUBGRID_SIZE

    @staticmethod
    def _get_all_possible_values() -> list[int]:
        """Gets a list of all possible value that a cell can have"""
        return list(range(MIN_VALUE, MAX_VALUE + 1))

    def _compute_all_domains(self) -> None:
        """Initialize the domain (total possible values given the rule constraints) of each cell in the grid"""
        self._domains = []

        for i in range(GRID_SIZE):
            row = []

            for j in range(GRID_SIZE):
                row.append(set())

            self._domains.append(row)

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                self._update_cell_domain(row, col)

    def _update_cell_domain(self, row: int, col: int) -> None:
        """Update _domains for a single cell."""
        if self.grid[row][col] is not None:
            self._clear_cell_domain(row, col)
            return

        domain = self._domains[row][col] = set(self._get_all_possible_values())
        self._update_cell_row_domain(domain, row)
        self._update_cell_col_domain(domain, col)
        self._update_cell_subgrid_domain(domain, row, col)

    def _clear_cell_domain(self, row: int, col: int) -> None:
        """Mark cell domain as empty (either can't have values or is already used)."""
        self._domains[row][col] = set()

    def _update_cell_row_domain(self, domain: set[int], row: int) -> None:
        """Update domain based on constraints of neighbors on row."""
        for col in range(GRID_SIZE):
            c = self.grid[row][col]

            if c in domain:
                domain.remove(c)

    def _update_cell_col_domain(self, domain: set[int], col: int) -> None:
        """Update domain based on constraints of neighbors on column."""
        for row in range(GRID_SIZE):
            c = self.grid[row][col]

            if c in domain:
                domain.remove(c)

    def _update_cell_subgrid_domain(self, domain: set[int], row: int, col: int) -> None:
        """Update domain based on constraints of neighbors on the same subgrid."""
        def subgrid_cb(row: int, col: int, value: int) -> None:
            if value in domain:
                domain.remove(value)

        self._foreach_subgrid(row, col, subgrid_cb)

    def _foreach_subgrid(self, row: int, col: int, cb: Callable[[int, int, int], None]):
        """Calls the given callback for all the cells in a subgrid"""
        row_offset, col_offset = self._get_subgrid_offsets(row, col)

        for i in range(SUBGRID_SIZE):
            for j in range(SUBGRID_SIZE):
                row = i + row_offset
                col = j + col_offset
                value = self.grid[row][col]
                cb(row, col, value)

    def _forward_propagate(self, row: int, col: int) -> None:
        """Forward constraint propagation for x, y and subgrid cells"""
        self._forward_propagate_row(row)
        self._forward_propagate_col(col)
        self._forward_propagate_subgrid(row, col)

    def _forward_propagate_row(self, row: int) -> None:
        """Propagate domain constraints to cells on the same row."""

        for col in range(GRID_SIZE):
            self._update_cell_domain(row, col)

    def _forward_propagate_col(self, col: int) -> None:
        """Propagate domain constraints to cells on the same column."""
        for row in range(GRID_SIZE):
            self._update_cell_domain(row, col)

    def _forward_propagate_subgrid(self, row: int, col: int) -> None:
        """Propagate domain constraints to cells on the same subgrid."""
        def cb(row: int, col: int, value: int) -> None:
            self._update_cell_domain(row, col)

        self._foreach_subgrid(row, col, cb)

    def _can_place(self, row: int, col: int, value: int) -> bool:
        """Whether possible to place value in a cell and not break the rules"""
        return (self._is_valid_row(row, value)
                and self._is_valid_col(col, value)
                and self._is_valid_subgrid(row, col, value))

    def _is_valid_row(self, row: int, check_value: Optional[int] = None) -> bool:
        """Check the rules on X"""
        visited: set[int] = { check_value }

        for col in range(GRID_SIZE):
            c = self.grid[row][col]

            if c is None:
                continue

            if c in visited:
                return False

            visited.add(c)

        return True

    def _is_valid_col(self, col: int, check_value: Optional[int] = None) -> bool:
        """Check the rules on Y"""
        visited: set[int] = { check_value }

        for row in range(GRID_SIZE):
            c = self.grid[row][col]

            if c is None:
                continue

            if c in visited:
                return False

            visited.add(c)

        return True

    def _is_valid_subgrid(self, subgrid_row: int, subgrid_col: int, check_value: Optional[int] = None) -> bool:
        """Check the rules for a subgrid."""
        visited: set[int] = { check_value }

        row_offset, col_offset = self._get_subgrid_offsets(subgrid_row, subgrid_col)

        for i in range(SUBGRID_SIZE):
            for j in range(SUBGRID_SIZE):
                row = i + row_offset
                col = j + col_offset
                c = self.grid[row][col]

                if c is None:
                    continue

                if c in visited:
                    return False

                visited.add(c)

        return True

    def _solve_backtracking(self, row: int, col: int) -> bool:
        """Simple backtracking implementation."""
        if row == GRID_SIZE - 1 and col == GRID_SIZE:
            return True

        if col == GRID_SIZE:
            row += 1
            col = 0

        if self.grid[row][col] is not None:
            return self._solve_backtracking(row, col + 1)

        for value in self._get_all_possible_values():
            if self._can_place(row, col, value):
                self.grid[row][col] = value

                if self._solve_backtracking(row, col + 1):
                    return True

            self.grid[row][col] = None

        return False

    def _solve_backtracking_with_constraint_propagation(self, row: int, col: int) -> bool:
        """Backtracking with forward propagation for the domains of cells."""
        if row == GRID_SIZE - 1 and col == GRID_SIZE:
            return True

        if col == GRID_SIZE:
            row += 1
            col = 0

        if self.grid[row][col] is not None:
            return self._solve_backtracking(row, col + 1)

        domains_snapshot = copy.deepcopy(self._domains)

        for value in self._domains[row][col]:
            if self._can_place(row, col, value):
                self.grid[row][col] = value
                self._forward_propagate(row, col)

                if self._solve_backtracking(row, col + 1):
                    return True

            self.grid[row][col] = None
            self._domains = domains_snapshot

        return False

    def _solve_constraint_propagation(self) -> bool:
        """Constraint propagation approach, pickes the values that are unique in domain of cells and proceeds to simple bakctracking."""
        progress_made = True

        while progress_made:
            progress_made = False
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    if self.grid[row][col] is None:
                        self._update_cell_domain(row, col)
                        domain_size = len(self._domains[row][col])

                        if domain_size == 1:
                            only_value = self._domains[row][col].pop()
                            self.grid[row][col] = only_value
                            self._clear_cell_domain(row, col)
                            progress_made = True
                            print(f'Propagated, applied {{{only_value}}} for ({col}, {row})')

                        # if impossible state
                        elif domain_size == 0:
                            return False

        # fallback to backtracking
        return self._solve_backtracking_with_constraint_propagation(0, 0)

    def _find_cell_with_least_domains(self) -> Optional[Tuple[int, int]]:
        """Find the cell with the fewest domain options available."""
        min_domain_size: int = MAX_VALUE
        cell_with_least_domains: Optional[Tuple[int, int]] = None

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.grid[row][col] is None:
                    domain_size = len(self._domains[row][col])

                    if 0 < domain_size < min_domain_size:
                        min_domain_size = domain_size
                        cell_with_least_domains = (row, col)

        return cell_with_least_domains

    def _solve_backtracking_with_constraint_propagation_heuristic(self) -> bool:
        """Solve using backtracking with constraint propagation and selecting cells with the fewest domains first."""
        cell = self._find_cell_with_least_domains()
        if cell is None:
            return True

        row, col = cell
        domains_snapshot = copy.deepcopy(self._domains)

        for value in self._domains[row][col]:
            if self._can_place(row, col, value):
                self.grid[row][col] = value
                self._forward_propagate(row, col)

                if self._solve_backtracking_with_constraint_propagation_heuristic():
                    return True

                self.grid[row][col] = None
                self._domains = domains_snapshot

        return False

    def _ac3(self) -> bool:
        """Enforce arc consistency on the grid."""
        queue = deque()

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.grid[row][col] is None:
                    neighbors = self._get_neighbors(row, col)
                    for neighbor in neighbors:
                        queue.append(((row, col), neighbor))

        while queue:
            (x1, y1), (x2, y2) = queue.popleft()

            if self._revise(x1, y1, x2, y2):
                if len(self._domains[x1][y1]) == 0:
                    return False

                neighbors = self._get_neighbors(x1, y1)
                for neighbor in neighbors:
                    if neighbor != (x2, y2):
                        queue.append(((x1, y1), neighbor))

        return True

    def _revise(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Revise the domain of cell (x1, y1) to satisfy arc consistency with cell (x2, y2)."""
        revised = False
        domain_x1 = self._domains[x1][y1]
        domain_x2 = self._domains[x2][y2]

        for value in set(domain_x1):
            if len(domain_x2) == 1 and value in domain_x2:
                domain_x1.remove(value)
                revised = True

        return revised

    def _get_neighbors(self, row: int, col: int) -> list[Tuple[int, int]]:
        """Get all neighboring cells in the same row, column, and subgrid."""
        neighbors = set()

        for c in range(GRID_SIZE):
            if c != col:
                neighbors.add((row, c))

        for r in range(GRID_SIZE):
            if r != row:
                neighbors.add((r, col))

        subgrid_row, subgrid_col = self._get_subgrid_offsets(row, col)
        for i in range(SUBGRID_SIZE):
            for j in range(SUBGRID_SIZE):
                r, c = subgrid_row + i, subgrid_col + j
                if (r, c) != (row, col):
                    neighbors.add((r, c))

        return list(neighbors)
