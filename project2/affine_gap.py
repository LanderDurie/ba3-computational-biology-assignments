import time
import gc
from Bio.SeqIO import parse
from numpy import True_, bool_, zeros

BLOSUM = {
    'A': {'A': 4, 'R': -1, 'N': -2, 'D': -2, "C": 0, 'Q': -1, 'E': -1, 'G': 0, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 0, 'W': -3, 'Y': -2, 'V': 0, 'B': -2,
          'Z': -1, 'X': 0, '*': -4},
    'R': {'A': -1, 'R': 5, 'N': 0, 'D': -2, "C": -3, 'Q': 1, 'E': 0, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 2, 'M': -1, 'F': -3, 'P': -2, 'S': -1, 'T': -1, 'W': -3, 'Y': -2, 'V': -3, 'B': -1, 'Z': 0,
          'X': -1, '*': -4},
    'N': {'A': -2, 'R': 0, 'N': 6, 'D': 1, "C": -3, 'Q': 0, 'E': 0, 'G': 0, 'H': 1, 'I': -3, 'L': -3, 'K': 0, 'M': -2, 'F': -3, 'P': -2, 'S': 1, 'T': 0, 'W': -4, 'Y': -2, 'V': -3, 'B': 3, 'Z': 0,
          'X': -1, '*': -4},
    'D': {'A': -2, 'R': -2, 'N': 1, 'D': 6, "C": -3, 'Q': 0, 'E': 2, 'G': -1, 'H': -1, 'I': -3, 'L': -4, 'K': -1, 'M': -3, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3, 'B': 4, 'Z': 1,
          'X': -1, '*': -4},
    'C': {'A': 0, 'R': -3, 'N': -3, 'D': -3, "C": 9, 'Q': -3, 'E': -4, 'G': -3, 'H': -3, 'I': -1, 'L': -1, 'K': -3, 'M': -1, 'F': -2, 'P': -3, 'S': -1, 'T': -1, 'W': -2, 'Y': -2, 'V': -1, 'B': -3,
          'Z': -3, 'X': -2, '*': -4},
    'Q': {'A': -1, 'R': 1, 'N': 0, 'D': 0, "C": -3, 'Q': 5, 'E': 2, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 1, 'M': 0, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -2, 'Y': -1, 'V': -2, 'B': 0, 'Z': 3,
          'X': -1, '*': -4},
    'E': {'A': -1, 'R': 0, 'N': 0, 'D': 2, "C": -4, 'Q': 2, 'E': 5, 'G': -2, 'H': 0, 'I': -3, 'L': -3, 'K': 1, 'M': -2, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2, 'B': 1, 'Z': 4,
          'X': -1, '*': -4},
    'G': {'A': 0, 'R': -2, 'N': 0, 'D': -1, "C": -3, 'Q': -2, 'E': -2, 'G': 6, 'H': -2, 'I': -4, 'L': -4, 'K': -2, 'M': -3, 'F': -3, 'P': -2, 'S': 0, 'T': -2, 'W': -2, 'Y': -3, 'V': -3, 'B': -1,
          'Z': -2, 'X': -1, '*': -4},
    'H': {'A': -2, 'R': 0, 'N': 1, 'D': -1, "C": -3, 'Q': 0, 'E': 0, 'G': -2, 'H': 8, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -1, 'P': -2, 'S': -1, 'T': -2, 'W': -2, 'Y': 2, 'V': -3, 'B': 0, 'Z': 0,
          'X': -1, '*': -4},
    'I': {'A': -1, 'R': -3, 'N': -3, 'D': -3, "C": -1, 'Q': -3, 'E': -3, 'G': -4, 'H': -3, 'I': 4, 'L': 2, 'K': -3, 'M': 1, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -3, 'Y': -1, 'V': 3, 'B': -3,
          'Z': -3, 'X': -1, '*': -4},
    'L': {'A': -1, 'R': -2, 'N': -3, 'D': -4, "C": -1, 'Q': -2, 'E': -3, 'G': -4, 'H': -3, 'I': 2, 'L': 4, 'K': -2, 'M': 2, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -2, 'Y': -1, 'V': 1, 'B': -4,
          'Z': -3, 'X': -1, '*': -4},
    'K': {'A': -1, 'R': 2, 'N': 0, 'D': -1, "C": -3, 'Q': 1, 'E': 1, 'G': -2, 'H': -1, 'I': -3, 'L': -2, 'K': 5, 'M': -1, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2, 'B': 0, 'Z': 1,
          'X': -1, '*': -4},
    'M': {'A': -1, 'R': -1, 'N': -2, 'D': -3, "C": -1, 'Q': 0, 'E': -2, 'G': -3, 'H': -2, 'I': 1, 'L': 2, 'K': -1, 'M': 5, 'F': 0, 'P': -2, 'S': -1, 'T': -1, 'W': -1, 'Y': -1, 'V': 1, 'B': -3,
          'Z': -1, 'X': -1, '*': -4},
    'F': {'A': -2, 'R': -3, 'N': -3, 'D': -3, "C": -2, 'Q': -3, 'E': -3, 'G': -3, 'H': -1, 'I': 0, 'L': 0, 'K': -3, 'M': 0, 'F': 6, 'P': -4, 'S': -2, 'T': -2, 'W': 1, 'Y': 3, 'V': -1, 'B': -3,
          'Z': -3, 'X': -1, '*': -4},
    'P': {'A': -1, 'R': -2, 'N': -2, 'D': -1, "C": -3, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -4, 'P': 7, 'S': -1, 'T': -1, 'W': -4, 'Y': -3, 'V': -2, 'B': -2,
          'Z': -1, 'X': -2, '*': -4},
    'S': {'A': 1, 'R': -1, 'N': 1, 'D': 0, "C": -1, 'Q': 0, 'E': 0, 'G': 0, 'H': -1, 'I': -2, 'L': -2, 'K': 0, 'M': -1, 'F': -2, 'P': -1, 'S': 4, 'T': 1, 'W': -3, 'Y': -2, 'V': -2, 'B': 0, 'Z': 0,
          'X': 0, '*': -4},
    'T': {'A': 0, 'R': -1, 'N': 0, 'D': -1, "C": -1, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 5, 'W': -2, 'Y': -2, 'V': 0, 'B': -1,
          'Z': -1, 'X': 0, '*': -4},
    'W': {'A': -3, 'R': -3, 'N': -4, 'D': -4, "C": -2, 'Q': -2, 'E': -3, 'G': -2, 'H': -2, 'I': -3, 'L': -2, 'K': -3, 'M': -1, 'F': 1, 'P': -4, 'S': -3, 'T': -2, 'W': 11, 'Y': 2, 'V': -3, 'B': -4,
          'Z': -3, 'X': -2, '*': -4},
    'Y': {'A': -2, 'R': -2, 'N': -2, 'D': -3, "C": -2, 'Q': -1, 'E': -2, 'G': -3, 'H': 2, 'I': -1, 'L': -1, 'K': -2, 'M': -1, 'F': 3, 'P': -3, 'S': -2, 'T': -2, 'W': 2, 'Y': 7, 'V': -1, 'B': -3,
          'Z': -2, 'X': -1, '*': -4},
    'V': {'A': 0, 'R': -3, 'N': -3, 'D': -3, "C": -1, 'Q': -2, 'E': -2, 'G': -3, 'H': -3, 'I': 3, 'L': 1, 'K': -2, 'M': 1, 'F': -1, 'P': -2, 'S': -2, 'T': 0, 'W': -3, 'Y': -1, 'V': 4, 'B': -3,
          'Z': -2, 'X': -1, '*': -4},
    'B': {'A': -2, 'R': -1, 'N': 3, 'D': 4, "C": -3, 'Q': 0, 'E': 1, 'G': -1, 'H': 0, 'I': -3, 'L': -4, 'K': 0, 'M': -3, 'F': -3, 'P': -2, 'S': 0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3, 'B': 4, 'Z': 1,
          'X': -1, '*': -4},
    'Z': {'A': -1, 'R': 0, 'N': 0, 'D': 1, "C": -3, 'Q': 3, 'E': 4, 'G': -2, 'H': 0, 'I': -3, 'L': -3, 'K': 1, 'M': -1, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2, 'B': 1, 'Z': 4,
          'X': -1, '*': -4},
    'X': {'A': 0, 'R': -1, 'N': -1, 'D': -1, "C": -2, 'Q': -1, 'E': -1, 'G': -1, 'H': -1, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -1, 'P': -2, 'S': 0, 'T': 0, 'W': -2, 'Y': -1, 'V': -1, 'B': -1,
          'Z': -1, 'X': -1, '*': -4},
    '*': {'A': -4, 'R': -4, 'N': -4, 'D': -4, "C": -4, 'Q': -4, 'E': -4, 'G': -4, 'H': -4, 'I': -4, 'L': -4, 'K': -4, 'M': -4, 'F': -4, 'P': -4, 'S': -4, 'T': -4, 'W': -4, 'Y': -4, 'V': -4, 'B': -4,
          'Z': -4, 'X': -4, '*': 1}
}


def calc_matrix(row_seq: str, col_seq: str):
    cols = len(col_seq)
    rows = len(row_seq)

    score_matrix = [[[0 for _ in range(rows + 1)] for _ in range(3)] for _ in range(2)]
    matrix_direction = [[zeros(cols + 1, dtype=bool_) for _ in range(rows + 1)] for _ in range(4)]

    for row in range(1, rows + 1):
        score_matrix[0][0][row] = -row - 10
        score_matrix[0][1][row] = -row - 10
        score_matrix[0][2][row] = -2147483647

    score_matrix[0][1][0] = -2147483647
    score_matrix[1][1][0] = -2147483647

    indexes_options = [[1, 0], [0, 1]]

    # fill matrix
    for col in range(1, cols + 1):
        indexes = indexes_options[col % 2]

        score_matrix[indexes[1]][0][0] = -col - 10
        score_matrix[indexes[1]][2][0] = -col - 10

        first_letter_dict = BLOSUM[col_seq[col - 1]]

        for row in range(1, rows + 1):

            middle_up, lower_up = score_matrix[indexes[1]][0][row - 1], score_matrix[indexes[1]][1][row - 1]

            if middle_up < lower_up + 10:
                lower_max = lower_up - 1
                # set lower max index
                matrix_direction[1][row][col] = True_
            else:
                lower_max = middle_up - 11

            middle_left, upper_left = score_matrix[indexes[0]][0][row], score_matrix[indexes[0]][2][row]

            if middle_left < upper_left + 10:
                upper_max = upper_left - 1
                # set upper max index
                matrix_direction[2][row][col] = True_
            else:
                upper_max = middle_left - 11

            # calculate and assign middle matrix

            # follow middle cost
            middle_max = score_matrix[indexes[0]][0][row - 1] + first_letter_dict.get(row_seq[row - 1])

            # compare lower and upper
            if lower_max < upper_max:
                if middle_max < upper_max:
                    middle_max = upper_max
                    # set middle max index
                    matrix_direction[0][row][col] = True_
                # set lower / upper direction
                matrix_direction[3][row][col] = True_
            else:
                if middle_max < lower_max:
                    middle_max = lower_max
                    # set middle max index
                    matrix_direction[0][row][col] = True_

            score_matrix[indexes[1]][0][row], score_matrix[indexes[1]][1][row], score_matrix[indexes[1]][2][row] = middle_max, lower_max, upper_max

    score = max((score_matrix[cols % 2][x][-1] for x in range(3)))

    return score, matrix_direction


def load_file(filename: str):
    data = list(parse(filename, "fasta"))
    first = str(data[0].seq)
    second = str(data[1].seq)
    return first, second


def global_alignment_score(filename: str):
    gc.disable()
    col_seq, row_seq = load_file(filename)
    score, _ = calc_matrix(row_seq, col_seq)
    return score


def global_alignment(filename: str):
    gc.disable()
    col_seq, row_seq = load_file(filename)
    _, matrix_dir = calc_matrix(row_seq, col_seq)

    row = len(row_seq)
    col = len(col_seq)

    # backtrack route
    row_align = ""
    col_align = ""

    current = 0

    while row > 0 and col > 0:

        direction = matrix_dir[current][row][col]

        if current == 0:
            if direction:
                if matrix_dir[3][row][col]:
                    current = 2
                else:
                    current = 1
            else:
                row_align = row_seq[row - 1] + row_align
                col_align = col_seq[col - 1] + col_align
                col -= 1
                row -= 1

        elif current == 1:
            row_align = row_seq[row - 1] + row_align
            col_align = '-' + col_align
            row -= 1
            if not direction:
                current = 0

        else:
            col_align = col_seq[col - 1] + col_align
            row_align = '-' + row_align
            col -= 1
            if not direction:
                current = 0

    # fill remaining
    if row > 0:
        for i in reversed(range(0, row)):
            row_align = row_seq[i] + row_align
            col_align = '-' + col_align
    if col > 0:
        for i in reversed(range(0, col)):
            col_align = col_seq[i] + col_align
            row_align = '-' + row_align

    return col_align, row_align


def affine_gap(filename: str):
    file = load_file(filename)
    print(f"first:  {file[0]}")
    print(f"second: {file[1]}")
    start = time.time()
    # print(global_alignment_score(filename))
    res = global_alignment(filename)
    print(f"seq01 : {res[0]}")
    print(f"seq02 : {res[1]}")
    print(f"ET: {time.time() - start}")


if __name__ == '__main__':
    affine_gap("data/data_1000_benchmark.fna")
    # affine_gap("data/data_19_affine.fna")
    # affine_gap("data/data_01_affine.fna")
