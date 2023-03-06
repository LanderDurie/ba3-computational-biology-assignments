from Bio import SeqIO
from Bio.Align import substitution_matrices, PairwiseAligner
import numpy as np


def score_settings():
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1
    return aligner


def print_matrix(matrix: np.array):
    for i in range(3):
        print("\n")
        for row in matrix:
            print(" ".join(["".join([" "] * (1 if item[i] >= 0 else 0)) + str(item[i]) + "".join(
                [" "] * (6 - (1 if item[i] >= 0 else 0) - len(str(item[i])))) for item in row]))


def calc_matrix(row_seq: str, col_seq: str):
    cols = len(col_seq)
    rows = len(row_seq)
    alignment = score_settings()

    matrix = np.zeros([rows + 1, cols + 1, 3])
    matrix_direction = np.zeros([rows + 1, cols + 1, 3], dtype=np.int32)

    # initialize
    for index in range(1, rows + 1):
        matrix[index][0][0] = alignment.open_gap_score + index * alignment.extend_gap_score
        matrix[index][0][1] = alignment.open_gap_score + index * alignment.extend_gap_score
        matrix[index][0][2] = -np.inf
    for index in range(1, cols + 1):
        matrix[0][index][0] = alignment.open_gap_score + index * alignment.extend_gap_score
        matrix[0][index][1] = -np.inf
        matrix[0][index][2] = alignment.open_gap_score + index * alignment.extend_gap_score

    # fill matrix
    for col in range(1, cols + 1):
        for row in range(1, rows + 1):
            # options
            down_options = [matrix[row - 1][col][0] + (alignment.open_gap_score + alignment.extend_gap_score),
                            matrix[row - 1][col][1] + alignment.extend_gap_score]

            right_options = [matrix[row][col - 1][0] + (alignment.open_gap_score + alignment.extend_gap_score),
                             matrix[row][col - 1][2] + alignment.extend_gap_score]

            # followed path
            followed_path = [0, np.array(down_options).argmax(), np.array(right_options).argmax()]

            # fill matrix
            matrix[row][col][1] = down_options[followed_path[1]]
            matrix[row][col][2] = right_options[followed_path[2]]
            matrix_direction[row][col][1] = followed_path[1]
            matrix_direction[row][col][2] = followed_path[2]

            # diagonal
            diag_options = [
                matrix[row - 1][col - 1][0] + alignment.substitution_matrix[row_seq[row - 1]][col_seq[col - 1]],
                matrix[row][col][1],
                matrix[row][col][2]]
            followed_path[0] = np.array(diag_options).argmax()

            matrix[row][col][0] = diag_options[followed_path[0]]
            matrix_direction[row][col][0] = followed_path[0]

    return matrix, matrix_direction


def load_file(filename: str):
    data = list(SeqIO.parse(filename, "fasta"))
    first = str(data[0].seq)
    second = str(data[1].seq)
    return first, second


def global_alignment_score(filename: str):
    col_seq, row_seq = load_file(filename)
    matrix, _ = calc_matrix(row_seq, col_seq)
    return int(max(matrix[-1][-1]))


def global_alignment(filename: str):
    import time
    col_seq, row_seq = load_file(filename)
    start = time.time()
    matrix, matrix_dir = calc_matrix(row_seq, col_seq)
    print(time.time() - start)

    row = len(row_seq)
    col = len(col_seq)

    # backtrack route
    row_align = []
    col_align = []

    current = 0

    while row > 0 and col > 0:

        direction = matrix_dir[row][col][current]

        if current == 0:

            if direction == 0:
                row_align = [row_seq[row - 1]] + row_align
                col_align = [col_seq[col - 1]] + col_align
                col -= 1
                row -= 1
            elif direction == 1:
                current = 1
            elif direction == 2:
                current = 2

        elif current == 1:
            row_align = [row_seq[row - 1]] + row_align
            col_align = ["-"] + col_align
            row -= 1
            if direction == 0:
                current = 0

        elif current == 2:
            col_align = [col_seq[col - 1]] + col_align
            row_align = ["-"] + row_align
            col -= 1
            if direction == 0:
                current = 0

    # fill remaining
    if row > 0:
        for i in reversed(range(0, row)):
            row_align = [row_seq[i]] + row_align
            col_align = ["-"] + col_align
    if col > 0:
        for i in reversed(range(0, col)):
            col_align = [col_seq[i]] + col_align
            row_align = ["-"] + row_align

    row_align = ''.join(row_align)
    col_align = ''.join(col_align)

    return col_align, row_align


def affine_gap(filename: str):
    import time
    start = time.time()
    file = load_file(filename)
    print(f"first:  {file[0]}")
    print(f"second: {file[1]}")
    print(global_alignment_score(filename))
    res = global_alignment(filename)
    print(f"seq01: {res[0]}")
    print(f"seq02: {res[1]}")
    print(f"ETA: {time.time() - start}")


affine_gap("data/data_06_affine.fna")
