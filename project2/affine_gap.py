from Bio import SeqIO
from Bio.Align import substitution_matrices, PairwiseAligner
import numpy as np


def score_settings():
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    return aligner


def calc_matrix(row_seq: str, col_seq: str):
    cols = len(col_seq)
    rows = len(row_seq)
    alignment = score_settings()
    print(alignment)
    matrix = np.zeros([rows + 1, cols + 1], dtype=np.int32)

    # initialize
    for index in range(1, rows + 1):
        matrix[index][0] = alignment.open_gap_score + index * alignment.extend_gap_score
    for index in range(1, cols + 1):
        matrix[0][index] = alignment.open_gap_score + index * alignment.extend_gap_score

    print(matrix)

    # fill matrix
    for col in range(1, cols + 1):
        for row in range(1, rows + 1):
            match = matrix[row - 1][col - 1] + alignment.substitution_matrix[col_seq[col - 1]][row_seq[row - 1]]
            gap_first = matrix[row][col - 1] + alignment.gap_score
            gap_second = matrix[row - 1][col] + alignment.gap_score
            matrix[row][col] = max(match, gap_first, gap_second)

    return matrix


def load_file(filename: str):
    data = list(SeqIO.parse(filename, "fasta"))
    data = sorted(data, key=lambda x: len(x.seq), reverse=True)
    first = str(data[0].seq)
    second = str(data[1].seq)
    return first, second


def global_alignment_score(filename: str):
    col_seq, row_seq = load_file(filename)
    matrix = calc_matrix(row_seq, col_seq)
    return int(matrix[-1][-1])


def global_alignment(filename: str):
    col_seq, row_seq = load_file(filename)
    matrix = calc_matrix(row_seq, col_seq)
    alignment = score_settings()

    row = len(row_seq)
    col = len(col_seq)

    # backtrack route
    row_align = ['-'] * col
    while row > 0 and col > 0:
        if matrix[row][col] - alignment.substitution_matrix[row_seq[row - 1]][col_seq[col - 1]] == matrix[row - 1][col - 1]:
            row_align[col - 1] = row_seq[row - 1]
            row -= 1
            col -= 1
        elif matrix[row][col] - alignment.gap_score == matrix[row][col - 1]:
            row_align[col - 1] = "-"
            col -= 1
        else:
            row -= 1

    row_align = ''.join(row_align)

    return col_seq, row_align


def affine_gap(filename: str):
    import time
    start = time.time()
    file = load_file(filename)
    print(f"first:  {file[0]}")
    print(f"second: {file[1]}")
    print(global_alignment_score(filename))
    # res = global_alignment(filename)
    # print(f"seq01: {res[0]}")
    # print(f"seq02: {res[1]}")
    print(f"ETA: {time.time() - start}")


affine_gap("data/data_01_affine.fna")
