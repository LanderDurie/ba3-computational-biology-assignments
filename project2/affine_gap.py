import sys
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

    """
    Calculate the score matrix and backtracking matrix.
    The score matrix has been reduced to 2 columns to reduce memory. We can now constantly switch between these 2 columns.
    During every position calculation step, The origin of the new value is stored in the backtracking (direction) matrix.
    We can use this matrix to backtrack through both strings.
    """

    # pre calc sizes
    cols = len(col_seq)
    rows = len(row_seq)
    total = cols * rows

    # create score and backtracking (direction) matrices

    score_cols_middle_new = [0]*(rows+1)
    score_cols_lower_new = [0]*(rows+1)
    score_cols_upper_new = [0]*(rows+1)
    score_cols_middle_old = [0]*(rows+1)
    score_cols_lower_old = [0]*(rows+1)
    score_cols_upper_old = [0]*(rows+1)

    matrix_direction_middle = zeros(total, dtype=bool_)
    matrix_direction_lower = zeros(total, dtype=bool_)
    matrix_direction_upper = zeros(total, dtype=bool_)
    matrix_direction_spec = zeros(total, dtype=bool_)
    # matrix_direction_middle = 0
    # matrix_direction_lower = 0
    # matrix_direction_upper = 0
    # matrix_direction_spec = 0

    # initialize score matrix
    for row in range(1, rows):
        score_cols_middle_new[row] = score_cols_lower_old[row] = - row - 10
        score_cols_upper_new[row] = -sys.maxsize
    score_cols_lower_new[0] = score_cols_lower_old[0] = -sys.maxsize

    # fill matrix
    # loop through cols
    total_index = -1
    for col in range(1, cols+1):

        # switch current cols
        score_cols_middle_old, score_cols_middle_new = score_cols_middle_new, score_cols_middle_old
        score_cols_lower_old, score_cols_lower_new = score_cols_lower_new, score_cols_lower_old
        score_cols_upper_old, score_cols_upper_new = score_cols_upper_new, score_cols_upper_old
        # init new value
        score_cols_middle_new[0] = score_cols_upper_new[0] = -col - 10
        # get dict with first letter (smaller dict later on)
        first_letter_dict = BLOSUM[col_seq[col - 1]]

        # loop through rows
        for row in range(1, rows+1):
            total_index += 1

            # calc new value for lower matrix
            middle_up, lower_up = score_cols_middle_new[row - 1], score_cols_lower_new[row - 1]
            if middle_up < lower_up + 10:
                lower_max = lower_up - 1
                # set lower max index
                matrix_direction_lower[total_index] = True_
                # matrix_direction_lower |= 1 << total_index
            else:
                lower_max = middle_up - 11

            # calc new value for upper matrix
            middle_left, upper_left = score_cols_middle_old[row], score_cols_upper_old[row]
            if middle_left < upper_left + 10:
                upper_max = upper_left - 1
                # set upper max index
                matrix_direction_upper[total_index] = True_
                # matrix_direction_upper |= 1 << total_index
            else:
                upper_max = middle_left - 11

            # calculate and assign middle matrix
            # follow middle cost
            middle_max = score_cols_middle_old[row - 1] + first_letter_dict.get(row_seq[row - 1])
            # compare lower and upper
            if lower_max < upper_max:
                if middle_max < upper_max:
                    middle_max = upper_max
                    # set middle max index
                    matrix_direction_middle[total_index] = True_
                    # matrix_direction_middle |= 1 << total_index
                # set lower / upper direction
                matrix_direction_spec[total_index] = True_
                # matrix_direction_spec |= 1 << total_index
            else:
                if middle_max < lower_max:
                    middle_max = lower_max
                    # set middle max index
                    matrix_direction_middle[total_index] = True_
                    # matrix_direction_middle |= 1 << total_index

            # assign values in score matrix
            score_cols_middle_new[row], score_cols_lower_new[row], score_cols_upper_new[row] = middle_max, lower_max, upper_max

    score = score_cols_middle_new[-1]
    return score, [matrix_direction_middle, matrix_direction_lower, matrix_direction_upper, matrix_direction_spec]


def load_file(filename: str):
    """
    Load both sequences from a fasta file using Bio.SeqIO
    """
    data = list(parse(filename, "fasta"))
    first = str(data[0].seq)
    second = str(data[1].seq)
    return first, second


def global_alignment_score(filename: str):
    """
    calculate the score matrix and return the score
    """
    col_seq, row_seq = load_file(filename)
    score, _ = calc_matrix(row_seq, col_seq)
    return score


def global_alignment(filename: str):
    """
    calculate the score matrix and backtrack through the calculated matrix.
    We can do this by looking at the value at a certain position and then backtracking to the previous one.
    """
    col_seq, row_seq = load_file(filename)
    _, matrix_dir = calc_matrix(row_seq, col_seq)

    rows = len(row_seq)
    row = len(row_seq)-1
    col = len(col_seq)-1

    # backtrack route
    row_align = ""
    col_align = ""

    current = 0
    while row >= 0 and col >= 0:
        index = rows * col + row
        direction = matrix_dir[current][index]
        if current == 0:
            if direction:
                if matrix_dir[3][index]:
                    current = 2
                else:
                    current = 1
            else:
                row_align = row_seq[row] + row_align
                col_align = col_seq[col] + col_align
                col -= 1
                row -= 1
        elif current == 1:
            row_align = row_seq[row] + row_align
            col_align = '-' + col_align
            row -= 1
            if not direction:
                current = 0
        else:
            col_align = col_seq[col] + col_align
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
    import time
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
    # affine_gap("data/data_1000_benchmark.fna")
    # affine_gap("data/data_19_affine.fna")
    affine_gap("data/data_01_affine.fna")
    # affine_gap("data/data_04_affine.fna")