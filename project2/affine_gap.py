import time
import gc
from Bio.SeqIO import parse
import numpy as np

substitution_matrix = {'AA': 4, 'AR': -1, 'AN': -2, 'AD': -2, 'AC': 0, 'AQ': -1, 'AE': -1, 'AG': 0,
                       'AH': -2, 'AI': -1, 'AL': -1, 'AK': -1, 'AM': -1, 'AF': -2, 'AP': -1,
                       'AS': 1, 'AT': 0, 'AW': -3, 'AY': -2, 'AV': 0, 'AB': -2, 'AJ': -1, 'AZ': -1,
                       'AX': -1, 'A*': -4, 'RA': -1, 'RR': 5, 'RN': 0, 'RD': -2, 'RC': -3, 'RQ': 1,
                       'RE': 0, 'RG': -2, 'RH': 0, 'RI': -3, 'RL': -2, 'RK': 2, 'RM': -1, 'RF': -3,
                       'RP': -2, 'RS': -1, 'RT': -1, 'RW': -3, 'RY': -2, 'RV': -3, 'RB': -1,
                       'RJ': -2, 'RZ': 0, 'RX': -1, 'R*': -4, 'NA': -2, 'NR': 0, 'NN': 6, 'ND': 1,
                       'NC': -3, 'NQ': 0, 'NE': 0, 'NG': 0, 'NH': 1, 'NI': -3, 'NL': -3, 'NK': 0,
                       'NM': -2, 'NF': -3, 'NP': -2, 'NS': 1, 'NT': 0, 'NW': -4, 'NY': -2, 'NV': -3,
                       'NB': 4, 'NJ': -3, 'NZ': 0, 'NX': -1, 'N*': -4, 'DA': -2, 'DR': -2, 'DN': 1,
                       'DD': 6, 'DC': -3, 'DQ': 0, 'DE': 2, 'DG': -1, 'DH': -1, 'DI': -3, 'DL': -4,
                       'DK': -1, 'DM': -3, 'DF': -3, 'DP': -1, 'DS': 0, 'DT': -1, 'DW': -4,
                       'DY': -3, 'DV': -3, 'DB': 4, 'DJ': -3, 'DZ': 1, 'DX': -1, 'D*': -4, 'CA': 0,
                       'CR': -3, 'CN': -3, 'CD': -3, 'CC': 9, 'CQ': -3, 'CE': -4, 'CG': -3,
                       'CH': -3, 'CI': -1, 'CL': -1, 'CK': -3, 'CM': -1, 'CF': -2, 'CP': -3,
                       'CS': -1, 'CT': -1, 'CW': -2, 'CY': -2, 'CV': -1, 'CB': -3, 'CJ': -1,
                       'CZ': -3, 'CX': -1, 'C*': -4, 'QA': -1, 'QR': 1, 'QN': 0, 'QD': 0, 'QC': -3,
                       'QQ': 5, 'QE': 2, 'QG': -2, 'QH': 0, 'QI': -3, 'QL': -2, 'QK': 1, 'QM': 0,
                       'QF': -3, 'QP': -1, 'QS': 0, 'QT': -1, 'QW': -2, 'QY': -1, 'QV': -2, 'QB': 0,
                       'QJ': -2, 'QZ': 4, 'QX': -1, 'Q*': -4, 'EA': -1, 'ER': 0, 'EN': 0, 'ED': 2,
                       'EC': -4, 'EQ': 2, 'EE': 5, 'EG': -2, 'EH': 0, 'EI': -3, 'EL': -3, 'EK': 1,
                       'EM': -2, 'EF': -3, 'EP': -1, 'ES': 0, 'ET': -1, 'EW': -3, 'EY': -2,
                       'EV': -2, 'EB': 1, 'EJ': -3, 'EZ': 4, 'EX': -1, 'E*': -4, 'GA': 0, 'GR': -2,
                       'GN': 0, 'GD': -1, 'GC': -3, 'GQ': -2, 'GE': -2, 'GG': 6, 'GH': -2, 'GI': -4,
                       'GL': -4, 'GK': -2, 'GM': -3, 'GF': -3, 'GP': -2, 'GS': 0, 'GT': -2,
                       'GW': -2, 'GY': -3, 'GV': -3, 'GB': -1, 'GJ': -4, 'GZ': -2, 'GX': -1,
                       'G*': -4, 'HA': -2, 'HR': 0, 'HN': 1, 'HD': -1, 'HC': -3, 'HQ': 0, 'HE': 0,
                       'HG': -2, 'HH': 8, 'HI': -3, 'HL': -3, 'HK': -1, 'HM': -2, 'HF': -1,
                       'HP': -2, 'HS': -1, 'HT': -2, 'HW': -2, 'HY': 2, 'HV': -3, 'HB': 0, 'HJ': -3,
                       'HZ': 0, 'HX': -1, 'H*': -4, 'IA': -1, 'IR': -3, 'IN': -3, 'ID': -3,
                       'IC': -1, 'IQ': -3, 'IE': -3, 'IG': -4, 'IH': -3, 'II': 4, 'IL': 2, 'IK': -3,
                       'IM': 1, 'IF': 0, 'IP': -3, 'IS': -2, 'IT': -1, 'IW': -3, 'IY': -1, 'IV': 3,
                       'IB': -3, 'IJ': 3, 'IZ': -3, 'IX': -1, 'I*': -4, 'LA': -1, 'LR': -2,
                       'LN': -3, 'LD': -4, 'LC': -1, 'LQ': -2, 'LE': -3, 'LG': -4, 'LH': -3,
                       'LI': 2, 'LL': 4, 'LK': -2, 'LM': 2, 'LF': 0, 'LP': -3, 'LS': -2, 'LT': -1,
                       'LW': -2, 'LY': -1, 'LV': 1, 'LB': -4, 'LJ': 3, 'LZ': -3, 'LX': -1, 'L*': -4,
                       'KA': -1, 'KR': 2, 'KN': 0, 'KD': -1, 'KC': -3, 'KQ': 1, 'KE': 1, 'KG': -2,
                       'KH': -1, 'KI': -3, 'KL': -2, 'KK': 5, 'KM': -1, 'KF': -3, 'KP': -1, 'KS': 0,
                       'KT': -1, 'KW': -3, 'KY': -2, 'KV': -2, 'KB': 0, 'KJ': -3, 'KZ': 1, 'KX': -1,
                       'K*': -4, 'MA': -1, 'MR': -1, 'MN': -2, 'MD': -3, 'MC': -1, 'MQ': 0,
                       'ME': -2, 'MG': -3, 'MH': -2, 'MI': 1, 'ML': 2, 'MK': -1, 'MM': 5, 'MF': 0,
                       'MP': -2, 'MS': -1, 'MT': -1, 'MW': -1, 'MY': -1, 'MV': 1, 'MB': -3, 'MJ': 2,
                       'MZ': -1, 'MX': -1, 'M*': -4, 'FA': -2, 'FR': -3, 'FN': -3, 'FD': -3,
                       'FC': -2, 'FQ': -3, 'FE': -3, 'FG': -3, 'FH': -1, 'FI': 0, 'FL': 0, 'FK': -3,
                       'FM': 0, 'FF': 6, 'FP': -4, 'FS': -2, 'FT': -2, 'FW': 1, 'FY': 3, 'FV': -1,
                       'FB': -3, 'FJ': 0, 'FZ': -3, 'FX': -1, 'F*': -4, 'PA': -1, 'PR': -2,
                       'PN': -2, 'PD': -1, 'PC': -3, 'PQ': -1, 'PE': -1, 'PG': -2, 'PH': -2,
                       'PI': -3, 'PL': -3, 'PK': -1, 'PM': -2, 'PF': -4, 'PP': 7, 'PS': -1,
                       'PT': -1, 'PW': -4, 'PY': -3, 'PV': -2, 'PB': -2, 'PJ': -3, 'PZ': -1,
                       'PX': -1, 'P*': -4, 'SA': 1, 'SR': -1, 'SN': 1, 'SD': 0, 'SC': -1, 'SQ': 0,
                       'SE': 0, 'SG': 0, 'SH': -1, 'SI': -2, 'SL': -2, 'SK': 0, 'SM': -1, 'SF': -2,
                       'SP': -1, 'SS': 4, 'ST': 1, 'SW': -3, 'SY': -2, 'SV': -2, 'SB': 0, 'SJ': -2,
                       'SZ': 0, 'SX': -1, 'S*': -4, 'TA': 0, 'TR': -1, 'TN': 0, 'TD': -1, 'TC': -1,
                       'TQ': -1, 'TE': -1, 'TG': -2, 'TH': -2, 'TI': -1, 'TL': -1, 'TK': -1,
                       'TM': -1, 'TF': -2, 'TP': -1, 'TS': 1, 'TT': 5, 'TW': -2, 'TY': -2, 'TV': 0,
                       'TB': -1, 'TJ': -1, 'TZ': -1, 'TX': -1, 'T*': -4, 'WA': -3, 'WR': -3,
                       'WN': -4, 'WD': -4, 'WC': -2, 'WQ': -2, 'WE': -3, 'WG': -2, 'WH': -2,
                       'WI': -3, 'WL': -2, 'WK': -3, 'WM': -1, 'WF': 1, 'WP': -4, 'WS': -3,
                       'WT': -2, 'WW': 11, 'WY': 2, 'WV': -3, 'WB': -4, 'WJ': -2, 'WZ': -2,
                       'WX': -1, 'W*': -4, 'YA': -2, 'YR': -2, 'YN': -2, 'YD': -3, 'YC': -2,
                       'YQ': -1, 'YE': -2, 'YG': -3, 'YH': 2, 'YI': -1, 'YL': -1, 'YK': -2,
                       'YM': -1, 'YF': 3, 'YP': -3, 'YS': -2, 'YT': -2, 'YW': 2, 'YY': 7, 'YV': -1,
                       'YB': -3, 'YJ': -1, 'YZ': -2, 'YX': -1, 'Y*': -4, 'VA': 0, 'VR': -3,
                       'VN': -3, 'VD': -3, 'VC': -1, 'VQ': -2, 'VE': -2, 'VG': -3, 'VH': -3,
                       'VI': 3, 'VL': 1, 'VK': -2, 'VM': 1, 'VF': -1, 'VP': -2, 'VS': -2, 'VT': 0,
                       'VW': -3, 'VY': -1, 'VV': 4, 'VB': -3, 'VJ': 2, 'VZ': -2, 'VX': -1, 'V*': -4,
                       'BA': -2, 'BR': -1, 'BN': 4, 'BD': 4, 'BC': -3, 'BQ': 0, 'BE': 1, 'BG': -1,
                       'BH': 0, 'BI': -3, 'BL': -4, 'BK': 0, 'BM': -3, 'BF': -3, 'BP': -2, 'BS': 0,
                       'BT': -1, 'BW': -4, 'BY': -3, 'BV': -3, 'BB': 4, 'BJ': -3, 'BZ': 0, 'BX': -1,
                       'B*': -4, 'JA': -1, 'JR': -2, 'JN': -3, 'JD': -3, 'JC': -1, 'JQ': -2,
                       'JE': -3, 'JG': -4, 'JH': -3, 'JI': 3, 'JL': 3, 'JK': -3, 'JM': 2, 'JF': 0,
                       'JP': -3, 'JS': -2, 'JT': -1, 'JW': -2, 'JY': -1, 'JV': 2, 'JB': -3, 'JJ': 3,
                       'JZ': -3, 'JX': -1, 'J*': -4, 'ZA': -1, 'ZR': 0, 'ZN': 0, 'ZD': 1, 'ZC': -3,
                       'ZQ': 4, 'ZE': 4, 'ZG': -2, 'ZH': 0, 'ZI': -3, 'ZL': -3, 'ZK': 1, 'ZM': -1,
                       'ZF': -3, 'ZP': -1, 'ZS': 0, 'ZT': -1, 'ZW': -2, 'ZY': -2, 'ZV': -2, 'ZB': 0,
                       'ZJ': -3, 'ZZ': 4, 'ZX': -1, 'Z*': -4, 'XA': -1, 'XR': -1, 'XN': -1,
                       'XD': -1, 'XC': -1, 'XQ': -1, 'XE': -1, 'XG': -1, 'XH': -1, 'XI': -1,
                       'XL': -1, 'XK': -1, 'XM': -1, 'XF': -1, 'XP': -1, 'XS': -1, 'XT': -1,
                       'XW': -1, 'XY': -1, 'XV': -1, 'XB': -1, 'XJ': -1, 'XZ': -1, 'XX': -1,
                       'X*': -4, '*A': -4, '*R': -4, '*N': -4, '*D': -4, '*C': -4, '*Q': -4,
                       '*E': -4, '*G': -4, '*H': -4, '*I': -4, '*L': -4, '*K': -4, '*M': -4,
                       '*F': -4, '*P': -4, '*S': -4, '*T': -4, '*W': -4, '*Y': -4, '*V': -4,
                       '*B': -4, '*J': -4, '*Z': -4, '*X': -4, '**': 1}


def print_matrix(matrix: np.array):
    for i in range(3):
        print("\n")
        for row in matrix:
            print(" ".join(["".join([" "] * (1 if item[i] >= 0 else 0)) + str(item[i]) + "".join(
                [" "] * (6 - (1 if item[i] >= 0 else 0) - len(str(item[i])))) for item in row]))


def calc_matrix(row_seq: str, col_seq: str):
    cols = np.int_(len(col_seq))
    rows = np.int_(len(row_seq))

    diag_matrix = [[[0 for _ in range(rows + 1)] for _ in range(3)] for _ in range(2)]
    matrix_direction = [[np.zeros(cols + 1, dtype=np.bool_) for _ in range(rows + 1)] for _ in range(4)]

    # fill matrix
    for diag in range(0, cols + rows):
        start_col = max(0, diag - rows)
        indexes = [(diag - 1) % 2, diag % 2]

        pos_list = [[min(rows, diag) - j - 1 + 1, start_col + j + 1] for j in
                    range(0, min(diag, (cols - start_col), rows))]

        for pos in pos_list:
            if diag_matrix[indexes[0]][0][pos[0] - 1] - diag_matrix[indexes[0]][1][pos[0] - 1] < 10:
                # set lower max
                diag_matrix[indexes[1]][1][pos[0]] = diag_matrix[indexes[0]][1][pos[0] - 1] - 1
                # set lower max index
                matrix_direction[1][pos[0]][pos[1]] = np.True_
            else:
                # set lower max
                diag_matrix[indexes[1]][1][pos[0]] = diag_matrix[indexes[0]][0][pos[0] - 1] - 11

            if diag_matrix[indexes[0]][0][pos[0]] - diag_matrix[indexes[0]][2][pos[0]] < 10:
                # set upper max
                diag_matrix[indexes[1]][2][pos[0]] = diag_matrix[indexes[0]][2][pos[0]] - 1
                # set upper max index
                matrix_direction[2][pos[0]][pos[1]] = np.True_
            else:
                # set upper max
                diag_matrix[indexes[1]][2][pos[0]] = diag_matrix[indexes[0]][0][pos[0]] - 11

            # calculate and assign middle matrix

            follow_score = diag_matrix[indexes[1]][0][pos[0] - 1] + substitution_matrix[row_seq[pos[0] - 1] + col_seq[pos[1] - 1]]

            # compare lower and upper
            if diag_matrix[indexes[1]][1][pos[0]] < diag_matrix[indexes[1]][2][pos[0]]:
                # compare follow and upper
                if follow_score < diag_matrix[indexes[1]][2][pos[0]]:
                    # set middle max
                    diag_matrix[indexes[1]][0][pos[0]] = diag_matrix[indexes[1]][2][pos[0]]
                    # set middle max index
                    matrix_direction[0][pos[0]][pos[1]] = np.True_
                else:
                    # set middle max
                    diag_matrix[indexes[1]][0][pos[0]] = follow_score
                # set lower / upper direction
                matrix_direction[3][pos[0]][pos[1]] = np.True_
            else:
                # compare follow and lower
                if follow_score < diag_matrix[indexes[1]][1][pos[0]]:
                    # set middle max
                    diag_matrix[indexes[1]][0][pos[0]] = diag_matrix[indexes[1]][1][pos[0]]
                    # set middle max index
                    matrix_direction[0][pos[0]][pos[1]] = np.True_
                else:
                    # set middle max
                    diag_matrix[indexes[1]][0][pos[0]] = follow_score

        # new init values
        side = -11 - diag * 1
        if diag < cols:
            diag_matrix[indexes[1]][0][0] = side
            diag_matrix[indexes[1]][1][0] = side
            diag_matrix[indexes[1]][2][0] = -32768
        if diag < rows:
            diag_matrix[indexes[1]][0][diag + 1] = side
            diag_matrix[indexes[1]][1][diag + 1] = -32768
            diag_matrix[indexes[1]][2][diag + 1] = side

    score = int(max((diag_matrix[(cols + rows - 1) % 2][x][-1] for x in range(3))))

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
    row_align = []
    col_align = []

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
                row_align = [row_seq[row - 1]] + row_align
                col_align = [col_seq[col - 1]] + col_align
                col -= 1
                row -= 1

        elif current == 1:
            row_align = [row_seq[row - 1]] + row_align
            col_align = ["-"] + col_align
            row -= 1
            if not direction:
                current = 0

        else:
            col_align = [col_seq[col - 1]] + col_align
            row_align = ["-"] + row_align
            col -= 1
            if not direction:
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
