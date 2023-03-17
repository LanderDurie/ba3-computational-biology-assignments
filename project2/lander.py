import sys

from Bio import SeqIO
import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices

SIGMA = 11
EPSILON = 1

def make_alignment_graph(str1, str2):
    rules = substitution_matrices.load("BLOSUM62")
    down_matrix = np.zeros((len(str1) + 1, len(str2) + 1))
    right_matrix = np.zeros((len(str1) + 1, len(str2) + 1))
    diagonal_matrix = np.zeros((len(str1) + 1, len(str2) + 1))

    for i in range(1, len(str1) + 1):
        right_matrix[i][0] = - np.inf
        down_matrix[i][0] = - SIGMA - ((i - 1) * EPSILON)
        diagonal_matrix[i][0] = - SIGMA - ((i - 1) * EPSILON)
    for i in range(1, len(str2) + 1):
        down_matrix[0][i] = - np.inf
        right_matrix[0][i] = - SIGMA - ((i - 1) * EPSILON)
        diagonal_matrix[0][i] = - SIGMA - ((i - 1) * EPSILON)

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            down_matrix[i][j] = max(
                [down_matrix[i - 1][j] - EPSILON, diagonal_matrix[i - 1][j] - SIGMA]
            )
            right_matrix[i][j] = max(
                [right_matrix[i][j - 1] - EPSILON, diagonal_matrix[i][j - 1] - SIGMA]
            )

            diagonal_matrix[i][j] = max(
                # niet zo zeker over de - 1 bij de rules indexering
                [down_matrix[i][j],
                 diagonal_matrix[i - 1][j - 1] + rules[str1[i - 1]][str2[j - 1]],
                 right_matrix[i][j]]
            )
    print("Down")
    print(down_matrix)
    print("Right")
    print(right_matrix)
    print("Diagonal")
    print(diagonal_matrix)
    return down_matrix, right_matrix, diagonal_matrix

tempSIGMA = 5

def global_alignment(fasta_file_location):
    seqs = [*SeqIO.parse(fasta_file_location, "fasta")]
    seq1 = seqs[0].seq
    seq2 = seqs[1].seq

    down_matrix, right_matrix, diagonal_matrix = make_alignment_graph(seq1, seq2)
    matrix = substitution_matrices.load("BLOSUM62")
    i = len(seq1)
    j = len(seq2)
    res1 = ""
    res2 = ""
    current_matrix = diagonal_matrix
    wich_matrix = "diagonal"
    while i > 0 or j > 0:
        if wich_matrix == "diagonal":

            schuin = current_matrix[i - 1][j - 1] + matrix[seq1[i - 1]][seq2[j - 1]]
            boven = down_matrix[i][j]
            links = right_matrix[i][j]
            if i > 0 and j > 0 and schuin >= links and schuin >= boven:
                res1 = seq1[i - 1] + res1
                res2 = seq2[j - 1] + res2
                i -= 1
                j -= 1
            elif i > 0 and boven >= links and boven >= schuin:
                current_matrix = down_matrix
                wich_matrix = "down"

            elif j > 0 and links >= boven and links >= schuin:
                current_matrix = right_matrix
                wich_matrix = "right"

        elif wich_matrix == "down":
            schuin = diagonal_matrix[i - 1][j] - SIGMA
            boven = down_matrix[i - 1][j] - EPSILON

            res1 = seq1[i - 1] + res1
            res2 = "-" + res2
            i -= 1

            if i > 0 and schuin >= boven:
                current_matrix = diagonal_matrix
                wich_matrix = "diagonal"

        else:
            schuin = diagonal_matrix[i][j - 1] - SIGMA
            links = right_matrix[i][j - 1] - EPSILON

            res1 = "-" + res1
            res2 = seq2[j - 1] + res2
            j -= 1

            if j > 0 and schuin >= links:
                current_matrix = diagonal_matrix
                wich_matrix = "diagonal"

    return res1, res2

def global_alignment_score(file):
    sequenties = list(SeqIO.parse(file, 'fasta'))
    str1 = sequenties[0].seq
    str2 = sequenties[1].seq
    down_matrix, right_matrix, diagonal_matri = make_alignment_graph(str1, str2)
    return int(diagonal_matri[len(str1)][len(str2)])


if __name__ == '__main__':
    rules = substitution_matrices.load("BLOSUM62")
    sequenties = list(SeqIO.parse("data/data_02.fna", 'fasta'))
    str1 = sequenties[0].seq
    str2 = sequenties[1].seq
    temp1 = " " + str1
    temp2 = " " + str2
    seq1, seq2 = global_alignment("data/data_02.fna")
    print(seq1)
    print('YHFDVPDCWAHRYWVENPQAIAQME-------QICFNWFPSMMMK-------QPHVF---KVDHHMSCRWLPIRGKKCSSCCTRMRVRTVWE')
    print(seq2)
    print('YHEDV----AHE------DAIAQMVNTFGFVWQICLNQFPSMMMKIYWIAVLSAHVADRKTWSKHMSCRWLPI----ISATCARMRVRTVWE')
    #print(make_alignment_graph(str1, str2))
    #global_alignment_score("data01.faa")
