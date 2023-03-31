import sys

import numpy as np


class UnrootedTree:
    def __init__(self, *argv):
        self.edges = argv

    def __repr__(self):
        return f"UnrootedTree{str(self)}"

    def __str__(self):
        return str(self.edges)

    @staticmethod
    def loadtxt(fname):

        edges = []
        with open(fname, 'r') as file:
            lines = file.readlines()

        for line in lines:
            res = line.replace('<->', ',').replace(':', ',').strip("\n").split(',')
            edges.append((int(res[0]), int(res[1]), float(res[2])))

        return UnrootedTree(*edges)

    def path(self, i, j, score=False):

        # find start
        start_edges = self.next_edges(i, [])

        for start_edge in start_edges:
            if start_edge[0] == i:
                res_path = self.DFS(start_edge[1], j, [start_edge], [i, start_edge[1]], start_edge[2])
            else:
                res_path = self.DFS(start_edge[0], j, [start_edge], [i, start_edge[0]], start_edge[2])
            if res_path is not None:
                if score:
                    return res_path[1]
                else:
                    return res_path[0]
        return 0

    def DFS(self, current_node, stop, edge_path, node_path, weight):

        if current_node == stop:
            return (node_path, weight)

        next_edges = self.next_edges(current_node, edge_path)

        if len(next_edges) == 0:
            return None

        for next_edge in next_edges:
            new_path = edge_path.copy()
            if next_edge[0] == current_node:
                res_path = self.DFS(next_edge[1], stop, new_path + [next_edge], node_path + [next_edge[1]], weight + next_edge[2])
            else:
                res_path = self.DFS(next_edge[0], stop, new_path + [next_edge], node_path + [next_edge[0]], weight + next_edge[2])
            if res_path is not None:
                return res_path


    def next_edges(self, i, path):
        return [edge for edge in self.edges if (edge[0] == i or edge[1] == i) and edge not in path]

    def distance_matrix(self):
        leaves = self.get_leaves(self.edges)
        matrix = []
        for i in leaves:
            row = []
            for j in leaves:
                row.append(self.path(i, j, score=True))
            matrix.append(row)
        return DistanceMatrix(matrix)

    def get_nodes(self, edges):
        nodes = []
        for edge in edges:
            nodes.append(edge[0])
            nodes.append(edge[1])
        return [*set(nodes)]

    def get_leaves(self, edges):
        leaves = []
        for node in self.get_nodes(edges):
            if len(self.next_edges(node, [])) == 1:
                leaves.append(node)
        return sorted(leaves)


class DistanceMatrix:

    def __init__(self, matrix, dtype=None, copy: bool = True, order: str = 'K', subok: bool = False,
                 ndmin: int = 0):
        self.matrix = np.array(matrix, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)

    def __repr__(self):
        return f"DistanceMatrix({str(self)})"

    def __str__(self):
        return str(self.matrix.tolist())

    def savetxt(self, fname, fmt='%.18e', delimiter=' ', newline='\n',
                header='', footer='', comments='# ', encoding=None):
        np.savetxt(
            fname=fname,
            X=self.matrix,
            fmt=fmt,
            delimiter=delimiter,
            newline=newline,
            header=header,
            footer=footer,
            comments=comments,
            encoding=encoding)

    @staticmethod
    def loadtxt(fname, dtype=np.dtype(float), comments='#', delimiter=None,
                converters=None, skiprows=0, usecols=None, unpack=False,
                ndmin=0, encoding='bytes', max_rows=None, *, quotechar=None,
                like=None):
        matrix = np.loadtxt(
            fname=fname,
            dtype=dtype,
            comments=comments,
            delimiter=delimiter,
            converters=converters,
            skiprows=skiprows,
            usecols=usecols,
            unpack=unpack,
            ndmin=ndmin,
            encoding=encoding,
            max_rows=max_rows,
            quotechar=quotechar,
            like=like
        )

        return DistanceMatrix(matrix)

    def limb_length(self, j):

        smallest = sys.maxsize
        for i in range(len(self.matrix)):
            for k in range(len(self.matrix)):
                if j != i and j != k:
                    value = (self.matrix[i][j] + self.matrix[j][k] - self.matrix[i][k]) / 2
                    if value < smallest:
                        smallest = value
        return smallest


    def total_distance(self, D, i):
        return sum(D[i][1:])


    def neighbour_joining(self):
        n = len(self.matrix)
        D = self.matrix.copy()

        # add row/col for node indexes
        D = np.vstack([np.arange(n), D])
        D = np.hstack([np.transpose([np.arange(-1, n)]), D])

        return UnrootedTree(*self.neighbour_joining_recursive(D, n, n-1))

    def neighbour_joining_recursive(self, D, n, node_index):

        print(D)

        if n == 2:
            return [(int(D[0, 1]), int(D[0, 2]), D[1][2])]

        Dnew = np.zeros(shape=(n, n))
        for i_it in range(n):
            for j_it in range(n):
                Dnew[i_it][j_it] = (n - 2) * D[i_it+1][j_it+1] - \
                                   self.total_distance(D, i_it+1) - \
                                   self.total_distance(D, j_it+1)
        print(Dnew)

        i = 0
        j = 0
        min_val = sys.maxsize
        for i_it in range(n):
            for j_it in range(n):
                if Dnew[i_it][j_it] < min_val and i_it != j_it:
                    min_val = Dnew[i_it][j_it]
                    i = i_it
                    j = j_it

        delta = (self.total_distance(D, i+1) - self.total_distance(D, j+1)) / (n-2)

        limb_length_i = 1/2 * (D[i+1][j+1] + delta)
        limb_length_j = 1/2 * (D[i+1][j+1] - delta)

        # add
        m = n+1
        D = np.vstack([D, np.zeros(n+1)])
        D = np.hstack([D, np.transpose([np.zeros(n+2)])])
        D[0][m] = D[m][0] = node_index+1


        for k in range(1, n+1):
            D[m][k] = D[k][m] = 1 / 2 * (D[k][i+1] + D[k][j+1] - D[i+1][j+1])

        node_index_i = D[0][i+1]
        node_index_j = D[0][j + 1]

        print(i, j)

        # remove
        if j < i:
            D = np.delete(D, i+1, 0)
            D = np.delete(D, j+1, 0)
            D = np.delete(D, i+1, 1)
            D = np.delete(D, j+1, 1)
        else:
            D = np.delete(D, j+1, 0)
            D = np.delete(D, i+1, 0)
            D = np.delete(D, j+1, 1)
            D = np.delete(D, i+1, 1)

        T = self.neighbour_joining_recursive(D, n-1, node_index+1)

        T += [(node_index_i, node_index+1, limb_length_i)]
        T += [(node_index_j, node_index+1, limb_length_j)]

        return T


if __name__ == '__main__':
    D = DistanceMatrix.loadtxt('data/distances.txt')
    print(D.neighbour_joining())