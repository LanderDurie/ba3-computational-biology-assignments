import sys

import numpy as np


def get_nodes(edges):
    nodes = []
    for edge in edges:
        nodes.append(edge[0])
        nodes.append(edge[1])
    return [*set(nodes)]


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
        with open(fname, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        for line in lines:
            res = line.replace('<->', ',').replace(':', ',').strip("\n").split(
                ',')
            edges.append((int(res[0]), int(res[1]), float(res[2])))

        return UnrootedTree(*edges)

    def path(self, i, j, score=False):

        # find start
        start_edges = self.next_edges(i, [])

        for start_edge in start_edges:
            if start_edge[0] == i:
                res_path = self.dfs(start_edge[1], j, [start_edge],
                                    [i, start_edge[1]], start_edge[2])
            else:
                res_path = self.dfs(start_edge[0], j, [start_edge],
                                    [i, start_edge[0]], start_edge[2])
            if res_path is not None:
                if score:
                    return res_path[1]
                return res_path[0]
        return 0

    def dfs(self, current_node, stop, edge_path, node_path, weight):

        if current_node == stop:
            return node_path, weight

        next_edges = self.next_edges(current_node, edge_path)

        if len(next_edges) == 0:
            return None

        for next_edge in next_edges:
            new_path = edge_path.copy()
            if next_edge[0] == current_node:
                res_path = self.dfs(next_edge[1], stop, new_path + [next_edge],
                                    node_path + [next_edge[1]],
                                    weight + next_edge[2])
            else:
                res_path = self.dfs(next_edge[0], stop, new_path + [next_edge],
                                    node_path + [next_edge[0]],
                                    weight + next_edge[2])
            if res_path is not None:
                return res_path, weight

    def next_edges(self, i, path):
        return [edge for edge in self.edges if
                i in (edge[0], edge[1]) and edge not in path]

    def distance_matrix(self):
        leaves = self.get_leaves(self.edges)
        matrix = []
        for i in leaves:
            row = []
            for j in leaves:
                row.append(self.path(i, j, score=True))
            matrix.append(row)
        return DistanceMatrix(matrix)

    def get_leaves(self, edges):
        leaves = []
        for node in get_nodes(edges):
            if len(self.next_edges(node, [])) == 1:
                leaves.append(node)
        return sorted(leaves)


class DistanceMatrix:

    def __init__(self, matrix, *args, **kwargs):
        self.matrix = np.array(matrix, *args, **kwargs)

    def __repr__(self):
        return f"DistanceMatrix({str(self)})"

    def __str__(self):
        return str(self.matrix)

    def savetxt(self, filename, *args, **kwargs):
        np.savetxt(
            fname=filename, *args, **kwargs)

    @staticmethod
    def loadtxt(filename, *args, **kwargs):
        matrix = np.loadtxt(
            fname=filename, *args, **kwargs
        )

        return DistanceMatrix(matrix)

    def limb_length(self, j):

        smallest = sys.maxsize
        for _, i in enumerate(self.matrix):
            for k in range(len(self.matrix)):
                if j not in (i, k):
                    value = (self.matrix[i][j] + self.matrix[j][k] -
                             self.matrix[i][k]) / 2
                    if value < smallest:
                        smallest = value
        return smallest

    def total_distance(self, D, i):
        return sum(D[i])

    def neighbour_joining(self):
        n = len(self.matrix)
        D = self.matrix.copy()

        # add row/col for node indexes

        M = []

        for i in np.arange(n):
            M += [i]

        return UnrootedTree(*self.neighbour_joining_recursive(D, M, n, n - 1))

    def neighbour_joining_recursive(self, D, M, n, node_index):

        # end condition
        if n == 2:
            return [(int(M[0]), int(M[1]), D[0][1])]

        # list of all possible distances
        distances = self.total_distance(D, np.arange(n))

        # calculate D' with broadcast
        def calc_Dnew(i, j):
            return - i - j
        Dnew = (n-2) * D + calc_Dnew(distances[:, np.newaxis], distances[np.newaxis, :])

        # find minimum indexes in D'
        i = 0
        j = 0
        min_val = sys.maxsize
        for i_it in range(n):
            for j_it in range(n):
                if Dnew[i_it][j_it] < min_val and i_it != j_it:
                    min_val = Dnew[i_it][j_it]
                    i = i_it
                    j = j_it
        if i > j:
            i, j = j, i

        delta = (distances[i] - distances[j]) / (n - 2)

        limb_length_i = 1 / 2 * (D[i][j] + delta)
        limb_length_j = 1 / 2 * (D[i][j] - delta)


        node_index_i = M[i]
        node_index_j = M[j]

        # add
        M[j] = node_index+1

        newRow = np.zeros(n)
        for k in range(n):
            newRow[k] = 1 / 2 * (D[k][i] + D[k][j] - D[i][j])
        D[j] = newRow
        for f in range(n):
            D[f][j] = newRow[f]

        del M[i]
        # remove
        D = np.delete(D, i, 0)
        D = np.delete(D, i, 1)


        T = self.neighbour_joining_recursive(D, M, n - 1, node_index + 1)

        T += [(node_index_i, node_index + 1, limb_length_i)]
        T += [(node_index_j, node_index + 1, limb_length_j)]

        return T


if __name__ == '__main__':
    import time
    start = time.time()
    D = DistanceMatrix.loadtxt('data/200_benchmark.txt')
    print(D.neighbour_joining())
    print(time.time() - start)
