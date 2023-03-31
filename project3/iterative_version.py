import sys
import numpy as np


def get_nodes(edges: list[tuple]):
    """
    get the used nodes from a list of edges
    :param filename: list of edges
    :return: list of nodes
    """

    nodes = []
    for edge in edges:
        nodes.append(edge[0])
        nodes.append(edge[1])
    return [*set(nodes)]


class UnrootedTree:
    """
    doctests for the UnrootedTree class

    >>> tree = UnrootedTree((3,5,7.0),(2,5,6.0),(4,5,4.0),(1,4,2.0),(0,4,11.0))
    >>> tree
    UnrootedTree((3,5,7.0),(2,5,6.0),(4,5,4.0),(1,4,2.0),(0,4,11.0))

    >>> tree = UnrootedTree((3,5,7.0),(2,5,6.0),(4,5,4.0),(1,4,2.0),(0,4,11.0))
    >>> tree.path(1, 3)
    [1, 4, 5, 3]
    """

    def __init__(self, *argv):
        """
        Initialize a new UnrootedTree
        :param argv: list of edges
        """
        self.edges = argv

    def __repr__(self):
        """
        create a representation of this class, format: UnrootedTree({edgelist})
        :return: string with the representation of this class
        """
        return f"UnrootedTree{str(self)}"

    def __str__(self):
        """
        create a string of this class, format: {edgelist}
        :return: string of this class
        """
        return str(self.edges)

    @staticmethod
    def loadtxt(fname: str):
        """
        create an unrooted tree based on a file with edges
        :param fname: name of a file with edges
        :return: new unrooted tree
        """
        edges = []
        with open(fname, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # parse lines
        for line in lines:
            res = line.replace("<->", ",").replace(":", ",").strip("\n").split(",")
            edges.append((int(res[0]), int(res[1]), float(res[2])))

        return UnrootedTree(*edges)

    def path(self, i: int, j: int, score: bool = False):
        """
        find a path between nodes i and j
        :param i: start node
        :param j: end node
        :param score: optionally get score of path (required for dodona)
        :return: path between 'i' and 'j'
        """
        # find start
        start_edges = self.next_edges(i, [])

        # loop throught all start posibilities
        for start_edge in start_edges:
            # run dfs
            if start_edge[0] == i:
                res_path = self.dfs(
                    start_edge[1], j, [start_edge], [i, start_edge[1]], start_edge[2]
                )
            else:
                res_path = self.dfs(
                    start_edge[0], j, [start_edge], [i, start_edge[0]], start_edge[2]
                )
            if res_path is not None:
                # path is found, return path, no need to check other paths
                if score:
                    # return path score
                    return res_path[1]
                # return path
                return res_path[0]
        # no path found
        return 0

    def dfs(
        self,
        current_node: int,
        stop: int,
        edge_path: list[tuple],
        node_path: list[int],
        weight: int,
    ):
        """
        recursive implementation of depth first search
        :param current_node: start node
        :param stop: dfs stop condition
        :param edge_path: currently followed edge path
        :param node_path: currently followed node path
        :param weight: weight of the current path
        :return: path between 'i' and 'j'
        """
        # stop condition
        if current_node == stop:
            return node_path, weight

        # get next edges from current node
        next_edges = self.next_edges(current_node, edge_path)

        # no remaining edges
        if len(next_edges) == 0:
            return None

        # search through all remaining edges
        for next_edge in next_edges:
            new_path = edge_path.copy()
            if next_edge[0] == current_node:
                res_path = self.dfs(
                    next_edge[1],
                    stop,
                    new_path + [next_edge],
                    node_path + [next_edge[1]],
                    weight + next_edge[2],
                )
            else:
                res_path = self.dfs(
                    next_edge[0],
                    stop,
                    new_path + [next_edge],
                    node_path + [next_edge[0]],
                    weight + next_edge[2],
                )
            if res_path is not None:
                return res_path
        return None

    def next_edges(self, i: tuple, path: list[tuple]):
        """
        get all edges that contain node i and are not yet visited
        :param i: current node
        :param path: path which has already been followed
        :return: all edges from the current node to a non visited node
        """
        return [
            edge for edge in self.edges if i in (edge[0], edge[1]) and edge not in path
        ]

    def distance_matrix(self):
        """
        :return: matix of distances between leaf nodes
        """
        leaves = self.get_leaves()
        matrix = []
        for i in leaves:
            row = []
            for j in leaves:
                row.append(self.path(i, j, score=True))
            matrix.append(row)
        return DistanceMatrix(matrix)

    def get_leaves(self):
        """
        :return: all nodes that are leaves
        """
        leaves = []
        for node in get_nodes(self.edges):
            if len(self.next_edges(node, [])) == 1:
                leaves.append(node)
        return sorted(leaves)


class DistanceMatrix:

    """
    doctests for the DistanceMatrix class

    >>> matrix = DistanceMatrix([[0,13,21,22],[13,0,12,13],[21,12,0,13],[22,13,13,0]])
    >>> matrix
    DistanceMatrix([[0,13,21,22],[13,0,12,13],[21,12,0,13],[22,13,13,0]])
    >>> print(matrix)
    [[0,13,21,22],[13,0,12,13],[21,12,0,13],[22,13,13,0]]

    >>> matrix = DistanceMatrix([[0,13,21,22],[13,0,12,13],[21,12,0,13],[22,13,13,0]])
    >>> matrix.savetxt('save_matrix.txt', fmt='%g')
    >>> new_matrix = DistanceMatrix.loadtxt('save_matrix.txt', fmt='%g')
    >>> new_matrix
    DistanceMatrix([[0.0,13.0,21.0,22.0],[13.0,0.0,12.0,13.0],[21.0,12.0,0.0,13.0],[22.0,13.0,13.0,0.0]])

    >>> DistanceMatrix.savetxt('save_matrix.txt', fmt='%g')
    >>> print(open('data/save_matrix.txt').read().rstrip())

    >>> matrix = DistanceMatrix.loadtxt('distances_01.txt')
    >>> float(matrix.limb_length(0))
    11.0

    >>> matrix = DistanceMatrix.loadtxt('distances_01.txt')
    >>> matrix.neighbour_joining()
    UnrootedTree((0, 4, 8.0),(3, 4, 12.0),(1, 5, 13.5),(2, 5, 16.5),(5, 4,2.0))
    """

    def __init__(self, matrix, *args, **kwargs):
        """
        Initialize a new DistanceMatrix
        :param matrix: initial matrix data
        """
        self.matrix = np.array(matrix, *args, **kwargs)

    def __repr__(self):
        """
        create a representation of this class, format: DistanceMatrix({matrix})
        :return: string with the representation of this class
        """
        return f"DistanceMatrix({str(self)})"

    def __str__(self):
        """
        create a string of this class, format: {matrix}
        :return: string of this class
        """
        return str(self.matrix.tolist())

    def savetxt(self, X: str, *args, **kwargs):
        """
        save the matrix from this class to a file
        :param X: filepath of where to store the matrix
        """
        np.savetxt(X, self.matrix, *args, **kwargs)

    @staticmethod
    def loadtxt(filename: str, *args, **kwargs):
        """
        load the matrix from a file to this class
        :param filename: filepath from where to load the matrix
        :return: new distance matrix with file content
        """
        matrix = np.loadtxt(fname=filename, *args, **kwargs)

        return DistanceMatrix(matrix)

    def limb_length(self, j: int):
        """
        calculate limb length from node i
        :param j: startnode
        :return: limb length
        """
        smallest = sys.maxsize
        for i, _ in enumerate(self.matrix):
            for k in range(len(self.matrix)):
                if j not in (i, k):
                    value = (
                        self.matrix[i][j] + self.matrix[j][k] - self.matrix[i][k]
                    ) / 2
                    if value < smallest:
                        smallest = value
        return smallest

    def neighbour_joining(self):
        """
        calculate an unrooted tree that fits the matrix that exists in this
        class under `self.matrix`
        """

        n = len(self.matrix)
        D = self.matrix.copy()

        # add row/col for node indexes

        M = []

        for i in np.arange(n):
            M += [i]

        node_index = n - 1
        T = []

        # iterative steps
        while n != 2:
            # list of all possible distances
            distances = np.sum(D, axis=0)

            # calculate D' with broadcast
            def calc_Dnew(i, j):
                return -i - j

            Dnew = (n - 2) * D + calc_Dnew(
                distances[:, np.newaxis], distances[np.newaxis, :]
            )

            # find minimum indexes in D'
            np.fill_diagonal(Dnew, np.inf)
            (i, j) = np.unravel_index(np.argmin(Dnew, axis=None), Dnew.shape)

            delta = (distances[i] - distances[j]) / (n - 2)

            limb_length_i = 1 / 2 * (D[i][j] + delta)
            limb_length_j = 1 / 2 * (D[i][j] - delta)

            # convert to int not to mix list and np ints
            node_index_i = M[int(i)]
            node_index_j = M[int(j)]

            # replace 1 row with new values
            M[j] = node_index + 1

            newRow = np.zeros(n)
            for k in range(n):
                newRow[k] = 1 / 2 * (D[k][i] + D[k][j] - D[i][j])
            D[j] = newRow
            for f in range(n):
                D[f][j] = newRow[f]

            # remove from map
            M.pop(i)

            # remove from matrix
            D = np.delete(D, i, 0)
            D = np.delete(D, i, 1)

            # add new edges to path
            T += [(node_index_i, node_index + 1, limb_length_i)]
            T += [(node_index_j, node_index + 1, limb_length_j)]

            # set new n and next node_index
            n -= 1
            node_index += 1

        # edge case
        T += [(int(M[0]), int(M[1]), D[0][1])]

        return UnrootedTree(*T)
