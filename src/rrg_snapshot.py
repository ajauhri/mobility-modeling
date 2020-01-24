from __future__ import division
from collections import Counter
import numpy as np
from scipy.stats import cauchy
import helpers
import networkx as nx

class RRGSnapshot(object):
    """
    Class captures properties of a Ride Request Graph(RRG) for a single snapshot
    of time. The RRG is a directed graph where nodes represent granular 
    geographic area (like 100 x 100 m^2 area can be represented by a node).
    """
    def __init__(self):
        self.out_weights = {} #contains weights on directed edges
        self.in_weights = {} 
        self.pairs = {} #unique node pairs in the graph
        self.C = [] # centroids for each unique pair of nodes
        self.pair_dists = []
        self.out_degree = Counter()
        self.in_degree = Counter()
        self.source_nodes = Counter()
        self.dest_nodes = Counter()
        self.n_edges = 0
        self.n_nodes = 0
        self.nxg = nx.Graph() # nx.clustering() only works on undirected graphs

    @staticmethod
    def _within_boundary(x, lb, ub):
        return helpers.gte_lb(x, lb) and helpers.lte_ub(x, ub)

    def init(self, S, D, lat_grids, lng_grids):
        """
        Takes the start and destination location of ride requests to find 
        the corresponding nodes of a graph

        :param S: 2-dimensional array with source lat and lngs
        :param D: 2-dimensional array with destination lat and lngs
        :param lat_grids: division of min. and max. 
        :param lng_grids: division of min. and max. 
        """
        if len(S) != len(D):
            return 
        for i in range(len(S)):
            '''
            Since the grids are in ascending order, argmax will 
            satisfy the following conditions:
                1) S[i, 0] >= lat_grids[src_lat_cell - 1] and S[i, 0] < lat_grids[src_lat_cell]
                2) S[i, 1] >= lng_grids[src_lng_cell - 1] and S[i, 1] < lng_grids[src_lng_cell]
            '''
            src_lat_cell = np.argmax(S[i, 0] < lat_grids) \
                if self._within_boundary(
                    S[i, 0], lat_grids[0], lat_grids[-1]) \
                else 1 if not helpers.gte_lb(S[i, 0], lat_grids[0]) \
                    else len(lat_grids) - 1
     
            src_lng_cell = np.argmax(S[i, 1] < lng_grids) \
                if self._within_boundary(
                    S[i, 1], lng_grids[0], lng_grids[-1]) \
                else 1 if not helpers.gte_lb(S[i, 1], lng_grids[0]) \
                    else len(lng_grids) - 1
             
            dest_lat_cell = np.argmax(D[i, 0] < lat_grids) \
                if self._within_boundary( 
                    D[i, 0], lat_grids[0], lat_grids[-1])  \
                else 1 if not helpers.gte_lb(D[i, 0], lat_grids[0]) \
                    else len(lat_grids) - 1
            
            dest_lng_cell = np.argmax(D[i, 1] < lng_grids) \
                if self._within_boundary(
                    D[i, 1], lng_grids[0], lng_grids[-1])  \
                else 1 if not helpers.gte_lb(D[i, 1], lng_grids[0]) \
                    else len(lng_grids) - 1

            source_node = (src_lat_cell - 1) * len(lng_grids) + \
                (src_lng_cell - 1)
            dest_node = (dest_lat_cell - 1) * len(lng_grids) + \
                (dest_lng_cell - 1)
            self.source_nodes[source_node] += 1
            self.dest_nodes[dest_node] += 1
            
            if source_node not in self.out_weights:
                self.out_weights[source_node] = Counter()
            self.out_weights[source_node][dest_node] += 1

            if dest_node not in self.in_weights:
                self.in_weights[dest_node] = Counter()
            self.in_weights[dest_node][source_node] += 1

            if not self.nxg.has_edge(source_node, dest_node):
                self.nxg.add_edge(source_node, dest_node, weight = 1)
            else:
                self.nxg[source_node][dest_node]['weight'] += 1

    def compute_nodes_and_edges(self):
        n_edges = 0
        for s, ds in self.out_weights.items():
            n_edges += len(ds)
        self.n_edges = n_edges
        self.n_nodes = len(np.union1d(list(self.source_nodes.keys()), 
            list(self.dest_nodes.keys())))
