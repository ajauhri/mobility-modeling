import utils.helpers as h
import utils.const as const
from applications.placement.algorithms.base import Algorithm

from collections import Counter
import logging
import numpy as np

class ShiftBand(Algorithm):
    def __init__(self, q, lat_grids, lng_grids, save_results=False):
        self.alg_id = 'sb'
        super(ShiftBand, self).__init__(q, lat_grids, lng_grids, save_results)

    def compute_regret(self):
        pickup_groups = self.g.bucket_by_time(0)
        dropoff_groups = self.g.bucket_by_time(4)

        gamma = 0.8
        alpha = 0.005
        beta = 0.8
        eta = 0.005 #too high values can lead to inf
        grids_per_bandit = 10
        W = np.ones((len(self.lat_grids), len(self.lng_grids)))
        N = np.zeros((len(self.lat_grids), len(self.lng_grids)))
        K = np.empty((len(self.lat_grids), len(self.lng_grids)))
        for i in xrange(0, len(self.lat_grids), grids_per_bandit):
            for j in xrange(0, len(self.lng_grids), grids_per_bandit):
                K[i:i+grids_per_bandit, j:j+grids_per_bandit] = \
                N[i:i+grids_per_bandit, j:j+grids_per_bandit].shape[0] * \
                N[i:i+grids_per_bandit, j:j+grids_per_bandit].shape[1]

        for t in sorted(dropoff_groups.keys()):
            success = 0
            for i in xrange(0, len(self.lat_grids), grids_per_bandit):
                for j in xrange(0, len(self.lng_grids), grids_per_bandit):
                    N[i:i+grids_per_bandit, j:j+grids_per_bandit] = np.sum(
                            W[i:i+grids_per_bandit, j:j+grids_per_bandit])

            P = ((1 - gamma) * W / N) + (gamma / K)
            X = np.zeros((len(self.lat_grids), len(self.lng_grids)))
            M = np.zeros((len(self.lat_grids), len(self.lng_grids)),
                    dtype=bool)


            dropoffs = dropoff_groups[t]
            pickups = pickup_groups[t+1] if t+1 in pickup_groups else []
            pickup_nodes = Counter()

            for i in pickups:
                node = h.get_node(self.lat_grids, self.lng_grids,
                        self.q.M[i, 2:4])[0]
                pickup_nodes[node] += 1

            for i in dropoffs:
                lat_cell, lng_cell = h.get_node(self.lat_grids, self.lng_grids,
                        self.q.M[i, 5:7])[1:3]
                idxs = self.g.get_boudaries(lat_cell, lng_cell)

                """
                1) Select a subset of space for the bandit problem
                """
                bandits = P[idxs[0]:idxs[1], idxs[2]: idxs[3]]
                M[idxs[0]:idxs[1], idxs[2]:idxs[3]] = True

                """
                2) Select bandit based on probabilities
                """
                rand = np.random.uniform()
                cum_p = bandits.cumsum().reshape(bandits.shape[0],
                        bandits.shape[1])
                if len(np.where(rand <= cum_p)[0]) == 0:
                    print(bandits)
                    print(W[start_lat_cell:end_lat_cell,
                        start_lng_cell: end_lng_cell])

                placement_lat_cell = np.where(rand <= cum_p)[0][0] + idxs[0]
                placement_lng_cell = np.where(rand <= cum_p)[1][0] + idxs[2]

                placement_node = placement_lat_cell * \
                        len(self.lng_grids) + placement_lng_cell
                """
                3) Collect reward, if any at placement node or the chosen bandit
                """
                if placement_node in pickup_nodes:
                    success += 1
                    if pickup_nodes[placement_node] == 1:
                        pickup_nodes.pop(node, None)
                    else:
                        pickup_nodes[placement_node] -= 1

                    p_i_t = P[placement_lat_cell, placement_lng_cell]
                    X[placement_lat_cell, placement_lng_cell] = 1/ p_i_t

            c = np.sqrt(const.time_windows_per_day*7*K[M]/20)
            W[M] = W[M] * np.exp(
                           eta * (
                                  X[M] + (alpha / \
                                          (P[M] * c)))) \
                           + N[M] * beta/K[M]
            if len(D) > 0:
                success_per = 100*success/len(D)
                print_str = ("time key: {0}, #good placements: {1}, """
                    """#percentage: {2:.2f}, #total: {3}""".format(t, success,
                        success_per, len(D)))
                self.regrets.append([t, success, len(D)])
                logging.debug(print_str)
        if self.save_results:
            self.save_regrets()
