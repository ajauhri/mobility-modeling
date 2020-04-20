import numpy as np

import utils.helpers as h
import utils.const as const
import applications.placement.algo_commons as alg_c

from collections import Counter
from applications.placement.algorithms.base import Algorithm


class MAB(Algorithm):
    def __init__(self, q, lat_grids, lng_grids, save_results=False):
        self.init_history = 3
        super(MAB, self).__init__(q, lat_grids, lng_grids, save_results)
    
    def _choice(self, P, idxs):
        r = np.random.uniform(0, 1)
        cum_p = 0
        for i in range(idxs[0], idxs[1]):
            for j in range(idxs[2], idxs[3]):
                cum_p += P[i, j]
                if cum_p >= r:
                    break
            else:
                continue
            break
        return i, j
    
    def find_history(self, time_keys):
        history = np.zeros((len(self.lat_grids), 
                len(self.lng_grids)))
       
        for i in time_keys:
            pickups = self.pickup_groups[i]
            if i in self.dropoff_groups:
                dropoffs = self.dropoff_groups[i]
            else:
                dropoffs = []
            
            """
            1) Find all pickups at t correspoding to nodes
            """
            for i in pickups:
                j,k = h.get_node(self.lat_grids, self.lng_grids, 
                        self.q.M[i, 2:4])[1:3]
                history[j,k] += 1 
            
            """
            2) Find all dropoffs at t corresponding to nodes
            """
            for i in dropoffs:
                j,k = h.get_node(self.lat_grids, self.lng_grids, 
                        self.q.M[i, 5:7])[1:3]
                history[j,k] += 1
        return history 

    def compute_success(self, t, placements):
            success = 0
            X_bar = np.zeros((len(self.lat_grids), len(self.lng_grids)))
            if t in self.pickup_groups:
                pickups = self.pickup_groups[t]
                for i in pickups:
                    node = h.get_node(self.lat_grids, self.lng_grids, 
                            self.q.M[i, 2:4])[0]
                    if node in placements:
                        success += 1
                        if placements[node] == 1:
                            placements.pop(node, None)
                        else:
                            placements[node] -= 1
                        lat_idx = int(node/len(self.lng_grids))
                        lng_idx = node % len(self.lng_grids)
                        X_bar[lat_idx, lng_idx] += 1
            return success, X_bar 
       
class Exp3(MAB):
    def __init__(self, q, lat_grids, lng_grids, save_results=False):
        self.alg_id = 'sb_' + str(const.static_boundaries) + '_exp3'
        super(Exp3, self).__init__(q, lat_grids, lng_grids, save_results)

    def compute_regret(self):
        W = np.ones((len(self.lat_grids), len(self.lng_grids)))
        K = const.region_width*const.region_width
        P = (1 - const.mab_gamma) * W / K + const.mab_gamma/K

        for t in sorted(self.dropoff_groups.keys())[self.init_history:]:
            placements = Counter()

            for i in self.dropoff_groups[t]:
                lat_idx, lng_idx = h.get_node(self.lat_grids, self.lng_grids, 
                        self.q.M[i, 5:7])[1:3]
                idxs = alg_c.get_boundaries(self.lat_grids, self.lng_grids,
                       lat_idx, lng_idx)
                W_i = W[idxs[0]:idxs[1], idxs[2]:idxs[3]] 
                n_bandits = (idxs[1] - idxs[0]) * (idxs[3] - idxs[2])
                P[idxs[0]:idxs[1], idxs[2]:idxs[3]] = (1 - const.mab_gamma) *\
                        W_i / np.sum(W_i) + const.mab_gamma/n_bandits

                placement_lat_idx, placement_lng_idx = self._choice(P, idxs) 
                node = placement_lat_idx * len(self.lng_grids) + \
                        placement_lng_idx
                placements[node] += 1 
            
            success, X_bar = self.compute_success(t+1, placements)
            
            if success > 0:
                X_bar = X_bar/np.max(X_bar)
                W = W * np.exp(const.mab_gamma*(X_bar/P)/K)
            self.add_result(t, success, self.dropoff_groups[t])
        self.save_regrets()

class Exp3WithContext(MAB):
    def __init__(self, q, lat_grids, lng_grids, save_results=False):
        self.alg_id = 'sb_' + str(const.static_boundaries) + '_exp3wcontext'
        super(Exp3WithContext, self).__init__(q, lat_grids, lng_grids, save_results)

    def compute_regret(self):
        W = np.ones((len(self.lat_grids), len(self.lng_grids)))
        C = self.find_history(range(self.init_history))
        W += C
        K = const.region_width*const.region_width
        P = (1 - const.mab_gamma) * W / K + const.mab_gamma/K

        for t in sorted(self.dropoff_groups.keys())[self.init_history:]:
            C = np.zeros((len(self.lat_grids), len(self.lng_grids)))
            placements = Counter()

            for i in self.dropoff_groups[t]:
                lat_idx, lng_idx = h.get_node(self.lat_grids, self.lng_grids, 
                        self.q.M[i, 5:7])[1:3]
                
                # get destinations from context
                C[lat_idx, lng_idx] += 1

                idxs = alg_c.get_boundaries(self.lat_grids, self.lng_grids,
                       lat_idx, lng_idx)
                W_i = W[idxs[0]:idxs[1], idxs[2]:idxs[3]] 
                n_bandits = (idxs[1] - idxs[0]) * (idxs[3] - idxs[2])
                P[idxs[0]:idxs[1], idxs[2]:idxs[3]] = (1 - const.mab_gamma) *\
                        W_i / np.sum(W_i) + const.mab_gamma/n_bandits

                placement_lat_idx, placement_lng_idx = self._choice(P, idxs) 
                node = placement_lat_idx * len(self.lng_grids) + \
                        placement_lng_idx
                placements[node] += 1
                
            if t in self.pickup_groups:
                for i in self.pickup_groups[t]:
                    lat_idx, lng_idx = h.get_node(
                                self.lat_grids, self.lng_grids, 
                                self.q.M[i, 2:4])[1:3]
                    C[lat_idx, lng_idx] += 1
           
            success, X_bar = self.compute_success(t+1, placements)            

            if success > 0:
                X_bar = X_bar/np.max(X_bar)
                W = W * np.exp(const.mab_gamma*(X_bar/P)/K)
                W += C
            self.add_result(t, success, self.dropoff_groups[t])
        self.save_regrets()

class Exp3AndFTL(MAB):
    def __init__(self, q, lat_grids, lng_grids, save_results=False):
        self.alg_id = 'sb_' + str(const.static_boundaries) + '_exp3_ftl'
        super(Exp3AndFTL, self).__init__(q, lat_grids, lng_grids, save_results)
 
    def find_ftl_history(self, time_keys):
        history = Counter()
       
        for i in time_keys:
            pickups = self.pickup_groups[i]
            if i in self.dropoff_groups:
                dropoffs = self.dropoff_groups[i]
            else:
                dropoffs = []
            
            """
            1) Find all pickups at t correspoding to nodes
            """
            for i in pickups:
                node = h.get_node(self.lat_grids, self.lng_grids, 
                        self.q.M[i, 2:4])[0]
                history[node] += 1 
            
            """
            2) Find all dropoffs at t corresponding to nodes
            """
            for i in dropoffs:
                node = h.get_node(self.lat_grids, self.lng_grids, 
                        self.q.M[i, 5:7])[0]
                history[node] += 1
        return history 


    def compute_regret(self):
        W = np.ones((len(self.lat_grids), len(self.lng_grids)))
        K = const.region_width*const.region_width
        P = (1 - const.mab_gamma) * W / K + const.mab_gamma/K
        history = self.find_ftl_history(range(self.init_history))

        for t in sorted(self.dropoff_groups.keys())[self.init_history:]:
            success = 0
            dropoffs = self.dropoff_groups[t]
            X_bar = np.zeros((len(self.lat_grids), len(self.lng_grids)))
            dropoff_nodes = [] 
            mab_placements = Counter()
            ftl_placements = Counter()

            for i in dropoffs:
                node, lat_idx, lng_idx = h.get_node(self.lat_grids, 
                        self.lng_grids, 
                        self.q.M[i, 5:7])
                dropoff_nodes.append(node) 
                
                if np.random.uniform() <= 0.1:
                    idxs = alg_c.get_boundaries(self.lat_grids, self.lng_grids,
                           lat_idx, lng_idx)
                    W_i = W[idxs[0]:idxs[1], idxs[2]:idxs[3]] 
                    n_bandits = (idxs[1] - idxs[0]) * (idxs[3] - idxs[2])
                    P[idxs[0]:idxs[1], idxs[2]:idxs[3]] = \
                            (1 - const.mab_gamma) * W_i / np.sum(W_i) + \
                            const.mab_gamma/n_bandits

                    placement_lat_idx, placement_lng_idx = self._choice(P, idxs) 
                    node = placement_lat_idx * len(self.lng_grids) + \
                            placement_lng_idx
                    mab_placements[node] += 1
                else:
                    node, freq = alg_c.find_leader(lat_idx, lng_idx,
                            self.lat_grids, self.lng_grids, history)
                    ftl_placements[node] += 1

            if t+1 in self.pickup_groups:
                pickups = self.pickup_groups[t+1]
                for i in pickups:
                    node = h.get_node(self.lat_grids, self.lng_grids, 
                        self.q.M[i, 2:4])[0]
                    if node in ftl_placements:
                        success += 1
                        if ftl_placements[node] == 1:
                            ftl_placements.pop(node, None)
                        else:
                            ftl_placements[node] -= 1
                    elif node in mab_placements:
                        success += 1
                        if mab_placements[node] == 1:
                            mab_placements.pop(node, None)
                        else:
                            mab_placements[node] -= 1
                        lat_idx = int(node/len(self.lng_grids))
                        lng_idx = node % len(self.lng_grids)
                        X_bar[lat_idx, lng_idx] += 1

            for n in dropoff_nodes:
                history[n] += 1

            if t in self.pickup_groups:
                for i in self.pickup_groups[t]:
                    nodes, lat_idx, lng_idx = h.get_node(
                                self.lat_grids, self.lng_grids, 
                                self.q.M[i, 2:4])
                    history[nodes] += 1

            if success > 0:
                X_bar = X_bar/np.max(X_bar)
                W = W * np.exp(const.mab_gamma*(X_bar/P)/K)
            
            self.add_result(t, success, dropoffs)
        self.save_regrets()
