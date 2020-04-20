import utils.helpers as h
import utils.const as const
from applications.placement.algorithms.base import Algorithm
#from utils.grouping import Group
import applications.placement.algo_commons as alg_c

from collections import Counter
import numpy as np

class FTL(Algorithm):
    def __init__(self, q, lat_grids, lng_grids, save_results):
        super(FTL, self).__init__(q, lat_grids, lng_grids, save_results)
   
    def find_histowwry(self, time_keys):
        history = Counter()
       
        for t in time_keys:
            pickups = self.pickup_groups[t]
            if t in self.dropoff_groups:
                dropoffs = self.dropoff_groups[t]
            else:
                dropoffs = []
            
            """
            1) Find all pickups at t correspoding to nodes
            """
            for p in pickups:
                node = h.get_node(self.lat_grids, self.lng_grids, 
                        self.q.M[p, 2:4])[0]
                history[node] += 1 
            
            """
            2) Find all dropoffs at t corresponding to nodes
            """
            """
            for d in dropoffs:
                node = h.get_node(self.lat_grids, self.lng_grids, 
                        self.q.M[d, 5:7])[0]
                history[node] += 1
            """
        return history 

    def compute_success(self, t, placements):
        success = 0
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
        return success
   
class FTLWithLimitedHistory(FTL):
    def __init__(self, q, lat_grids, lng_grids, n_history=1, 
            save_results=False):
        self.alg_id = 'sb_' + str(const.static_boundaries) + \
                '_ftl_w_lim_hist_' + str(n_history)
        self.n_history = n_history
        super(FTLWithLimitedHistory, self).__init__(q, lat_grids, lng_grids,
                save_results)
            
    def compute_regret(self):
        for t in sorted(self.dropoff_groups.keys())[self.n_history:]:
            placements = Counter()
            if t%self.n_history == 0:
                history = self.find_history(range((t - self.n_history), t))
            for i in self.dropoff_groups[t]:
                lat_cell, lng_cell = h.get_node(
                            self.lat_grids, self.lng_grids, 
                            self.q.M[i, 5:7])[1:3]
                leader_node, freq = alg_c.find_rand_leader(lat_cell, lng_cell,
                        self.lat_grids, 
                        self.lng_grids, 
                        history)
                placements[leader_node] += 1
            
            success = self.compute_success(t+1, placements)
            self.add_result(t, success, self.dropoff_groups[t])
        self.save_regrets()

class FTLWithCompHistory(FTL):
    def __init__(self, q, lat_grids, lng_grids, save_results=False):
        super(FTLWithCompHistory, self).__init__(q, lat_grids, lng_grids,
                save_results)
        self.alg_id = 'sb_' + str(const.static_boundaries) + '_ftl_w_comp_hist'
        self.init_history = 3

    def compute_regret(self):
        history = self.find_history(range(self.init_history))
        for t in sorted(self.dropoff_groups.keys())[self.init_history:]:
            placements = Counter()
            dropoff_nodes = []
            
            for i in self.dropoff_groups[t]:
                node, lat_idx, lng_idx = h.get_node(self.lat_grids, 
                        self.lng_grids, 
                        self.q.M[i, 5:7])
                dropoff_nodes.append(node)
                leader_node, freq = alg_c.find_rand_leader(lat_idx,
                        lng_idx, self.lat_grids, 
                        self.lng_grids, history)
                placements[leader_node] += 1
            
            # update history from t snapshot
            for n in dropoff_nodes:
                history[n] += 1
             
            if t in self.pickup_groups:
                for i in self.pickup_groups[t]:
                    node = h.get_node(
                                self.lat_grids, self.lng_grids, 
                                self.q.M[i, 2:4])[0]
                    history[node] += 1
            success = self.compute_success(t+1, placements)
            self.add_result(t, success, self.dropoff_groups[t])
        self.save_regrets()

class FTLWithPrevHistory(FTL):
    def __init__(self, q, p, lat_grids, lng_grids, save_results=False):
        super(FTLWithPrevHistory, self).__init__(q, lat_grids, lng_grids,
                save_results)
        self.p = p 
#        g = Group(self.p.start_date, self.p.end_date,
#                self.p.M, self.p.city, width=const.time_window_secs)
        self.g = []
        self.prev_pickup_groups = g.bucket_by_time(0)
        self.prev_dropoff_groups = g.bucket_by_time(4) 
 
        self.alg_id = 'sb_' + str(const.static_boundaries) + '_ftl_w_prev_hist'
        self.init_history = 3

    def find_history_from_prev(self, time_keys):
        history = Counter()
       
        for i in time_keys:
            pickups = self.prev_pickup_groups[i]
            if i in self.prev_dropoff_groups:
                dropoffs = self.prev_dropoff_groups[i]
            else:
                dropoffs = []
            
            """
            1) Find all pickups at t correspoding to nodes
            """
            for i in pickups:
                node = h.get_node(self.lat_grids, self.lng_grids, 
                        self.p.M[i, 2:4])[0]
                history[node] += 1 
        return history 



    def compute_regret(self):
        for t in sorted(self.dropoff_groups.keys())[self.init_history:]:
            if t+1 not in self.prev_pickup_groups:
                continue
            history = self.find_history_from_prev([t+1])
            placements = Counter()
            dropoff_nodes = []
            
            for i in self.dropoff_groups[t]:
                node, lat_idx, lng_idx = h.get_node(self.lat_grids, 
                        self.lng_grids, 
                        self.q.M[i, 5:7])
                dropoff_nodes.append(node)
                leader_node, freq = alg_c.find_rand_leader(lat_idx,
                        lng_idx, self.lat_grids, 
                        self.lng_grids, history)
                placements[leader_node] += 1
            
            # update history from t snapshot
            #for n in dropoff_nodes:
            #    history[n] += 1
            """
            if t in self.prev_dropoff_groups:
                for i in self.prev_dropoff_groups[t]:
                    node = h.get_node(
                                self.lat_grids, self.lng_grids, 
                                self.p.M[i, 5:7])[0]
                    history[node] += 1
            if t in self.prev_pickup_groups:
                for i in self.prev_pickup_groups[t]:
                    node = h.get_node(
                                self.lat_grids, self.lng_grids, 
                                self.p.M[i, 2:4])[0]
                    history[node] += 1
            """
            success = self.compute_success(t+1, placements)
            self.add_result(t, success, self.dropoff_groups[t])
        self.save_regrets()


class PertFTLWithCompHistory(FTL):
    def __init__(self, q, prev, lat_grids, lng_grids, save_results=False):
        self.alg_id = 'sb_' + str(const.static_boundaries) + \
                '_pert_ftl_w_comp_hist'
        self.init_history = 3
        super(PertFTLWithCompHistory, self).__init__(q, lat_grids, lng_grids,
                save_results)

    def compute_regret(self):
        history = self.find_history(range(self.init_history))

        for t in sorted(self.dropoff_groups.keys())[self.init_history:]:
            placements = Counter()
            dropoff_nodes = []

            for i in self.dropoff_groups[t]:
                node, lat_idx, lng_idx = h.get_node(self.lat_grids, 
                        self.lng_grids, self.q.M[i, 5:7])
                dropoff_nodes.append(node)
                leader_node = alg_c.find_perturbed_leader(
                        lat_idx, 
                        lng_idx, 
                        self.lat_grids, 
                        self.lng_grids, 
                        history 
                        )
                placements[leader_node] += 1
            
            """
            update history
            """
            for n in dropoff_nodes:
                history[n] += 1
            
            if t in self.pickup_groups:
                pickups = self.pickup_groups[t]
                for i in pickups:
                    pickup_node = h.get_node(
                                self.lat_grids, self.lng_grids, 
                                self.q.M[i, 2:4])[0]
                    history[pickup_node] += 1
            
            success = self.compute_success(t+1, placements)
            self.add_result(t, success, self.dropoff_groups[t])
        self.save_regrets()

class FTLBestInHindsight(FTL):
    def __init__(self, q, lat_grids, lng_grids, save_results=False):
        super(FTLBestInHindsight, self).__init__(q, lat_grids, lng_grids,
                save_results)
        self.alg_id = 'sb_' + str(const.static_boundaries) + \
                '_ftl_best_in_hindsight'
    
    def compute_regret(self):
        history = self.find_history(range(np.max(self.pickup_groups.keys())))
        
        for t in sorted(self.dropoff_groups.keys()):
            placements = Counter()

            for i in self.dropoff_groups[t]:
                lat_idx, lng_idx = h.get_node(self.lat_grids, 
                        self.lng_grids, self.q.M[i, 5:7])[1:3]
                leader_node = alg_c.find_leader(
                        lat_idx, 
                        lng_idx, 
                        self.lat_grids, 
                        self.lng_grids, 
                        history)[0]
                placements[leader_node] += 1
            
            success = self.compute_success(t+1, placements)
            self.add_result(t, success, self.dropoff_groups[t])
        self.save_regrets()
