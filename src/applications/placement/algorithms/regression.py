import utils.helpers as h
import utils.const as const

from applications.placement.algorithms.base import Algorithm
import applications.placement.algo_commons as alg_c


from collections import Counter
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class Regression(Algorithm):
    def __init__(self, q, lat_grids, lng_grids, n_history=1,
            save_results=False):
        self.alg_id = 'sb_' + str(const.static_boundaries) + \
                '_reg_' + str(n_history)
        self.n_history = n_history
        super(Regression, self).__init__(q, lat_grids, lng_grids,
                save_results)
    
    def _get_training(self, pickup_groups, dropoff_groups, t):
        centers = Counter()
        for i in t:
            pickups = pickup_groups[i]
            dropoffs = dropoff_groups[i] if i in dropoff_groups else []
            S = self.q.M[pickups, 2:4]
            D = self.q.M[dropoffs, 5:7]
        
            """
            1) Find all pickups at t correspoding to nodes
            """
            for s_i in range(len(S)):
                lat_idx, lng_idx = h.get_node(self.lat_grids, 
                        self.lng_grids, S[s_i])[1:3]
                lat_c = (self.lat_grids[lat_idx] + self.lat_grids[lat_idx+1])/2
                lng_c = (self.lng_grids[lng_idx] + self.lng_grids[lng_idx+1])/2
                centers[(lat_c, lng_c)] += 1
            
            """
            2) Find all dropoffs at t corresponding to nodes
            """
            for d_i in range(len(D)):
                lat_idx, lng_idx = h.get_node(self.lat_grids, 
                        self.lng_grids, D[d_i])[1:3]
                lat_c = (self.lat_grids[lat_idx] + self.lat_grids[lat_idx+1])/2
                lng_c = (self.lng_grids[lng_idx] + self.lng_grids[lng_idx+1])/2
                centers[(lat_c, lng_c)] += 1
        
        training_centers = []
        for k,v in centers.iteritems():
            training_centers.append([k[0], k[1], v])
        return np.array(training_centers)

    def compute_regret(self):
        pickup_groups = self.g.bucket_by_time(0)
        dropoff_groups = self.g.bucket_by_time(4) 

        for t in sorted(pickup_groups.keys())[self.n_history:]:
            success = 0
            placements = Counter()
            dropoffs = [] 

            training_centers = self._get_training(pickup_groups, dropoff_groups,
                    range(t-self.n_history, t))
            #reg = linear_model.LogisticRegression()
            reg = RandomForestRegressor()
            reg.fit(training_centers[:,:2], training_centers[:,2])

            if t in dropoff_groups:
                dropoffs = dropoff_groups[t]
                for i in dropoffs:
                    dropoff_lat_cell, dropoff_lng_cell = h.get_node(
                                self.lat_grids, self.lng_grids, 
                                self.q.M[i, 5:7])[1:3]
                    test_centers = alg_c.find_bandit_centers(dropoff_lat_cell,
                            dropoff_lng_cell, self.lat_grids, 
                            self.lng_grids)
                    p = test_centers[np.argmax(reg.predict(test_centers))]
                    argmax_node = h.get_node(
                            self.lat_grids, self.lng_grids, p)[0]
                    placements[argmax_node] += 1
            
            if (t+1) in pickup_groups:
                pickups = pickup_groups[t+1]
                for i in pickups:
                    node = h.get_node(self.lat_grids, self.lng_grids, 
                            self.q.M[i, 2:4])[0]
                    if node in placements:
                        success += 1
                        if placements[node] == 1:
                            placements.pop(node, None)
                        else:
                            placements[node] -= 1
            self.add_result(t, success, dropoffs)
        self.save_regrets()
