from collections import Counter
import numpy as np

import utils.helpers as h
import applications.placement.algo_commons as alg_c
import utils.const as const
from applications.placement.algorithms.base import Algorithm


class Random(Algorithm):
    def __init__(self, q, lat_grids, lng_grids, save_results):
        self.alg_id = 'sb_' + str(const.static_boundaries) + '_rndm'
        super(Random, self).__init__(q, lat_grids, lng_grids, save_results)

    def compute_regret(self):
        pickup_groups = self.g.bucket_by_time(0)
        dropoff_groups = self.g.bucket_by_time(4)

        for t in sorted(dropoff_groups.keys()):
            success = 0
            placements = Counter()
            dropoffs = dropoff_groups[t]
            for i in dropoffs:
                lat_idx, lng_idx = h.get_node(
                            self.lat_grids, self.lng_grids, 
                            self.q.M[i, 5:7])[1:3]
                idxs = alg_c.get_boundaries(self.lat_grids, self.lng_grids,
                        lat_idx, lng_idx)
                rndm_lat_idx = np.random.randint(idxs[0], idxs[1])
                rndm_lng_idx = np.random.randint(idxs[2], idxs[3])
                rndm_node = rndm_lat_idx * len(self.lng_grids) + \
                        rndm_lng_idx
                placements[rndm_node] += 1
            
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
