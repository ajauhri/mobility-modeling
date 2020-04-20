import utils.helpers as h
import utils.const as const
from applications.placement.algorithms.base import Algorithm
import applications.placement.algo_commons as alg_c

from collections import Counter
import logging
import numpy as np

class Opt(Algorithm):
    def __init__(self, q, lat_grids, lng_grids, save_results=False):
        self.alg_id = 'sb_' + str(const.static_boundaries) + '_opt'
        super(Opt, self).__init__(q, lat_grids, lng_grids, save_results)
    
    def compute_regret(self):
        pickup_groups = self.g.bucket_by_time(0)
        dropoff_groups = self.g.bucket_by_time(4) 

        for t in sorted(dropoff_groups.keys()):
            success = 0
            future_pickups = Counter()
            n_dropoffs = len(dropoff_groups[t])
            if (t+1) in pickup_groups:
                pickups = pickup_groups[t+1]
                for i in pickups:
                    node = h.get_node(self.lat_grids, self.lng_grids, 
                            self.q.M[i, 2:4])[0]
                    future_pickups[node] += 1
            
            for i in dropoff_groups[t]:
                lat_idx, lng_idx = h.get_node(
                            self.lat_grids, self.lng_grids, 
                            self.q.M[i, 5:7])[1:3]
                
                idxs = alg_c.get_boundaries(self.lat_grids, self.lng_grids,
                        lat_idx, lng_idx)
                
                for j in range(idxs[0], idxs[1]):
                    for k in range(idxs[2], idxs[3]):
                        node = j * len(self.lng_grids) + k
                        if node in future_pickups:
                            success += 1
                            print(t, i, node)
                            if future_pickups[node] == 1:
                                future_pickups.pop(node, None)
                            else:
                                future_pickups[node] -= 1
                            break
                    else:
                        continue
                    break
            if n_dropoffs > 0:
                good_per = 100*success/n_dropoffs
                print_str = ("time key: {0}, #good placements: {1}, """
                    """#percentage: {2:.2f}, #total: {3}""".format(t, 
                    success, 100*success/n_dropoffs, n_dropoffs))
                logging.debug(print_str)
                self.regrets.append([t, success, n_dropoffs])
            if t == 3:
                break
        self.save_regrets()

    def compute_pickups_and_eta(self):
        """
        header information contains id, request time, begin time, being lat,
        begin lng, dropoff time, dropoff lat, dropoff lng, actual eta, 
        predicted eta
        """
        c = Group(self.q.start_date, self.q.end_date, self.q.M, self.q.city, 
                const.time_window_secs)
        pickup_groups = c.bucket_by_time(0)
        dropoff_groups = c.bucket_by_time(4) 

        for k, dropoffs in dropoff_groups.iteritems():
            etas = []
            n_dropoffs = len(dropoffs)
            n_close_pickups = 0
            n_close_poolable_pickups = 0
            n_no_close_pickups = 0
            if (k + 1) in pickup_groups:
                pickups = pickup_groups[k+1]
                n_pickups = len(pickups) #TODO: some erroneous(with zero speed, see calculate_eta) trips need to be removed from being counted
                while len(dropoffs) > 0 and len(pickups) > 0:
                    rides_tbd = []
                    d = alg_c.orthodromic_dist(
                            self.q.M[np.ix_(pickups, [2,3])],
                            self.q.M[dropoffs[0], [5,6]])
                    
                    close_pickups = pickups[d <= const.max_pickup_radius]
                    
                    g = ClusterByDist(self.q.start_date, self.q.end_date, 
                            self.q.M, self.q.city, self.q.ids, 
                            const.src_radius, const.dest_radius)
                    poolable_rides = g.find_poolability(k+1, 
                            [2,3], [5,6], close_pickups)[2]
                    if len(poolable_rides) >= 2:
                        rides_tbd = poolable_rides
                        r = alg_c.calculate_eta(dropoffs[0], poolable_rides,
                                self.q)
                        etas += r
                        n_close_poolable_pickups += len(r)
                    elif len(poolable_rides) == 0 and len(close_pickups) >= 1:
                        rides_tbd = [close_pickups[0]]
                        r = alg_c.calculate_eta(dropoffs[0], max_poolable,
                                self.q)
                        etas += r
                        n_close_pickups += len(r)
                    elif len(close_pickups) == 0:
                        n_no_close_pickups += 1
                    
                    #remove all pickups which were close
                    idx = []
                    for i in rides_tbd:
                        idx.append(np.where(pickups==i)[0][0])
                    pickups = np.delete(pickups, idx)
                    dropoffs = dropoffs[1:]

            if n_dropoffs > 0: 
                self.results[k+1] = etas
                self.stats[k+1] = [n_close_pickups, 
                        n_close_poolable_pickups, n_pickups, 
                        n_no_close_pickups, n_dropoffs]
                
        if self.save_results:
            with open(os.path.join(const.results_dir, 
                'baseline_' + self.q.city + '.pk1'), 'wb') as fd:
                pickle.dump(self.results, fd, -1)
                pickle.dump(self.stats, fd, -1)
                fd.close()
