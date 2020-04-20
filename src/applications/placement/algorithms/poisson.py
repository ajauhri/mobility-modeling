from collections import Counter
from applications.placement.algorithms.base import Algorithm
from scipy.special import factorial

import applications.placement.algo_commons as alg_c
import utils.helpers as helper
import numpy as np
import utils.const as const


class PointProcess(Algorithm):
    def __init__(self, q, lat_grids, lng_grids, n_history=1,
            save_results=False):
        self.alg_id = 'sb_' + str(const.static_boundaries) + '_ppn_' + str(n_history)
        self.n_history = n_history
        self.arrivals = {}
        super(PointProcess, self).__init__(q, lat_grids, lng_grids, 
                save_results)
    
    def _neg_log_likelihood(self, params, data):
        """ the negative log-Likelohood-Function"""
        lnl = - np.sum(np.log(self._poisson(data, params[0])))
        return lnl

    def _poisson(self, k, lamb):
        """poisson pdf, parameter lamb is the fit parameter"""
        return (lamb**k/factorial(k)) * np.exp(-lamb)

    def _fit_poisson_pp(self, pickup_groups, dropoff_groups, t):
        #arrivals = {}
        lam = {}
        for i in t:
            pickups = pickup_groups[i]
            dropoffs = dropoff_groups[i] if i in dropoff_groups else []

            for j in pickups:
                node = helper.get_node(self.lat_grids,
                        self.lng_grids, self.q.M[j, 2:4])[0]
                if node in self.arrivals:
                    self.arrivals[node].append(self.q.M[j, 0])
                else:
                    self.arrivals[node] = [self.q.M[j,0]]
            for j in dropoffs:
                node = helper.get_node(self.lat_grids,
                        self.lng_grids, self.q.M[j, 5:7])[0]
                if node in self.arrivals:
                    self.arrivals[node].append(self.q.M[j, 4])
                else:
                    self.arrivals[node] = [self.q.M[j, 4]]
        for k,v in self.arrivals.iteritems():
            v = np.sort(v).tolist()
            """
            inter_arrival_times = []
            for i in range(len(v[1:])):
                inter_arrival_time = v[i+1] - v[i]
                inter_arrival_times.append(
                        inter_arrival_time/const.time_window_secs)
            inter_arrival_times = np.array(inter_arrival_times)
            """
            self.arrivals[k] = v
            """
            if len(inter_arrival_times) > const.pp_min_samples:
                result = minimize(self._neg_log_likelihood, 
                        x0=np.ones(1), 
                        args=(inter_arrival_times,), 
                        method='L-BFGS-B',
                        bounds=[(0,None)]
                        )
                x_plot = np.linspace(0, np.max(inter_arrival_times), 10000)
                plt.hist(inter_arrival_times, normed=True)
                plt.plot(x_plot, self._poisson(x_plot, result.x), 'r-', lw=2)
                plt.show()
                print 1 - math.pow(np.e, -result.x), result.x
                #lam[k] = result.x
                #lam[k] = np.mean(inter_arrival_times)
                lam[k] = len(inter_arrival_times)/np.sum(inter_arrival_times)
                #print lam[k]
             """
        #return lam
                

    def compute_regret(self):
        pickup_groups = self.g.bucket_by_time(0)
        dropoff_groups = self.g.bucket_by_time(4)
        for t in sorted(dropoff_groups.keys())[self.n_history:]:
            self.arrivals = {}
            self._fit_poisson_pp(pickup_groups, dropoff_groups,
                range(t-self.n_history, t))

            success = 0
            placements = Counter()
            
            """
            lamb = self._fit_poisson_pp(pickup_groups, dropoff_groups,
                    range(t-self.n_history, t))
            """ 
            for i in dropoff_groups[t]:
                dropoff_lat_cell, dropoff_lng_cell = helper.get_node(
                        self.lat_grids, self.lng_grids, 
                        self.q.M[i, 5:7])[1:3]
                placement_node = alg_c.find_mle_node(dropoff_lat_cell,
                        dropoff_lng_cell, self.lat_grids, self.lng_grids,
                        self.arrivals)
                placements[placement_node] += 1
            
            if (t+1) in pickup_groups:
                pickups = pickup_groups[t+1]
                for i in pickups:
                    node = helper.get_node(self.lat_grids, self.lng_grids,
                            self.q.M[i, 2:4])[0]
                    if node in placements:
                        success += 1
                        if placements[node] == 1:
                            placements.pop(node, None)
                        else:
                            placements[node] -= 1
            #self._fit_poisson_pp(pickup_groups, dropoff_groups,
            #    range(t, t+1))
            self.add_result(t, success, dropoff_groups[t])
        self.save_regrets()
