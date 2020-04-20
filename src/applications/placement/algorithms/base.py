#from utils.grouping import Group
import utils.const as const

import os
import numpy as np
import logging

class Algorithm(object):
    def __init__(self, q, lat_grids, lng_grids, save_results):
        self.results = {}
        self.stats = {}
        self.q = q
        self.lat_grids = lat_grids
        self.lng_grids = lng_grids
        self.save_results = save_results
        self.regrets = []
#        self.g = Group(self.q.start_date, self.q.end_date,
#                self.q.M, self.q.city, width=const.time_window_secs)
        self.g = []
        self.pickup_groups = self.g.bucket_by_time(0)
        self.dropoff_groups = self.g.bucket_by_time(4) 
  
    def save_regrets(self):
        if self.save_results:
            self.regrets = np.array(self.regrets)
            np.savetxt(os.path.join(const.results_dir, 
                self.alg_id + "_" + self.q.city + ".csv"), 
                self.regrets, delimiter=',')


    def add_result(self, t, success, dropoffs):
       if len(dropoffs) > 0:
            success_per = 100*success/len(dropoffs)
            print_str = ("alg: {0}, time key: {1}, #good placements: {2}, """
                """#percentage: {3:.2f}, #total: {4}""".format(self.alg_id, t, 
                    success, success_per, len(dropoffs)))
            logging.debug(print_str)
            self.regrets.append([t, success, len(dropoffs)])
