import applications.placement.algorithms.ftl as ftl
#import applications.placement.algorithms.mru as mru
#import applications.placement.algorithms.prob as prob
from applications.placement.algorithms.poisson import PointProcess
from applications.placement.algorithms.regression import Regression
from applications.placement.algorithms.shiftband import ShiftBand
from applications.placement.algorithms.baseline import Random
from applications.placement.algorithms.opt import Opt
#from applications.placement.algorithms.sleeping_experts import SleepingExperts
import applications.placement.regret_plots as plots
import applications.placement.algorithms.mab as mab
import utils.const as const

import sys
import argparse
import pandas
import logging

const.default_speed_ms = 5 # default speed in meters/sec
const.time_window_mins = 3 # 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
const.time_window_secs = const.time_window_mins*60
const.time_windows_per_day = int(24 * 60 / const.time_window_mins)
const.src_radius = 100
const.dest_radius = 1000
const.max_pickup_radius = 500
const.region_width = 5 # 12, 14, 16, 18, 20, 22, 24, 25, 26, 28, 30
const.pp_min_samples = 3
const.static_boundaries = 0
const.pftl_epsilon = 0.01 
const.mab_gamma = 0.8
const.node_length_meters = 100
const.input_dir = "./input/"
const.results_dir = "./results/"
const.figures_dir = "./figs/"

const.k = 10
const.gamma = 0.5


def main():
    print
    return
    logging.basicConfig(format='%(asctime)s main: %(message)s', 
            level=logging.DEBUG)
    
    parser = argparse.ArgumentParser(description='Parse logs')
    parser.add_argument("-s", "--save_results", help="save results",
            action="store_true")
    parser.add_argument("-p", "--save_plots", help="regenerate and save plots",
            action="store_true")
    parser.add_argument("start", help="start date (format YYYYMMDD)")
    parser.add_argument("-i", "--input", help="input file",  
        default="")
    parser.add_argument("-n", "--days", type=int, help="time period over days",
        default=1)

    args = parser.parse_args(sys.argv[1:])
    year = int(args.start[:4])
    month = int(args.start[4:6])
    day = int(args.start[6:])

    df = pandas.read_csv(args.input, sep=',')
    for index, row in df.iterrows():
        city = row['city']
        if args.save_plots:
            #plots.compare_ftl(city)
            plots.compare_algs(city)
            #plots.fd_stats(city)
        else:
            file_exists = True
            q = Queries(year, month, day, args.days, city.strip())
            #p = Queries(year, month, 1, args.days, city.strip())
            lat_grids, lng_grids = h.grid_city(row['lat_min'], row['lat_max'], 
                    row['lng_min'], row['lng_max'])
            a = ftl.FTLWithLimitedHistory(q,
                    lat_grids, 
                    lng_grids, 
                    10, 
                    args.save_results
                    )
            a = prob.PDistWithCompHistory(q, 
                    lat_grids, 
                    lng_grids, 
                    args.save_results
                    )
            a = ftl.FTLWithCompHistory(q,
                    lat_grids, 
                    lng_grids, 
                    args.save_results
                    )
            a = ftl.FTLWithPrevHistory(q, p,
                    lat_grids, 
                    lng_grids, 
                    args.save_results
                    )
            a = mru.MRUWithLimitedHistory(q,
                    lat_grids, 
                    lng_grids, 
                    5,
                    args.save_results
                    )
            a = ftl.PertFTLWithCompHistory(q, 
                    lat_grids, 
                    lng_grids, 
                    args.save_results
                    )
            a = ftl.FTLBestInHindsight(q, 
                    lat_grids, 
                    lng_grids, 
                    args.save_results
                    )
            a = SleepingExperts(q, 
                    lat_grids, 
                    lng_grids, 
                    args.save_results
                    )

            a = mab.Exp3WithContext(q, lat_grids, lng_grids, args.save_results)
            a = mab.Exp3(q, lat_grids, lng_grids, args.save_results)
            a = mab.Exp3AndFTL(q, lat_grids, lng_grids, args.save_results)
            a = Opt(q, lat_grids, lng_grids, args.save_results)
            a = ShiftBand(q, lat_grids, lng_grids, args.save_results)
            a = Regression(q, lat_grids, lng_grids, 20, args.save_results)
            a = Random(q, lat_grids, lng_grids, args.save_results)
            a = PointProcess(q, lat_grids, lng_grids, 20, 
                    args.save_results)
            a.compute_regret()
        break

if __name__ == "__main__":
    main()
"""
To execute: ./main.py <start date in yyyymmdd> -n <#of days> -i <input file>
"""
