#!/usr/bin/env python3.6
import argparse
import logging

import numpy as np
import pandas as pd

import utils.const as const
import utils.helpers as helpers
from modelling.fractals import Spatial
from modelling.temporal import Temporal

const.stats_dir = './stats'


def compute_stats(args, params):
    df = pd.read_csv(params.fname, sep=',')
    reqs_ts = df.loc[:, ['request_timestamp']].values.astype(np.float64)
    P = df.loc[:, ['pickup_latitude', 'pickup_longitude']].values.astype(
        np.float64)
    D = df.loc[:, ['dropoff_latitude', 'dropoff_longitude']].values.astype(
        np.float64)
    time_bin_bounds = helpers.get_time_bin_bounds(
        reqs_ts,
        args.time_bin_width)
    reqs_over_time = helpers.bucket_by_time(time_bin_bounds, reqs_ts)
    if args.fractal_analysis:
        a = Spatial(P, D, reqs_ts, reqs_over_time, args, params)
    else:
        a = Temporal(P, D, reqs_ts, reqs_over_time, args, params)
    a.compute_stats()


def main():
    """
    To execute:
    `./main.py`
    """
    parser = argparse.ArgumentParser(description="DPL plots for cities")
    parser.add_argument("-s", "--save_results",
                        help="save plots and stats",
                        action="store_true")
    parser.add_argument("-f", "--fractal_analysis",
                        help="run fractal analysis",
                        action="store_true")
    parser.add_argument("-i", "--input", help="input file",
                        default="cities.csv")
    parser.add_argument("-n",
                        "--cities",
                        help="number of cities to read from the input file",
                        type=int,
                        default=1)
    parser.add_argument("--min_node_len",
                        help="miniumum length of node (meters)",
                        type=int,
                        default=100)
    parser.add_argument("--max_node_len",
                        help="maximum length of node (meters)",
                        type=int,
                        default=300)
    parser.add_argument("--time_bin_width",
                        help="time bin width in seconds",
                        type=int,
                        default=300)
    parser.add_argument("--skip_night_hours",
                        help="Skip midnight to 6am for analysis",
                        action="store_true")
    parser.add_argument(
        "--max_time_bin",
        help="maximum number of time bins of width (--time_bin_width) to read",
        type=int)
    parser.add_argument(
        "-d", "--debug", help="debug logging level",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.INFO)
    args = parser.parse_args()

    if args.max_time_bin is None:
        args.max_time_bin = int((60 * 60 * 7 * 24) / args.time_bin_width)

    logging.basicConfig(
        format=("[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s "
                "- %(message)s"),
        level=args.loglevel
    )

    df = pd.read_csv(args.input, sep=',', dtype={
        'start_lat': float,
        'end_lat': float,
        'start_lng': float,
        'end_lng': float,
        'cons_ts': int})

    for i, r in df.iterrows():
        if args.cities == i:
            break
        p = helpers.Params(r)
        compute_stats(args, p)

if __name__ == "__main__":
    main()

