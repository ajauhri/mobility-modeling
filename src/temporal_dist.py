#!/usr/bin/env python3.6
import utils.const as const
import utils.helpers as helpers
from modelling.temporal import Temporal
from utils.rrg_snapshot import RRGSnapshot

import argparse
import sys
import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pylab as plt


class Temporal:
    def __init__(self, P, D, reqs_ts, reqs_over_time, args, params):
        self.P = P
        self.D = D
        self.reqs_ts = reqs_ts
        self.reqs_over_time = reqs_over_time
        self.args = args
        self.params = params

    def compute_stats(self, week, ax):
        logging.info(
            """Processing city {} with """
            """time_bin_width={}secs, """
            """max_time_bin={}""".format(
                self.params.prefix,
                self.args.time_bin_width,
                self.args.max_time_bin))


        x = []
        y = []
        for t, idxs in self.reqs_over_time.items():
            if t == self.args.max_time_bin:
                break
            x.append(t)
            y.append(len(idxs))

        labels = ['week n', 'week n+1']
        markers = ['k-', 'r--']
        plt.plot(x, y, markers[week], linewidth=1, label=labels[week])

        if week == 1:
            plt.yticks([])
            plt.legend()
            plt.xlabel("time", fontsize=18)
            plt.xticks(np.arange(min(x), max(x) + 1, 96))
            ax.set_xticklabels(
                ["Fri", "Sat", "Sun", "Mon", "Tue", "Wed", "Thur"],
                size='medium')
            ax.xaxis.set_tick_params(width=1.5, length=9)
            plt.xlim([0, 680])
            plt.ylim([0, 8000])
            plt.ylabel("volume of ride requests", fontsize=18)
            plot_path = os.path.join(const.plot_dir,
                "volume_time_series_{}.pdf".format(self.params.prefix))
            plt.savefig(plot_path, format='pdf', dpi=800, bbox_inches='tight')
            logging.info("Plot saved at {}".format(plot_path))
            plt.clf()


def compute_stats(args, params, week, ax):
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
    a = Temporal(P, D, reqs_ts, reqs_over_time, args, params)
    a.compute_stats(week, ax)


def main():
    """
    To execute:
    `./src/temporal_dist.py`
    """
    parser = argparse.ArgumentParser(description="DPL plots for cities")
    parser.add_argument("-i", "--input", help="input file",
                        default="temporal_plot_cities.csv")
    parser.add_argument("--time_bin_width",
                        help="time bin width in seconds",
                        type=int,
                        default=60*15)
    parser.add_argument(
        "--max_time_bin",
        help="maximum number of time bins of width (--time_bin_width) to read",
        type=int)
    parser.add_argument(
        "-d", "--debug", help="debug logging level",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.INFO)
    args = parser.parse_args()

    if args.max_time_bin == None:
        args.max_time_bin = int((60 * 60 * 7 * 24) / args.time_bin_width)

    logging.basicConfig(
        format=("[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s "
                "- %(message)s"),
        level=args.loglevel
    )

    df = pd.read_csv(args.input, sep=',', dtype={'start_lat':float,
        'end_lat':float, 'start_lng':float, 'end_lng':float, 'cons_ts':int})

    week_i = 0
    for i, r in df.iterrows():
        # we assume two consecutive rows are present for each city
        if i % 2 == 0:
            week_i = 0
            fig, ax = plt.subplots()
        p = helpers.Params(r)
        compute_stats(args, p, week_i, ax)
        week_i += 1


if __name__ == "__main__":
    main()
