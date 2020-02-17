#!/usr/bin/env python3.6
import argparse
import sys
import numpy as np
import pandas
from collections import Counter
import pandas as pd
import matplotlib.pylab as plt
import os
import logging
from pathlib import Path

"""
-- number of nodes active relative to the maximum number of nodes
-- distances of tiles 
-- clustering tiles
"""

from rrg_snapshot import RRGSnapshot
import const
import helpers

const.plot_dir = './plots'
const.stats_dir = './stats'

def enable_log_scale():
    plt.xscale('log')
    plt.yscale('log')


def dpl_plot(city, fname, n_nodes, n_edges, fit_func, p):
    enable_log_scale()
    plt.xlabel('Node count', fontsize=25)
    plt.ylabel('Edge count', fontsize=25)
    plt.tick_params(axis='both', labelsize=15)
    plt.ylim([1, 10**4])
    plt.xlim([1, 10**4])
    plt.subplots_adjust(top=0.88)
    g = plt.plot(n_nodes, n_edges, 'kx', markersize=3)[0]
    plt.plot(n_nodes, fit_func(n_nodes, p), color='r')
    plt.title("C=%.3f, alpha=%.3f" % (p[0], p[1]), fontsize=25)
    dirname = os.path.join(const.plot_dir, city, "dpl")
    Path(dirname).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        os.path.join(dirname, "{}.png".format(fname)),
        format='png', dpi=500, bbox_inches='tight')
    plt.clf()


def node_degree_plot(city, fname, degree, in_degree):
    if in_degree:
        label = 'in_degree'
    else:
        label = 'out_degree'
    vals = [[k, v] for k, v in degree.items()]
    vals = np.array(vals)
    enable_log_scale()
    plt.scatter(vals[:, 0], vals[:, 1], marker='x', s=20, c='k')
    plt.xlabel('Node {}'.format(label), fontsize=25)
    plt.ylabel('Number of nodes', fontsize=25)
    plt.tick_params(axis='both', labelsize=15)
    dirname = os.path.join(const.plot_dir, city, label)
    Path(dirname).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        os.path.join(dirname, "{}.png".format(fname)),
        format='png', dpi=500, bbox_inches='tight')
    plt.clf()


def dpl(args, params):
    if args.save_results:
        logging.info("Saving results")
        out_file = os.path.join(
            const.stats_dir, 'dpl_{}_{}s_{}m_{}m'.format
            (
                params.prefix,
                args.time_bin_width,
                args.min_node_len,
                args.max_node_len
            )
        )
        out_fd = open(out_file, 'w')
        out_fd.write(
            "node_len,c,alpha,r2,mean_nodes,mean_edges,tot_possible_nodes\n")
    logging.info(
        """Processing city {} with min_node_len={}m, """
        """max_nodel_len={}m, time_bin_width={}secs, max_time_bin={}""".format(
            params.prefix, 
            args.min_node_len, 
            args.max_node_len, 
            args.time_bin_width,
            args.max_time_bin
        )
    )
    df = pandas.read_csv(params.fname, sep=',')
    request_ts_vec = df.loc[:, ['request_timestamp']].values.astype(np.float64)
    P = df.loc[:, ['pickup_latitude', 'pickup_longitude']].values.astype(
        np.float64)
    D = df.loc[:, ['dropoff_latitude', 'dropoff_longitude']].values.astype(
        np.float64)
    time_bin_bounds = helpers.get_time_bin_bounds(
        request_ts_vec,
        args.time_bin_width)
    data_time_buckets = helpers.bucket_by_time(time_bin_bounds, request_ts_vec)
    
    for node_len in range(args.min_node_len, args.max_node_len+1, 50):
        n_nodes = []
        n_edges = []
        in_degree = Counter()
        out_degree = Counter()
        
        lat_grids, lng_grids = helpers.grid_area(
            params.start_lat,
            params.end_lat,
            params.start_lng, params.end_lng, 
            node_len)
        tot_nodes = len(lat_grids) * len(lng_grids)

        for t, idxs in data_time_buckets.items():
            if len(idxs) <= 10:
                continue
            if t == args.max_time_bin:
                break
            rrg_t = RRGSnapshot()
            rrg_t.init(P[idxs, :], D[idxs, :], lat_grids, lng_grids)
            rrg_t.compute_nodes_and_edges()
            rrg_t.compute_node_degree()
            n_nodes.append(rrg_t.n_nodes)
            n_edges.append(rrg_t.n_edges)
            in_degree += rrg_t.in_degree
            out_degree += rrg_t.out_degree
            logging.debug(
                """node len={}m, time_bin={}, num_nodes={}, """
                """num_edges={}, #rides={}""".format(
                    node_len, t, n_nodes[-1], n_edges[-1], len(idxs)
                )
            )
        p, infodict = helpers.compute_least_sq(n_nodes, n_edges)
        r2 = helpers.compute_r2(n_edges, infodict)
        logging.info(
            """city prefix=%s, node_len=%dm, C=%.3f, alpha=%.3f, r2=%.3f, """
            """mean nodes=%d, mean edges=%d, total possible nodes=%d"""
            % (params.prefix, node_len, p[0], p[1], r2, np.mean(n_nodes), 
                np.mean(n_edges),
                tot_nodes))
        if args.save_results:
            out_fd.write("%d, %.3f, %.3f, %.3f, %d, %d, %d\n".format(
                node_len, p[0], p[1], r2, np.mean(n_nodes), np.mean(n_edges),
                tot_nodes))
            fname = "n{}_t{}".format(node_len, args.time_bin_width)
            dpl_plot(params.prefix,
                     fname, n_nodes, n_edges, helpers.fit_func, p)
            node_degree_plot(params.prefix, fname, in_degree, True)
            node_degree_plot(params.prefix, fname, out_degree, False)
    out_fd.close()


def main():
    """
    To execute:
    `./main.py`
    """
    parser = argparse.ArgumentParser(description="DPL plots for cities")
    parser.add_argument("-s", "--save_results", help="save plots and stats",
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
                        default=50)
    parser.add_argument("--max_node_len", 
                        help="maximum length of node (meters)",
                        type=int,
                        default=500)
    parser.add_argument("--time_bin_width", 
                        help="time bin width (seconds)",
                        type=int,
                        default=300)
    parser.add_argument(
        "--max_time_bin",
        help="maximum number of time bins of width (--time_bin_width) to read",
        type=int,
        default=2016)
    parser.add_argument(
        "-d", "--debug", help="debug logging level",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.INFO)
    args = parser.parse_args()

    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] - %(message)s',
        level=args.loglevel
    )

    df = pd.read_csv(args.input, sep=',')

    for i, r in df.iterrows():
        if args.cities == i:
            break
        p = helpers.Params(r)
        dpl(args, p)

if __name__ == "__main__":
    main()
