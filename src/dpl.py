import numpy as np
import logging
from collections import Counter
import os

from rrg_snapshot import RRGSnapshot
import plot_helpers as ph
import helpers
import const
import time


def compute_stats(P, D, request_ts_vec, reqs_over_time, args, params):
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
    for node_len in range(args.min_node_len, args.max_node_len+1, 50):
        n_nodes = []
        n_edges = []
        in_degree = Counter()
        out_degree = Counter()
        diameter = []
        n_rides = []

        lat_grids, lng_grids = helpers.grid_area(
            params.start_lat,
            params.end_lat,
            params.start_lng, params.end_lng, 
            node_len)

        tot_nodes = len(lat_grids) * len(lng_grids)

        for t, idxs in reqs_over_time.items():
            if len(idxs) <= 10 or \
                (args.skip_night_hours and
                helpers.is_night_hour(request_ts_vec[idxs][0],
                    params.time_zone)):
                continue
            if t == args.max_time_bin:
                break

            rrg_t = RRGSnapshot()
            rrg_t.init(P[idxs, :], D[idxs, :], lat_grids, lng_grids)
            rrg_t.compute_nodes_and_edges()
            rrg_t.compute_node_degree()

            n_nodes.append(rrg_t.n_nodes)
            n_edges.append(rrg_t.n_edges)
            n_rides.append(len(idxs))

            in_degree += rrg_t.in_degree
            out_degree += rrg_t.out_degree

            diameter.append(
                helpers.compute_diameter_effective(rrg_t.out_weights))

            logging.debug(
                    """node len={}m, time_bin={}, num_nodes={}, """
                    """num_edges={}, diameter={}, #rides={}""".format(
                        node_len, t, n_nodes[-1], n_edges[-1], diameter[-1],
                        len(idxs))
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
            ph.dpl_plot(params.prefix,
                        fname, n_nodes, n_edges, helpers.fit_func, p)
            ph.node_degree_plot(params.prefix, fname, in_degree, True)
            ph.node_degree_plot(params.prefix, fname, out_degree, False)
            ph.effective_diameter(params.prefix, fname, n_nodes, diameter)
    out_fd.close()
