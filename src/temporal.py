import numpy as np
import logging
from collections import Counter
import os
import networkx as nx

from rrg_snapshot import RRGSnapshot
import plot_helpers as ph
import helpers
import const
import time

class Temporal:
    def _compute_real_node_degree_exp(self, key):
        if len(self.rrg_t[key].in_degree) >= 2:
            degree_mle_exp = helpers.real_degree_exp(self.rrg_t[key])
            self.node_degree_exp[key].append(degree_mle_exp)


    def _update_rrg_and_attrs(self, key, P, D, lat_grids, lng_grids, t):
        self.rrg_t[key].init(P, D, lat_grids, lng_grids)
        self.rrg_t[key].compute_nodes_and_edges()
        self.rrg_t[key].compute_node_degree()
        self.n_nodes[key].append(self.rrg_t[key].n_nodes)
        self.n_edges[key].append(self.rrg_t[key].n_edges)
        self._compute_real_node_degree_exp(key)

    def _compute_dpl(self, key):
        p, infodict = helpers.compute_least_sq(
            self.n_nodes[key][10:],
            self.n_edges[key][10:])
        r2 = helpers.compute_r2(self.n_edges[key][10:], infodict)
        return r2, p


    def _print_debug_stats(self, node_len, t, n):
        logging.debug(
            """node len={}m, time_bin={}, #rides={};\n {}: num_nodes={}, """
            """num_edges={}, degree_exp={:.3f};\n {}: num_nodes={}, """
            """num_edges={}, degree_exp={:.3f}; """.format(
                node_len,
                t, n,
                "each_ts",
                self.n_nodes['each_ts'][-1],
                self.n_edges['each_ts'][-1],
                self.node_degree_exp['each_ts'][-1],
                'every_n_ts',
                self.n_nodes['every_n_ts'][-1]
                    if len(self.n_nodes['every_n_ts']) else 0,
                self.n_edges['every_n_ts'][-1]
                    if len(self.n_edges['every_n_ts']) else 0,
                self.node_degree_exp['every_n_ts'][-1]
                    if len(self.node_degree_exp['every_n_ts']) else 0))


    def _print_info_stats(self, key, dpl_params, r2, theor_deg_exp,
        node_len, tot_nodes):
        logging.info(
            """%s for %s; node_len=%dm;\n dpl:C=%.3f, exp=%.3f, """
            """r2=%.3f; mean nodes=%d, edges=%d, tot possible nodes=%d;\n """
            """degree_exp:theor=%.3f real(avg.)=%.3f"""
            % (key,
               self.params.prefix,
               node_len,
               dpl_params[0],
               dpl_params[1],
               r2,
               np.mean(self.n_nodes[key]),
               np.mean(self.n_edges[key]),
               tot_nodes,
               theor_deg_exp,
               np.mean(self.node_degree_exp[key])))


    def _generate_plots(self, key, dpl_params, node_len, t):
        fname = "{}_{}_n{}_t{}".format(key, t, node_len,
            self.args.time_bin_width)
        ph.dpl_plot(self.params.prefix,
            "dpl_" + fname, self.n_nodes[key],
            self.n_edges[key],
            helpers.fit_func,
            dpl_params)
        ph.node_degree_exp_plot(self.params.prefix, "degree_" + fname,
            self.n_nodes[key], self.node_degree_exp[key])


    def __init__(self, P, D, reqs_ts, reqs_over_time, args, params):
        self.P = P
        self.D = D
        self.reqs_ts = reqs_ts
        self.reqs_over_time = reqs_over_time
        self.args = args
        self.params = params

        """
        `each_ts` -- refers to all attributes related to generation of graph for each time snaphsot (ts)
        `every_n_ts` -- refers to all attributes related to generation of graph every n ts
        """
        self.n_nodes = {'each_ts': [], 'every_n_ts': []}
        self.n_edges = {'each_ts': [], 'every_n_ts': []}
        self.rrg_t = {'each_ts': RRGSnapshot(), 'every_n_ts': RRGSnapshot()}
        self.node_degree_exp = {'each_ts': [], 'every_n_ts': []}

        if args.save_results:
            logging.info("Saving results")
            out_file = os.path.join(
                const.stats_dir, 'temporal_{}_{}s_{}m_{}m'.format
                (
                    self.params.prefix,
                    self.args.time_bin_width,
                    self.args.min_node_len,
                    self.args.max_node_len
                )
            )
            self.out_fd = open(out_file, 'w')
            self.out_fd.write(
                "node_len,c,alpha,r2,mean_nodes,mean_edges,tot_possible_nodes\n")


    def compute_stats(self):
        logging.info(
            """Processing city {} with min_node_len={}m, """
            """max_nodel_len={}m, time_bin_width={}secs, """
            """max_time_bin={}""".format(
                self.params.prefix,
                self.args.min_node_len,
                self.args.max_node_len,
                self.args.time_bin_width,
                self.args.max_time_bin))

        for node_len in range(self.args.min_node_len,
            self.args.max_node_len+1,
            50):

            n_rides = []

            lat_grids, lng_grids = helpers.grid_area(
                self.params.start_lat,
                self.params.end_lat,
                self.params.start_lng,
                self.params.end_lng,
                node_len)

            tot_nodes = len(lat_grids) * len(lng_grids)

            for t, idxs in self.reqs_over_time.items():
                if len(idxs) <= 10 or (self.args.skip_night_hours and \
                    helpers.is_night_hour(
                        self.reqs_ts[idxs[0]],
                        self.params.time_zone)):
                    continue


                if t == self.args.max_time_bin:
                    break

                if len(self.n_nodes['every_n_ts']) == 60:
                    r2, p = self._compute_dpl('every_n_ts')
                    theor_deg_exp = helpers.theor_degree_exp(p[1])
                    self._print_info_stats('every_n_ts', p,
                        r2, theor_deg_exp, node_len, tot_nodes)

                    if self.args.save_results:
                        self._generate_plots('every_n_ts',
                            p, node_len, t)

                    self.n_nodes['every_n_ts'] = []
                    self.n_edges['every_n_ts'] = []
                    self.node_degree_exp['every_n_ts'] = []
                    self.rrg_t['every_n_ts'] = RRGSnapshot()

                # Generate a new graph for each time snapshot (ts)
                self.rrg_t['each_ts'] = RRGSnapshot()
                self._update_rrg_and_attrs('each_ts', self.P[idxs, :],
                    self.D[idxs, :], lat_grids, lng_grids, t)

                # Not to generate a new graph for every n time snapshots analysis
                self._update_rrg_and_attrs('every_n_ts', self.P[idxs, :],
                    self.D[idxs, :], lat_grids, lng_grids, t)

                self._print_debug_stats(node_len, t, len(idxs))

                n_rides.append(len(idxs))
                #d1, d2 = helpers.compute_diameter_effective(rrg_t.out_weights)

            r2, p = self._compute_dpl('each_ts')
            theor_deg_exp = helpers.theor_degree_exp(p[1])
            self._print_info_stats('each_ts', p, r2, theor_deg_exp,
                node_len, tot_nodes)

            if self.args.save_results:
                self.out_fd.write("%d, %.3f, %.3f, %.3f, %d, %d, %d\n".format(
                    node_len, p[0], p[1], r2,
                    np.mean(self.n_nodes['each_ts']),
                    np.mean(self.n_edges['each_ts']),
                    tot_nodes))
                self._generate_plots('each_ts', p, node_len, t)

                #ph.node_degree_plot(params.prefix, fname, node_degree, True)
                #ph.node_degree_plot(params.prefix, fname, in_degree, True)
                #ph.node_degree_plot(params.prefix, fname, out_degree, False)
                #ph.effective_diameter(params.prefix, fname, n_nodes, diameter)
                self.out_fd.close()
