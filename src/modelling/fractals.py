import logging
import os
import numpy as np

import utils.const as const
import utils.helpers as helpers
import utils.plot_helpers as ph

from utils.rrg_snapshot import RRGSnapshot

fractal_limits = {
    "new_york": [450, 2500],
    "san_francisco": [450, 2500],
    "los_angeles": [1500, 4000],
    "chicago": [600, 3000],
    "new_york_yellow": [450, 2500],
    "mexico_city": [600, 2500],
    "rio_de_janeiro": [600, 2500],
    "london": [600, 2500],
    "toronto": [500, 2500],
    "boston": [600, 2500],
    "paris": [900, 4000],
    "sao_paulo": [900, 4000],
    "miami": [600, 2500],
    "new_jersey": [900, 2500],
    "new_delhi": [900, 3000],
}


class Spatial:
    def __init__(self, P, D, reqs_ts, reqs_over_time, args, params):
        self.P = P
        self.D = D
        self.reqs_ts = reqs_ts
        self.reqs_over_time = reqs_over_time
        self.args = args
        self.params = params

    def compute_stats(self):
        logging.info("Performing fractal analysis for {}".format(self.params.prefix))
        if self.params.prefix in fractal_limits:
            min_len = fractal_limits[self.params.prefix][0]
            max_len = fractal_limits[self.params.prefix][1]
        else:
            min_len = 450
            max_len = 2500

        logging.info(
            """Processing city {} with min_node_len={}m, """
            """max_nodel_len={}m, time_bin_width={}secs, max_time_bin={}""".format(
                self.params.prefix,
                min_len,
                max_len,
                self.args.time_bin_width,
                self.args.max_time_bin
            )
        )

        time_idx = []
        d2 = []
        d0 = []

        if self.args.save_results:
            logging.info("Saving results")
            out_file = os.path.join(
                const.stats_dir, 'ss_{}_{}s_{}m_{}m_details'.format
                (
                    self.params.prefix,
                    self.args.time_bin_width,
                    min_len,
                    max_len
                )
            )
            out_fd = open(out_file, 'w')
            out_fd.write(
                "time_idx,d0_constant,d0_alpha,d2_constant,d2_alpha,count\n")

        for t, idxs in self.reqs_over_time.items():
            if t == self.args.max_time_bin:
                break

            if len(idxs) <= 10 or \
                (self.args.skip_night_hours and
                 helpers.is_night_hour(self.reqs_ts[idxs[0]],
                                       self.params.time_zone)):
                continue

            epsilon = []
            p_dest = []
            box_count_dest = []
            p_src = []
            box_count_src = []
            """
            Step value for node_len will alter the results for instance a step size
            of 100 meters will give a different exponent for D2 in comparison to
            a step size of 50 meters.
            """
            for node_len in range(min_len, max_len+1, 50):
                lat_grids, lng_grids = helpers.grid_area(
                    self.params.start_lat,
                    self.params.end_lat,
                    self.params.start_lng,
                    self.params.end_lng,
                    node_len)

                rrg_t = RRGSnapshot()
                rrg_t.init(self.P[idxs, :], self.D[idxs, :], lat_grids, lng_grids)
                epsilon.append(node_len)
                #box_count.append(len(rrg_t.dest_nodes))
                prob = np.array(list(rrg_t.dest_nodes.values()))/len(idxs)
                #print("aj ", prob, sum(prob), sum(prob**0), len(rrg_t.dest_nodes))
                box_count_dest.append(
                    np.sum(
                        (np.array(list(rrg_t.dest_nodes.values()))/len(idxs))**0
                    )
                )
                box_count_src.append(
                    np.sum(
                        (np.array(list(rrg_t.source_nodes.values()))/len(idxs))**0
                    )
                )
                p_dest.append(
                    np.sum(
                        (np.array(list(rrg_t.dest_nodes.values()))/len(idxs))**2
                    )
                )
                p_src.append(
                    np.sum(
                        (np.array(list(rrg_t.source_nodes.values()))/len(idxs))**2
                    )
                )

            epsilon = np.array(epsilon)
            box_count_dest = np.array(box_count_dest)
            p_dest = np.array(p_dest)
            box_count_src = np.array(box_count_src)
            p_src = np.array(p_src)

            d0_params, _ = helpers.compute_least_sq(epsilon, box_count_dest)
            d2_params_dest, _ = helpers.compute_least_sq(epsilon, p_dest)
            d2_params_src, _ = helpers.compute_least_sq(epsilon, p_src)
            #print('aj 1 ', np.polyfit(np.log(box_count), np.log(epsilon), 1))
            #print('aj 1 ', np.linalg.lstsq(np.log(box_count), np.log(epsilon)))
            #print('aj 2 ', np.polyfit(np.log(p), np.log(epsilon), 1))
            #print("aj ", d3_params[1]/2)
            if self.args.save_results:
                """
                ph.fractal_plot(
                    self.params.prefix, 'd0_t{}'.format(t), epsilon,
                    box_count_dest,
                    helpers.fit_func,
                    d0_params,
                    xlabel=r'$\log \epsilon$', ylabel=r'$\log \sum_i p_i^2$',
                    prefix='d0_' + str(t),
                    xlim=[10**2, 10**3.65],
                    ylim=[10**1, 10**3])
                ph.fractal_plot(
                    self.params.prefix, 'dest_d2_t{}'.format(t), epsilon,
                    p_dest,
                    helpers.fit_func,
                    d2_params_dest,
                    xlabel=r'$\log \epsilon$', ylabel=r'$\log \sum_i p_i^2$',
                    xlim=[10**2, 10**3.65],
                    ylim=[10**2, 10**4])
                ph.fractal_plot(
                    self.params.prefix, 'src_d2_t{}'.format(t), epsilon,
                    p_src,
                    helpers.fit_func,
                    d2_params_src,
                    xlabel=r'$\log \epsilon$', ylabel=r'$\log \sum_i p_i^2$',
                    xlim=[10**2, 10**3.65],
                    ylim=[10**2, 10**4])
                """
                out_fd.write("%d, %.3f, %.3f, %.3f, %.3f, %d\n" % (
                    t,
                    d0_params[0],
                    -d0_params[1],
                    d2_params_dest[0],
                    d2_params_dest[1],
                    len(idxs)))

            logging.debug("city %s, time snapshot %d, D0 %.3f D2 %.3f #rides %d" % (
                self.params.prefix,
                t,
                -d0_params[1],
                d2_params_dest[1],
                len(idxs)))
            time_idx.append(t)
            d0.append(d0_params[1])
            d2.append(d2_params_dest[1])
        d0 = -np.array(d0)
        d2 = np.array(d2)

        l = """city %s,
               D0 mean %.3f, min %.3f, max %.3f
               D2 mean %.3f, min %.3f, max %.3f """ % (
            self.params.prefix,
            np.mean(d0),
            np.mean(d0),
            np.max(d0),
            np.mean(d2),
            np.min(d2),
            np.max(d2))

        logging.info(l)
        summary_fd = open(os.path.join(
            const.stats_dir, 'ss_{}_{}s_{}m_{}m_summary'.format(
                self.params.prefix,
                self.args.time_bin_width,
                min_len,
                max_len
            )), 'w')
        summary_fd.write(l)
        summary_fd.close()
