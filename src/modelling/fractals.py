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
}


def compute_stats(P, D, reqs_ts, reqs_over_time, args, params):
    logging.info("Performing fractal analysis for {}".format(params.prefix))
    if params.prefix in fractal_limits:
        min_len = fractal_limits[params.prefix][0]
        max_len = fractal_limits[params.prefix][1]
    else:
        min_len = 450
        max_len = 2500

    logging.info(
        """Processing city {} with min_node_len={}m, """
        """max_nodel_len={}m, time_bin_width={}secs, max_time_bin={}""".format(
            params.prefix,
            min_len,
            max_len,
            args.time_bin_width,
            args.max_time_bin
        )
    )

    time_idx = []
    d2 = []
    d0 = []

    if args.save_results:
        logging.info("Saving results")
        out_file = os.path.join(
            const.stats_dir, 'ss_{}_{}s_{}m_{}m_details'.format
            (
                params.prefix,
                args.time_bin_width,
                min_len,
                max_len
            )
        )
        out_fd = open(out_file, 'w')
        out_fd.write(
            "time_idx,d0_constant,d0_alpha,d2_constant,d2_alpha,count\n")

    for t, idxs in reqs_over_time.items():
        if t == args.max_time_bin:
            break

        if len(idxs) <= 10 or \
            (args.skip_night_hours and
             helpers.is_night_hour(reqs_ts[idxs[0]],
                                   params.time_zone)):
            continue

        epsilon = []
        p = []
        box_count = []
        """
        Step value for node_len will alter the results for instance a step size
        of 100 meters will give a different exponent for D2 in comparison to
        a step size of 50 meters.
        """
        for node_len in range(min_len, max_len+1, 50):
            lat_grids, lng_grids = helpers.grid_area(
                params.start_lat,
                params.end_lat,
                params.start_lng,
                params.end_lng,
                node_len)

            rrg_t = RRGSnapshot()
            rrg_t.init(P[idxs, :], D[idxs, :], lat_grids, lng_grids)
            epsilon.append(node_len)
            #box_count.append(len(rrg_t.dest_nodes))
            box_count.append(
                np.sum(
                    (np.array(list(rrg_t.dest_nodes.values()))/len(idxs))**0
                )
            )
            p.append(
                np.sum(
                    (np.array(list(rrg_t.dest_nodes.values()))/len(idxs))**2
                )
            )

        epsilon = np.array(epsilon)
        box_count = np.array(box_count)
        p = np.array(p)

        d0_params, _ = helpers.compute_least_sq(epsilon, box_count)
        d2_params, _ = helpers.compute_least_sq(epsilon, p)
        if args.save_results:
            ph.fractal_plot(
                params.prefix, 'd0_t{}'.format(t), epsilon,
                box_count,
                helpers.fit_func,
                d0_params,
                xlabel=r'$\log \epsilon$', ylabel=r'$\log N(\epsilon)$',
                prefix='d0_' + str(t),
                xlim=[10**2, 10**3.65],
                ylim=[10**1, 10**3])
            ph.fractal_plot(
                params.prefix, 'd2_t{}'.format(t), epsilon,
                p,
                helpers.fit_func,
                d2_params,
                xlabel=r'$\log \epsilon$', ylabel=r'$\log S2$',
                xlim=[10**2, 10**3.65],
                ylim=[10**2, 10**4])
        if args.save_results:
            out_fd.write("%d, %.3f, %.3f, %.3f, %.3f, %d\n" % (
                t,
                d0_params[0],
                -d0_params[1],
                d2_params[0],
                d2_params[1],
                len(idxs)))

        logging.debug("city %s, time snapshot %d, D0 %.3f D2 %.3f #rides %d" % (
            params.prefix,
            t,
            -d0_params[1],
            d2_params[1],
            len(idxs)))
        time_idx.append(t)
        d0.append(d0_params[1])
        d2.append(d2_params[1])
    d0 = -np.array(d0)
    d2 = np.array(d2)

    l = """city %s, D0 mean %.3f,
           D2 mean %.3f D0 max %.3f,
           D2 max %.3f """ % (
        params.prefix,
        np.mean(d0),
        np.mean(d2),
        np.max(d0),
        np.max(d2))

    logging.info(l)
    summary_fd = open(os.path.join(
        const.stats_dir, 'ss_{}_{}s_{}m_{}m_summary'.format(
            params.prefix,
            args.time_bin_width,
            min_len,
            max_len
        )), 'w')
    summary_fd.write(l)
    summary_fd.close()
