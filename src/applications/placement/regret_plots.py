import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pandas as pd
import numpy as np

import utils.const as const
import utils.helpers as helpers

def compare_ftl(city, t=[3, 10, 20]):
    markers = ['b--', 'g-.', 'r--', 'c--', 'm--']
    n_aggs = 20
    aggregations = np.arange(0, (7*24*60)/const.time_window_mins, n_aggs)
    for i in range(len(t)):
        fname = os.path.join(const.results_dir, 
                'ftl_' + str(t[i]) + '_' + city + ".csv")
        A = pd.read_csv(fname, sep=',').values
        per = []
        x = []
        for a_i in aggregations:
            h = np.where((a_i <= A[:,0]) & (A[:,0] < (a_i + n_aggs)))[0]
            per.append(np.sum(A[h,1]) / np.sum(A[h,2]))
            x.append(a_i)
        print(per)
        plt.plot(x, per, markers[i], label=str(t[i]) + ' history',
                linewidth=1)
    plt.legend()
    plt.title(city)
    plt.show()

def compare_algs(city, ftl_default='3', reg_default='20', pp_default='20'):
    prefix = 'sb_' + str(const.static_boundaries)
    algs = {
            #0: '_opt', 
            #0: '_opt_10', 
            #1: '_opt_12', 
            #2: '_opt_14', 
            #3: '_opt_16', 
            #4: '_opt_18', 
            #5: '_opt_20', 
            #6: '_opt_30', 
            #1: '_ftl_best_in_hindsight', 
            2: '_ftl_w_comp_hist', 
            #9: '_ftl_w_prev_hist', 
            #10: '_mru_w_lim_hist_10', 
            #5: '_mru_w_lim_hist_5', 
            #3: '_pprob_w_comp_hist', 
            #10: '_ftl_w_lim_hist_' + ftl_default, 
            #5: '_pdist_w_comp_hist', 
            #4: '_sleeping_expert',
            #5: '_pert_ftl_w_comp_hist',
            #6: '_exp3_ftl',
            #7: '_exp4wcontext',
            #8: '_exp3wcontext',
            #9: '_exp3_w_contxt_ftl',
            #10: '_exp3',
            #12: '_reg_' + reg_default,
            4: '_ppn_' + pp_default,
            13: '_rndm',
            }
    labels = {
            #0: r'OPT',
            #0: '10', 
            #1: '12', 
            #2: '14', 
            #3: '16', 
            #4: '18', 
            #5: '20', 
            #6: '30', 
            #1: '_ftl_best_in_hindsight', 
            2: r'FTL-CH', 
            #9: r'FTL-PH', 
            10: 'MRU-LM-10', 
            5: 'MRU-LM-5', 
            10: r'FTL-LH', 
            #3: '_pprob_w_comp_hist', 
            #10: r'FTL-LM', 
            #5: '_pdist_w_comp_hist', 
            #4: r'SE-LM',
            #5: '_pert_ftl_w_comp_hist',
            #6: '_exp3_ftl',
            #7: '_exp4wcontext',
            #8: '_exp3wcontext',
            #9: '_exp3_w_contxt_ftl',
            #10: '_exp3',
            13: r'URand-NH',
            #12: '_reg_' + reg_default,
            4: r'PP-LH' 
            }
    style = {
            #0: '-',
            #0: '10', 
            #1: '12', 
            #2: '14', 
            #3: '16', 
            #4: '18', 
            #5: '20', 
            #6: '30', 
            #1: '_ftl_best_in_hindsight', 
            2: '-', 
            #9: '-', 
            10: '-', 
            5: '-', 
            #10: '--', 
            #3: '_pprob_w_comp_hist', 
            #10: r'FTL-LM', 
            #5: '_pdist_w_comp_hist', 
            #4: r'SE-LM',
            #5: '_pert_ftl_w_comp_hist',
            #6: '_exp3_ftl',
            #7: '_exp4wcontext',
            #8: '_exp3wcontext',
            #9: '_exp3_w_contxt_ftl',
            #10: '_exp3',
            #12: '_reg_' + reg_default,
            4: '-',
            13: '-'
            }



    n_aggs = 20
    aggregations = np.arange(0, (7*24*60)/3, n_aggs)
    colors = cm.rainbow(np.linspace(0, 1, 14))
    e = []
    s = []
    for i in algs.keys():
        fname = os.path.join(const.results_dir, 
                prefix + algs[i] + "_" + city + ".csv")
        A = pd.read_csv(fname, sep=',').values
        per = []
        x = []
        d = 0
        n = 0
        for a_i in aggregations:
            h = np.where((a_i <= A[:,0]) & (A[:,0] < (a_i + n_aggs)))[0]
            per.append(np.sum(A[h,1]) / np.sum(A[h,2]))
            d +=  np.sum(A[h,2])
            n +=  np.sum(A[h,1])
            x.append(a_i)
        if 'hindsight' in algs[i]:
            plt.plot(x, per, label=labels[i], linewidth=1.5, c='k')
        else:
            plt.plot(x, per, style[i], label=labels[i], linewidth=1.5, c=colors[i])
        print(city, labels[i], n/d)
        #e.append(int(labels[i])*50)
        #s.append(n/d)
    plt.xlabel("time snapshot")
    plt.ylim([0, 0.25])
    #plt.ylim([0, 0.7])
    plt.ylabel("reward percentage")
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(const.figures_dir, "alg_comparison_{0}.png".format(
        city, len(algs))), format='png', dpi=500, bbox_tight='tight')
    plt.clf()
    """
    e = np.array(e)
    s = np.array(s)
    params, res = helpers.compute_least_sq(e, s)
    print params
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([10**2.5, 10**3.1])
    plt.ylim([10**-.5, 10**0])
    plt.xlabel(r'$\log \epsilon^\prime$', fontsize=35)
    plt.ylabel(r'$\log \mathbb{E}[R_t]$', fontsize=35)
    plt.tick_params(axis='both', labelsize=15)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.plot(e, s, 'k.', markersize=6)
    plt.plot(e, helpers.fit_func(e, params), color='r', linewidth=1)
    plt.savefig(os.path.join(const.figures_dir, "regret_average_opt.png"),
        format='png', dpi=500, bbox_tight='tight')
    """
def fd_stats(city):
    fd_fname = os.path.join(const.results_dir,
            "fd_" + city + ".csv")
    T = pd.read_csv(fd_fname, sep=',').values
    #0:t 1:square_size 2:d0_c 3:d0_m 4:d2_c 5:d2_m 6:dropoffs 7:pickups at t+1
    d0 = T[:, 3]
    d2 = T[:, 5]
    d2 = d2[d2 > np.percentile(d2, 30)]
    s = """city {0}, D2 min {1}, max{2}, mean {3}""".format(city, 
            np.min(d2), 
            np.max(d2),
            np.mean(d2))
    print(s)

    """
    e = 100
    e_b = 1000
    rew_mu = 0
    tot_pts = 0
    for a_i in aggregations:
        h = np.where((a_i <= T[:,0]) & (T[:,0] < (a_i + n_aggs)))[0]
        e_norm = e_b/(T[h,1]*1000)
        rew_mu += np.sum((T[h, 6] - 1) * e_norm**T[h,5]*e_norm**-T[h,3]/T[h,2])
        #rew_mu += np.sum(e_norm**-T[h,3])
        #print e_norm**T[h,3]/T[h,2] * (T[h, 6] - 1) * e_norm**T[h,5]
        tot_pts += np.sum(T[h,6])
    print 'theor', rew_mu
    """
