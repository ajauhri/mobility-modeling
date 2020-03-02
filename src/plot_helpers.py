import matplotlib.pylab as plt
import numpy as np
from pathlib import Path
import os

import const

const.plot_dir = './plots'


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
    plt.plot(n_nodes, n_edges, 'kx', markersize=3)
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


def fractal_plot(city, fname, epsilon, d, fit_func, params, **args):
    enable_log_scale()
    plt.xlabel(args['xlabel'], fontsize=25)
    plt.ylabel(args['ylabel'], fontsize=25)
    plt.tick_params(axis='both', labelsize=15)
    plt.subplots_adjust(top=0.88)
    plt.xlim(args['xlim'])
    plt.ylim(args['ylim'])
    plt.plot(epsilon, d, 'k.', markersize=6)
    plt.plot(epsilon, fit_func(epsilon, params), color='r', linewidth=2)
    plt.title("C=%.3f, alpha=%.3f" % (params[0], params[1]), fontsize=25)
    dirname = os.path.join(const.plot_dir, city, "fd")
    Path(dirname).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        os.path.join(dirname, "{}.png".format(fname)),
        format='png', dpi=500, bbox_tight='tight')
    plt.clf()


def effective_diameter(city, fname, n_nodes, diameter):
    plt.xlabel('Node count', fontsize=25)
    plt.ylabel('Effective Diamter', fontsize=25)
    plt.plot(n_nodes, diameter, 'kx', markersize=3)
    dirname = os.path.join(const.plot_dir, city, "ed")
    Path(dirname).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        os.path.join(dirname, "{}.png".format(fname)),
        format='png', dpi=500, bbox_inches='tight')
    plt.clf()
