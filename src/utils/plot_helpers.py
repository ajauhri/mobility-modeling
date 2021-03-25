import matplotlib
matplotlib.use('pgf')

import matplotlib.pylab as plt
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [],                    # use latex default serif font
    "font.sans-serif": ["DejaVu Sans"],  # use a specific sans-serif font
    "pgf.rcfonts": False,
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
})

import numpy as np
from pathlib import Path
import os

import utils.const as const

const.plot_dir = './plots'


def enable_log_scale():
    plt.xscale('log')
    plt.yscale('log')


def _save_fig(dirname, fname):
    Path(dirname).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(dirname, "{}.png".format(fname)))


def dpl_plot(city, fname, n_nodes, n_edges, fit_func, p):
    enable_log_scale()
    plt.xlabel('Number of nodes', fontsize=25)
    plt.ylabel('Number of edges', fontsize=25)
    plt.tick_params(axis='both', labelsize=15)
    #plt.ylim([1, 10**4])
    #plt.xlim([1, 10**4])
    plt.subplots_adjust(top=0.88)
    plt.plot(n_nodes, n_edges, 'kx', markersize=3)
    plt.plot(n_nodes, fit_func(n_nodes, p), color='r')
    plt.title("C=%.3f, alpha=%.3f" % (p[0], p[1]), fontsize=25)
    dirname = os.path.join(const.plot_dir, city, "dpl")
    _save_fig(dirname, fname)
    plt.clf()


def node_degree_exp_plot(city, fname, n_nodes, exp, theor_exp):
    plt.xlabel('Number of nodes', fontsize=25)
    plt.ylabel('Degree exponent', fontsize=25)
    plt.tick_params(axis='both', labelsize=15)
    plt.subplots_adjust(top=0.88)
    plt.plot(n_nodes, exp, 'kx', markersize=3)
    plt.plot([np.min(n_nodes), np.max(n_nodes)], [theor_exp, theor_exp], 'r--')
    dirname = os.path.join(const.plot_dir, city, "degree_exp")
    _save_fig(dirname, fname)
    plt.clf()


def node_degree_plot(city, fname, degree):
    vals = [[k, v] for k, v in degree.items()]
    vals = np.array(vals)
    enable_log_scale()
    plt.scatter(vals[:, 0], vals[:, 1], marker='x', s=20, c='k')
    plt.xlabel('Node degree', fontsize=25)
    plt.ylabel('Number of nodes', fontsize=25)
    plt.tick_params(axis='both', labelsize=15)
    dirname = os.path.join(const.plot_dir, city, "degree")
    _save_fig(dirname, fname)
    plt.clf()


def degree_ratio_plot(city, fname, degree_ratio, time_snapshots):
    x = np.array(time_snapshots)
    y = np.array(degree_ratio)
    xcoords = []
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    labels = ["Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
    for i in range(1, len(x)):
    	if x[i] - x[i-1] != 1:
            xcoords.append((x[i]+x[i-1])/2)
    if len(xcoords) > 0:
        for xc, c, l in zip(xcoords, colors, labels):
            plt.axvline(x=xc, label='{}'.format(l), c=c, ls='-.')

    vals = np.column_stack((x, y))
    plt.scatter(vals[:, 0], vals[:, 1], marker='x', s=10, c='k')
    plt.xlabel('Time snapshots over a week', fontsize=25)
    plt.ylabel(f'Out and in degree ratio', fontsize=25)
    plt.tick_params(axis='both', labelsize=15)
    dirname = os.path.join(const.plot_dir, city, f"degree_ratio")
    plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.9))
    _save_fig(dirname, fname)
    plt.clf()


def avg_degree_plot(city, fname, avg_degree, degree_type, time_snapshots):
    x = np.array(time_snapshots)
    y = np.array(avg_degree)
    xcoords = []
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    labels = ["Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
    for i in range(1, len(x)):
    	if x[i] - x[i-1] != 1:
            xcoords.append((x[i]+x[i-1])/2)
    if len(xcoords) > 0:
        for xc, c, l in zip(xcoords, colors, labels):
            plt.axvline(x=xc, label='{}'.format(l), c=c, ls='-.')

    vals = np.column_stack((x, y))
    plt.scatter(vals[:, 0], vals[:, 1], marker='x', s=10, c='k')
    plt.xlabel('Time snapshots over a week', fontsize=25)
    plt.ylabel(f'Average {degree_type} degree', fontsize=25)
    plt.tick_params(axis='both', labelsize=15)
    plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.9))
    dirname = os.path.join(const.plot_dir, city, f"avg_degree_{degree_type}")
    _save_fig(dirname, fname)
    plt.clf()


def fractal_plot(city, fname, epsilon, d, fit_func, params, **args):
    enable_log_scale()
    plt.xlabel(args['xlabel'], fontsize=25)
    plt.ylabel(args['ylabel'], fontsize=25)
    plt.tick_params(axis='both', labelsize=15)
    plt.subplots_adjust(top=0.88)
    #plt.xlim(args['xlim'])
    #plt.ylim(args['ylim'])
    plt.plot(epsilon, d, 'k.', markersize=6)
    plt.plot(epsilon, fit_func(epsilon, params), color='r', linewidth=2)
    plt.title("C=%.3f, alpha=%.3f" % (params[0], params[1]), fontsize=25)
    dirname = os.path.join(const.plot_dir, city, "fd")
    _save_fig(dirname, fname)
    plt.clf()


def effective_diameter(city, fname, n_nodes, diameter):
    plt.xlabel('Node count', fontsize=25)
    plt.ylabel('Effective Diameter', fontsize=25)
    plt.plot(n_nodes, diameter, 'kx', markersize=3)
    dirname = os.path.join(const.plot_dir, city, "ed")
    Path(dirname).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        os.path.join(dirname, "{}.png".format(fname)),
        format='png', dpi=500, bbox_inches='tight')
    plt.clf()
