#!/usr/bin/env python3.6
import sys
import numpy as np
from collections import Counter

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

import utils.helpers as helpers


def syn_dpl(edge_p, p, node):
    coords_queue = [x for x in range(node)]
    rich = []
    total_edges = 0
    while len(coords_queue) > 1:
        source = np.random.choice(coords_queue)
        coords_queue.remove(source)
        edges = np.random.geometric(edge_p)
        total_edges += edges
        for i in range(edges):
            if (np.random.rand() <= p or len(rich) == 0) and len(coords_queue) > 0:
                dest = np.random.choice(coords_queue)
                coords_queue.remove(dest)
            else:
                dest = np.random.choice(rich)
        rich.append(source)
    # print(node, total_edges, round(node/total_edges,2))
    return node, total_edges


def main():
    edge_p_arr = [x for x in np.arange(0.5, 1, 0.05)]
    p_arr = [x for x in np.arange(0.9, 1, .01)]
    nodes = [x for x in range(500, 2500, 250)]
    done = False
    for edge_p in edge_p_arr:
        for p in p_arr:
            n_nodes = []
            n_edges = []
            for k in nodes:
                n, e = syn_dpl(edge_p, p, k)
                n_nodes.append(n)
                n_edges.append(e)
            params, info_dict = helpers.compute_least_sq(n_nodes, n_edges)
            print(round(edge_p, 2), round(p, 2), params)
            if params[1] > 1.01:
                done = True
        if done:
            break
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of nodes', fontsize=25)
    plt.ylabel('Number of edges', fontsize=25)
    plt.tick_params(axis='both', labelsize=15)
    plt.plot(n_nodes, n_edges, 'kx', markersize=3)
    plt.plot(n_nodes, helpers.fit_func(n_nodes, params), color='r')
    plt.title("C=%.3f, alpha=%.3f" % (params[0], params[1]), fontsize=25)
    plt.tight_layout()
    plt.savefig("graph_generator_with_random_data.png")

if __name__ == "__main__":
    main()

