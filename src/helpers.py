from __future__ import division

import numpy as np
import logging
import scipy.optimize as scimin
import matplotlib
matplotlib.use('agg', warn=False, force=True)
import matplotlib.pylab as plt
import matplotlib.cm as cm
import os
import sys

latitude_meters = 111.2 * 1000
longitude_meters = 111 * 1000
earth_radius_km = 6371.009 

def gte_lb(x, lb):
    return x >= lb
    
def lte_ub(x, ub):
    return x <= ub

def within_boundary(x, lb, ub):
    return gte_lb(x, lb) and lte_ub(x, ub)

def get_time_bin_bounds(vec, time_bin_width_secs):
    """
    Compute boundaries of time bins based on time ranges in the data file
    """
    # to ensure the last upper bound is greater than the actual max. time
    bias = time_bin_width_secs
    return np.array(range(int(np.min(vec)), int(np.max(vec)) + bias, 
        time_bin_width_secs))

def bucket_by_time(time_bin_bounds, vec):
    time_bins = {}
    for i in range(len(time_bin_bounds) - 1):
        time_bins[i] = np.where(np.logical_and(
            vec >= time_bin_bounds[i],
            vec < time_bin_bounds[i+1]))[0]
    logging.debug('Loaded {0} time buckets'.format(
        len(time_bins)))
    return time_bins

def grid_area(lat_min, lat_max, lng_min, lng_max, 
        length_meters):
    """
    Divides a spatial area into equally sized cells.
    """
    
    lat_steps = int(abs(lat_max - lat_min) 
        * latitude_meters / length_meters)
    lat_grids = np.linspace(lat_min, lat_max, lat_steps)
    
    lng_steps = int(abs(lng_max - lng_min) 
        * longitude_meters / length_meters)
    lng_grids = np.linspace(lng_min, lng_max, lng_steps)

    return lat_grids, lng_grids


def get_node(lat_grids, lng_grids, p):
    lat_cell = 0
    lng_cell = 0
    
    if within_boundary(p[0], lat_grids[0], lat_grids[-1]):
        if within_boundary(p[1], lng_grids[0], lng_grids[-1]):
            lat_cell = np.argmax(p[0] < lat_grids)
            lng_cell = np.argmax(p[1] < lng_grids)
    
    node = (lat_cell - 1) * len(lng_grids) + (lng_cell - 1)
    return node, lat_cell - 1, lng_cell - 1

def compute_least_sq(x, y):
    guess = [1,1]
    params, cov, infodict, mesg, iter = scimin.leastsq(
        resi, guess, args=(x, y),
        full_output=True)
    return params, infodict

def fit_func(x, p):
    """
    Linear fit function which empirically which relates number of edges 
    to number of nodes

    :param x: domain of the function
    :param p: parameters of the function
    :return: range of the function
    """
    c, l = p
    return c*x**l

def resi(p, x, y):
    """
    Finds the residual between the fitted function and the ground truth

    :param p: parameters for the fit function
    :param x
    :param y 
    :return: residual vector of same dimenstion as n_nodes
    """
    return y - fit_func(x, p)

class Params(object):
    def __init__(self, r):
        self.prefix = r['prefix']
        self.start_lat = r['start_lat']
        self.end_lat = r['end_lat']
        self.start_lng = r['start_lng']
        self.end_lng = r['end_lng']
        self.fname = r['file_name']


def compute_r2(n_edges, infodict):
    ss_err = (infodict['fvec']**2).sum()
    ss_tot = np.sum(n_edges - np.mean(n_edges)**2)
    rsquared = 1 - (ss_err/ss_tot)
    return rsquared


def compute_diameter(graph_weights):
    """
    Find the effective diamater of the graph defines as minimum moves needed
    to move between any two connected edges

    :param graph_weights: graph

    """
    least_max_diameter = 0
    for node_id in graph_weights.keys():
        visited_set = {}
        distance = {}
        for src in graph_weights.keys():
            distance.update({k:sys.maxsize for k in graph_weights[src].keys()})
            distance[src] = sys.maxsize

        num_of_nodes = len(distance)
        distance[node_id] = 0

        while len(visited_set) != num_of_nodes:
            vertex_node = -1
            min_val = sys.maxsize
            for k, v in distance.items():
                if v < min_val and k not in visited_set:
                    min_val = v
                    vertex_node = k

            if vertex_node == -1:
                break

            visited_set[vertex_node] = '' 
            if vertex_node in graph_weights:
                for k, v in graph_weights[vertex_node].items():
                    if distance[vertex_node] + 1 < distance[k]:
                        distance[k] = distance[vertex_node] + 1

        for k,v in distance.items():
            if v < sys.maxsize and least_max_diameter < v:
                least_max_diameter = v

    return least_max_diameter
