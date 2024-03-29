from dateutil import tz
import numpy as np
import logging
import scipy.optimize as scimin
import sys
import pytz
import datetime

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
    return np.array(
        range(
            int(np.min(vec)),
            int(np.max(vec)) + bias,
            time_bin_width_secs
        ))


def bucket_by_time(time_bin_bounds, vec):
    time_bins = {}
    for i in range(len(time_bin_bounds) - 1):
        time_bins[i] = np.where(np.logical_and(
            vec >= time_bin_bounds[i],
            vec < time_bin_bounds[i+1]))[0]
    logging.debug('Loaded {0} time buckets'.format(
        len(time_bins)))
    return time_bins


def grid_area(lat_min, lat_max, lng_min, lng_max, length_meters):
    """
    Divides a spatial area into equally sized cells.
    """

    lat_steps = int(
        abs(lat_max - lat_min) * latitude_meters / length_meters)
    lat_grids = np.linspace(lat_min, lat_max, lat_steps)

    lng_steps = int(
        abs(lng_max - lng_min) * longitude_meters / length_meters)
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


def compute_least_sq(x, y, with_const=True):
    guess = [1, 1]
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
        self.cons_ts = r['cons_ts']
        self.fname = r['file_name']
        self.time_zone = r['time_zone']


def compute_r2(n_edges, infodict):
    ss_err = (infodict['fvec']**2).sum()
    ss_tot = np.sum(n_edges - np.mean(n_edges)**2)
    rsquared = 1 - (ss_err/ss_tot)
    return rsquared


def compute_diameter_effective(graph_weights):
    """
    Find the effective diamater of the graph defines as minimum moves needed
    to move between any two connected edges

    :param graph_weights: graph

    """
    dists = []
    max_min_diameter = 0
    for node_id in graph_weights.keys():
        visited_set = {}
        distance = {}
        distance_map = {}

        distance_map[0] = set([node_id])
        distance[node_id] = 0
        while len(distance_map) > 0:

            min_val_node = min(distance_map.keys())
            node_set = distance_map[min_val_node]
            vertex = node_set.pop()

            visited_set[vertex] = ''

            if vertex in graph_weights:
                for k, v in graph_weights[vertex].items():
                    if k not in visited_set:
                        if k in distance:
                            if distance[vertex] + 1 < distance[k]:
                                new_dist = distance[vertex] + 1

                                # Add new distance value in map for k
                                add_distance_map(distance_map, new_dist, k)

                                # Clear old distance value from map for k
                                clean_distance_map(distance_map, distance[k], k)

                                distance[k] = new_dist
                        else:
                            new_dist = distance[vertex] + 1
                            distance[k] = new_dist

                            # Add new distance value in map for k
                            add_distance_map(distance_map, new_dist, k)

                        dists.append(distance[k])
                        if distance[k] > max_min_diameter:
                            max_min_diameter = distance[k]

            # Clear old distance value from map for vertex
            if len(node_set) == 0:
                del distance_map[min_val_node]
            del distance[vertex]


    ed = np.percentile(dists, 98)
    #print(ed, max_min_diameter)
    return ed, max_min_diameter

def add_distance_map(dist_map, new_dist, node):
    if new_dist in dist_map:
        dist_map[new_dist].add(node)
    else:
        dist_map[new_dist] = set([node])


def clean_distance_map(dist_map, old_distance, node):
    dist_old_set = dist_map[old_distance]
    if len(dist_old_set) == 1:
        del dist_map[old_distance]
    else:
        dist_old_set.remove(node)


def is_night_hour(epoch, time_zone):
    dt = datetime.datetime.utcfromtimestamp(epoch)
    dt = pytz.utc.localize(dt)
    to_zone = tz.gettz(time_zone)
    hour = dt.astimezone(to_zone).hour
    if hour >= 0 and hour <= 7:
        return True
    else:
        return False


def theor_degree_exp(alpha):
    return 2/alpha

def time_varying_theor_degree_exp(alpha, nodes):
    nodes = np.array(nodes)
    n = 4*nodes**(alpha - 1) - 1
    d = 2*nodes**(alpha - 1) - 1
    return n/d


def real_degree_exp(rrg_t):
    node_degree = rrg_t.in_degree + rrg_t.out_degree
    node_degree_arr = np.array([[k, v] for k, v in node_degree.items()])
    return compute_mle(node_degree_arr[:, 0])

def compute_mle(x):
    n = len(x)
    x_min = np.min(x)
    return 1 + (n * (1 / (np.sum(np.log(x/x_min)))))
