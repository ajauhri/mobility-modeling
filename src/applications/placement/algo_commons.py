import utils.const as const
import numpy as np
import math
import json
from httplib2 import Http

graph_hopper_url = """http://localhost:8989/route?point={0}%2C{1}&point={2}%2C{3}&vehicle=car"""
const.earth_radius = 6371.009 #in km

def get_boundaries(lat_grids, lng_grids, lat_idx, lng_idx):
    """
    Given a pair of lat. and lng. index, this function returns the boundaries
    of the possible actions which can be taken. The return is of dtype in the
    half-open interval i.e. [start_lat_idx, end_lat_idx) and so fort for
    longitudes.
    """
    if const.static_boundaries:
        start_lat_idx = lat_idx - lat_idx % const.region_width
        start_lng_idx = lng_idx - lng_idx % const.region_width
        end_lat_idx = start_lat_idx + const.region_width
        end_lng_idx = start_lng_idx + const.region_width
    else:
        start_lat_idx = lat_idx - int(const.region_width / 2)
        start_lng_idx = lng_idx - int(const.region_width / 2)
        end_lat_idx = start_lat_idx + const.region_width
        end_lng_idx = start_lng_idx + const.region_width

    if start_lat_idx <= 0:
        start_lat_idx = 0

    if start_lng_idx <= 0:
        start_lng_idx = 0

    if end_lat_idx > len(lat_grids) - 1:
        end_lat_idx = len(lat_grids) - 1

    if end_lng_idx > len(lng_grids) - 1:
        end_lng_idx = len(lng_grids) - 1

    return start_lat_idx, end_lat_idx, start_lng_idx, end_lng_idx


def find_bandit_centers(lat_cell, lng_cell, lat_grids, lng_grids):
    idxs = get_boundaries(lat_grids, lng_grids, lat_cell, lng_cell)
    centers = []
    for i in range(idxs[0], idxs[1]):
        for j in range(idxs[2], idxs[3]):
            centers.append([(lat_grids[i] + lat_grids[i+1]) / 2,
                (lng_grids[j] + lng_grids[j+1]) / 2])
    return centers

def find_mle_node(lat_cell, lng_cell, lat_grids, lng_grids, arrivals):
    idxs = get_boundaries(lat_grids, lng_grids, lat_cell, lng_cell)
    placement_node = -1
    max_p = 0
    for i in range(idxs[0], idxs[1]):
        for j in range(idxs[2], idxs[3]):
            node = i * len(lng_grids) + j
            if node in arrivals and len(arrivals[node]) > const.pp_min_samples:
                inter_arrivals = np.ediff1d(arrivals[node]) / \
                        const.time_window_secs
                lamb = len(inter_arrivals)/np.sum(inter_arrivals)
                p = 1 - math.pow(np.e, -lamb)
                if p >= max_p:
                    placement_node = node
                    max_p = p
    if placement_node == -1:
        lat_cell = np.random.randint(idxs[0], idxs[1])
        lng_cell = np.random.randint(idxs[2], idxs[3])
        placement_node = lat_cell * len(lng_grids) + lng_cell
    else:
        #lamb.pop(placement_node, None)
        arrivals.pop(placement_node, None)
        #arrivals[placement_node] = arrivals[placement_node][1:]
    return placement_node

def orthodromic_dist(a, b):
    lat1, lng1 = np.radians(a[:, 0]), np.radians(a[:, 1])
    lat2, lng2 = np.radians(b[0]), np.radians(b[1])

    sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)
    sin_lat2, cos_lat2 = np.sin(lat2), np.cos(lat2)

    delta_lng = lng2 - lng1
    cos_delta_lng, sin_delta_lng = np.cos(delta_lng), np.sin(delta_lng)

    d = np.arctan2(np.sqrt((cos_lat2 * sin_delta_lng) ** 2 +
                   (cos_lat1 * sin_lat2 -
                    sin_lat1 * cos_lat2 * cos_delta_lng) ** 2),
              sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)
    return const.earth_radius * d * 1000

def calculate_eta(dropoff, pickups, q):
    """
    Order of pickups based on distance
    """
    res = []
    d = orthodromic_dist(
        q.M[np.ix_(pickups, [2, 3])],
        q.M[dropoff, [5, 6]])

    h = Http()
    start_lat, start_lng = q.M[dropoff, 5], q.M[dropoff, 6]
    for i in np.argsort(d):
        new_pickup_url = graph_hopper_url.format(start_lat,
                                                 start_lng,
                                                 q.M[pickups[i], 2],
                                                 q.M[pickups[i], 3])
        resp, content = h.request(new_pickup_url, "GET")
        json_obj = get_json(content)
        if 'paths' in json_obj:
            pickup_distance = json_obj['paths'][0]["distance"]
        else:
            res.append([q.M[pickups[i], 1] - q.M[pickups[i], 0]] * 2)
            start_lat, start_lng = q.M[pickups[i], 2], q.M[pickups[i], 3]
            continue

        """
        Original dropoff used to predict estimated pickup time
        """
        dropoff_url = graph_hopper_url.format(q.M[pickups[i], 2],
                                              q.M[pickups[i], 3],
                                              q.M[pickups[i], 5],
                                              q.M[pickups[i], 6])
        resp, content = h.request(dropoff_url, "GET")
        json_obj = get_json(content)
        if 'paths' in json_obj:
            dropoff_speed = json_obj['paths'][0]["distance"] / \
                            (q.M[pickups[i], 4] - q.M[pickups[i], 1])
        else:
            dropoff_speed = const.default_speed_ms

            # not a valid trip
        if dropoff_speed == 0:
            continue

        eta = pickup_distance / dropoff_speed
        new_arrival_time = q.M[dropoff, 4] + eta
        request_time = q.M[pickups[i], 0]
        if new_arrival_time >= request_time:
            res.append([new_arrival_time - request_time,
                        q.M[pickups[i], 1] - q.M[pickups[i], 0]])
        else:
            res.append([60, q.M[pickups[i], 1] - q.M[pickups[i], 0]])

        """
        For next pickup location, use pickup location of last pickup
        """
        start_lat, start_lng = q.M[pickups[i], 2], q.M[pickups[i], 3]
    return res

def find_rand_leader(lat_cell, lng_cell, lat_grids, lng_grids,
        history, update=True):
    idxs = get_boundaries(lat_grids, lng_grids, lat_cell, lng_cell)
    counts = {}
    for i in range(idxs[0], idxs[1]):
        for j in range(idxs[2], idxs[3]):
            node = i * len(lng_grids) + j
            if node in history:
                if history[node] in counts:
                    counts[history[node]].append(node)
                else:
                    counts[history[node]] = [node]

    if len(counts) == 0:
        leader_node = idxs[0] * len(lng_grids) + idxs[2]
        return leader_node, 0
    else:
        freq = max(counts.keys())
        cands = counts[freq]
        leader_node = np.random.choice(cands)
        if update:
            history[leader_node] -= 1
            #history.pop(leader_node, 0)
        return leader_node, freq

def find_perturbed_leader(lat_cell, lng_cell,
        lat_grids, lng_grids, nodes):
    idxs = get_boundaries(lat_grids, lng_grids, lat_cell, lng_cell)
    max_reward = -1
    leader_node = -1
    for i in range(idxs[0], idxs[1]):
        for j in range(idxs[2], idxs[3]):
            node = i * len(lng_grids) + j
            u = np.random.uniform(0, 1 + 1/const.pftl_epsilon)
            reward = (nodes[node] + u) if node in nodes else u
            if reward > max_reward:
                max_reward = reward
                leader_node = node
    return leader_node

def get_json(s):
    return json.loads(s)