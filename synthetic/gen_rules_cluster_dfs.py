from __future__ import print_function, division
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import random
import sys
from sklearn.utils import shuffle
import utils
from scipy.spatial import distance
# from automata import Automata
from scipy.special import softmax
import pandas as pd
import pickle
from config import Config
from sklearn.cluster import KMeans

hidden_size = Config.stateSize


def my_add_dict(d, k, v):
    if k not in d:
        d[k] = [v]
    else:
        d[k].append(v)


def my_add_count(d, k, v=1):
    if k not in d:
        d[k] = v
    else:
        d[k] += v


def count_support(dup_count_point, points):
    return sum([dup_count_point[pid] for pid in points])


class Point:
    def __init__(self, id, prev_id, prefix, trans_sym, data, lg_sm):
        self.id = id
        self.prev_id = prev_id
        self.prefix = prefix
        self.trans_sym = trans_sym
        self.data = data
        self.lg_sm = lg_sm
        self.cluster = -1


df = pd.read_csv('long_states_df.csv', nrows=100000)
pred_th = 0.95

states_org = utils.process_df1(df['states'])
states_lg = utils.process_df1(df['logit_seq'])
state_sm = softmax(utils.process_df1(df['logit_seq']), axis=1)
digit_seq = df['words'].values
ind_seq = df['ind_seq'].values
reset_seq = df['reset_bool'].values

point0 = Point(0, -1, "", None, np.zeros(hidden_size), np.ones(hidden_size) / hidden_size)
# point_list = []

point_dict = {}
id_to_points = {}
dup_count_seq, dup_count_point = {}, {}
count = 1
prune_th = 0
curr_seq = ""
curr_point = point0
point_dict[curr_seq] = 0
id_to_points[0] = point0
# curr_seq = str(digit_seq[0])
for i in range(len(states_org)):
    curr_seq += str(digit_seq[i])
    if curr_seq not in point_dict:
        p = Point(len(point_dict), curr_point.id, curr_seq, digit_seq[i], states_org[i], state_sm[i])
        id_to_points[len(point_dict)] = p
        point_dict[curr_seq] = len(point_dict)
    # my_add_count(dup_count_point, p.id)
    else:
        p = id_to_points[point_dict[curr_seq]]
    my_add_count(dup_count_seq, curr_seq)
    my_add_count(dup_count_point, p.id)

    curr_point = p
    if ind_seq[i] == 1:
        curr_seq = ""
        curr_point = point0
        my_add_count(dup_count_point, 0)
no_of_clusters = 10
kmeans = KMeans(n_clusters=no_of_clusters)

pos_points = [pid for pid in id_to_points if id_to_points[pid].lg_sm[0] >= pred_th]
other_points = [pid for pid in id_to_points if id_to_points[pid].lg_sm[0] < pred_th]
point_data = [id_to_points[pid].data for pid in other_points]
kmeans.fit(point_data)
y_kmeans = kmeans.predict(point_data)
for i in range(len(point_data)):
    id_to_points[other_points[i]].cluster = y_kmeans[i]

result_paths = []
def dfs(focal_points, path):
    prev_point_dict, prev_point_count, focal_point_dict, prev_point_cluster = {}, {}, {}, {}
    prev_ids, current_ids = [],[]
    for pid in focal_points:
        if id_to_points[pid].prev_id not in focal_points and id_to_points[pid].prev_id != -1:
            prev_ids.append(id_to_points[pid].prev_id)
            current_ids.append(pid)

    for i, prev_id in enumerate(prev_ids):
        my_add_dict(prev_point_cluster, id_to_points[prev_id].cluster, prev_id)

    # prune_clusters
    keep_clusters = set([k for k in prev_point_cluster if len(prev_point_cluster[k]) > prune_th])

    for i, prev_id in enumerate(prev_ids):
        if id_to_points[prev_id].cluster not in keep_clusters:
            continue
        my_add_dict(prev_point_dict, id_to_points[prev_id].trans_sym, prev_id)
        my_add_dict(focal_point_dict, id_to_points[prev_id].trans_sym, current_ids[i])
        my_add_count(prev_point_count, id_to_points[prev_id].trans_sym, dup_count_point[prev_id])



    sort_prevs = sorted(prev_point_count, key=lambda k: focal_point_dict[k])

    for k in sort_prevs:
        if prev_point_dict[k] == [0]:
            # print("path = ", path[::-1], count_support(dup_count_point,focal_point_dict[k]))
            print("path = ", path[::-1])
            result_paths.append(path[::-1])
        # stack.append((prev_point_dict[k], path + [k]))
        dfs(prev_point_dict[k], path + [k])
# print("end while")
dfs(pos_points, [])
#optional - sort the patterns from short to long so the merging is easier
result_paths = sorted(result_paths,key=len)
print("result paths = ",result_paths)

#write the patterns
path_file = open("inputs.txt","w")
for p in result_paths:
    writepath = ",".join([str(e) for e in p])
    print("writepath = ",writepath)
    path_file.write("O," + writepath + ",P\n")
print('Done')
