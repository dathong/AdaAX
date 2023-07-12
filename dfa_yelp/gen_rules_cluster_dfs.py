from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import sys
from sklearn.utils import shuffle
from scipy.spatial import distance
from scipy.special import softmax
import pandas as pd
import utils
import pickle

def my_add_dict(d,k,v):
    # print("d,k,v: ",d,k,v)
    if k in d:
        d[k].append(v)
    else:
        d[k] = [v]

def my_add_dict1(d,k,v1,v2):
    # print("d,k,v1,v2: ",d,k,v1,v2)
    if k not in d:
        d[k] = {v1:v2}
    else:
        d[k][v1] = v2

tf.compat.v1.disable_eager_execution()

df = pd.read_csv('long_states_df.csv',nrows=50000)

states_df = utils.process_df2(df['states'])
states_lg = utils.process_df1(df['logit_seq'])
digit_seq = df['words'].values
ind_seq = df['ind_seq'].values
reset_seq = df['reset_bool'].values

alphabet = pickle.load(open("alphabet","rb"))
print("states_df = ",states_df.shape)

# np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
state_list_sm = softmax(states_lg, axis=1)
fig, ax = plt.subplots()
ax.set_xlim([min(states_df[:,0])-0.1, max(states_df[:,0])+0.1])
ax.set_ylim([min(states_df[:,1])-0.1, max(states_df[:,1])+0.1])
ax.scatter(states_df[:,0], states_df[:,1], c=state_list_sm[:, 1],s=10 )
# ax.scatter(states_df[:,0], states_df[:,1], c=state_list_sm[:, 1],s=10 )
# plt.show()
# print('df = ',df)
df['state_list_sm'] = list(state_list_sm)
# print('df = ',df)
df.to_csv('long_states_df_sm.csv', index=False, header=True)



def drawArrow(A, B, label, color='red'):
	# print('label = ',label)
	ax.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
			 head_width=0.02, length_includes_head=True, label=label, color=color)
	ax.annotate(label, ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2))

# df = pd.read_csv('short_states.csv')
# print('df = ', df)
# short_states = [[0,0]]
# short_states.extend(utils.process_df(df['state']))
# seq = df['digit'].values
# print('short states = ',short_states)

import pickle
# for i in range(len(short_states) - 1):
# 	drawArrow(short_states[i], short_states[i + 1], label=seq[i],color='blue')
# with open ('./centers_pos', 'rb') as fp:
# 	centers_pos = pickle.load(fp)
# ax.plot(centers_pos[:, 0], centers_pos[:, 1], 'kx', markersize=15, color='blue')
# plt.show()
print('-----generate dictionaries----')
# d1 = {}
# for i,state in enumerate(states_df):
# 	d1[i] = tuple(state)

print("[db] states_df = ",states_df)
state_size = len(states_df[0])
id_to_p, p_to_id, prefixes, suffixes = {0:(tuple([0]*state_size),0,0)},{tuple([0]*state_size):0},{0:{}},{0:{}}
P_states, NP_states = set(),set([0])
for i in range(len(states_df)-1,-1,-1):
    id_to_p[i+1] = (tuple(states_df[i]),state_list_sm[i],reset_seq[i],ind_seq[i])
    p_to_id[tuple(states_df[i])] = i+1
    if state_list_sm[i][0] > 0.9:
        P_states.add(i+1)
    else:
        NP_states.add(i+1)
    if reset_seq[i] == 1:
        my_add_dict1(prefixes,i+1,0, digit_seq[i])
        my_add_dict1(suffixes, 0, i+1,digit_seq[i])
    else:
        my_add_dict1(prefixes, i+1, i, digit_seq[i])
        my_add_dict1(suffixes, i, i+1, digit_seq[i])




print("----clustering----")
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

no_of_clusters = 1
kmeans = KMeans(n_clusters=no_of_clusters)
# y_dict = {i:0 for i in list(NP_states)}

NP_states_list = list(NP_states)
NP_state_vals = [list(id_to_p[id][0]) for id in NP_states_list]
kmeans.fit(NP_state_vals)
y_kmeans = kmeans.predict(NP_state_vals)
print('y_kmeans = ',y_kmeans)
y_dict = {NP_states_list[i]:y_kmeans[i] for i in range(len(NP_states_list))}
y_dict1 = {}
for k in y_dict:
    my_add_dict(y_dict1,y_dict[k],k)

centers_pos = kmeans.cluster_centers_

print('----computing mutual distances----')
import scipy.spatial
dist = scipy.spatial.distance.cdist(centers_pos,centers_pos)
avgDist = np.sum(dist)/(dist.shape[0]*(dist.shape[0] - 1))
with open('avgClusterDist','wb') as fp:
    pickle.dump(avgDist,fp)
with open('centers_pos','wb') as fp:
    pickle.dump(centers_pos,fp)

print('avgDist = ',avgDist)

ax.plot(centers_pos[:, 0], centers_pos[:, 1], 'kx', markersize=15, color='blue')
plt.show()


# drawPoints = np.array([id_to_p[id][0] for id in P_states])
# print("[db] drawPoints = ",drawPoints)
# fig, ax = plt.subplots()
# ax.set_xlim([min(states_df[:, 0]) - 0.1, max(states_df[:, 0]) + 0.1])
# ax.set_ylim([min(states_df[:, 1]) - 0.1, max(states_df[:, 1]) + 0.1])
# ax.scatter(drawPoints[:,0], drawPoints[:,1],s=10 )
# plt.show()
#
# drawPoints = np.array([id_to_p[id][0] for id in NP_states])
# fig, ax = plt.subplots()
# ax.set_xlim([min(states_df[:, 0]) - 0.1, max(states_df[:, 0]) + 0.1])
# ax.set_ylim([min(states_df[:, 1]) - 0.1, max(states_df[:, 1]) + 0.1])
# ax.scatter(drawPoints[:,0], drawPoints[:,1],s=10 )
# plt.show()



import pickle

with open('y_kmeans', 'wb') as fp:
    pickle.dump(y_kmeans, fp)

print("----back forwarding----")

def countFreq(points,y_dict):
    d1,d2 = {},{}
    for p in points:
        if p not in y_dict:
            continue
        # print("d,p,y_dict[p] ",d,p,y_dict[p])
        if y_dict[p] in d1:
            d1[y_dict[p]]+=1
            d2[y_dict[p]].append(p)
        else:
            d1[y_dict[p]] = 1
            d2[y_dict[p]] = [p]
    return d1,d2

def countFreq1(points,suff):
    d1,d2 = {},{}
    for p in points:
        k = list(suff[p].keys())[0]
        v = suff[p][k]
        # [k,v] = list(prefixes[p].keys())
        if v in d1:
            d1[v]+=1
            d2[v].append(p)
        else:
            d1[v] = 1
            d2[v] = [p]
    return d1,d2
res = set()

testPoints = {}
for k in suffixes:
    if k not in NP_states:
        continue
    testPoints[k] = suffixes[k]
print("db1 test points...")
drawPoints = np.array([id_to_p[id][0] for id in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
fig, ax = plt.subplots()
ax.set_xlim([min(states_df[:, 0]) - 0.1, max(states_df[:, 0]) + 0.1])
ax.set_ylim([min(states_df[:, 1]) - 0.1, max(states_df[:, 1]) + 0.1])
ax.scatter(drawPoints[:,0], drawPoints[:,1],s=10 )
for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    plt.annotate(str(i), (drawPoints[i][0], drawPoints[i][1]))
plt.show()
print("end db1...")
def back_forward(points,path):

    # currPoints = [list(prefixes[p].keys())[0] for p in points if list(prefixes[p].keys())[0] in NP_states]
    currPoints = []
    for p in points:
        if len(prefixes[p]) > 0:
            prev = list(prefixes[p].keys())[0]
            if prev == 0:
                newPath = path + "," + str(suffixes[0][p])[::-1]
                res.add(newPath[::-1])
                continue
            if prev in NP_states:
                currPoints.append(prev)
    # currPoints = [p for p in currPoints if p in NP_states]
    print("[db] len currPoints ",len(currPoints))
    print("[db] path = ",path)
    if len(currPoints) <= 0:
        # res.add(path)
        return

    # drawPoints = np.array([id_to_p[id][0] for id in currPoints])
    # fig, ax = plt.subplots()
    # ax.set_xlim([min(states_df[:, 0]) - 0.1, max(states_df[:, 0]) + 0.1])
    # ax.set_ylim([min(states_df[:, 1]) - 0.1, max(states_df[:, 1]) + 0.1])
    # ax.scatter(drawPoints[:,0], drawPoints[:,1],s=10 )
    # plt.show()

    d, dd = countFreq(currPoints,y_dict)
    d_sorted = sorted(d.items(), key=lambda item: item[1],reverse=True)
    for t in d_sorted:
        newP = [pid for pid in currPoints if pid in y_dict1[t[0]]]
        d1,d2 = countFreq1(newP,suffixes)
        print("db...")
        d1_sorted = sorted(d1.items(), key=lambda item: item[1], reverse=True)
        for t1 in d1_sorted:
            s = t1[0]
            # if s != 1:
            #     continue
            newP1 = []
            for pid in newP:
                if len(suffixes[pid]) > 0 and list(suffixes[pid].items())[0][1] == s:
                    newP1.append(pid)

            # newP1 = [pid for pid in newP if list(prefixes[pid].items())[0][1] == s]
            back_forward(newP1,path + "," + str(s[::-1]))

# P_prevPoints = [list(prefixes[p].keys())[0] for p in P_states]
# currPoints = [list(prefixes[p].keys())[0] for p in P_states]
back_forward(P_states,"")
print("res = ",res)
res_list = list(res)
new_list = sorted(res_list, key=len)
opf = open("inputs_sorted.txt","w")

def readData():
    x = open("x.csv","r")
    y = open("y.csv","r")
    x_res,y_res = [],[]
    for r in x:
        r_elems = r.strip().split(",")
        x_res.append([str(e) for e in r_elems])
    for r in y:
        y_res.append(r.strip())
    x.close()
    y.close()
    return x_res,y_res

x,y = readData()

x_str = [",".join(e) + "," for e in x]
yFile = open("y.csv","r")
yLbl = [int(y) for y in yFile]
rule_freq = {r:0 for r in new_list}
rule_freq1 = {r:0 for r in new_list}
for r in new_list:
    rule = "," + r.strip()
    for i,e in enumerate(x_str):

        elem = "," + e.strip(",") + ","
        # if 'crowded' in elem:

        if elem.startswith(rule) and yLbl[i] == 1:
            # if 'crowded' in elem:
            #     print("[db] crowded elem: ", elem)
            #     print("[db] crowded r: ", r)
            rule_freq[r] += 1
        if elem.startswith(rule) and yLbl[i] == 0:
            rule_freq1[r] += 1
    opf.write("O," + r + "P \n")

print("rule_freq = ",rule_freq)
d_sort = sorted(rule_freq.items(), key=lambda item: item[1],reverse=True)
opf.close()

opf = open("inputs_details.txt","w")
opf1 = open("inputs.txt","w")
for e in d_sort:
    if e[1] == 0:
        continue
    opf.write(str(e[0]) + "," + str(e[1]) + "," + str(rule_freq1[e[0]]) + "\n")
    opf1.write("O," + str(e[0]) + "P \n")
opf.close()
print('Done')
