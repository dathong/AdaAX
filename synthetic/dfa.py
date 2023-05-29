from pythomata import SimpleDFA
import re
import copy
import tensorflow as tf
import numpy as np
from config import Config
from scipy.special import softmax
import pickle

state_size = Config.stateSize
batch_size = 1
num_layers = 1

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
alphabets = pickle.load(open("alphabet","rb"))
print("len alphabet: ",len(alphabets))
y_lbl = ['1','0']
num_classes = len(y_lbl)

batchX_placeholder = tf.compat.v1.placeholder(tf.float32, [batch_size,None,len(alphabets)])
# batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
y_lbl_placeholder = tf.compat.v1.placeholder(tf.int64, [None, num_classes])

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# Unpack columns
# inputs_series = tf.split(batchX_placeholder, tf.shape(batchX_placeholder)[1], axis=1)
# labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward passes
lstm = tf.contrib.rnn.BasicLSTMCell(state_size)
cell = tf.contrib.rnn.MultiRNNCell([lstm for _ in range(num_layers)])
init_state = cell.zero_state(batch_size, tf.float32)

states_series, current_state = tf.nn.dynamic_rnn(cell, batchX_placeholder, initial_state=init_state)


logits_series = tf.matmul(tf.squeeze(states_series,axis=0),W2)
prediction_series = tf.nn.softmax(logits_series,axis=1)

# logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
# predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

output_logits = logits_series[-1]
y_pred = prediction_series[-1]

# losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits_series,labels_series)]
# losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
#             for logits, labels in zip(logits_series,labels_series)]

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_lbl_placeholder, logits=output_logits), name='loss')

# total_loss = tf.reduce_mean(losses)

train_step = tf.compat.v1.train.AdagradOptimizer(0.1).minimize(loss)

def my_add_dict(d, k, v):
    # print("d,k,v: ",d,k,v)
    if k in d:
        d[k].append(v)
    else:
        d[k] = [v]


def convert_data_x(x, alphabets):
    d = {a: i for i, a in enumerate(alphabets)}
    res = []

    res1 = []
    for e in x:
        if e == "\n":
            continue
        v = [0] * len(d)
            # print("e = ",e)
            # print("d = ",d)
        v[d[e]] = 1
        res1.append(v)

    return res1


def convert_data_y(y, alphabets):
    d = {a: i for i, a in enumerate(alphabets)}
    res = []
    for r in y:
        for e in r:
            if e == "\n":
                continue
            v = [0] * len(d)
            v[d[e]] = 1
            res.append(v)
            break
    return res

def merge_one(dfa, prefixes, suffixes, s1, s2, visitedStates):
    dfa2 = copy.deepcopy(dfa)
    prefixes1, suffixes1 = copy.deepcopy(prefixes), copy.deepcopy(suffixes)
    # print("[db] s2, prefixes = ", s2, prefixes)
    p2 = prefixes[s2][0]
    pState2, pSym2 = p2[0], p2[1]
    dfa2[pState2][pSym2] = s1
    for sym in dfa2[s1]:
        sSym1, sState1 = (sym, dfa2[s1][sym])
        break
    for sym in dfa2[s2]:
        sSym2, sState2 = (sym, dfa2[s2][sym])
        break
    del prefixes1[s2]
    visitedStates.add(s2)
    for p in prefixes1:
        if p == 'S':
            continue
        if prefixes1[p][0][0] == s2:
            prefixes1[p][0] = (s1, prefixes1[p][0][1])
    if sSym1 != sSym2:
        dfa2[s1][sSym2] = sState2
        dfa3 = dfa2
    else:
        if dfa2[s2][sSym2] not in visitedStates:
            dfa3 = merge_one_pre(dfa2, s1, sSym2, dfa2[s1][sSym1], dfa2[s2][sSym2], prefixes1, visitedStates)
        else:
            dfa3 = dfa2
    dfa4 = reset_dfa(dfa3)
    return dfa4, prefixes1


def merge_one_pre(dfa, pState, pSym, s1, s2, prefixes1, visitedStates):
    print("merge_one_pre")
    dfa[pState][pSym] = s1
    if s1 == 'E':
        return change_state_dest(dfa, s2)
    if s2 == 'E':
        return change_state_dest(dfa, s1)
    for sym in dfa[s1]:
        sSym1, sState1 = (sym, dfa[s1][sym])
        break
    for sym in dfa[s2]:
        sSym2, sState2 = (sym, dfa[s2][sym])
        break
    del prefixes1[s2]
    visitedStates.add(s2)
    for p in prefixes1:
        if p == 'S':
            continue
        if prefixes1[p][0][0] == s2:
            prefixes1[p][0] = (s1, prefixes1[p][0][1])
    if sSym1 != sSym2:
        dfa[s1][sSym2] = sState2
        return dfa
    else:
        if dfa[s2][sSym2] not in visitedStates:
            return merge_one_pre(dfa, sState1, sSym2, dfa[s1][sSym1], dfa[s2][sSym2], prefixes1, visitedStates)
        else:
            return dfa


def merge_gen_pre(dfa, s1, s2, visitedStates):
    print("merge_gen_pre")
    if s1 == 'E':
        return change_state_dest(dfa, s2)
    if s2 == 'E':
        return change_state_dest(dfa, s1)
    visitedStates.add(s2)
    for sym in dfa[s2]:
        sSym2, sState2 = (sym, dfa[s2][sym])
        break
    s1syms = {}
    for sym in dfa[s1]:
        sSym1, sState1 = (sym, dfa[s1][sym])
        s1syms[sSym1] = sState1
    if sSym2 not in s1syms:
        dfa[s1][sSym2] = sState2
        return dfa
    else:
        if dfa[s2][sSym2] not in visitedStates:
            return merge_gen_pre(dfa, s1syms[sSym2], dfa[s2][sSym2], visitedStates)
        else:
            return dfa


def merge_gen(dfa, prefixes, suffixes, s1, s2, visistedStates):
    dfa2 = copy.deepcopy(dfa)
    prefixes1, suffixes1 = copy.deepcopy(prefixes), copy.deepcopy(suffixes)
    p2 = prefixes[s2][0]
    pState2, pSym2 = p2[0], p2[1]
    # print("----merge_gen---")
    # print("[db] dfa: ",dfa)
    # print("[db] pState2: ", pState2)
    # print("[db] pSym2: ", pSym2)
    # print("[db] s1: ", s1)
    # print("[db] s2: ", s2)
    dfa2[pState2][pSym2] = s1
    for sym in dfa2[s2]:
        if dfa2[s2][sym] == s2:
            continue
        sSym2, sState2 = (sym, dfa2[s2][sym])
        break
    s1syms = {}
    for sym in dfa2[s1]:
        sSym1, sState1 = (sym, dfa2[s1][sym])
        s1syms[sSym1] = sState1
    if sSym2 not in s1syms:
        dfa2[s1][sSym2] = sState2
        dfa3 = dfa2
    else:
        dfa3 = merge_gen_pre(dfa2, s1syms[sSym2], dfa2[s2][sSym2], visistedStates)
    dfa4 = reset_dfa(dfa3)
    return dfa4


def change_state_dest(dfa, s1):
    print("change state dest")
    resDfa = {'S': {}}
    stateQueue = ['S']
    visited = set()
    while len(stateQueue) > 0:
        # print("in while loop")
        # print("dfa = ", dfa)
        state1 = stateQueue.pop(0)
        visited.add(state1)
        resDfa[state1] = {}
        for sym in dfa[state1]:
            resDfa[state1][sym] = dfa[state1][sym]
            if dfa[state1][sym] == s1:
                if s1 != 'S' and s1 in dfa:
                    dfa['E'] = dfa.pop(s1)
                    dfa['E'] = {}
                    dfa[state1][sym] = 'E'
                    resDfa[state1][sym] = 'E'
                else:
                    dfa['E'] = {}
                    dfa[state1][sym] = 'E'
                    resDfa[state1][sym] = 'E'
            if dfa[state1][sym] not in visited:
                stateQueue.append(dfa[state1][sym])
    return resDfa


def reset_dfa(dfa):
    resDfa = {'S': {}}
    stateQueue = ['S']
    visited = set()
    while len(stateQueue) > 0:
        state1 = stateQueue.pop(0)
        if state1 not in resDfa:
            resDfa[state1] = {}
        for sym in dfa[state1]:
            if tuple([state1, sym, dfa[state1][sym]]) in visited:
                continue
            resDfa[state1][sym] = dfa[state1][sym]
            stateQueue.append(dfa[state1][sym])
            visited.add(tuple([state1, sym, dfa[state1][sym]]))
    return resDfa


def updatePS(tFunc):
    prefixes, suffixes = {}, {}

    def dfa(v, visited):
        if v == 'E':
            return
        if v in visited:
            return
        visited.add(v)
        for sym in tFunc[v]:
            my_add_dict(prefixes, tFunc[v][sym], (v, sym))
            my_add_dict(suffixes, v, (sym, tFunc[v][sym]))
            dfa(tFunc[v][sym], visited)
        visited.remove(v)

    visited = set()
    dfa('S', visited)
    return prefixes, suffixes


def verifyDfa(tFunc):
    # print("[db1] tFunc = ",tFunc)
    paths = genAllSyms(tFunc)
    # print("[db1] paths = ",paths)
    logit_seqs,pred_seqs = [],[]
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.initialize_all_variables())
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, './my_test_model')
    for i in range(len(paths)):

        x = paths[i]
        # print("[db1] x = ",x)
        x0 = convert_data_x(x, alphabets)
        y0 = convert_data_y([y_lbl[0]], y_lbl)
        # print("[db1] i = ", i,x0,y0)
        # print("x = ",x)

        _states_series, _current_state, _y_pred, _logits_series, _prediction_series = sess.run(
                [states_series, current_state, y_pred, logits_series, prediction_series],
                feed_dict={
                    batchX_placeholder: [x0],
                    y_lbl_placeholder: y0,
                })



        logit_seqs.append(_logits_series)
        pred_seqs.append(_prediction_series)


        # print("[db] _y_pred,y = ", _y_pred, y)
    # print("[db1] logit_seq = ",logit_seqs)
    # print("[db1] pred_seq = ",pred_seqs)
    th = 0.6
    for pred_seq in pred_seqs:
        for p in pred_seq[:-1]:
            if p[0] >= th:
                # print("[db1] return False")
                return False
        if pred_seq[-1][0]< th:
            # print("[db1] return False")
            return False
    # print("[db1] return True")
    return True

def build_dfa(ips):
    alphabet, states = set(), set(["S", "E"])
    tFunc = {"S": {}, "E": {}}
    initialState = "S"
    acceptingStates = {"E"}
    stateCount = 1
    ipf = open(ips, "r")
    for lCount, line in enumerate(ipf):
        print("lCount = ",lCount)
        if lCount > 30:
            break
        l = line.strip("\n")
        syms = l.split(",")
        state1, state2 = "S", "S"
        print("syms = ", syms)
        prefixes, suffixes = {'S': []}, {}
        stateList = list(tFunc.keys())
        newStates = ['S'] if lCount == 0 else []
        # print("syms = ",syms)
        # print("syms[1:-1] = ",syms[1:-1])
        alphabet = alphabet.union(set([s for s in syms[1:-1]]))
        # print("[db] alphabet 1 : ", alphabet)
        for s in syms[1:-2]:
            # print("s = ",s)
            if s in tFunc[state1]:
                my_add_dict(prefixes, tFunc[state1][s], (state1, s))
                my_add_dict(suffixes, state1, (tFunc[state1], s))
                state1 = tFunc[state1][s]
                states.add(state1)
            else:
                my_add_dict(prefixes, str(stateCount), (state1, s))
                my_add_dict(suffixes, state1, (str(stateCount), s))
                tFunc[state1][s] = str(stateCount)
                state1 = str(stateCount)
                if state1 not in tFunc:
                    newStates.append(state1)
                states.add(state1)
                tFunc[state1] = {}
                stateCount += 1

        # print("alphabet = ", alphabet)
        s = syms[-2]
        # print("tF = ", state1, s)
        # print("preFixes = ", prefixes)
        # print("newStates = ", newStates)
        # print("[db] tFunc = ", tFunc)
        # print("[db] state1 = ", state1)
        # print("[db] syms = ", syms)
        tFunc[state1][syms[-2]] = "E"
        my_add_dict(suffixes, state1, ('E', s))

        for i in range(0, len(newStates)):
            for j in range(i, len(newStates)):
                if i != j and newStates[i] in tFunc and newStates[j] in tFunc:
                    visitedStates = set()
                    tFunc1, prefixes1 = merge_one(tFunc, prefixes, suffixes, newStates[i], newStates[j], visitedStates)

                    if verifyDfa(tFunc1):
                        prefixes = prefixes1
                        tFunc = tFunc1
                        del suffixes[newStates[j]]

        if lCount == 0:
            continue

        for s2 in newStates:
            for s1 in stateList:
                if s1 == 'E':
                    continue
                if not (s1 in tFunc and s2 in tFunc and s2 in prefixes):
                    continue
                dfa = SimpleDFA(states, alphabet, initialState, acceptingStates, tFunc)
                graph = dfa.to_graphviz()
                graph.render("vis1")
                visitedStates = set()
                # print("----next merge_gen---")
                # print("[db] tFunc: ", tFunc)
                # print("[db] prefixes: ", prefixes)
                # print("[db] suffixes: ", suffixes)

                tFunc1 = merge_gen(tFunc, prefixes, suffixes, s1, s2, visitedStates)

                if verifyDfa(tFunc1):
                    tFunc = tFunc1
                    prefixes, suffixes = updatePS(tFunc)
                    dfa = SimpleDFA(states, alphabet, initialState, acceptingStates, tFunc)
                    graph = dfa.to_graphviz()
                    graph.render("vis2")
                    break

        count,count0 = 0,0
        print("tFunc = ",tFunc)
        for k in tFunc:
            count+=1
            count0+=1
            for d in tFunc[k]:
                count += 1
        print("DFA size: ", count0,count)

    tFunc = reset_dfa(tFunc)
    states = set(tFunc.keys())
    states.add('E')
    dfa = SimpleDFA(states, alphabet, initialState, acceptingStates, tFunc)
    # print("tFunc = ", tFunc)
    #---acc---
    # compute accuracy---

    y_train = [int(y.strip("\n").split(",").index("1")) for y in open("y.csv", "r")]
    x_train = [x.strip("\n") for x in open("x.csv", "r")]
    yPred = []
    y_train = [int(y) for y in y_train]
    for x in x_train:
        # print("x = ",x)
        currState = "S"
        trueFlag, done = True, False
        for s in x.split(","):
            if s not in tFunc[currState]:
                trueFlag = False
                break
            nextState = tFunc[currState][s]
            currState = nextState
        # print("currState = ",currState)
            if currState in acceptingStates:
                yPred.append(1)
                done = True
                break
        if done:
            continue
        if trueFlag and currState in acceptingStates:
            yPred.append(1)
            break
        else:
            yPred.append(0)
    # print("x_t")
    print("xTrain = ", x_train)
    print("yTrain = ", y_train)
    print("yPred  = ", yPred)
    print("len yPred = ", len(yPred))

    match = [1 if y_train[i] == yPred[i] else 0 for i in range(len(y_train))]
    print("match = ", match)
    print("acc = ", sum(match) / len(match))
    #----
    graph = dfa.to_graphviz()
    graph.render("vis2")
    return dfa


def genCycles(tFunc):
    res1, res2 = [], []

    def cyclicUtil(v, path1, path2, visited, recStack):

        visited.add(v)
        recStack.add(v)
        for sym in tFunc[v]:
            if tFunc[v][sym] not in visited:
                cyclicUtil(tFunc[v][sym], path1 + [str(tFunc[v][sym])], path2 + [str(sym)], visited, recStack)
            elif tFunc[v][sym] in recStack:
                res1.append(path1 + [str(tFunc[v][sym])])
                res2.append(path2 + [str(sym)])
        recStack.remove(v)

    visited = set()
    recStack = set()
    for v in tFunc:
        if v not in visited:
            cyclicUtil(v, [v], ['S'], visited, recStack)
    return res1, res2


def genAllSyms(tFunc):
    vSeq, symSeq = genCycles(tFunc)
    res1, res2 = [], []
    vDict = {}
    for i in range(len(vSeq)):
        lastInd = vSeq[i].index(vSeq[i][-1])
        s1 = vSeq[i][lastInd:]
        s2 = symSeq[i][lastInd + 1:]
        res1.append(s1)
        res2.append(s2)
        my_add_dict(vDict, vSeq[i][-1], s2)

    res = []

    def dfa(v, path, visited):
        if v == 'E':
            res.append(path)
            return
        if v in visited:
            return
        visited.add(v)
        for sym in tFunc[v]:
            if v in vDict and len(vDict[v]) > 0:
                for c in vDict[v]:
                    nextPath = path + c
                    dfa(tFunc[v][sym], path + [str(sym)], visited)
                    dfa(tFunc[v][sym], nextPath + [str(sym)], visited)
            else:
                dfa(tFunc[v][sym], path + [str(sym)], visited)
        visited.remove(v)

    dfa('S', [], set())
    return list(res)


if __name__ == "__main__":
    dfa = build_dfa("inputs.txt")
    graph = dfa.minimize().trim().to_graphviz()
    graph = dfa.to_graphviz()
    graph.render("vis")
    print("Done")
