import re
import numpy as np
import random

def process_df(field):
    res = []
    for s in field:
        s = s.replace('[','').replace(']','').strip()
        # print('s = ', s)
        first = float(s.split()[0])
        second = float(s.split()[1])
        res.append([first,second])
    return np.array(res)

def process_df1(field):
    res = []
    for s in field:
        s = s.replace('[','').replace(']','').replace(',','').strip()
        # print('s = ', s)
        first = float(s.split()[0])
        second = float(s.split()[1])
        res.append([first,second])
    return np.array(res)

def process_df2(field):
    res = []
    for s in field:
        s = s.replace('[','').replace(']','').replace(',','').strip()
        # print('s = ', s)
        s_elems = s.split()
        res1 = []
        for e in s_elems:
            val = float(e)
            res1.append(val)
        res.append(res1)
    return np.array(res)

def index_of(sub_arr,arr):
    arr = arr.tolist()
    jump = len(sub_arr)
    # print('sub_arr = ',sub_arr)
    # print('arr = ',arr)
    for i in range(0,len(arr[0])):
        if arr[0][i:i+jump] == sub_arr:
            return i + jump
    return -1


def generate_data1(total_series_no = 20000):
    def arr_to_string(ipArr):
        returnStr = ""
        for e in ipArr:
            returnStr += str(e)
        return returnStr

    def str_to_arr(ipStr):
        return [int(e) for e in ipStr]

    seq_length = 15
    max_length = 100
    # total_series_no = 20000
    inp_arr = []
    inp_arr_str = []
    lbl_arr = []
    rdn = random.randint(1,10)

    for i in range(0,total_series_no):
        inp_a = arr_to_string(np.random.choice(2,max_length))
        inp_arr_str.append(inp_a[:seq_length])

    # pos_sub_str = ["111","1111","11111"]
    # neg_sub_str = ["1"*c for c in range(6,15)]

    count_pos = 0
    # _not_tomita_3 = re.compile("((0|1)*0)*1(11)*(0(0|1)*1)*0(00)*(1(0|1)*)*$")
    for a in inp_arr_str:
        if re.search("11111", a) != None:
        # if '11111' in a or '101010' in a:
            lbl_arr.append([1,0])
            count_pos+=1
        else:
            lbl_arr.append([0,1])
        inp_arr.append(str_to_arr(a))
    print('count pos, neg: ',count_pos,len(lbl_arr) - count_pos)
    return np.array(inp_arr),np.array(lbl_arr)

def generate_data2():
    alphabets = ['0','1']
    paths, lbls = [],[]
    maxlength = 10
    count = [0,0]
    def genstr(s):
        if len(s) > 0:
            paths.append(s)
            # if '101010' in s:
            # if re.search("1(10)+1", s) != None:
            if '11111' in s or "101010" in s or "001100" in s:
            # if (s.count("10") >= 2):
                lbls.append([1, 0])
                count[0]+=1
            else:
                lbls.append([0, 1])
                count[1] += 1
        if len(s) > maxlength:
            return

        # if re.search("(1)*", s) != None:

        for d in alphabets:
            genstr(s + d)

    genstr("")
    print('count ',count)
    return paths, lbls

def convert_data_x_todigit(x,alphabets):
    res = []
    for ip in x:
        res1 = []
        for e in ip:
            for i,d in enumerate(e):
                if str(d) == '1':
                    res1.append(alphabets[i])
        res.append(res1)
    return res

def convert_data_x(x,alphabets=['1','0']):
    d = {a:i for i,a in enumerate(alphabets)}
    res = []
    for ip in x:
        res1 = []
        for e in ip:
            v = [0] * len(d)
            v[d[e]] = 1
            res1.append(v)
        res.append(res1)
    return res

if __name__ == "__main__":
    print("test")
    X_arr, Y_arr = generate_data1()
    print("X_arr = ",X_arr)
    print("Y_arr = ",Y_arr)
    # X_arr.tofile("x.csv",delimiter=',',format='%.0f')
    np.savetxt('x.csv', X_arr, delimiter=",",fmt="%d")
    np.savetxt('y.csv', Y_arr, delimiter=",", fmt="%d")