import torch as ts
import numpy as np
import time
import random
import os

def enforce_rep(seed=42):
    # Sets seed manually for both CPU and CUDA
    ts.manual_seed(seed)
    ts.cuda.manual_seed_all(seed)
    # For atomic operations there is currently 
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    ts.backends.cudnn.deterministic = True
    ts.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)

def num2str(n,w):
    i0 = 10 ** (w - 1)
    res = ""
    while n < i0:
        res += "0"
        i0 /= 10
    
    res += str(n)
    return res

def get_time():
    return time.time()

def comp_time(t0,time_fun):
    if time_fun is None:
        time_fun = lambda x : x

    used_time = time_fun(round(time.time() - t0,2))
    measure = "seconds"
    if used_time >= 60.0:
        used_time /= 60.0
        #used_time = round(used_time,2)
        measure = "minutes"
    if used_time >= 60.0:
        used_time /= 60.0
        #used_time = round(used_time,2)
        measure = "hours"
    used_time = round(used_time,2)
    return str(used_time) + " " + measure

def get_device():
    device = ts.device("cpu")
    if ts.cuda.is_available():
        print("has cuda")
        device = ts.device("cuda")
    return device

def save_acc(model_name,n_epochs,n_splits,accs,avg_loss):
    with open("models/" + model_name + ".acc.txt","a") as f:
        res  = "[" + str(n_epochs) + "*" + str(n_splits) + "] :\n"
        res += "   -low acc  : " + str(accs.min()) + "\n"
        res += "   -high acc : " + str(accs.max()) + "\n"
        res += "   -avg acc  : " + str(accs.mean()) + "\n"
        res += "   -avg loss : " + str(avg_loss) + "\n"
        f.write(res)

def load_seed(sname):
    res = []
    with open("seeds/" + sname,"r") as f:
        res = f.read().split("</s>")
    return [x.strip() + " " for x in res]

def save_poems(fname,txt):
    path0 = "out"
    fname += ".txt"

    #n_fs = os.listdir(path0)
    #n_fs = len([x for x in n_fs if x == fname])
    #fname = num2str(n_fs + 1,4) + ".txt"

    with open("out/" + fname,"w") as f:
        f.write(txt)

if __name__ == "__main__":
    save_poems("noget")
