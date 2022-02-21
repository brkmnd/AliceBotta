import sys
sys.path.append("../")

from lexer import toks,create_tokens
from corpus import save_data


corpus_files = { 1 : "alice.txt"
               , 2 : "dickinson.txt"
               #, 3 : "plath.txt"
               , 4 : "poe.txt"
               , 5 : "ulysses.txt"
               , 6 : "whitman.txt"
               }

def corpus_len(c0):
    i = 0
    for k in c0:
        for t in c0[k]:
            i += 1
    return i

def find_splits(i):
    res = [j for j in range(1,50) if i % j == 0]
    return res

def load_txt(fpath):
    res = ""
    with open(fpath,"r") as f:
        res = f.read()
    return res

def append1(d,k,v):
    if k not in d:
        d[k] = []
    d[k].append(v)

def compile_c():
    fs = [corpus_files[i] for i in corpus_files]
    txts = {f:load_txt("txts/" + f) for f in fs}
    res = {}

    for f0 in fs:
        txts0 = txts[f0].split(toks["split"])
        for txt0 in txts0:
            ts0 = create_tokens(txt0)
            if len(ts0) > 0:
                append1(res,f0,ts0)

    return res





    


def main():
    r0 = compile_c()
    r0_len = corpus_len(r0)
    r0_splits = find_splits(r0_len)
    print("number of seqs    : " + str(r0_len))
    print("reasonable splits : " + ",".join([str(x) for x in r0_splits]))
    save_data("corpus1.json",r0)


if __name__ == "__main__":
    main()
