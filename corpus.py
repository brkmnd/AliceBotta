import json
import lexer as lx
import random

class Vocab:
    def __init__(this,seqs):
        this.__create_vocab(seqs)
        this.__create_maps()
        this.size = len(this.vocab)

        this.has = lambda x: x in this.vocab

    # __ for private
    def __create_vocab(this,seqs):
        v0 = [lx.toks[k] for k in lx.toks]
        for s0 in seqs:
            for t0 in s0:
                v0.append(t0)

        this.vocab = list(set(v0))
        this.vocab.sort()

    def __create_maps(this):
        word2id = {this.vocab[i]:i for i in range(len(this.vocab))} 
        id2word = this.vocab

        def word2id_map(w):
            if w in word2id:
                return word2id[w]
            return word2id[lx.toks["unk"]]
    
        def id2word_map(i):
            if i < 0 or i >= len(id2word):
                return lx.toks["unk"]
            return id2word[i]

        this.word2id = word2id_map
        this.id2word = id2word_map

class Corpus:
    def __init__(this,D):
        this.D = D
        this.__load_seqs()
        this.vocab = Vocab(this.seqs)


    # __ for private
    def __load_seqs(this):
        this.seqs = []
        
        for k in this.D:
            this.seqs = this.seqs + this.D[k]
        
        this.n_seqs = len(this.seqs)
        this.n_train = -1
        this.n_val = -1

    def shuffle(this):
        random.shuffle(this.seqs)

    def create_splits(this,n_splits):
        # returns the resulting splits
        res = []
        n_seqs = this.n_seqs
        n_val = round(n_seqs / n_splits)
        n_train = n_seqs - n_val
        seqs = this.seqs

        if n_seqs % n_splits == 0:
            print("splits are even")
        else:
            print("split uneven (" + str(n_seqs) + " seqs), good split sizes: " + find_split_sizes(n_seqs))
            return None

        for i in range(n_splits):
            seqs_val = seqs[i * n_val:(i + 1) * n_val]
            seqs_train = seqs[:i * n_val] + seqs[(i + 1) * n_val:]
            res.append((seqs_train,seqs_val))

        this.n_train = n_train
        this.n_val = n_val
        return res

    def has_seq(this,s0):
        res = []

        for k in this.D:
            for i in range(len(this.D[k])):
                xs = this.D[k][i]
                si = sublist_ind(xs,s0)
                if si > -1:
                    res.append((k,i,si))

        return res

    def slice_seq(this,tid,sid,s0,s1):
        None

    def get_txt(this,k):

        if k in this.D:
            return this.D[k]
        return None

def sublist_ind(xs,sub_xs):
    n_xs = len(xs)
    n_subs = len(sub_xs)

    if n_subs > n_xs:
        return -1
    elif n_subs == 0:
        return 0
    elif xs == sub_xs:
        return 0
    else:
        for i in range(n_xs - n_subs + 1):
            s0 = xs[i:i+n_subs]
            if s0 == sub_xs:
                return i
        return -1

def save_data(fname,res):
    with open(fname,"w",encoding="utf8") as f:
        json.dump(res,f,ensure_ascii=False)

def load_data(fname):
    res = None
    with open("datasets/" + fname,"r") as f:
        res = json.load(f)
    return res


def load_corpus(cname):
    D = load_data(cname)
    c0 = Corpus(D)
    return c0

if __name__ == "__main__":
    c0 = load_corpus("corpus1.json")
    c0_splits = c0.create_splits(7)
    print("vocab size   : " + str(c0.vocab.size))
    print("nr seqs      : " + str(c0.n_seqs))
    print("nr of trains : " + str(c0.n_train))
    print("nr of vals   : " + str(c0.n_val))
    
    print(c0.vocab.has("<eot>"))
    print(c0.vocab.has("<unk>"))
