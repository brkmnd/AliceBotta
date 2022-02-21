from utils import load_seed,get_device,save_poems,list_seeds,list_seed_subdirs
from lexer import create_tokens
from models import get_model,model_dict
from run_model import hyper_params,pred_model
from corpus import load_corpus

from random import randint

starters = { 0 : "i"
           , 1 : "you"
           , 2 : "him"
           , 3 : "her"
           , 4 : "they"
           }

def segment_poem(p):
    i0 = 0
    res = []

    for i in range(len(p)):
        if i0 >= len(p):
            break
        split_i = randint(1,5) 
        l0 = p[i0:i0 + split_i]
        res.append(l0)
        i0 += split_i

    return res

def segpoem2str(p):
    return "\n".join([" ".join(x) for x in p])

def main_gen_poems(m0,model,vocab,args,device):
    seed_fname = args["seed_fname"]
    start_w = args["start_w"]
    gen_n = args["gen_n"]

    seed0 = load_seed(seed_fname)
    seed0_toks = [create_tokens(s) for s in seed0]
    seed_i = 1
    out_fname = seed_fname.replace("/","-") + "-" + start_w

    res  = "--generating poems for seeds '" + seed_fname + "'\n"
    res += "--model   = " + m0["model_name"] + "\n"
    res += "--gen_n   = " + str(gen_n) + "\n"
    res += "--start_w = '" + start_w + "'\n"
    res += "--n_seeds = " + str(len(seed0_toks)) + "\n"
    res += "\n"

    for ts0 in seed0_toks:
        poem0 = pred_model(m0,model,vocab,ts0,start_w,gen_n,device)
        poem0 = segment_poem(poem0)
        res += "--seed_i  = " + str(seed_i) + "\n"
        res += "--seed    = '" + " ".join(ts0) + "'\n"
        res += "********\n"
        res += segpoem2str(poem0)
        res += "\n\n"
        seed_i += 1

    save_poems(out_fname,res)
    print("poems saved to " + out_fname)



def main(device):
    m0 = model_dict["m2"]
    corpus = load_corpus(m0["corpus"])
    vocab = corpus.vocab
    m0["vocab_size"] = vocab.size
    m0["hyper_params"] = hyper_params
    model = get_model(m0,True,device)

    seed_subdirs = list_seed_subdirs()

    print("")
    print("****poem generator")

    seed_dir = input("write seed sub-dir [" + ",".join(seed_subdirs) + "] (q for quit) :\n")

    if seed_dir == "q":
        print("quitted poem generator")
    elif seed_dir in seed_subdirs:
        args = { "seed_fname":None
               , "start_w":None
               , "gen_n":50
               }

        for k0 in starters:
            args["start_w"] = starters[k0]
            for seed0 in list_seeds(seed_dir):
                args["seed_fname"] = seed_dir + "/" + seed0
                main_gen_poems(m0,model,vocab,args,device)
    else:
        print("did not understand seed sub-dir : " + seed_dir)

if __name__ == "__main__":
    device = get_device()
    main(device)
