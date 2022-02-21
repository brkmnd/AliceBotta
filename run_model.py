import torch as ts
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import comp_time,get_time,get_device,enforce_rep,save_acc
from models import model_dict,get_model,save_model
from corpus import load_corpus
from lexer import toks
import numpy as np

def concat_words(words,word2id,device):
    return ts.tensor([word2id(w) for w in words],dtype=ts.long).to(device)

def train_model(m0,model,seqs,vocab,device):
    hyper_params = m0["hyper_params"]
    l_rate = hyper_params["l_rate"]
    b_size = hyper_params["batch_size"]
    n_epochs = hyper_params["n_epochs"]
    word2id = vocab.word2id
    seqs_train,seqs_val = seqs

    loss_f = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr=l_rate)

    est_time_n = 0
    n = len(seqs_train)
    epoch_insts = []

    for epoch in range(1,n_epochs + 1):
        avg_loss = []
        model.train()
        start_time = get_time()

        for xs in seqs_train:
            hiddens = model.init_hiddens(b_size,device)
            model.zero_grad()

            ys = xs[1:] + [toks["eot"]]
            inputs = concat_words(xs,word2id,device)
            targets = concat_words(ys,word2id,device)

            hiddens = [h.detach() for h in hiddens]
            logits,hiddens = model(inputs,hiddens)

            loss = loss_f(logits,targets)
            avg_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            est_time_n += 1
            if est_time_n == 1000 and 1 == 2:
                print("est-time per epoch:" + comp_time(start_time,lambda t0: t0 * n / 1000))
                print("")

        acc,err_rate = eval_model(m0,model,seqs_val,vocab,device)
        avg_loss = np.array(avg_loss)
        epoch_insts.append((acc,avg_loss.mean(),model.state_dict()))

        print("accuracy[" + str(epoch) + "] : " + str(acc))
        print("loss[" + str(epoch) + "]     : " + str(avg_loss.mean()))
        print("----took " + comp_time(start_time,None))
    
    epoch_insts.sort(key=lambda x: x[0])
    best_acc,best_loss,best_state_dict = epoch_insts[-1]
    model.load_state_dict(best_state_dict)

    return model,best_loss,best_acc

def eval_model(m0,model,seqs,vocab,device):
    word2id = vocab.word2id

    n_correct = 0
    n_incorrect = 0
    n_total = 0

    model.eval()

    with ts.no_grad():
        for s0 in seqs:
            hiddens = model.init_hiddens(m0["hyper_params"]["batch_size"],device)
            xs = concat_words(s0,word2id,device)
            preds,_ = model(xs,hiddens)
            ys = s0[1:] + [toks["eot"]]

            for pred,y in zip(preds,ys):
                y_hat = pred.argmax().item()
                y = word2id(y)

                n_total += 1
                if y_hat == y:
                    n_correct += 1
                else:
                    n_incorrect += 1

    acc = n_correct / n_total
    err_r = n_incorrect / n_total

    return acc,err_r

def pred_model(m0,model,vocab,s0,start_w,gen_n,device):
    word2id = vocab.word2id
    id2word = vocab.id2word

    model.to(device)
    model.eval()
    
    res = [start_w]

    with ts.no_grad():
        hiddens = model.init_hiddens(m0["hyper_params"]["batch_size"],device)
        xs_start = concat_words(s0,word2id,device)
        _,hiddens = model(xs_start,hiddens)

        for _ in range(gen_n):
            xs = concat_words(res,word2id,device)
            preds,_ = model(xs,hiddens)
            pred = preds[-1].argmax().item()
            res.append(id2word(pred))

    return res

def main_eval(m0,model,seqs_splits,vocab,device):
    for _,seqs_eval in seqs_split:
        None

def main_train(m0,model,seqs_splits,vocab,device):
    model_name = m0["model_name"]
    n_epochs = m0["hyper_params"]["n_epochs"]
    n_splits = len(seqs_splits)
    
    split_i = 0
    start_time = get_time()
    accs = []
    avg_loss = []

    print("")
    print("--training")
    print("  nr splits : " + str(n_splits))
    print("  nr epochs : " + str(n_epochs))

    for seqs in seqs_splits:
        split_i += 1
        print("")
        print("split           : "+ str(split_i))
        print("train split len : " + str(len(seqs[0])))
        print("test split len  : " + str(len(seqs[1])))
        print("")

        model,loss,acc = train_model(m0,model,seqs,vocab,device)
        save_model(m0,model)
        accs.append(acc)
        avg_loss.append(loss)

        print("")
        print("best acc for split : " + str(acc))

    avg_loss = np.array(avg_loss)
    accs = np.array(accs)
    save_acc(m0["model_name"],n_epochs,n_splits,accs,avg_loss.mean())

    print("")
    print("--training in all took " + comp_time(start_time,None))
    print("avg loss : " + str(avg_loss.mean()))
    print("avg acc  : " + str(accs.mean()))
    print("--stats saved")
    print("")

# these parameters are added along those
# found in model_dict in models.py
hyper_params =  { "n_epochs":5
                , "l_rate":1 / 10 ** 4
                , "batch_size":1
                , "dropout_p":0.5
                }

def main(device):
    load_model = False
        
    m0 = model_dict["m0"]

    corpus = load_corpus(m0["corpus"])
    corpus.shuffle()
    seqs = corpus.create_splits(m0["n_splits"])
    vocab = corpus.vocab

    m0["hyper_params"] = hyper_params
    m0["vocab_size"] = vocab.size
    model = get_model(m0,load_model,device)

    print("")
    print("model name            : " + m0["model_name"])
    print("vocab size            : " + str(vocab.size))
    print("number of seqs        : " + str(corpus.n_seqs))
    print("  nr of train in seqs : " + str(corpus.n_train))
    print("  nr of val in seqs   : " + str(corpus.n_val))
    print("  train ratio         : " + str(corpus.n_train / corpus.n_seqs))
    print("")

    n_epochs = hyper_params["n_epochs"]
    n_splits = m0["n_splits"]

    menu = ( "pick one:\n"
           + "  1) train for " + str(n_epochs) + " epochs and " + str(n_splits) + " splits\n"
           + "  2) eval on splits\n"
           + "  q) quit\n"
           )
    a0 = input(menu)

    if a0 == "1":
        a1 = input("write number of rounds (0-10):\n")
        if a1 in [str(x) for x in range(11)]:
            for _ in range(int(a1)):
                main_train(m0,model,seqs,vocab,device)
        else:
            print("did not understand number of rounds: " + a1)
    elif a0 == "2":
        print("not yet")
    elif a0 == "q":
        print("quitting")
    else:
        print("did not understand choice: " + a0)

    #r0 = eval_model(m0,model,seqs[0][1],vocab,device)
    #print(r0)


if __name__ == "__main__":
    device = get_device()
    enforce_rep()
    main(device)
