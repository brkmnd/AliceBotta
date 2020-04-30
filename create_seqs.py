import re
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

# vars
# word position of where the tokens start.
tokens_start = 0
# length of words from tokens_start. None = all
tokens_len = None

n_gram,model_name, in_filename = 40,"poe_40","txt/poe.txt"

def load_doc(filename):
    file = open(filename,"r")
    text = file.read()
    file.close()
    return text

# create_tokens splits input into lowercase words
def create_tokens(txt):
    # replace special chars
    txt = txt.replace("&","and")
    # join words into one word if seperated by one of
    txt = txt.replace("-","")
    txt = txt.replace("'","")
    txt = txt.replace("’","")
    # allowed alphabet for words
    rx = re.compile("[a-zA-Z0-9]+")
    t = rx.findall(txt)
    return [word.lower() for word in t]

# create sequences
def create_seqs(tokens):
    length = n_gram + 1
    seqs = list()
    for i in range(length,len(tokens)):
        seq = tokens[i-length:i]
        line = " ".join(seq)
        seqs.append(line)
    return seqs

def save_doc(lines,filename):
    data = "\n".join(lines)
    file = open(filename,"w")
    file.write(data)
    file.close()

doc = load_doc(in_filename)

# index is the number of words to add.
if tokens_len == None:
    tokens = create_tokens(doc)
else:
    tokens = create_tokens(doc)[seq_start:seq_start + seq_len]

seqs = create_seqs(tokens)
save_doc(seqs,"models/"+model_name+"_seqs.txt")

#print status
print("----------------------------done")
print("tokens-len:")
print(len(tokens))
print("num unique tokens:")
print(len(set(tokens)))
