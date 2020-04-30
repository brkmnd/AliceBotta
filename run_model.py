import re
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
import keras.preprocessing.sequence as seq
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.models import load_model
from pickle import load
from random import randint

# variables
model_name = "alice_botta_40"
seed_src = "poe_40"

# number of words per text to generate
n_to_gen = 40
# number of texts to generate
txt_to_gen = 1
# custrom seed text to use as generation base
# if = None, then the original text is used
#custom_seed_txt = None

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
def save_doc(txt,filename):
    file = open(filename,"w")
    file.write(txt)
    file.close()

# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
            # encode the text as integer
            encoded = tokenizer.texts_to_sequences([in_text])[0]
            # truncate sequences to a fixed length
            encoded = seq.pad_sequences([encoded], maxlen=seq_length, truncating='pre')
            # predict probabilities for each word
            yhat = model.predict_classes(encoded, verbose=0)
            # map predicted word index to word
            out_word = ''
            for word, index in tokenizer.word_index.items():
                if index == yhat:
                    out_word = word
                    break
            # append to input
            in_text += ' ' + out_word
            result.append(out_word)
	return result

# load sequences into list
def load_seqs(mname):
    doc = load_doc("models/" + mname + "_seqs.txt")
    return doc.split("\n")

def get_seed_txt(lines):
    return lines[randint(0,len(lines))]

def create_linebreaks(words):
    line_min = 3
    line_max = 8
    res = ""
    line_len = randint(line_min,line_max)
    for w in words:
        res += w
        line_len -= 1
        if line_len == 0:
            res += "\n"
            line_len = randint(line_min,line_max)
        else:
            res += " "
    return res


# load cleaned text sequences
lines = load_seqs(seed_src)
seq_length = len(lines[0].split()) - 1

# load the model
model = load_model("models/"+model_name+".h5")
 
# load the tokenizer
tokenizer = load(open("models/"+model_name+"_tokenizer.pkl", "rb"))

print("-------------------------------------")
print("----model:")
print(model_name)
print("----seed text:")
print(seed_src)
print("-------------------------------------")

#resultat af genererede tekster der skrives til en fil
res_to_file = ""

for i in range(txt_to_gen):
    seed_text = get_seed_txt(lines)
    seed_out = "SEED #"+str(i)+"\n"+seed_text+"\n"
    gen_words = generate_seq(model, tokenizer, seq_length, seed_text, n_to_gen)
    gen_out = "GEN #"+str(i)+"\n"+create_linebreaks(gen_words)+ "\n"
    all_out = seed_out + gen_out
    print(all_out)
    print("----------------------------------------------------")
    res_to_file += all_out + "\n\n"

save_doc(res_to_file,"models/"+model_name+"_out.txt")
