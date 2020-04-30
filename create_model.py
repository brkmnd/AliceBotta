import re
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

# model_name must match the sequence txt file name
# in converse the sequence file must have same name but with added _seqs
model_name = "alice_botta_50"
# the n in n-gram has been taken care of when creating sequences.

def load_doc(filename):
    file = open(filename,"r")
    text = file.read()
    file.close()
    return text

def save_doc(lines,filename):
    data = "\n".join(lines)
    file = open(filename,"w")
    file.write(data)
    file.close()

def model_create(file_name):
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(X, y, batch_size=128, epochs=100)
    # save the model to file
    model.save("models/"+file_name+".h5")
    # save the tokenizer
    dump(tokenizer, open("models/"+file_name+"_tokenizer.pkl", 'wb'))

seqs_filename = "models/"+model_name+"_seqs.txt"
doc = load_doc(seqs_filename)
lines = doc.split("\n")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

# size of vocab - number of different words
vocab_size = len(tokenizer.word_index) + 1
	
# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

#these are just as found in the guide
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
model_create(model_name)
