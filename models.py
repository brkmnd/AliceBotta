import torch as ts
import torch as ts
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import comp_time

model_dict = {
          "m0":{
                "model_name":"pmodel_lstm0"
              , "emb_dim":128
              , "inter_dim":0
              , "lstm_dim":128
              , "n_layers":2
              , "corpus":"corpus1.json"
              , "n_splits":7
              }
        }

models_path = "models/"

class MlsLSTM(nn.Module):
    def __init__(self,model_dict):
        super(MlsLSTM,self).__init__()
        
        self.n_layers = model_dict["n_layers"]
        self.lstm_dim = model_dict["lstm_dim"]
        self.emb_dim = model_dict["emb_dim"]
        self.inter_dim = model_dict["inter_dim"]
        self.vocab_size = model_dict["vocab_size"]
        self.dropout_p = model_dict["dropout_p"]
        
        # layers
        self.emb = nn.Embedding(self.vocab_size,self.emb_dim)
        self.lstm = nn.LSTM(self.emb_dim,self.lstm_dim,self.n_layers,dropout=self.dropout_p)
        if self.inter_dim == 0:
            self.lin0 = nn.Linear(self.lstm_dim,self.vocab_size)
        else:
            self.lin1 = nn.Linear(self.lstm_dim,self.inter_dim)
            self.lin2 = nn.Linear(self.inter_dim,self.vocab_size)

    def forward(self,inputs,hiddens):
        e = self.emb(inputs).view(len(inputs),1,-1)

        lstm_out,hiddens = self.lstm(e,hiddens)
        lstm_out = lstm_out.view(len(inputs),-1)

        if self.inter_dim == 0:
            u = self.lin0(lstm_out)
        else:
            h = F.relu(self.lin1(lstm_out))
            u = self.lin2(h)

        return F.log_softmax(u,dim=1),hiddens

    def init_hiddens(self,b_size,device):
        weight = next(self.parameters())
        return ( weight.new_zeros(self.n_layers,b_size,self.lstm_dim).to(device)
               , weight.new_zeros(self.n_layers,b_size,self.lstm_dim).to(device)
               )

def get_model(m0,load_model,device):
    m0["dropout_p"] = m0["hyper_params"]["dropout_p"]
    model_name = m0["model_name"]
    model = MlsLSTM(m0)

    if load_model:
        model_name += ".ptm"
        model.load_state_dict(ts.load(models_path + model_name))
        print("model loaded from " + model_name)
    
    model.to(device)
    return model

def save_model(m0,model):
    model_name = m0["model_name"]
    model_name += ".ptm"
    ts.save(model.state_dict(),models_path + model_name)
    print("model saved as '" + model_name + "'")
