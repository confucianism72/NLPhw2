import torch
import torch.nn as nn
from data_process import *

class LSTM_our(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()


        self.lstm = nn.LSTM(**cfg.LSTM)
        # linear layer only takes the last dimension of the input
        self.linear = nn.Linear(cfg.LSTM.hidden_size, cfg.output_dim)
        # input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None
        self.activation = nn.LogSoftmax(dim = -1)
        self.dropout = nn.Dropout(p= cfg.dropout)

        self.init_state = None
        self.use_previous_state = cfg.use_previous_state
        
        assert cfg.LSTM.input_size >= 300
        custom_len = cfg.LSTM.input_size - 300
        # self.embedding = nn.Embedding(10000, cfg.LSTM.input_size, scale_grad_by_freq=cfg.scale_grad_by_freq)

        vocab = get_vocab()
        vocab_embed_dict = get_vocab_embed_dict()
        weight = torch.zeros(10000, cfg.LSTM.input_size)
        for i, word in enumerate(vocab):
            weight[i, 0:300] = torch.from_numpy(vocab_embed_dict[word])

        weight = weight.detach()
        # padding_idx =  list(range(9714)) if cfg.fix_glove else None

        self.embedding = nn.Embedding.from_pretrained(weight, freeze=False,  scale_grad_by_freq=cfg.scale_grad_by_freq)
        embedding_weight = self.embedding.weight.detach()
        if cfg.fix_glove:
            embedding_weight[:9714].requires_grad = False
        if cfg.LSTM.input_size > 300:
            embedding_weight[:, 300: ].requires_grad = True
        self.embedding.weight = nn.Parameter(embedding_weight)

    def start_epoch(self):
        self.init_state = None
    
    def end_batch(self):
        if self.use_previous_state:
            self.init_state = self.out_state[0].detach(), self.out_state[1].detach() 
        self.out_state = None

    def forward(self, x):
        x = self.embedding(x)
        x, self.out_state = self.lstm(x, self.init_state)
        N, S, H = x.shape
        x = x.reshape(-1, H)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        # x = x.view(N, S, -1)
        return x
    
import hydra
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    model = LSTM_our(cfg = cfg.model, input_size = 300, hidden_size = 512, num_layers = 3, batch_first = True, output_dim= 10000)
    print(model)
    input = torch.zeros(3, 5, dtype= torch.int64)
    output = model(input)
    print(output.shape)
    print(output)

if __name__ == '__main__':
    main()
    
    
