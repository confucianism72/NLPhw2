from data_process import DataLoader
import torch
import torch.nn as nn
from model import LSTM_our as LSTM
from tqdm import tqdm

import hydra
import wandb
import os
import datetime

# def init():
#     hidden_dim = 512
#     use_previous_state = False
#     num_layers = 3
#     # dropout=0.0, bidirectional=False,
#     num_epochs = 10
#     learning_rate = 0.001
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    

def train(cfg, name):
    device = cfg.device
    train_data  = DataLoader(cfg.data.train, **cfg.train.train_dataLoader)
    valid_data  = DataLoader(cfg.data.valid)
    model = LSTM(cfg.model).to(device)
    # input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None
    loss_func = nn.NLLLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate, weight_decay= cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = cfg.train.num_epochs)
    
    logger = {'train_loss': [], 'valid_loss':[], 'valid_ppl':[]}
    path = f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/result_{name}.txt'
    with open(path, 'w') as f:
        f.write('')
    # wandb.init(project='NLhw2')
    
    epoch = -1
    while (True):
        for _ in range(cfg.train.num_epochs):
            epoch+=1
            model.train()
            model.start_epoch()
            train_loss = 0
            for input_data, target in tqdm(train_data):
                input_data, target = input_data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(input_data)
                loss = loss_func(output, target)
                loss.backward(retain_graph=True)
                optimizer.step()
                model.end_batch()

                train_loss += loss.item()
                # logger['train_loss'].append(loss.item())
                # wandb.log({'train_loss': loss.item()})
            # scheduler.step()
            train_loss /= train_data.get_len()
            valid_loss = valid(model, valid_data, cfg)
            print(f'epoch: {epoch}, valid_loss: {valid_loss.item()}, train_loss: {train_loss}, path: {path}')
            # logger['valid_loss'].append(valid_loss.item())
            # wandb.log({'valid_loss': valid_loss.item()})
            with open(path, 'a') as f:
                f.write(f"{str(epoch)} %  {str(valid_loss.item())} % {str(train_loss)} \n")

        torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()},
                    f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/final.pt')
        print(f'output dir is {path}')
        
        # end = input('Continue try? [y/n]: ')
        # if end =='n':
            # break
        break
    
    # os.
    # with open(path, 'w') as f:
    #     for i, r in enumerate(logger['valid_loss']):
    #         f.write(f"{str(i)}  {str(r)} \n")


def valid(model, valid_data, cfg):
    device = cfg.device
    model.eval()
    model.start_epoch()
    loss_func = nn.NLLLoss()
    loss = 0
    with torch.no_grad():
        for input, target in tqdm(valid_data):
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss += loss_func(output, target)
            model.end_batch()
            ...
    return loss/valid_data.get_len()

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    print(cfg)
    # print(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    train(cfg, name = '')
    # for input_size in [300, 350, 400, 500, 600]:
    #     cfg.model.LSTM.input_size = input_size
    #     for hidden_size in [64, 128, 256, 512, 1024, 2048]:
    #         cfg.model.LSTM.hidden_size = hidden_size
    #         for dropout in [0,0.01,0.05,0.1,0.3,0.5,0.7]:
    #             cfg.model.dropout = dropout
    #             for weight_decay in [0,1e-6,1e-5, 3e-5, 6e-5, 1e-4]:
    #                 cfg.train.weight_decay = weight_decay
                    
                    # train(cfg, name = f'{input_size}_{hidden_size}_{dropout}_{weight_decay}')
    ...


if __name__ == '__main__':
    main()
    