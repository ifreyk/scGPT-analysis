# Libraries
import torch
from torch.utils.data import Dataset, DataLoader
import math
from torch import nn
import transformers
import neptune

import scanpy as sc
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm_notebook

from email.message import EmailMessage
import ssl
import smtplib

# Load tokens
with open('./scdata/geneformer_token_dictionary.pkl','rb') as file1:
    dct_tokens = pickle.load(file1)

# Train and test data h5ad
train_data = sc.read_h5ad('./scdata/train_data.h5ad')
test_data = sc.read_h5ad('./scdata/test_data.h5ad')

# Choose device
device = torch.device('cuda')

# basic parametres
EPOCHS = 8
lr = 5e-5
wd = 1e-7
batch_size = 24 # м.б, можно больше, если gpu позволяет
MAX_LEN = 2048
num_train_steps = 80000
freeze_layers = None # пока решил не замораживать

# Prepare data columns
x_train = train_data.X
x_test = test_data.X
y_train = train_data.obs['orig.ident'].values
y_test = test_data.obs['orig.ident'].values
un_values = np.array(y_train.unique())
dct_targets = dict(zip(un_values, np.arange(len(un_values))))
y_train = np.array([dct_targets[i] for i in y_train])
y_test = np.array([dct_targets[i] for i in y_test])

# class to use for DataLoader
class Un_data(Dataset):
    def __init__(self,data,targets, max_len, mode='train'):
        self.data = torch.IntTensor(data)[:,:max_len] # в случае, если подаю необрезанные токены
        self.targets = torch.FloatTensor(targets) if mode == 'train' else None
        self.mode = mode
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        if self.mode == 'train':
            tokens = self.data[idx,:]
            target = self.targets[idx]
            at_mask = (tokens != 0).long()
            return tokens, target,at_mask
        else:
            return self.data[idx,:]

# Load Data to DataLoader
train_data_ = Un_data(data=x_train, targets=y_train,max_len=MAX_LEN, mode='train')
test_data_ = Un_data(data=x_test,targets=y_test, max_len=MAX_LEN, mode='train')
train_dataloader = DataLoader(train_data_, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data_, batch_size=batch_size, shuffle=True)

# Function that was used for something (don't know for what)
def forward(  # меняю forward, т.к. BertForSequenceClassification не поддерживает average pooling(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
):

    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )

    lhs = outputs.last_hidden_state
    lhs[(~attention_mask.bool())] = 0
    pooled_output = lhs.sum(-2)
    mean_factor = attention_mask.sum(-1).reshape(-1, 1)
    pooled_output = pooled_output / mean_factor

    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    return logits  # (loss), logits, (hidden_states), (attentions)


config = { # он здесь не нужен, но пусть пока побудет
    "hidden_size": 256,
    "num_hidden_layers": 6,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-12,
    "attention_probs_dropout_prob": 0.02,
    "hidden_dropout_prob": 0.02,
    "intermediate_size": 512,
    "hidden_act": 'relu',
    "max_position_embeddings": 2048,
    "model_type": "bert",
    "num_attention_heads": 4,
    "vocab_size": len(dct_tokens),  # genes+2 for <mask> and <pad> tokens
    'num_labels':1,
    "pad_token_id": dct_tokens.get("<pad>")
}

config = transformers.BertConfig(**config)
# Здесь похоже надо будет вместо BertForSequenceClassification выбрать другой параметр для регрессии уже
transformers.BertForSequenceClassification.forward = forward
model = transformers.BertForSequenceClassification.from_pretrained('ctheodoris/Geneformer'
                                                                   ,num_labels=1,output_attentions=False,)
# These parametres would be used in train() function
model.to(device)
criterion = nn.BCEWithLogitsLoss()
acc_cr = nn.Sigmoid()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=num_train_steps)

# Тут какой-то нептун, чтобы сделать какой-то класс run. Видимо на этом месте у нас свой api
api = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMTVkMjE3NC02OTQzLTQ0YWQtYmFjZi0zZjU3YThlNzM3NTMifQ=='
project = 'bushsebzz/CMAP-BERT'
run = neptune.init_run(project=project,api_token=api)
params = {
    "max_epochs": EPOCHS,
    'batch_size': batch_size,
    'Num_genes': 2048,
    "optimizer": "Adam",
    "dropout": 0.02,
    'max_lr': lr,
    'warmup_steps': 0,
    'scheduler_steps':num_train_steps ,
    'wd': 1e-7,
    'Active_func': 'RELU',
    'Num_gpu': 1,
    'Pooling_strategy':'right_average'

}
run["parameters"] = params


def train(model,train_dataloader,test_dataloader ,criterion, optimizer,device,EPOCHS):
    print('Nice, train is starting!')
    train_lst_loss = []
    test_lst_loss = []
    for epoch in range(1, EPOCHS + 1):
        run["Epoch"].append(epoch)
        lst_acc = []
        model.train()
        temp_loss_lst = []
        print(f'Epoch {epoch}')
        it = 0
        for batch,targets,mask in tqdm_notebook(train_dataloader, desc='Training'):
            optimizer.zero_grad()
            batch = batch.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            pred = model(batch,attention_mask=mask)
            loss = criterion(pred[:,0],targets)
            acc_ = (torch.round(acc_cr(pred[:,0])) == targets)
            acc = sum(acc_)/len(acc_)
            lst_acc.append( acc_)

            it += 1
            if (it % 100) == 0:
                print(f'loss: {loss}')
                print(f'accuracy: {acc}')

            run["train/cur_acc"].append(acc)
            run["train/loss"].append(loss)
            run['train/cur_lr'].append(optimizer.param_groups[0]['lr'])
            loss.backward()
            optimizer.step()
            scheduler.step()
            temp_loss_lst += [(loss.item())]
        lst_acc = torch.concat(lst_acc)
        lst_acc = lst_acc.sum()/len(lst_acc)
        run["train/global_acc"].append(lst_acc)
        train_lst_loss += [sum(temp_loss_lst)/len(temp_loss_lst)]


        model.eval()
        temp_loss_lst = []
        lst_acc = []

        with torch.no_grad():
            for batch,targets,mask in tqdm_notebook(test_dataloader, desc='Validating'):
                batch = batch.to(device)
                targets = targets.to(device)
                mask = mask.to(device)
                pred = model(batch,attention_mask=mask)
                loss = criterion(pred[:,0],targets)

                run["test/loss"].append(loss)
                acc_ = (torch.round(acc_cr(pred[:,0])) == targets)
                acc = sum(acc_)/len(acc_)
                lst_acc.append( acc_)
                run["test/cur_acc"].append(acc)
                run["test/loss"].append(loss)


                temp_loss_lst += [(loss.item())]

            test_lst_loss += [sum(temp_loss_lst)/len(temp_loss_lst)]
        lst_acc = torch.concat(lst_acc)
        lst_acc = lst_acc.sum()/len(lst_acc)
        run["test/global_acc"].append(lst_acc)
        torch.save(model,f'./checkpoints/Gcl_{epoch}.pth') # выбери директорию, куда можно сохранять моделб
    return model

# Actual training
model = train(model,train_dataloader,test_dataloader ,criterion, optimizer,device,EPOCHS)