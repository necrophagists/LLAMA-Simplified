from create_dataloader import PretrainedDataset
from llama.model import Transformer

import logging
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import shutil
from  typing import Optional,List
from Config import Args
device ='cuda' if torch.cuda.is_available() else 'cpu'

logger =logging.Logger("Trainer")
set_handle =logging.StreamHandler()
logger.addHandler(set_handle)
logger.setLevel(logging.DEBUG)

class LlamaPreTrainer:
    def __init__(self,args:Args):
      self.args =args
      logger.info("start init model")
      self.model=self.init_model(args)
      logger.info("start init dataloader")
      self.train_loader,self.test_loader =self.init_dataloader()
      logger.info("start training!")
      self.train_and_val()
    def init_model(self,args):

        if self.args.init_mode == "new": #如果是new 那就是从头开始
           logger.info("from new")
           model =Transformer(args=args).to(device)
        else:
            pass

        return model
    def init_dataloader(self):
        logger.info("start init trainloader")
        train_dataset =PretrainedDataset(self.args.train_data_path_list, self.model.params.max_seq_len, self.args.memmap)
        train_loader=DataLoader(train_dataset, pin_memory=False, batch_size=self.args.max_batch_size, shuffle=False)
        logger.info("finish init trainloader")
        test_loader =None
        if self.args.test_data_path_list is not None:
            logger.info("start init testloader")
            test_dataset =PretrainedDataset(self.args.test_data_path_list, self.model.params.max_seq_len, self.args.memmap)
            test_loader = DataLoader(test_dataset, pin_memory=False, batch_size=self.args.max_batch_size, shuffle=False)
            logger.info("finish init testloader")
        return train_loader,test_loader
    def train_and_val(self):
        total_step=1

        loss_func =nn.CrossEntropyLoss(reduction='mean')
        optimizer =AdamW(self.model.parameters(), lr=self.args.lr)
        steps =len(self.train_loader)
        lr_schedule=CosineAnnealingLR(optimizer=optimizer, T_max=self.args.epochs * steps)
        self.args.save_steps = self.args.epochs*len(self.train_loader)
        for epoch in range(self.args.epochs):
            epoch_loss=0
            self.model.train()

            for step,(X,Y) in enumerate(tqdm(self.train_loader)):

                optimizer.zero_grad()
                X =X.to(device)
                Y=Y.to(device)
                logits=self.model(X).transpose(-2,-1)
                step_loss =loss_func(logits,Y)
                epoch_loss += step_loss.data.item()
                step_loss.backward()
                optimizer.step()
                lr_schedule.step()

                if total_step %self.args.save_steps ==0:
                   temp_path = self.args.ckpt_path + f'/checkpoint-{total_step}'
                   if os.path.exists(self.args.ckpt_path) == False:
                       os.mkdir(self.args.ckpt_path)
                   if os.path.exists(temp_path)==False:
                       os.mkdir(temp_path)
                   if self.args.is_overwrite ==True:
                         clear_ckpt_file(self.args.ckpt_path)
                   self.save_ckpt(temp_path,total_step)
                   logger.info("already save ckpt!!")
                total_step += 1
            print(epoch,epoch_loss,optimizer.param_groups[0]['lr'])
            if epoch ==10:
                break
            if self.test_loader !=None:
               self.model.eval()
    def save_ckpt(self,path,cur_step):
        path =path+'/model_state_dict.pt'
        torch.save(self.model.state_dict(),path)
        logger.info(f"already save the {cur_step} steps ckpt!")
    def load_ckpt(self):
        pass
    def export_bin_file(self):
        pass


def clear_ckpt_file(path):
    shutil.rmtree(path)
    logger.info("already remove ckpt!")



