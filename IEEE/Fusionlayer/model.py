# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 11:14:39 2020

@author: Administrator
"""

import argparse
import torch
import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss,BCELoss,L1Loss,BCEWithLogitsLoss
from torch.utils.data import DataLoader,SequentialSampler, RandomSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel)
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
class R2_loss(nn.Module):
    def __init__(self):
        super().__init__()   #没有需要保存的参数和状态信息
        
    def forward(self, preds, labels):  # 定义前向的函数运算即可\
        #求出loss和acc
        RSS=torch.sum((labels-preds)*(labels-preds), 0)
        mean_target=torch.mean(labels)
        TSS=torch.sum((preds-mean_target)*(preds-mean_target), 0)    
        return RSS/TSS
class WeightedMultilabel(nn.Module):
    def __init__(self,weights):
      super(WeightedMultilabel, self).__init__()
      self.loss = nn.BCEWithLogitsLoss()
      # 
      self.weights = weights

    def forward(self,outputs, targets):
        return torch.sum(self.loss(outputs, targets) * self.weights)
class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, args):
        super().__init__()
        self.norm= nn.BatchNorm1d(args.out_size)
        self.dense = nn.Linear(args.out_size, args.linear_layer_size[0])
        self.norm_1= nn.BatchNorm1d(args.linear_layer_size[0])
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.dense_1 = nn.Linear(args.linear_layer_size[0], args.linear_layer_size[1])  
        self.norm_2= nn.BatchNorm1d(args.linear_layer_size[1])
        self.out_proj = nn.Linear(args.linear_layer_size[1], args.num_label)

    def forward(self, features, **kwargs):
        x = self.norm(features)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(self.norm_1(x))
        x = self.dropout(x)
        x = self.dense_1(x)
        x = torch.relu(self.norm_2(x))
        x = self.dropout(x)        
        x = self.out_proj(x)
        return x
class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()
        args.out_size=len(args.dense_features)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args=args

        #创建BERT模型，并且导入预训练模型
        config = RobertaConfig.from_pretrained(args.pretrained_model_path) 
        config.output_hidden_states=True
        args.hidden_size=config.hidden_size
        args.num_hidden_layers=config.num_hidden_layers
        self.text_layer=RobertaModel.from_pretrained(args.pretrained_model_path,config=config) 
        self.text_linear=nn.Linear(args.text_dim+args.vocab_dim_v1*len(args.text_features), args.hidden_size)
        logger.info("Load linear from %s",os.path.join(args.pretrained_model_path, "linear.bin"))
        self.text_linear.load_state_dict(torch.load(os.path.join(args.pretrained_model_path, "linear.bin")))           
        logger.info("Load embeddings from %s",os.path.join(args.pretrained_model_path, "embeddings.bin"))
        self.text_embeddings=nn.Embedding.from_pretrained(torch.load(os.path.join(args.pretrained_model_path, "embeddings.bin"))['weight'],freeze=True)             
        args.out_size+=args.hidden_size*2
        
        #创建fusion-layer模型，随机初始化
        config = RobertaConfig()        
        config.num_hidden_layers=4
        config.intermediate_size=2048
        config.hidden_size=512
        config.num_attention_heads=16
        config.vocab_size=5
        self.text_layer_1=RobertaModel(config=config)
        self.text_layer_1.apply(self._init_weights)
        self.text_linear_1=nn.Linear(args.text_dim_1+args.hidden_size, 512) 
        self.text_linear_1.apply(self._init_weights)  
        self.norm= nn.BatchNorm1d(args.text_dim_1+args.hidden_size)
        args.out_size+=1024    

        #创建分类器，随机初始化
        self.classifier=ClassificationHead(args)
        self.classifier.apply(self._init_weights)
        # self.norm= nn.BatchNorm1d(args.out_size)
        # self.dense = nn.Linear(args.out_size, args.linear_layer_size[0])
        # self.dense.apply(self._init_weights)
        # self.norm_1= nn.BatchNorm1d(args.linear_layer_size[0])
        # self.dropout = nn.Dropout(args.hidden_dropout_prob)
        # self.dense_1 = nn.Linear(args.linear_layer_size[0], args.linear_layer_size[1])
        # self.dense_1.apply(self._init_weights)  
        # self.norm_2= nn.BatchNorm1d(args.linear_layer_size[1])

        # self.dense_2 = nn.Linear(args.linear_layer_size[1], args.linear_layer_size[2])
        # self.dense_2.apply(self._init_weights)  
        # self.norm_3= nn.BatchNorm1d(args.linear_layer_size[2])



        # # self.out_proj = nn.Linear(args.linear_layer_size[1], args.num_label)
        # self.out_proj = nn.Linear(args.linear_layer_size[2], 1)
        # self.out_proj.apply(self._init_weights) 
        # self.R2_loss=R2_loss()
        
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            
    def forward(self,dense_features,text_features,text_ids,text_masks,text_features_1,text_masks_1,labels=None):
        outputs=[]
        #获取浮点数，作为分类器的输入
        outputs.append(dense_features.float()) 
        #print(torch.isnan(dense_features.float()).any())        
        #获取BERT模型的hidden state，并且做max pooling和mean pooling作为分类器的输入
        text_masks=text_masks.float()
        text_embedding=self.text_embeddings(text_ids).view(text_ids.size(0),text_ids.size(1),-1)
        text_features=torch.cat((text_features.float(),text_embedding),-1)
        text_features=torch.relu(self.text_linear(self.dropout(text_features)))
        hidden_states=self.text_layer(inputs_embeds=text_features,attention_mask=text_masks)[0]
        embed_mean=(hidden_states*text_masks.unsqueeze(-1)).sum(1)/text_masks.sum(1).unsqueeze(-1)
        embed_mean=embed_mean.float()
        embed_max=hidden_states+(1-text_masks).unsqueeze(-1)*(-1e10)
        embed_max=embed_max.max(1)[0].float()
        outputs.append(embed_mean)
        outputs.append(embed_max)
        #获取fusion-layer的hidden state，并且做max pooling和mean pooling作为分类器的输入
        text_masks_1=text_masks_1.float()
        text_features_1=torch.cat((text_features_1.float(),hidden_states),-1)
        bs,le,dim=text_features_1.size()
        text_features_1=self.norm(text_features_1.view(-1,dim)).view(bs,le,dim)
        text_features_1=torch.relu(self.text_linear_1(text_features_1))
        # print("*******",text_features_1.size(),"******",text_masks_1.size())
        hidden_states=self.text_layer_1(inputs_embeds=text_features_1,attention_mask=text_masks_1)[0]
        embed_mean=(hidden_states*text_masks_1.unsqueeze(-1)).sum(1)/text_masks_1.sum(1).unsqueeze(-1)
        embed_mean=embed_mean.float()
        embed_max=hidden_states+(1-text_masks_1).unsqueeze(-1)*(-1e10)
        embed_max=embed_max.max(1)[0].float()
        outputs.append(embed_mean)
        outputs.append(embed_max)             

        #将特征输入分类器，得到2分类的logits
        final_hidden_state=torch.cat(outputs,-1)
        logits=self.classifier(final_hidden_state)

        
        #print(final_hidden_state.size())
        
        # x = self.norm(final_hidden_state)
        # x = self.dropout(x)
        
        # x = self.dense(x)
        
        # #print("dense",self.args.out_size, self.args.linear_layer_size[0])
        # x = torch.relu(self.norm_1(x))
        # x = self.dropout(x)
        # x = self.dense_1(x)
        # x = torch.relu(self.norm_2(x))

        # x = self.dense_2(x)
        # x = torch.relu(self.norm_3(x))
        # x = self.dropout(x)     
        # lsigmoid = nn.LogSigmoid()   
        # logits =lsigmoid(self.out_proj(x) )      
         #other另一个版本
        # logits =torch.relu(self.out_proj(x) ) 
        # m = nn.Threshold(0, 0)
        # logits = m(logits)
        #《end other另一个版本
        #labels[0]表示是否升级，labels[1]表示回归预测的数值
        
        
        # prob=torch.softmax(logits,-1)
        # is_escalated=torch.argmax(prob, dim=1)#得到是否升级，0表示从不，1表示升级
        # escalated_probs=logits[:,1]*is_escalated.type_as(logits)
        # escalated_probs=logits[:,1]
        #返回loss或概率结果
        if labels is not None:
            # weights=torch.tensor([0.04,0.96]).to(self.args.device)
            # loss_fct = WeightedMultilabel(weights) 
            # loss1 = loss_fct(is_escalated.float(), labels[0].float())
            loss_fct1=MSELoss()
            # loss_fct1=R2_loss()
            #print("*******",escalated_probs.size(),labels[1].size())
            loss2= loss_fct1(logits, labels[1].float())
            # print("真实值：",labels[1],"预测：",logits)
            return loss2.to(torch.float32)
        else:
            
            return logits#返回预测目标


            

