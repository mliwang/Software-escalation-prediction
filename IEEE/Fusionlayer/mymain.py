# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 16:33:20 2020

@author: Administrator
"""
import os
import gc
import torch
import logging
import argparse
import ctrNets as ctrNet
import pickle
import gensim
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from data_loader import TextDataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler,TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
base_path="../middle"
#定义浮点数特征 包含统计特征和原始特征处理成数值的
# dense_features=['milestones_size','milestones_unique','time_unique','milestone_times_sum','milestone_times_mean','milestone_times_std',
#                 'REFERENCEID', 'SITECODE', 'SITENAME', 'CONTACTCOMPANY', 'COMPANYID',
#        'COMPANYOID', 'INITIALUSERGROUPID', 'INITIALUSERGROUPDESC',
#        'MEDPROJECTID', 'MEDPROJECTAREA', 'MEDPROJOPENBY', 'IWAYJIRAISSUEID',
#        'NEWIWAYJIRAISSUEID', 'PREMIUMCODE', 'PRIMARYPRODUCTFAMILYID',
#        'PRIMARYPRODUCTFAMILYDESC', 'PRIMARYPRODUCTID', 'PRIMARYPRODUCTDESC',
#        'PRIMARYRELEASEID', 'PRIMARYRELEASEDESC', 'PRIMARYPRODUCTAREAID',
#        'PRIMARYPRODUCTAREADESC', 'PRIMARYPRODUCTOSFAMILYID',
#        'PRIMARYPRODUCTOSFAMILYDESC', 'PRIMARYPRODUCTOSPLATFORMID',
#        'PRIMARYPRODUCTOSPLATFORMDESC', 'CONTACTMETHODFLAG', 'CUSTOMERFIRST',
#        'CUSTOMERMIDDLE', 'CUSTOMERLAST', 'CUSTOMERNAME', 'CONTACTPHONE',
#        'CONTACTMOBILEPHONE', 'CONTACTEMAIL', 'IBICUSTOMERFIRST',
#        'IBICUSTOMERMI', 'IBICUSTOMERLAST', 'IBICUSTOMERNAME', 'IBIPHONE',
#        'IBIMOBILE', 'IBIEMAIL', 'CUSTOMERLABEL', 'BRANCHCODE', 'COUNTRY',
#        'AGENT_ID', 'AGENT_NAME', 'ASM_NAME', 'SITECOMPANYNAME',
#        'SITECOMPANY_OID', 'SITECOMPANYID', 'PRIMARYPRODUCTVERSION',
#        'CASENUM', 'PROJNUM', 'IWAYJIRA', 'PNOTARGET', 'PNOCRITICALUSER',
#        'ISPREMIUM', 'CUSTOMER_NAME', 'CUSTOMER_LABEL', 'GLOBAL_ID',
#        'CUSTOMER_PHASE', 'P1PHONE', 'SITEINSTALLYEAR','SECONDS_SINCE_CASE_START', 'SEVERITY', 'ISESCALATE',
#        'formerbugtimes','formercomtimes']
statistic_features=['milestones_size','milestones_unique','time_unique','milestone_times_sum','milestone_times_mean','milestone_times_std']
features_dict=pickle.load(open('middle/densefeature.pkl', 'rb'))
s=['MILESTONEDESCRIPTION'+f'b_svd_{i}' for i in range(20)]+['NOTEDESCRIPTION'+f'b_svd_{i}' for i in range(20)]
dense_features=features_dict['newone']+s#+features_dict['myself']
# orignal=['SECONDS_SINCE_CASE_START','INV_TIME_TO_NEXT_ESCALATION','milestone_SECONDS_SINCE_CASE_START','comments_SECONDS_SINCE_CASE_START',
# 'comments_CREATED_BY']
# for f in dense_features:
#     if f in orignal:
#         dense_features.remove(f)
# case_metadata=pd.read_csv("data/IBI_case_metadata_anonymized.csv")
# orignal=list(case_metadata.columns)
# for f in features_dict['myself']:
#     if f not in orignal:
#         dense_features.append(f)
dense_features.remove('REFERENCEID')

# dense_features.extend(statistic_features)
base_path1="word2vecf"
text_features=[      
    [base_path1+"/milestone_seq.256d",'milestone_seq',256],
    # [base_path1+"/CREATED_BY_seq.256d",'CREATED_BY_seq',256],
    [base_path1+"/UPDATED_BY_seq.256d",'UPDATED_BY_seq',256]
    ]
#定义用户点击的人工构造序列特征
text_features_1=[       
[base_path1+"/myself.100d",'milestone_seq',100]    
]


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=512)
    parser.add_argument('--max_len_text', type=int, default=128)
    parser.add_argument('--num_hidden_layers', type=int, default=6)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--vocab_size_v1', type=int, default=500000)
    parser.add_argument('--vocab_dim_v1', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--lr', type=float, default=8e-5)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--display_steps', type=int, default=100)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--eval_batch_size', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--num_label', type=int, default=1)    
   
    args = parser.parse_args()
    
    #设置参数
    args.hidden_size=sum([x[-1] for x in text_features])
    logger.info("Argument %s", args)    
    args.vocab=pickle.load(open(os.path.join(args.pretrained_model_path, "vocab.pkl"),'rb'))
    args.vocab_size_v1=len(args.vocab)
    args.text_features=text_features
    args.text_features_1=text_features_1
    args.dense_features=dense_features
    args.linear_layer_size=[1024,512]
    args.text_dim=sum([x[-1] for x in text_features])
    args.text_dim_1=sum([x[-1] for x in text_features_1])
    args.output_dir=args.output_path+"/index_{}".format(args.index)
    
    #读取word2vector模型
    args.embeddings_tables={}
    for x in args.text_features:
        if x[0] not in args.embeddings_tables:
            try:
                args.embeddings_tables[x[0]]=gensim.models.KeyedVectors.load_word2vec_format(x[0],binary=False)  
            except:
                args.embeddings_tables[x[0]]=pickle.load(open(x[0],'rb'))


    args.embeddings_tables_1={}
    for x in args.text_features_1:
        if x[0] not in args.embeddings_tables_1:
            try:
                args.embeddings_tables_1[x[0]]=gensim.models.KeyedVectors.load_word2vec_format(x[0],binary=False)  
            except:
                args.embeddings_tables_1[x[0]]=pickle.load(open(x[0],'rb'))
    
    #读取数据       
    train_df=pd.read_pickle('middle/train_all1.pkl')
    
    train_df['nescated']=train_df.apply(lambda x: 0 if x['INV_TIME_TO_NEXT_ESCALATION']==0 else 1,axis=1)

    # res_id=list(train_df[train_df['INV_TIME_TO_NEXT_ESCALATION']>0]['REFERENCEID'].unique())
    # train_df=train_df[train_df['REFERENCEID'].isin(res_id)]


    train_df=train_df[train_df['milestones_size']>0]
    negdf=train_df[train_df['INV_TIME_TO_NEXT_ESCALATION']>0]
    # train_df=train_df[train_df['INV_TIME_TO_NEXT_ESCALATION']>0]
    train_df=train_df.append(negdf)
    train_df=train_df.append(negdf)
    #想办法删掉一部分负样本



   
   
    test_df=pd.read_pickle('middle/test_all1.pkl')
    # train_df['decision_time']=train_df['decision_time']/(24*3600)
    # test_df['decision_time']=test_df['decision_time']/(24*3600)
    test_df['nescated']=0
    # if "REFERENCEID" in args.dense_features:
    #   args.dense_features.remove("REFERENCEID")
    # df=train_df[args.dense_features].append(test_df[args.dense_features])
    
    # ss=StandardScaler()
    # ss.fit(df[args.dense_features])
    # train_df[args.dense_features]=ss.transform(train_df[args.dense_features])
    # test_df[args.dense_features]=ss.transform(test_df[args.dense_features])
    # needcolums=[]
    # for c in args.dense_features:
    #   if train_df[c].isnull().any():
    #     needcolums.append(c)

    # del df
    # df=train_df[args.dense_features].append(test_df[args.dense_features])
    # for column in needcolums:
    #     mean_val = df[column].mean()
    #     train_df[column].fillna(mean_val, inplace=True)
    #     test_df[column].fillna(mean_val, inplace=True)
    # for c in args.dense_features:
    #   if test_df[c].isnull().any():
    #       mean_val = train_df[c].mean( )
    #       test_df[c].fillna(mean_val, inplace=True )
    # print("train has nan:",train_df[args.dense_features].isnull().any().any())
    # print("test has nan:",test_df[args.dense_features].isnull().any().any())
    test_dataset = TextDataset(args,test_df)    
    
    #建立模型
    skf=StratifiedKFold(n_splits=5,random_state=2020,shuffle=True)
    model=ctrNet.ctrNet(args)
    
    #训练模型
    for i,(train_index,test_index) in enumerate(skf.split(train_df,train_df['nescated'].values)):
        if i!=args.index:
            continue
        logger.info("Index: %s",args.index)
        train_dataset = TextDataset(args,train_df.iloc[train_index])
        dev_dataset=TextDataset(args,train_df.iloc[test_index])
        model.train(train_dataset,dev_dataset)
        dev_df=train_df.iloc[test_index]
    
    #输出结果
    accs=[]
    
    
    
    for f,num in [('escalated',2)]:
        model.reload(f)
        dev_preds=model.infer(dev_dataset)
        dev_df['{}_{}'.format(f,num)]=dev_preds
        acc=model.eval(dev_df['target'].values,dev_preds)['eval_R2']
        accs.append(acc)
        test_preds=model.infer(test_dataset)

        logger.info("Test %s %s",f,np.mean(test_preds,0))
        logger.info("R2 %s %s",f,round(acc,5))
        out_fs=['REFERENCEID','decision_time','predict_{}'.format(f)]
        test_df['predict_{}'.format(f)]=test_preds
        try:
            os.system("mkdir submission")
        except:
            pass

        test_df[out_fs].to_csv('result/submission_test_{}_{}_{}.csv'.format(f,args.index,round(acc,5)),index=False)
        
    logger.info("  best_r2 = %s",round(sum(accs),4))
   

