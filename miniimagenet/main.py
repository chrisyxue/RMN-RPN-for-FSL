# ——————————————————————————————————
# Author : Zhiyu Xue
# Reference :  Learning to Compare: Relation Network for Few-Shot Learning
# ——————————————————————————————————

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import task_generator as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats
from models import *
from utils import *

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5) # N-Way
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5) # N-shot
parser.add_argument("-b","--batch_num_per_class",type = int, default = 10) # b images per class
parser.add_argument("-e","--episode",type = int, default= 500000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-v","--length_of_vector",type=int,default=50)
args = parser.parse_args()

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
LEN_VEC = args.length_of_vector
VAL_EPISODE = 300



def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    # scipy.stats.sem() -> calculate the standard error
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def weights_init_kaiming(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv2d') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)

"""save data"""
def save_data(loss_list,acc_list,loss,acc):
    loss_list.append(loss)
    acc_list.append(acc)
    return loss_list,acc_list
 
def main():

    save_path = make_dir(way=CLASS_NUM,shot=SAMPLE_NUM_PER_CLASS,batch_size=BATCH_NUM_PER_CLASS)
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders,metatest_folders,metaval_folders = tg.mini_imagenet_folders()

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    RP = RP_Network(GPU=GPU)
    RM = RM_Network(64,HIDDEN_UNIT,GPU=GPU) 


    RP.apply(weights_init_kaiming)
    RM.apply(weights_init_kaiming)

    feature_encoder.cuda(GPU)
    RP.cuda(GPU)
    RM.cuda(GPU)
    
    # Implement Optimizer and Scheduler
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = ReduceLROnPlateau(feature_encoder_optim,mode="max",factor=0.5,patience=2,verbose=True)

    RP_optim = torch.optim.Adam(RP.parameters(),lr=LEARNING_RATE)
    RP_scheduler = ReduceLROnPlateau(RP_optim,mode="max",factor=0.5,patience=2,verbose=True)

    RM_optim = torch.optim.Adam(RM.parameters(),lr=LEARNING_RATE)
    RM_optim_scheduler = ReduceLROnPlateau(RM_optim,mode='max',factor=0.5,patience=2,verbose=True) 
    

    # somethings to save the data 
    loss_list = []
    acc_list = []
    episode_list = []

    print("Training...")

    last_accuracy = 0.0

    for episode in range(EPISODE):

        #feature_encoder_scheduler.step(episode)
        # RM_scheduler.step(episode)
        
        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
        batch_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True,train_query_argue=True)

        # sample datas
        samples,sample_labels = sample_dataloader.__iter__().next() #25*3*84*84
        batches,batch_labels = batch_dataloader.__iter__().next()

        # calculate features
        sample_features = feature_encoder(Variable(samples).cuda(GPU)) # 25*64*19*19
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,19,19)
        sample_features = torch.sum(sample_features,1).squeeze(1)
        batch_features = feature_encoder(Variable(batches).cuda(GPU)) # 20x64*5*5
        

        """--------------- Phrase of RPN ----------------------"""
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
        batch_features_ext = torch.transpose(batch_features_ext,0,1)
        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
        Att = RP(relation_pairs)
        """----------------------------------------------------------------"""

        
        """------------------Phrase of RMN-----------------------------------------"""
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        sample_features_ext = sample_features_ext.view(-1,FEATURE_DIM,19,19).contiguous() 
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1) 
        batch_features_ext = torch.transpose(batch_features_ext,0,1) 
        batch_features_ext = batch_features_ext.contiguous().view(-1,FEATURE_DIM,19,19) 
        
        batch_features_ext_att =  batch_features_ext + batch_features_ext*Att.expand_as(batch_features_ext)
        
        relations = RM(sample_features_ext,batch_features_ext_att).view(-1,CLASS_NUM)   
        """----------------------------------------------------------------"""
        
        # BP and Optimize
        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1).cuda(GPU))
        loss = mse(relations,one_hot_labels)


        # training
        feature_encoder.zero_grad()
        RP.zero_grad()
        RM.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(RP.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(RM.parameters(),0.5)

        feature_encoder_optim.step()
        RP_optim.step()
        RM_optim.step()


        if (episode+1)%100 == 0:
           print("episode:",episode+1,"loss",loss.item())

        # Validation
        if episode%5000 == 0:
            print("validation...")
            accuracies_val = []
            for i in range(VAL_EPISODE):
                total_rewards = 0
                task = tg.MiniImagenetTask(metaval_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,15)
                sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
                num_per_class = 5
                val_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=num_per_class,split="test",shuffle=False)

                sample_images,sample_labels = sample_dataloader.__iter__().next()
                for val_images,val_labels in val_dataloader:
                    val_images,val_labels = val_images.cuda(GPU),val_labels.cuda(GPU)
                    batch_size = val_labels.shape[0]
                    # calculate features
                    sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
                    sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,19,19)
                    sample_features = torch.sum(sample_features,1).squeeze(1)
                    val_features = feature_encoder(Variable(val_images).cuda(GPU)) # 20x64
                    
                    """---------------RPN----------------------"""
                    # calculate relations
                    # each batch sample link to every samples to calculate relations
                    # to form a 100x128 matrix for relation network
                    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)
                    val_features_ext = val_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)
                    val_features_ext = torch.transpose(val_features_ext,0,1)
                    relation_pairs = torch.cat((sample_features_ext,val_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
                    # relations =    RP(sample_features_ext,test_features_ext).view(-1,CLASS_NUM)
                    Att = RP(relation_pairs)
                    # print(Att)
                    """-----------------------------------------------"""
                    
                    """------------------RMN-----------------------------------------"""
                    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)
                    sample_features_ext = sample_features_ext.view(-1,FEATURE_DIM,19,19) 

                    val_features_ext = val_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)
                    val_features_ext = torch.transpose(val_features_ext,0,1)
                    val_features_ext = val_features_ext.contiguous().view(-1,FEATURE_DIM,19,19)

                    val_features_ext_att = val_features_ext + val_features_ext*Att.expand_as(val_features_ext)

                    relations = RM(sample_features_ext,val_features_ext_att).view(-1,CLASS_NUM)
                    """-----------------------------------------------"""
                    



                    _,predict_labels = torch.max(relations.data,1)

                    rewards = [1 if predict_labels[j]==val_labels[j] else 0 for j in range(batch_size)]
                    
                    total_rewards += np.sum(rewards)


                acc_val = total_rewards/1.0/CLASS_NUM/15
                accuracies_val.append(acc_val)

            val_accuracy,h = mean_confidence_interval(accuracies_val)
            feature_encoder_scheduler.step(val_accuracy)
            RP_scheduler.step(val_accuracy)
            RM_optim_scheduler.step(val_accuracy)
            print("Acc_Val",val_accuracy,'Episode',episode)


        if episode%500 == 0:

            # test
            print("Testing...")
            accuracies = []
            for i in range(TEST_EPISODE):
                total_rewards = 0
                task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,15)
                sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
                num_per_class = 5
                test_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=num_per_class,split="test",shuffle=False)

                sample_images,sample_labels = sample_dataloader.__iter__().next()
                for test_images,test_labels in test_dataloader:
                    test_images,test_labels = test_images.cuda(GPU),test_labels.cuda(GPU)
                    batch_size = test_labels.shape[0]
                    # calculate features
                    sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
                    sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,19,19)
                    sample_features = torch.sum(sample_features,1).squeeze(1)
                    test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 20x64

                        
                    """---------------RPN----------------------"""
                    # calculate relations
                    # each batch sample link to every samples to calculate relations
                    # to form a 100x128 matrix for relation network
                    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)
                    test_features_ext = test_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)
                    test_features_ext = torch.transpose(test_features_ext,0,1)
                    relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
                    # relations =    RP(sample_features_ext,test_features_ext).view(-1,CLASS_NUM)
                    Att = RP(relation_pairs)
                    """-----------------------------------------------"""
                    
                    """------------------RMN-----------------------------------------"""
                    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)
                    sample_features_ext = sample_features_ext.view(-1,FEATURE_DIM,19,19) 

                    test_features_ext = test_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)
                    test_features_ext = torch.transpose(test_features_ext,0,1)
                    test_features_ext = test_features_ext.contiguous().view(-1,FEATURE_DIM,19,19)

                    test_features_ext_att = test_features_ext + test_features_ext*Att.expand_as(test_features_ext)

                    relations = RM(sample_features_ext,test_features_ext_att).view(-1,CLASS_NUM)
                    """-----------------------------------------------"""

                    _,predict_labels = torch.max(relations.data,1)

                    rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(batch_size)]
                    
                    total_rewards += np.sum(rewards)


                accuracy = total_rewards/1.0/CLASS_NUM/15
                accuracies.append(accuracy)


            test_accuracy,h = mean_confidence_interval(accuracies)

            print("test accuracy:",test_accuracy,"h:",h)
            
            acc_list.append(test_accuracy)
            episode_list.append(episode)

            if test_accuracy > last_accuracy:

                # save networks
                torch.save(feature_encoder.state_dict(),save_path+"encoder.pkl")
                torch.save(RP.state_dict(),save_path+"RP.pkl")
                torch.save(RM.state_dict(),save_path+"RM.pkl")
                print("save networks for episode:",episode)

                last_accuracy = test_accuracy

        #save data
        np.savetxt(save_path+"acc.txt",acc_list)
        np.savetxt(save_path+"episode.txt",episode_list)  



if __name__ == '__main__':
    main()
