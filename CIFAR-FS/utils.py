# ——————————————————————————————————
# Author : Zhiyu Xue
# Reference :  Learning to Compare: Relation Network for Few-Shot Learning
# ——————————————————————————————————

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as F
from torch.nn import init
import math
import argparse
import scipy as sp
import scipy.stats
from models import *
import os

def make_dir(way,shot,batch_size):
    name = "cifar_" + str(way) + "_way_" + str(shot) + "_shot_" +str(batch_size) + "_batch"
    if os.path.exists("results/"+ name + "/") == False:
        os.mkdir("results/"+ name + "/")
    path = "results/"+ name + "/"
    return path
    

"Initialization"
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

