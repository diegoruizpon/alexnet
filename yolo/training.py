import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from yolo import yolo


class yolo_loss(nn.Module):
    
    def __init__(self, l_coord = 5, l_noobj = .5 ):
        super(yolo_loss, self).__init__()

        # structure of output

        
        self.l_coord = l_coord
        self.l_noobj = l_noobj

        self.coord_loss_1 = nn.MSELoss()
        



    def forward(self, predictions, targets):
        
        return predictions