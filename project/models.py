import os
import torch
import numpy as np
import netCDF4 as nc
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from datetime import datetime
import torch.optim as optim
import dataset as windData
from tqdm import tqdm
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time): # idk if they use the same embedding in the paper's model
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings
    
    def plot(self, T):
        all_t = torch.arange(1, T+1)
        test_embed = SinusoidalPositionEmbeddings(dim=32)
        output = test_embed(all_t)

        plt.figure(figsize=(10,4))
        plt.imshow(output, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title('Time embedding Visualization')
        plt.ylabel('Time step')
        plt.show()

### Rajouter des Attention layer comme dans le paper (test)
# Faudrait peut-être mettre aussi, un attention sur les channels jsp

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        # Average and Max pooling features
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]

        # Concatenate along channel dimension
        concat = torch.cat((avg_out, max_out), dim=1)

        # Apply convolution to create attention map
        attention = torch.sigmoid(self.conv(concat))

        # Multiply attention map with input features
        return x * attention


class DownBlock(nn.Module):
    '''Downsampling block used to build Unet'''
    def __init__(self, in_ch, out_ch, ks, drop_p, time_emb_dim, attention=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, ks, padding='same')
        self.conv2 = nn.Conv2d(out_ch, out_ch, ks, padding='same')

        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)

        self.drop = nn.Dropout(p=drop_p)

        self.relu = nn.ReLU()

        self.avgpool = nn.AvgPool2d(kernel_size=6, padding=2, stride=2)

        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        
        self.attention = False
        if attention:
            self.attention = True
            #self.channel_attention = ChannelAttention(out_ch)
            self.spatial_attention = SpatialAttention()
        

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))

        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb

        h = self.bnorm2(self.relu(self.conv2(h)))
        
        if self.attention:
            #h = self.channel_attention(h)
            h = self.spatial_attention(h)

        h = self.avgpool(h)

        h = self.drop(h)     # not sure we wanna use dropout

        return h
    
class UpBlock(nn.Module):
    '''Upampling block used to build Unet'''
    def __init__(self, in_ch, out_ch, ks, drop_p, time_emb_dim, shape, attention=False):
        super().__init__()

        self.upsamp = nn.Upsample(size=shape, mode='bilinear')

        # 2*in_ch because residual connection
        self.conv1 = nn.Conv2d(2*in_ch, out_ch, ks, padding='same')
        self.conv2 = nn.Conv2d(out_ch, out_ch, ks, padding='same')

        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)

        self.drop = nn.Dropout(p=drop_p)

        self.relu = nn.ReLU()

        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        
        self.attention = False
        if attention:
            self.attention = True
            #self.channel_attention = ChannelAttention(out_ch)
            self.spatial_attention = SpatialAttention()


    def forward(self, x, t):
        h = self.upsamp(x)

        h = self.drop(h)     # not sure we wanna use dropout

        h = self.bnorm1(self.relu(self.conv1(h)))

        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb

        h = self.bnorm2(self.relu(self.conv2(h)))
        
        if self.attention:
            #h = self.channel_attention(h)
            h = self.spatial_attention(h)

        return h
    

class Bottleneck(nn.Module):
    '''Bottleneck block used to build Unet'''
    def __init__(self, in_ch, out_ch, ks, time_emb_dim, attention=False):
        super().__init__()

        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)

        self.conv1 = nn.Conv2d(in_ch, out_ch, ks, padding='same')
        self.conv2 = nn.Conv2d(out_ch, out_ch, ks, padding='same')

        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        
        self.attention = False
        if attention:
            self.attention = True
            #self.channel_attention = ChannelAttention(out_ch)
            self.spatial_attention = SpatialAttention()

        self.relu = nn.ReLU()


    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))

        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb

        h = self.bnorm2(self.relu(self.conv2(h)))
        
        if self.attention:
            #h = self.channel_attention(h)
            h = self.spatial_attention(h)

        return h
    

class DenoisingUnet(nn.Module):
    '''Unet architecture denoising model'''
    def __init__(self, nbr_channels, input_channels, output_channels, kernel_sizes, time_emb_dim, shape, dropout=0.1, attention=False):
        super().__init__()
        self.nbr_channels = nbr_channels
        self.input_channels = input_channels
        self.dropout = dropout
        self.time_emb_dim = time_emb_dim
        self.dropout = dropout
        self.attention = attention

        self.down_channels = tuple(nbr_channels)
        self.up_channels = tuple(nbr_channels[::-1])

        # Not sure this is the right way to combine the blocks but I think so because
        # they say they have 4 blocks on each side and they give 4 different numbers of
        # channels

        self.down0 = DownBlock(
            input_channels, 
            self.down_channels[0], 
            kernel_sizes[0],
            dropout, 
            time_emb_dim,
            attention=attention
            )

        dbList = []
        for i in range(len(self.down_channels)-1):
            dbList.append(DownBlock(
                                    self.down_channels[i],
                                    self.down_channels[i+1], 
                                    kernel_sizes[i+1],
                                    dropout, 
                                    time_emb_dim
                                    ))

        self.downs = nn.ModuleList(dbList)

        self.bottleneck = Bottleneck(
                                self.down_channels[-1], 
                                self.down_channels[-1], 
                                kernel_sizes[-1],
                                time_emb_dim, 
                                attention=attention)

        shapes = []
        for i in range(len(self.up_channels)):
            shapes.append((shape[0]//2 **(len(nbr_channels)-i-1), shape[1]//2 **(len(nbr_channels)-i-1)))

        upList = []
        for i in range(len(self.up_channels)-1):
            upList.append(UpBlock(
                                    self.up_channels[i],
                                    self.up_channels[i+1], 
                                    kernel_sizes[i],
                                    dropout, 
                                    time_emb_dim, 
                                    shapes[i], 
                                    attention=attention
                                    ))

        self.ups = nn.ModuleList(upList)

        self.output = UpBlock(
            self.up_channels[-1], 
            output_channels, 
            kernel_sizes[-1],
            dropout, 
            time_emb_dim, 
            shapes[-1],
            attention=attention
            )

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )

    def forward(self, x, timestep):
        # Embed time
        t = self.time_mlp(timestep)

        residuals = []

        x = self.down0(x, t)

        residuals.append(x)

        for down in self.downs:
            x = down(x, t)
            residuals.append(x)

        x = self.bottleneck(x, t)

        for up in self.ups:
            residual_x = residuals.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        residual_x = residuals.pop()
        x = torch.cat((x, residual_x), dim=1)
        x = self.output(x, t)

        return x
    

def callback_lr_wd(optimizer, epoch, num_epochs):
    '''Callback function that adjusts both the learning rate and weight decay'''
    lr = 1e-04 * (0.1 ** (epoch / num_epochs)) # Learning rate decay from 1e-04 to 1e-05

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#############################################
# UNet model from the youtube tutorial
#############################################
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.2, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        self.spatial_attention = SpatialAttention()

        # self.drop = nn.Dropout(p=dropout)

    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        h = self.spatial_attention(h)
        # Down or Upsample
        h = self.transform(h)

        # h = self.drop(h)
        return h

class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, unet_channels=(64, 128, 256, 512, 1024), input_channels=3, output_channels=3, time_emb_dim=32, dropout=0.2):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.down_channels = unet_channels
        self.up_channels = unet_channels[::-1]
        self.time_emb_dim = time_emb_dim
        self.dropout = dropout

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim),
                nn.ReLU()
            )

        # Initial projection
        self.conv0 = nn.Conv2d(self.input_channels, self.down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(self.down_channels[i], self.down_channels[i+1], \
                                    self.time_emb_dim, dropout=self.dropout) \
                    for i in range(len(self.down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(self.up_channels[i], self.up_channels[i+1], \
                                        self.time_emb_dim, dropout=self.dropout, up=True) \
                    for i in range(len(self.up_channels)-1)])

        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(self.up_channels[-1], self.output_channels, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)
    

