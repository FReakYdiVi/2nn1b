import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader 

import numpy as np
import matplotlib.pyplot as plt

# Load the config YAML file
import yaml
with open("config_model.yaml", "r") as file:
    config = yaml.safe_load(file)

# Extract the parameters from the config
number_of_points = config["number_of_points"]
number_of_functions = config["number_of_functions"]

bias = config["bias"]
in_channels1 = config["in_channels1"]
out_channels1 = config["out_channels1"]
kernel_size1 = config["kernel_size1"]
out_channels2 = config["out_channels2"]
kernel_size2 = config["kernel_size2"]
#latent_dim = config["latent_dim"]

# define the input dimensions
input_shape = (1, number_of_points, 2)
dimInt = (number_of_points-kernel_size1[0]+1-kernel_size2[0]+1)

'''
there is modification in the given model because the main purpose behind making CVAE model is to make condition applied to both encoder
and decoder but previously condition is passing only through  decoder not encoder so there should be concatenation od condition in the
encoder part but with original shape not reshaped data because the x has dimensions has 4

'''

# define the encoder network
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        #defining the convulution layers with input channels=2 as we had concatenated the x and cond_data along dim=1
        self.conv1 = nn.Conv2d(2, out_channels1, kernel_size=kernel_size1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size=kernel_size2, bias=bias)
        self.fc1_mean = nn.Linear(out_channels2 * dimInt, latent_dim)
        self.fc1_logvar = nn.Linear(out_channels2 * dimInt, latent_dim)
        

    def forward(self, x , cond_data):
        #concatenating the x and cond_data along dim=1
        x=torch.cat([x,cond_data],dim=1)

        #appling activation function=relu
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))
        
        x = x.view(x.size(0), -1)  # Flatten the feature map to pass through linear layer 

        #defining z_mean and z_logvar to pass through sampling layer
        z_mean = self.fc1_mean(x)
        z_logvar = self.fc1_logvar(x)
        
        return z_mean, z_logvar

# define the sampling layer
class Sampling(nn.Module):  #The same approach as reparametrization trick
    def __init__(self):
        super(Sampling, self).__init__()

    def forward(self, z_mean, z_logvar):
        batch_size, latent_dim = z_mean.size()
        epsilon = torch.randn(batch_size, latent_dim).to(z_mean.device)
        std = torch.exp(0.5 * z_logvar)
        z = z_mean + std * epsilon
        return z

# define the decoder network
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
       #defining the linear layers to input output features of encoder part
        self.fc1 = nn.LazyLinear(out_channels2 * dimInt)
        #defining the convulution transpose due to resizing the input to its original shape
        self.conv_transpose1 = nn.ConvTranspose2d(out_channels2, out_channels1, kernel_size=kernel_size2, bias=bias)
        self.conv_transpose2 = nn.ConvTranspose2d(out_channels1, in_channels1, kernel_size=kernel_size1, bias=bias)
        #appling Leakyrelu as activation function as it is more useful in activating neurons than ReLU
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, z, cond):
        #concatenating the z and cond along dim=1
        x = torch.cat([z, cond], dim=1)
        x = F.relu(self.fc1(x))
        batch_size, latent_dim = z.size()
        #reshaping to its original size
        x = torch.reshape(x, (batch_size, out_channels2, dimInt, 1))  # Reshape to (batch_size, channels, height, width)
        
        #passing through convulution networks
        x = self.conv_transpose1(x) 
        x = self.conv_transpose2(x)
        
        x = self.relu(x)
        return x

# define the CVAE model
class CVAE(nn.Module):
    def __init__(self,latent_dim = 8):
        super(CVAE, self).__init__()
        
        self.encoder = Encoder(latent_dim)
        self.sampling = Sampling()
        self.decoder = Decoder(latent_dim)

    def forward(self, x, cond_data , cond):
        z_mean, z_logvar = self.encoder(x,cond_data)
        z = self.sampling(z_mean, z_logvar)
        x_recon = self.decoder(z, cond)
        return x_recon, z_mean, z_logvar

'''
In this Model there is addition of one more Convulution layer and this is just experiment and main motive behind this model is
to increase the number of input features 
'''

# class Encoder(nn.Module):
#     def __init__(self,latent_dim):
#         super(Encoder,self).__init__()

#         self.conv1=nn.Conv2d(2,16,kernel_size=[8,2],bias=True)
#         self.bn1=nn.BatchNorm2d(16) #applying batch normalization to avoid overfitting 
        
#         self.conv2=nn.Conv2d(16,32,kernel_size=[8,1],bias=True)
#         self.bn2=nn.BatchNorm2d(32)

#         # this is one more addition to the existing network
#         self.conv3=nn.Conv2d(32,64,kernel_size=[3,1],bias=True)
#         self.bn3=nn.BatchNorm2d(64)
        
#         self.fc1_mean=nn.Linear(64*34,latent_dim)
#         self.fc2_logvar=nn.Linear(64*34,latent_dim)

#         self.leaky_relu=nn.LeakyReLU(0.01)

#     def forward(self,x,cond_data):
#         #concatenating x and cond_data
#         x=torch.cat([x,cond_data],dim=1)
#         #passing through convulution network 
#         x=self.leaky_relu(self.bn1(self.conv1(x)))
#         x=self.leaky_relu(self.bn2(self.conv2(x)))
#         x=self.leaky_relu(self.bn3(self.conv3(x)))
        
#         #flatten x to pass through linear layer
#         x = x.view(x.size(0), -1)
#         #definig z_mean and z_logvar
#         z_mean = self.fc1_mean(x)
#         z_logvar = self.fc2_logvar(x)

#         return z_mean,z_logvar
    
# class Sampling(nn.Module):  #The same approach as reparametrization trick
#     def __init__(self):
#         super(Sampling, self).__init__()

#     def forward(self, z_mean, z_logvar):
#         batch_size, latent_dim = z_mean.size()
#         epsilon = torch.randn(batch_size, latent_dim).to(z_mean.device)
#         std = torch.exp(0.5 * z_logvar)
#         z = z_mean + std * epsilon
#         return z
    

# class Decoder(nn.Module):
#     def __init__(self, latent_dim):
#         super(Decoder,self).__init__()
#         #defining layers 
#         self.fc1=nn.LazyLinear(64*34)
#         self.conv_transpose1=nn.ConvTranspose2d(64,32,kernel_size=[3,1],bias=True)
#         self.bn1=nn.BatchNorm2d(32)
        
#         self.conv_transpose2=nn.ConvTranspose2d(32,16,kernel_size=[8,1],bias=True)
#         self.bn2=nn.BatchNorm2d(16)
       
#         self.conv_transpose3=nn.ConvTranspose2d(16,1,kernel_size=[8,2],bias=True)
#         self.relu=nn.LeakyReLU(0.01)

#     def forward(self, z, cond):
#         x = torch.cat([z, cond], dim=1)
#         x = F.relu(self.fc1(x))
#         batch_size, latent_dim = z.size()
        
#         x = torch.reshape(x, (batch_size, 64, 34, 1))  # Reshape to (batch_size, channels, height, width)
        
#         x=self.bn1(self.conv_transpose1(x))
#         x=self.bn2(self.conv_transpose2(x))
#         x=self.conv_transpose3(x)
        
#         x = self.relu(x)
#         return x
    

# class CVAE_ex(nn.Module):
#     def __init__(self,latent_dim):
#         super(CVAE_ex, self).__init__()
        
#         self.encoder = Encoder(latent_dim)
#         self.sampling = Sampling()
#         self.decoder = Decoder(latent_dim)

#     def forward(self, x, cond_data,cond):
#         z_mean, z_logvar = self.encoder(x,cond_data)
#         z = self.sampling(z_mean, z_logvar)
#         x_recon = self.decoder(z, cond)
#         return x_recon, z_mean, z_logvar

'''
This model is based on reasearch paper included in README
'''

class Encoder1(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder1, self).__init__()
        
        self.cond_dim = 100  #cond_dim=1*50*2=100 
        self.fc1 = nn.Linear(self.cond_dim, 128)  
        # Second fully connected layer
        self.fc2 = nn.Linear(128, 128) 
        # Fully connected layers for mean and log variance of z1
        self.fc3_mean = nn.Linear(128, latent_dim)
        self.fc3_logvar = nn.Linear(128, latent_dim)

    def forward(self, cond):
        # Process cond through fully connected layers
        h = F.relu(self.fc1(cond))
        h = F.relu(self.fc2(h))
        z1_mean = self.fc3_mean(h)
        z1_logvar = self.fc3_logvar(h)
        return z1_mean, z1_logvar


class Encoder2(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder2, self).__init__()
 
        # Infer cond_dim from the number of functions in the config
        # self.cond_dim = config["number_of_functions"]
        self.cond_dim = 100
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels1, out_channels1, kernel_size=kernel_size1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size=kernel_size2, bias=bias)
        
        # Calculate the size of the feature map after the convolutions
        self.conv_output_size = out_channels2 * dimInt
        
        # The size of the input to the first fully connected layer
        self.fc_input_size = self.conv_output_size + self.cond_dim
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 100)  # Example dimension from the diagram
        self.fc2_mean = nn.Linear(100, latent_dim)
        self.fc2_logvar = nn.Linear(100, latent_dim)

    def forward(self, x, cond):
        # Process x through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten the output of the convolutional layers
        x = x.view(x.size(0), -1)
        
        # Concatenate the flattened feature map with the condition (cond)
        x = torch.cat((x, cond), dim=1)
        
        # Process the concatenated vector through fully connected layers
        x = F.relu(self.fc1(x))
        z2_mean = self.fc2_mean(x)
        z2_logvar = self.fc2_logvar(x)
        
        return z2_mean, z2_logvar


# define the sampling layer
class Sampling(nn.Module):  #The same approach as reparametrization trick
    def __init__(self):
        super(Sampling, self).__init__()

    def forward(self, z_mean, z_logvar):
        batch_size, latent_dim = z_mean.size()
        epsilon = torch.randn(batch_size, latent_dim).to(z_mean.device)
        std = torch.exp(0.5 * z_logvar)
        z = z_mean + std * epsilon
        return z

# define the decoder network
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        #defining the layers 
        self.fc1 = nn.LazyLinear(out_channels2 * dimInt)
        self.conv_transpose1 = nn.ConvTranspose2d(out_channels2, out_channels1, kernel_size=kernel_size2, bias=bias)
        self.conv_transpose2 = nn.ConvTranspose2d(out_channels1, in_channels1, kernel_size=kernel_size1, bias=bias)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        x = F.relu(self.fc1(x))
        batch_size, latent_dim = z.size()
        
        x = torch.reshape(x, (batch_size, out_channels2, dimInt, 1))  # Reshape to (batch_size, channels, height, width)
        
        x = self.conv_transpose1(x)
        
        x = self.conv_transpose2(x)
        
        x = self.relu(x)
        return x

# define the CVAE model
class CVAE_re(nn.Module):
    def __init__(self,latent_dim = 8):
        super(CVAE_re, self).__init__()
        
        self.encoder1 = Encoder1(latent_dim)
        self.encoder2 = Encoder2(latent_dim)
        self.sampling = Sampling()
        self.decoder = Decoder(latent_dim)

    def forward(self, x, cond):
        z_mean1, z_logvar1 = self.encoder1(cond)
        z_mean2, z_logvar2 = self.encoder2(x, cond)
        z1 = self.sampling(z_mean1, z_logvar1)
        z2 = self.sampling(z_mean2, z_logvar2)
        x_recon = self.decoder(z2, cond)
        return x_recon, z_mean1, z_logvar1, z_mean2, z_logvar2
    



