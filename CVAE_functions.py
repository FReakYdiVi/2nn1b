#importing necesaary libraries 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader 

import matplotlib.pyplot as plt
import numpy as np

#defining this to make seprate tensor for x and y
class SignalDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx][0] #load pair x0/x1
        y = self.data[idx][1]
        return x, y  

# define the loss function
def loss_function(recon_x, x, cond_data, mu, logvar, beta, wx, wy, fun_list):
    #defining Kullback-Leibler (KL) divergence loss (mu=z_mean1,logvar=z_logvar)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    
    
    # recon_loss_fn = torch.nn.L1Loss(reduction='mean')
    # recon_loss_fn = torch.nn.L1Loss(reduction='sum')
    
    #taking MSE as loss function for reconstruction loss 
    
    recon_loss_fn = torch.nn.MSELoss()
    x_loss  = recon_loss_fn(x, recon_x)
    
   
    
    # Calculate the next-wise-element functions in fun_list
    results_list = []
    x0 = recon_x[:,0,:,0].cpu().detach().numpy().flatten()
    x1 = recon_x[:,0,:,1].cpu().detach().numpy().flatten()
    
    #applying 
    for fun in fun_list:
        result = fun(x0, x1)
        results_list.append(result)
    
    Nw = recon_x.size(-2)
    #converting numpy array
    recon_cond_data = np.vstack([results_list]).T.reshape(len(cond_data), Nw*len(fun_list))
    #converting numpy array into pytorch tensors
    recon_cond_data = torch.Tensor(np.array(recon_cond_data)).type(torch.float)    
    # if torch.cuda.is_available():
    #     recon_cond_data = recon_cond_data.cuda()
    if torch.backends.mps.is_available():
        recon_cond_data = recon_cond_data.to(torch.device('mps'))


    y_loss =  recon_loss_fn(cond_data, recon_cond_data)
  
    #calculating total loss and assigning weights to differenty losses 
    total_loss = (beta * KLD + wx * x_loss + wy * y_loss).mean()
    
    return total_loss, KLD, x_loss, y_loss


def train_cvae_br(cvae, train_loader, optimizer, beta, wx, wy, epoch, fun_list, step_to_print=1):
    cvae.train()
    train_loss = 0.0
    KLD_loss = 0.0
    recon_loss = 0.0
    cond_loss = 0.0

    for batch_idx, (data, cond_data) in enumerate(train_loader):
        Nw = data.size(-2)
        cond_data = torch.reshape(cond_data, (len(cond_data), Nw * len(fun_list)))
        if torch.backends.mps.is_available():
            cond_data = cond_data.to(torch.device('mps'))
            data = data.to(torch.device('mps'))
        # if torch.cuda.is_available():
        #     cond_data = cond_data.cuda()
        #     data = data.cuda()

        # ===================forward=====================
        recon_data, z_mean, z_logvar = cvae(data, cond_data)
        loss, loss_KDL, loss_x, loss_y = loss_function(recon_data, data, cond_data, z_mean, z_logvar, beta, wx, wy,
                                                       fun_list)
        train_loss += loss.item()
        KLD_loss += loss_KDL.item()
        recon_loss += loss_x.item()
        cond_loss += loss_y.item()
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    KLD_loss /= len(train_loader)
    recon_loss /= len(train_loader)
    cond_loss /= len(train_loader)

    result_dict = {
        "epoch": epoch,
        "average_loss": train_loss,
        "KLD_loss": KLD_loss,
        "x_loss": recon_loss,
        "y_loss": cond_loss
    }

    if epoch % step_to_print == 0:
        print('Train Epoch {}: Average Loss: {:.6f}, KDL: {:3f}, x_loss: {:3f}, y_loss: {:3f}'.format(epoch, train_loss,
                                                                                                    KLD_loss,
                                                                                                    recon_loss, cond_loss))

    return result_dict
    
    

def test_cvae_br(cvae, test_loader, beta, wx, wy,fun_list):
    cvae.eval()
    test_loss = 0.0

    with torch.no_grad():
        
        for batch_idx, (data, cond_data) in enumerate(test_loader):
            Nw = data.size(-2)
            cond_data =  torch.reshape(cond_data, (len(cond_data), Nw* len(fun_list)))
            # if torch.cuda.is_available():
            #     cond_data =  cond_data.cuda()
            #     data = data.cuda()
            #     cond_data = cond_data.cuda()
            if torch.backends.mps.is_available():
                cond_data = cond_data.to(torch.device('mps'))
                data = data.to(torch.device('mps'))

            recon_data, z_mean, z_logvar = cvae(data, cond_data)

            loss,_,_,_ = loss_function(recon_data, data, cond_data, z_mean, z_logvar, beta, wx, wy, fun_list)

            test_loss += loss.item()

    test_loss /= len(test_loader)
    print('Test Loss: {:.6f}'.format(test_loss))
    return test_loss
    
    

    
def generate_samples_br(cvae, num_samples, given_y, input_shape, zmult = 1):
    
    cvae.eval()
    samples = []
    givens = []
    
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Generate random latent vector
            z_rand = (torch.randn(*input_shape)*zmult)
            # if torch.cuda.is_available():
            #     z_rand = z_rand.cuda()
            if torch.backends.mps.is_available():
                z_rand = z_rand.to(torch.device('mps'))
                
            num_args = cvae.encoder.forward.__code__.co_argcount
            if num_args > 2 :
                z = cvae.sampling(*cvae.encoder(z_rand.unsqueeze(0), given_y))
            else: 
                z = cvae.sampling(*cvae.encoder(z_rand.unsqueeze(0)))
            # Another way to generate random latent vector
            #z = torch.randn(1, latent_dim).cuda()
            
            # Set conditional data as one of the given y 
            # Generate sample from decoder under given_y
            sample = cvae.decoder(z, given_y)
            samples.append(sample)
            givens.append(given_y)

    
    samples = torch.cat(samples, dim=0)   
    givens = torch.cat(givens, dim=0) 
    return samples, givens





'''
These functions are defined for models that are passing condition both through Encoder and Decoder 
'''

def train_cvae(cvae, train_loader, optimizer, beta, wx, wy, epoch, fun_list, step_to_print=1):
    cvae.train()
    #intialsing all losses to zero
    train_loss = 0.0
    KLD_loss = 0.0
    recon_loss = 0.0
    cond_loss = 0.0

    for batch_idx, (data, cond_data) in enumerate(train_loader):
        Nw = data.size(-2) #nw=50
        cond_data_reshape = torch.reshape(cond_data, (len(cond_data), Nw * len(fun_list)))#reshaping the cond_data
        if torch.backends.mps.is_available():
            cond_data = cond_data.to(torch.device('mps'))
            data = data.to(torch.device('mps'))
        # if torch.cuda.is_available():
        #     cond_data = cond_data.cuda()
        #     data = data.cuda()

        # ===================forward=====================
        recon_data, z_mean, z_logvar = cvae(data, cond_data, cond_data_reshape)
        loss, loss_KDL, loss_x, loss_y = loss_function(recon_data, data, cond_data_reshape, z_mean, z_logvar, beta, wx, wy,
                                                       fun_list)
        train_loss += loss.item()
        KLD_loss += loss_KDL.item()
        recon_loss += loss_x.item()
        cond_loss += loss_y.item()
        # ===================backward=====================
        #optimizer to optimize losses to minimise
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ##adding up the loss 
    train_loss /= len(train_loader)
    KLD_loss /= len(train_loader)
    recon_loss /= len(train_loader)
    cond_loss /= len(train_loader)
     
    #storing all losses into a dictionary
    result_dict = {
        "epoch": epoch,
        "average_loss": train_loss,
        "KLD_loss": KLD_loss,
        "x_loss": recon_loss,
        "y_loss": cond_loss
    }
     
    #printing all losses
    if epoch % step_to_print == 0:
        print('Train Epoch {}: Average Loss: {:.6f}, KDL: {:3f}, x_loss: {:3f}, y_loss: {:3f}'.format(epoch, train_loss,
                                                                                                    KLD_loss,
                                                                                                    recon_loss, cond_loss))

    return result_dict
    
    

def test_cvae(cvae, test_loader, beta, wx, wy,fun_list):
    cvae.eval()
    #intailising value of test to be zero
    test_loss = 0.0

    with torch.no_grad():
        
        for batch_idx, (data, cond_data) in enumerate(test_loader):
            Nw = data.size(-2)
            cond_data_reshape =  torch.reshape(cond_data, (len(cond_data), Nw* len(fun_list))) #reshaping the cond_data
            # if torch.cuda.is_available():
            #     cond_data =  cond_data.cuda()
            #     data = data.cuda()
            #     cond_data = cond_data.cuda()
            if torch.backends.mps.is_available():
                cond_data = cond_data.to(torch.device('mps'))
                data = data.to(torch.device('mps'))

            recon_data, z_mean, z_logvar = cvae(data, cond_data, cond_data_reshape)

            loss,_,_,_ = loss_function(recon_data, data, cond_data_reshape, z_mean, z_logvar, beta, wx, wy, fun_list)

            test_loss += loss.item()

    test_loss /= len(test_loader)
    print('Test Loss: {:.6f}'.format(test_loss))
    return test_loss



#this function is defined to generate the x from given y 
def generate_samples(cvae, num_samples, given_y, given_y_shape, input_shape, zmult = 1):
    
    cvae.eval()
    #intialising list to store reconstructed x and y
    samples = []
    givens = []
    
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Generate random latent vector
            z_rand = (torch.randn(*input_shape)*zmult)
            # if torch.cuda.is_available():
            #     z_rand = z_rand.cuda()
            if torch.backends.mps.is_available():
                z_rand = z_rand.to(torch.device('mps'))
                
            num_args = cvae.encoder.forward.__code__.co_argcount
            if num_args > 2 :
                z = cvae.sampling(*cvae.encoder(z_rand.unsqueeze(0), given_y))
            else: 
                z = cvae.sampling(*cvae.encoder(z_rand.unsqueeze(0)))
            # Another way to generate random latent vector
            #z = torch.randn(1, latent_dim).cuda()
            
            # Set conditional data as one of the given y 
            # Generate sample from decoder under given_y
            sample = cvae.decoder(z, given_y_shape)
            samples.append(sample)
            givens.append(given_y)

    
    samples = torch.cat(samples, dim=0)   
    givens = torch.cat(givens, dim=0) 
    return samples, givens

'''
These functions are redefined beacuse there is change in model architecture as there is addition of one more encoder so all functions
are redefined to satisfy the architecture of model that is referenced to reaserch paper

'''


def loss_function_re(recon_x, x, cond_data, mu1, logvar1, mu2, logvar2, beta, wx, wy, fun_list):
    
    

    # KLD loss for z1 (latent variable from Encoder1)
    KLD_z1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())

    # KLD loss for z2 (latent variable from Encoder2)
    KLD_z2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())

    # KLD loss between z1 and z2
    KLD_latent = 0.5 * (torch.exp(logvar2 - logvar1) + (mu1 - mu2)**2 / torch.exp(logvar2) - 1 + logvar1 - logvar2).sum()

    # Total KLD loss is the sum of KLD_z1, KLD_z2, and KLD_latent
    KLD = KLD_z1 + KLD_z2 + KLD_latent

    
    # recon_loss_fn = torch.nn.L1Loss(reduction='mean')
    # recon_loss_fn = torch.nn.L1Loss(reduction='sum')
    recon_loss_fn = torch.nn.MSELoss()
    x_loss  = recon_loss_fn(x, recon_x)
   
    
    # Calculate the next-wise-element functions in fun_list
    results_list = []
    x0 = recon_x[:,0,:,0].cpu().detach().numpy().flatten()
    x1 = recon_x[:,0,:,1].cpu().detach().numpy().flatten()
    
    for fun in fun_list:
        result = fun(x0, x1)
        results_list.append(result)
    
    Nw = recon_x.size(-2)
    recon_cond_data = np.vstack([results_list]).T.reshape(len(cond_data), Nw*len(fun_list))
    recon_cond_data = torch.Tensor(np.array(recon_cond_data)).type(torch.float)    
    # if torch.cuda.is_available():
    #     recon_cond_data = recon_cond_data.cuda()
    if torch.backends.mps.is_available():
        recon_cond_data = recon_cond_data.to(torch.device('mps'))

    #now cross entropy loss is used to reconstruction for y_loss
    

    y_loss = recon_loss_fn(cond_data,recon_cond_data)

    #calculating total loss  
    total_loss = (beta * KLD + wx * x_loss + wy * y_loss).mean()
    
    return total_loss, KLD, x_loss, y_loss




def train_cvae_re(cvae, train_loader, optimizer, beta, wx, wy, epoch, fun_list, step_to_print=1):
    cvae.train()
    #intialising losses
    train_loss = 0.0
    KLD_loss = 0.0
    recon_loss = 0.0
    cond_loss = 0.0

    for batch_idx, (data, cond_data) in enumerate(train_loader):
        Nw = data.size(-2)
        cond_data = torch.reshape(cond_data, (len(cond_data), Nw * len(fun_list)))
        if torch.backends.mps.is_available():
            cond_data = cond_data.to(torch.device('mps'))
            data = data.to(torch.device('mps'))
        # if torch.cuda.is_available():
        #     cond_data = cond_data.cuda()
        #     data = data.cuda()

        # ===================forward=====================
        #
        recon_data, z_mean1, z_logvar1, z_mean2, z_logvar2 = cvae(data, cond_data)
        
        loss, loss_KDL, loss_x, loss_y = loss_function_re(recon_data, data, cond_data, z_mean1, z_logvar1, z_mean2, z_logvar2, beta, wx, wy,
                                                       fun_list)
        train_loss += loss.item()
        KLD_loss += loss_KDL.item()
        recon_loss += loss_x.item()
        cond_loss += loss_y.item()
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ## adding up the loss 
    train_loss /= len(train_loader)
    KLD_loss /= len(train_loader)
    recon_loss /= len(train_loader)
    cond_loss /= len(train_loader)
    ## storing the losses in dictionary 
    result_dict = {
        "epoch": epoch,
        "average_loss": train_loss,
        "KLD_loss": KLD_loss,
        "x_loss": recon_loss,
        "y_loss": cond_loss
    }

    if epoch % step_to_print == 0:
        print('Train Epoch {}: Average Loss: {:.6f}, KDL: {:3f}, x_loss: {:3f}, y_loss: {:3f}'.format(epoch, train_loss,
                                                                                                    KLD_loss,
                                                                                                    recon_loss, cond_loss))

    return result_dict
    
    

def test_cvae_re(cvae, test_loader, beta, wx, wy,fun_list):
    cvae.eval()
    test_loss = 0.0

    with torch.no_grad():
        
        for batch_idx, (data, cond_data) in enumerate(test_loader):
            Nw = data.size(-2)
            cond_data =  torch.reshape(cond_data, (len(cond_data), Nw* len(fun_list)))
            # if torch.cuda.is_available():
            #     cond_data =  cond_data.cuda()
            #     data = data.cuda()
            #     cond_data = cond_data.cuda()
            if torch.backends.mps.is_available():
                cond_data = cond_data.to(torch.device('mps'))
                data = data.to(torch.device('mps'))

            # recon_data, z_mean, z_logvar = cvae(data, cond_data)
            recon_data, z_mean1, z_logvar1, z_mean2, z_logvar2 = cvae(data, cond_data)

            # loss,_,_,_ = loss_function(recon_data, data, cond_data, z_mean, z_logvar, beta, wx, wy, fun_list)
            loss,_,_,_ = loss_function_re(recon_data, data, cond_data, z_mean1, z_logvar1, z_mean2, z_logvar2, beta, wx, wy, fun_list)


            test_loss += loss.item()

    test_loss /= len(test_loader)
    print('Test Loss: {:.6f}'.format(test_loss))
    return test_loss
    
    

#there is modification that now y will be passing througfh encoder1   
def generate_samples_re(cvae, num_samples, given_y, input_shape, zmult = 1):
    
    cvae.eval()
    samples = []
    givens = []
    
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Generate random latent vector
            z_rand = (torch.randn(*input_shape)*zmult)
            # if torch.cuda.is_available():
            #     z_rand = z_rand.cuda()
            if torch.backends.mps.is_available():
                z_rand = z_rand.to(torch.device('mps'))
                
            num_args = cvae.encoder1.forward.__code__.co_argcount
            if num_args > 2 :
                z = cvae.sampling(*cvae.encoder1(z_rand.unsqueeze(0), given_y))
            else: 
                # z = cvae.sampling(*cvae.encoder1(z_rand.unsqueeze(0)))
                z = cvae.sampling(*cvae.encoder1(given_y))
            # Another way to generate random latent vector
            #z = torch.randn(1, latent_dim).cuda()
            
            # Set conditional data as one of the given y 
            # Generate sample from decoder under given_y
            sample = cvae.decoder(z, given_y)
            samples.append(sample)
            givens.append(given_y)

    
    samples = torch.cat(samples, dim=0)   
    givens = torch.cat(givens, dim=0) 
    return samples, givens
    



def plot_samples(x, y, num_samples , n_cols = 10, fig_size = 2): 
    # Truncate the samples to the desired number
    x = x[0:num_samples]
    y = y[0:num_samples]
    # Calculate the number of rows needed for the grid
    n_rows = round(len(x)/n_cols)
    # Create subplots with the specified number of columns and rows
    plt.rcdefaults()
    f, axarr = plt.subplots(n_rows, n_cols, figsize=(1.25*n_cols*fig_size, n_rows*fig_size))

     # Iterate over each subplot
    for j, ax in enumerate(axarr.flat):
        # Extract x and y data for the current sample
        x0 = x[j,0,:,0].cpu().detach().numpy().flatten()
        x1 = x[j,0,:,1].cpu().detach().numpy().flatten()
        y0 = y[j,0,:,0].cpu().detach().numpy().flatten()
        y1 = y[j,0,:,1].cpu().detach().numpy().flatten()
        
        
        #y_gen = x0*x1
         # Plot the data
        ax.plot(range(50),x0)
        ax.plot(range(50),x1)
        #ax.plot(range(50),y_gen)
        ax.plot(range(50),y0, color = 'r', linestyle = 'dotted')
        ax.plot(range(50),y1, color = 'b', linestyle = 'dotted') 
        
         # Set axis ticks to empty
        ax.set_xticks([])
        ax.set_yticks([])
        
# Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show() 
    
    

def plot_samples_stacked(x_given, x, y, fun_list, n_cols = 4, fig_size = 3): 
   # Set default plotting parameters
    plt.rcdefaults()
    # Determine the number of X and Y samples
    x_num = x.size(-1)
    y_num = y.size(-1)
    n_cols = x_num + y_num 
    
    # Create subplots with the specified number of columns
    f, axs = plt.subplots(1, n_cols, figsize=(1.25*n_cols*fig_size, fig_size))
     
    # Iterate over each sample
    for j in range(len(x)):
        # Plot each X sample
        for i in range(x_num + y_num):
            if i < x_num :
                x_i = x[j,0,:,i].cpu().detach().numpy().flatten()
                x_i_given = x_given[:,0,:,i].cpu().detach().numpy().flatten()
                axs[i].plot(range(50), x_i)
                axs[i].plot(range(50), x_i_given, color = 'r')
                axs[i].set_title(f'X{i}') 
                axs[i].set_ylim(0,1)
            else: 
                
                y0 = y[j,0,:,i-x_num].cpu().detach().numpy().flatten()
                
                axs[i].plot(range(50), y0, color = 'r')
                axs[i].set_title(f'Y{i-x_num}') 
                x0 = x[j,0,:,0].cpu().detach().numpy().flatten()
                x1 = x[j,0,:,1].cpu().detach().numpy().flatten()
                fun = fun_list[i-x_num]
                ygen = fun(x0,x1)
                axs[i].plot(range(50),ygen, color = 'g', linestyle = 'dotted')
                    
            # Remove ticks from the axes
            axs[i].set(xticks=[], yticks=[])    
    
    
    # Show the plots
    plt.show()           