"""All functions and modules related to model definition."""
import torch
import torch.nn as nn
import logging
import numpy as np
from model_code.unet import UNetModel
from model_code import torch_dct
import math
from torchvision.transforms import functional as F
from diffusers import DDIMScheduler


class DCTBlur(nn.Module):

    def __init__(self, blur_sigmas, image_size, device):
        super(DCTBlur, self).__init__()
        self.blur_sigmas = torch.tensor(blur_sigmas).to(device)
        freqs = np.pi*torch.linspace(0, image_size-1,
                                     image_size).to(device)/image_size
        self.frequencies_squared = freqs[:, None]**2 + freqs[None, :]**2

    def forward(self, x, fwd_steps):
        if len(x.shape) == 4:
            sigmas = self.blur_sigmas[fwd_steps][:, None, None, None]
        elif len(x.shape) == 3:
            sigmas = self.blur_sigmas[fwd_steps][:, None, None]
        t = sigmas**2/2
        dct_coefs = torch_dct.dct_2d(x, norm='ortho')
        dct_coefs = dct_coefs * torch.exp(- self.frequencies_squared * t)
        return torch_dct.idct_2d(dct_coefs, norm='ortho')


def create_forward_process_from_sigmas(config, sigmas, device):
    forward_process_module = DCTBlur(sigmas, config.data.image_size, device)
    return forward_process_module

"""
To create a new forward process create an auxiliary function and then a final create_forward_process like function as above. The latter should take config and device 
as inputs. Config has details such as size of image and device ensures that operations take place in one device whether cpu or gpu. 

Sample details are below

Keep same arguments for forward process as below.



"""

class IdentityMap(nn.Module):
    def __init__(self,image_size,device):

        """Identity map forward process"""
        super(IdentityMap, self).__init__()
        self.image_size = image_size
        self.T = torch.eye(image_size).to(device)

    def forward(self, x, fwd_steps):
        return torch.matmul(x, self.T)
    
def create_forward_process_identity(config, device):
    forward_process_module = IdentityMap(config.data.image_size, device)
    return forward_process_module


class FixedRandomMap(nn.Module):
    def __init__(self,image_size,K,device):

        """Fixed random map forward process
        K : number of timesteps
        image_size : size of image
        device : cpu or gpu

        Here for each time step we multiply with a fixed random matrix for all data points (otherwise T is non deterministic)

        An alternative could be to loop through each time step and multiply with a random matrix for each timestep. That paramaterization is below
        """
        super(FixedRandomMap, self).__init__()
        self.image_size = image_size
        self.timesteps = K 
        self.T = [torch.randn(image_size,image_size).to(device) for i in range(self.timesteps)]

    def forward(self, x, fwd_steps):
        
        return torch.cat([torch.matmul(self.T[int(fwd_steps[i]-1)],x[i]).unsqueeze(0).to(x.device) for i in range(len(fwd_steps))])
    
def create_forward_process_fixed_random(config, device):
    forward_process_module = FixedRandomMap(config.data.image_size, config.model.K, device)
    return forward_process_module

class FixedRandomMap_alt(nn.Module):
    def __init__(self,image_size,K,device):

        """Fixed random map forward process alternate
        K : number of timesteps
        image_size : size of image
        device : cpu or gpu

        Retuns nan loss. Not yet resolved.


        """
        super(FixedRandomMap_alt, self).__init__()
        self.image_size = image_size
        self.timesteps = K 
        self.T = [(torch.randn(image_size,image_size).to(device) > 0).type(torch.uint8) for i in range(self.timesteps)]

    def forward(self, x, fwd_steps):

        for i in range(len(fwd_steps)):
            for j in range(int(fwd_steps[i])):

                x[i] = torch.matmul(self.T[j].float(),x[i])
        return x
    
def create_forward_process_fixed_random_alt(config, device):
    forward_process_module = FixedRandomMap_alt(config.data.image_size, config.model.K, device)
    return forward_process_module


class Ramp_Map(nn.Module):
    def __init__(self,image_size,K,device):
        super(Ramp_Map, self).__init__()
        self.image_size = image_size
        self.timesteps = K
        #Add adjustment to ensure image does not explode or vanish
        c = 0.001/K
        self.T = [-i * c*torch.eye(image_size).to(device) for i in range(self.timesteps)]

    def forward(self, x, fwd_steps):
        for i in range(len(fwd_steps)):

            x[i] = x[i] -self.T[int(fwd_steps[i]-1)]*torch.eye(self.image_size).to(x.device)
        return x
        #return torch.cat([torch.matmul(self.T[int(fwd_steps[i]-1)],x[i]).unsqueeze(0).to(x.device) for i in range(len(fwd_steps))])
    
def create_forward_process_ramp(config, device):
    forward_process_module = Ramp_Map(config.data.image_size, config.model.K, device)
    return forward_process_module


class Decelerating_Ramp_Map(nn.Module):
    def __init__(self,image_size,K,device):
        super(Decelerating_Ramp_Map, self).__init__()
        self.image_size = image_size
        self.timesteps = K
        c = 0.001/K
        #Kinematics equation for deceleration
        
        self.T = [(-i * c + c*math.pow(i,2)/(2*K*K))*torch.eye(image_size).to(device) for i in range(self.timesteps)]
    def forward(self, x, fwd_steps):
        for i in range(len(fwd_steps)):

            x[i] = x[i] -self.T[int(fwd_steps[i]-1)]*torch.eye(self.image_size).to(x.device)
        
        return x
        
    
def create_forward_process_decelerating_ramp(config, device):
    forward_process_module = Decelerating_Ramp_Map(config.data.image_size, config.model.K, device)
    return forward_process_module


class Decelerating_Ramp_Map_Reversal(nn.Module):
    def __init__(self,image_size,K,device):
        super(Decelerating_Ramp_Map_Reversal, self).__init__()
        self.image_size = image_size
        self.timesteps = K
        #Kinematics equation for deceleration with reversal
        c = 0.01/K

        
        self.T = [(-i * c + c*math.pow(i,2)/(K*K))*torch.eye(image_size).to(device) for i in range(self.timesteps)]
    def forward(self, x, fwd_steps):
        for i in range(len(fwd_steps)):

            x[i] = x[i] -self.T[int(fwd_steps[i]-1)]*torch.eye(self.image_size).to(x.device)
        
        return x
    
def create_forward_process_decelerating_ramp_reversal(config, device):
    forward_process_module = Decelerating_Ramp_Map_Reversal(config.data.image_size, config.model.K, device)
    return forward_process_module


class Darkening_Map(nn.Module):
    def __init__(self,image_size,K,device):
        super(Darkening_Map, self).__init__()
        self.image_size = image_size
        self.timesteps = K

        c = -0.7/K

        self.T = [c*i+1 for i in range(self.timesteps)]

    def forward(self, x, fwd_steps):
        for i in range(len(fwd_steps)):

            x[i] = F.adjust_brightness(x[i] ,self.T[int(fwd_steps[i]-1)])
        
        return x
    
def create_forward_process_darkening(config, device):
    forward_process_module = Darkening_Map(config.data.image_size, config.model.K, device)
    return forward_process_module

class Cooling_Map(nn.Module):
    def __init__(self,image_size,K,device):
        super(Cooling_Map, self).__init__()
        self.image_size = image_size
        self.timesteps = K

        self.rand_seed = torch.randn(image_size).to(device)

        self.scheduler = DDIMScheduler.from_config("CompVis/ldm-celebahq-256", subfolder="scheduler")

        self.T = [math.exp((-i/(K-i))+0.02)*torch.eye(image_size).to(device) for i in range(self.timesteps)] #Add temperature parameter

        self.actual_time = K  #Add to config later on

    def forward(self, x, fwd_steps):
        finale = self.actual_time*torch.ones(len(fwd_steps)).long().to(x.device)
        xf = self.scheduler.add_noise(x, self.rand_seed, finale)
        
        for i in range(len(fwd_steps)):

            

            

            x[i] = xf[i] + (x[i]-xf[i])*self.T[int(fwd_steps[i]-1)]

        return x
    
def create_forward_process_cooling(config, device):
    forward_process_module = Cooling_Map(config.data.image_size, config.model.K, device)
    return forward_process_module


class Cooling_Map_Dark(nn.Module):
    def __init__(self,image_size,K,device):
        super(Cooling_Map_Dark, self).__init__()
        self.image_size = image_size
        self.timesteps = K

        self.rand_seed = torch.randn(image_size).to(device)

        self.scheduler = DDIMScheduler.from_config("CompVis/ldm-celebahq-256", subfolder="scheduler")

        self.T = [math.exp((-i/(K-i))+0.02)*torch.eye(image_size).to(device) for i in range(self.timesteps)] #Add temperature parameter

        self.actual_time = 500  #Add to config later on

    def forward(self, x, fwd_steps):
        #finale = self.actual_time*torch.ones(len(fwd_steps)).long().to(x.device)
        #xf = self.scheduler.add_noise(x, self.rand_seed, finale)
        xf = 0.6*x
        for i in range(len(fwd_steps)):

            

            

            x[i] = xf[i] + (x[i]-xf[i])*self.T[int(fwd_steps[i]-1)]

        return x
    
def create_forward_process_cooling_dark(config, device):
    forward_process_module = Cooling_Map_Dark(config.data.image_size, config.model.K, device)
    return forward_process_module


class Darkening_Map_Sink(nn.Module):
    def __init__(self,image_size,K,device):
        super(Darkening_Map_Sink, self).__init__()
        self.image_size = image_size
        self.timesteps = K

        self.eta = 0.0

        self.T = [(1-self.eta)*math.cos(math.pi*i/2*K)+self.eta for i in range(self.timesteps)]

    def forward(self, x, fwd_steps):
        for i in range(len(fwd_steps)):

            x[i] = F.adjust_brightness(x[i] ,self.T[int(fwd_steps[i]-1)])
        
        return x
    
def create_forward_process_darkening_map_sink(config, device):
    forward_process_module = Darkening_Map_Sink(config.data.image_size, config.model.K, device)
    return forward_process_module

class Darkening_Map_No_Sink(nn.Module):
    def __init__(self,image_size,K,device):
        super(Darkening_Map_No_Sink, self).__init__()
        self.image_size = image_size
        self.timesteps = K
        self.eta = 0.7

        self.T = [(1-self.eta)*math.cos(2*math.pi*i/K)+self.eta for i in range(self.timesteps)]
    
    def forward(self, x, fwd_steps):
        for i in range(len(fwd_steps)):

            x[i] = F.adjust_brightness(x[i] ,self.T[int(fwd_steps[i]-1)])
        
        return x

def create_forward_process_darkening_map_no_sink(config, device):
    forward_process_module = Darkening_Map_No_Sink(config.data.image_size, config.model.K, device)
    return forward_process_module

class Darkening_Map_No_Sink_Periodic(nn.Module):
    def __init__(self,image_size,K,device):
        super(Darkening_Map_No_Sink_Periodic, self).__init__()
        self.image_size = image_size
        self.timesteps = K
        self.eta = 0.8

        self.period = 10

        self.T = [(1-self.eta)*math.cos(2*self.period*math.pi*i/K)+self.eta for i in range(self.timesteps)]
    
    def forward(self, x, fwd_steps):
        for i in range(len(fwd_steps)):

            x[i] = F.adjust_brightness(x[i] ,self.T[int(fwd_steps[i]-1)])
        
        return x
    
def create_forward_process_darkening_map_no_sink_periodic(config, device):
    forward_process_module = Darkening_Map_No_Sink_Periodic(config.data.image_size, config.model.K, device)
    return forward_process_module






#Add any additional transformations to the function map below. This allows for cleaner code and beter bookkeeping

Function_map = {

    'identity' : create_forward_process_identity,
    'fixed_random' : create_forward_process_fixed_random,
    'fixed_random_alt' : create_forward_process_fixed_random_alt,
    'heat' : create_forward_process_from_sigmas,
    'ramp': create_forward_process_ramp,
    'decelerating_ramp': create_forward_process_decelerating_ramp,
    'decelerating_ramp_reversal': create_forward_process_decelerating_ramp_reversal,
    'darkening': create_forward_process_darkening,
    'cooling': create_forward_process_cooling,
    'cooling_dark': create_forward_process_cooling_dark,
    'darkening_sink': create_forward_process_darkening_map_sink,
    'darkening_no_sink': create_forward_process_darkening_map_no_sink,
    'darkening_no_sink_periodic': create_forward_process_darkening_map_no_sink_periodic,}



#------------------------------ DO NOT MODIFY THE CODE BELOW THIS LINE UNLESS DISTRIBUTION OF NOISE CHANGES ---------------------------------------------

"""Utilities related to log-likelihood evaluation"""


def KL(dists, sigma0, sigma1, dim):
    # Calculates a matrix of KL divergences between spherical Gaussian distributions
    # with distances dists between their centers, where dists is a matrix
    return 0.5 * ((sigma0**2/sigma1**2)*dim + (dists)**2/sigma1**2 - dim + 2*dim*np.log(sigma1/sigma0))


def L_K_upperbound(K, trainloader, testloader, blur_module, sigma_inf, sigma_prior, dim,
                   train_size, test_size, device='cpu'):

    # Calculates the upper bound for the term E_q[KL[q(x_K|x_0)|p(x_K)]]
    # in a memory-efficient way, that is, calculates the distances between
    # test and training data points in batches, and uses those distances to calculate
    # the upper bound

    KL_div_upper_bound = torch.zeros(test_size, device=device)
    testdata_count = 0
    count = 0
    for testbatch in testloader:
        logging.info("Batch {}".format(count))
        count += 1
        blur_fwd_steps_test = [K] * len(testbatch[0])
        testbatch = blur_module(testbatch[0].to(
            device), blur_fwd_steps_test).reshape(len(testbatch[0]), -1)
        dists = torch.zeros(train_size, len(testbatch), device=device)
        traindata_count = 0
    # Get distances between the test batch and training data
        for trainbatch in trainloader:
            blur_fwd_steps_train = [K] * len(trainbatch[0])
            trainbatch = blur_module(trainbatch[0].to(
                device), blur_fwd_steps_train).reshape(len(trainbatch[0]), -1)
            dists[traindata_count:traindata_count +
                  len(trainbatch), :] = torch.cdist(trainbatch, testbatch)
            traindata_count += len(trainbatch)
        # Calculate the upper bounds on the KL divergence for each test batch element
        kl_divs = KL(dists, sigma_inf, sigma_prior, testbatch.shape[-1])
        inference_entropy = dim*0.5 * \
            torch.log(
                2*np.pi*torch.exp(torch.tensor([1]))*sigma_inf**2).to(device)
        cross_entropies = kl_divs + inference_entropy
        # log-sum-exp trick
        log_phi = -kl_divs - torch.logsumexp(-kl_divs, 0)[None, :]
        phi = torch.exp(log_phi)
        KL_div_upper_bound_batch = -inference_entropy + \
            (phi * (cross_entropies + log_phi + np.log(train_size))).sum(0)
        KL_div_upper_bound[testdata_count:testdata_count +
                           len(testbatch)] = KL_div_upper_bound_batch
        testdata_count += len(testbatch)
    return KL_div_upper_bound


def neg_ELBO(config, trainloader, testloader, blur_module, sigma, delta, image_size,
             train_size, test_size, model, device='cpu', num_epochs=10):
    """Estimates the terms in the negative evidence lower bound for the model
    num_epochs: Used for the estimation of terms L_k: How many epochs through these?"""

    logging.info("Calculating the upper bound for L_K...")
    L_K_upbound = L_K_upperbound(config.model.K, trainloader, testloader, blur_module, sigma,
                               delta, image_size**2, train_size, test_size, device)
    logging.info("... done! Value {}, len {}".format(
      L_K_upbound, len(L_K_upbound)))

    model_fn = get_model_fn(model, train=False)
    num_dims = image_size**2 * next(iter(trainloader))[0].shape[1]

    # There are K - 1 intermediate scales
    L_others = torch.zeros(config.model.K, device=device)
    mse_losses = torch.zeros(config.model.K, device=device)

    logging.info("Calculating the other terms...")
    with torch.no_grad():
        # Go through the set a few times for more accuracy, not just once
        for i in range(num_epochs):
            for testbatch in testloader:
                testbatch = testbatch[0].to(device).float()
                batch_size = len(testbatch)
                fwd_steps = torch.randint(
                    1, config.model.K, (batch_size,), device=device)
                blurred_batch = blur_module(testbatch, fwd_steps).float()
                less_blurred_batch = blur_module(testbatch, fwd_steps-1).float()
                noise = torch.randn_like(blurred_batch) * sigma
                perturbed_data = noise + blurred_batch
                diff = model_fn(perturbed_data, fwd_steps)
                prediction = perturbed_data + diff
                mse_loss = ((less_blurred_batch - prediction)
                            ** 2).sum((1, 2, 3))
                loss = mse_loss / delta**2
                loss += 2*num_dims*np.log(delta/sigma)
                loss += sigma**2/delta**2*num_dims
                loss -= num_dims
                loss /= 2
                # Normalize so that the significance of these terms matches with L_K and L_0
                # This way, we only go through once for each data point
                loss *= (config.model.K-1)
                mse_loss *= (config.model.K-1)
                L_others.scatter_add_(0, fwd_steps, loss)
                mse_losses.scatter_add_(0, fwd_steps, mse_loss)

        L_others = L_others / (test_size*num_epochs)
        mse_losses = mse_losses / (test_size*num_epochs)

        # Calculate L_0
        for testbatch in testloader:
            testbatch = testbatch[0].to(device).float()
            batch_size = len(testbatch)
            blurred_batch = blur_module(testbatch, [1]).float()
            non_blurred_batch = testbatch
            fwd_steps = torch.ones(batch_size, device=device)
            noise = torch.randn_like(blurred_batch) * sigma
            perturbed_data = noise + blurred_batch
            diff = model_fn(perturbed_data, fwd_steps)
            prediction = perturbed_data + diff
            mse_loss = ((non_blurred_batch - prediction)**2).sum((1, 2, 3))
            loss = 0.5*mse_loss/delta**2
            # Normalization constant
            loss += num_dims*np.log(delta*np.sqrt(2*np.pi))
            L_others[0] += loss.sum()
            mse_losses[0] += mse_loss.sum()
        L_others[0] = L_others[0] / test_size
        mse_losses[0] = mse_losses[0] / test_size

    logging.info("... Done! Values {}".format(L_others))
    return L_K_upbound.detach().cpu(), L_others.detach().cpu(), mse_losses.detach().cpu()


"""The next two functions based on https://github.com/yang-song/score_sde"""


def create_model(config, device_ids=None):
    """Create the model."""
    model = UNetModel(config)
    model = model.to(config.device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model


def get_model_fn(model, train=False):
    """A wrapper for using the model in eval or train mode"""
    def model_fn(x, fwd_steps):
        """Args:
                x: A mini-batch of input data.
                fwd_steps: A mini-batch of conditioning variables for different levels.
        """
        if not train:
            model.eval()
            return model(x, fwd_steps)
        else:
            model.train()
            return model(x, fwd_steps)
    return model_fn
