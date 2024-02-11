# Conditional Varitional Autoencoder for Deep generative models of Subsurface model

This is pytorch implementation of:
* Conditional Varitional Autoencoders(CVAE) using Convulution network 

## What is a CVAE?

A **CVAE** is a form of variational autoencoder that is
conditioned on an observation, where in our case the observation is a function y.

The autoencoders from which variational autoencoders
are derived are typically used for problems involving
image reconstruction and/or dimensionality reduction.

An Autoencoders is composed of two neural netwroks **Encoder** and **Decoder** where an Encoders takes an input prededined by the user and convert it into a low dimensionality space known as **latent space** and This latent space is passed through decoder to convert it to the original input size.

Additionally,
the decoder simultaneously learns to decode the latent
space representation and reconstruct that data back to
its original input. 

In **Varitional Autoencoders** this latent space is interpreted as a set of parameters governing statistical distributions In proceeding to the decoder
network, samples from the latent space (z) are randomly
drawn from these distributions and fed into the decoder,
therefore adding an element of variation into the process.
So in this way Varitional Autoencoders are used for generating samples of input which are similar to the given input(x) and **Conditional Autoencoder** is one step up in which we also add a condition that is to be fulfilled and this condition is passes through both encoder and decoder

![Alt text]('https://miro.medium.com/v2/resize:fit:4800/format:webp/1*zcqQjh9NsvJD72PU8xv9Rg.png')


## Model Structure


| Neural Network | Number of CNN Layers | Number of Linear Layers | Activation Function       |
|-----------|----------------------|-------------------------|---------------------------|
| Encoder   | 2                    | 1                       |  ReLU     |
| Decoder   | 2                    | 1                       | LeakyReLU (0.01 |


### Network Structure

Refer to [CVAE_example.py](https://github.com/FReakYdiVi/2nn1b/blob/main/CVAE_example.py) for model code and structure names as CVAE 


## Model Structure related to research paper 

| Component   | Number of CNN Layers | Number of Linear Layers | Activation Functions Used |
|-------------|----------------------|-------------------------|---------------------------|
| Encoder1    | 2                    | 2                      | ReLu          |
| Encoder2    | 0                    | 2                       | -          |
| Decoder     | 2                    | 1                       | LeakyRelu          |

### Network Structure

Refer to [CVAE_example.py](https://github.com/FReakYdiVi/2nn1b/blob/main/CVAE_example.py) for model code and structure names as CVAE_re

If you want to know more about the structure , just read this [reaserch paper](https://arxiv.org/abs/2101.06685
)

## Usage Of Libararies
### pre requistes
1. pytorch , numpy ,matplot , pandas
### Built In Library
1. CVAE_example, CVAE_functions 

## Blockers

This was tough and new kind of problem for me as it was related to geneartive ai and through this journey i got to many insights towards how autoencoders work or particulary how Conditional variational autoencoders work at generating samples and there were many hurdles towards how to make this project a better performer so I am gonna share a few and how i solved it -

1. the first problem was with **tuning of hyperparameters**
and I was using **optuna** for this but i was not getting good results with the optuna so then i thought to drop the idea of hyperparameter tuning and move towards upgrading the model structure of given model and thought i would be more relelvant.

2. I just modify the given model by adding leaky relu and it greatly reduces the overall loss of the model and secondly i also added condition to the encoder part as it was firstly not passing through encoder.

3. I also modify the given structure by adding one more convulution layer and you can see this code in highlighted from in [CVAE_example.py](https://github.com/FReakYdiVi/2nn1b/blob/main/CVAE_example.py)

4. I got know about the research paper then i tried to build the similar structure just to make **two encoders and decoder** so firstly it was hard to make such structure and also updating the functions according to the given structure  and i also changed the y_loss function to cross entropy which was more better than MSE but finally i made the similar model but it took too much time as it was very tough to make

