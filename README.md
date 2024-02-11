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
| Encoder   | 2                    | 1                       |  LeakyReLU (0.01)    |
| Decoder   | 2                    | 1                       | LeakyReLU (0.01 |



### Network Structure

Refer to 


## Model Structure related to research paper 


