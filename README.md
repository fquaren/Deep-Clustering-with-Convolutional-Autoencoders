# Baseline Implementation of Deep Clustering with Convolutional Autoencoders for Medical Imaging Classification

In this work we propose the baseline implementation of an unsupervised deep clustering approach as a tool for automated image classification. Two different algorithms are tested, DEC and DCEC. Both are constituted by a first stage, in which robust features are learned by training an autoencoder, and by a second stage, in which learned features are encouraged to be cluster-oriented by finetuning the encoder utilizing the prediction of $k$-means centers for initialization. The approach is tested on a small dataset composed of 2D medical images from 3 different sources (CT, MRI, PET). The methodology intends to be applied for the automated staging and classification of idiopathic pulmonary fibrosis (IPF) on high-resolution CT (HRCT), where early and accurate staging tools are needed. Various deep learning approaches have been proposed for the same task so far, all requiring image annotation for supervised training. Our approach aims to the same goal, but without any annotation or supervision.

![alt text](https://github.com/fquaren/Deep-Clustering-with-Convolutional-Autoencoders/blob/master/reports/figures/DEC.svg)

![alt text](https://github.com/fquaren/Deep-Clustering-with-Convolutional-Autoencoders/blob/master/reports/figures/DCEC.png)
