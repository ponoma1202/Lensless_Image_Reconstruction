# Basic_Transformer

This repo contains the most basic Vision Transformer (ViT) architecture, mostly following the "An Image is Worth 16x16 Words" paper. The model currently achieves 86% accuracy on CIFAR10 dataset.

Originally, I coded up an "Attention Is All You Need" encoder/decoder Transformer in llm_model.py. It has not been tested. 

For conceptual overview, I found the following article series helpful: 
https://medium.com/@hunter-j-phillips/overview-the-implemented-transformer-eafd87fe9589

ViT implementation:
https://github.com/s-chh/PyTorch-Scratch-Vision-Transformer-ViT/tree/main 

Classic Transformer implementation used for NLP Transformer: 
https://github.com/brandokoch/attention-is-all-you-need-paper/tree/master.

### Datasets 

Training on CIFAR10 dataset for classification. 
Mirflickr lensless imaging data (in progress).
