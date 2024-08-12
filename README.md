# Lensless Image Reconstruction Model

This repo builds off of the work of "Image Reconstruction for Transformer" by Xiuxi Pan et al. The authors create their own Transformer architecture with multiple AxialAttention + convolution encoder blocks. They also use OverlapPatchEmbedding to embed the image.

In this repo, I experiment with SwinTransformer and ConvNeXT models for phase mask-based lensless image reconstruction. The aim is to improve image reconstruction and remove image artifacts (long term goal).

### Datasets 

Mirflickr lensless imaging data.
