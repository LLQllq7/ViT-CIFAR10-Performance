# README

## Vision Transformer for CIFAR-10 Classification and Model Comparison

*The University of Science & Technology of China*, *Leqi Li*

This project implements and evaluates the Vision Transformer (ViT) model for image classification on the CIFAR-10 dataset. The main objectives are to replicate the Transformer model originally proposed in "Attention is All You Need" and adapt it for computer vision tasks. The Vision Transformer (ViT) is compared with other popular pretrained models, including ResNet18 and an enhanced version of ViT (ViT+ToMe), in terms of classification accuracy and training performance.

### Key Features:

- **Transformer-based Vision Model**: Implemented the Vision Transformer (ViT) architecture, which splits images into patches and applies self-attention mechanisms for classification tasks.
- **Performance Comparison**: Evaluated ViT against ResNet18 and ViT+ToMe on CIFAR-10, analyzing their performance on a small-scale dataset.
- **Experiment Design**: Conducted comprehensive experiments, testing models across different training set sizes (500, 10000, 20000 samples).
- **Visualization**: Visualized attention maps and training curves to understand the model's behavior and performance improvements over time.

### Results:

- **ViT** demonstrated significantly better accuracy than ResNet18 on CIFAR-10, especially in terms of classification accuracy with increasing training data.
- **ViT+ToMe** achieved even higher performance with reduced training time, showing the advantages of the Token-Merge technique.
