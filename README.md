# Adversarial Image Detection

This project aims to detect adversarial attacks on Convolutional Neural Networks (CNNs) using analysis in the Fourier domain. The goal is to build a robust detector against various adversarial attack methods by leveraging spectral characteristics.

## Project Overview

This project involves training VGG-16 models on CIFAR-10, CIFAR-100 and MNIST datasets, generating adversarial examples using different attack methods, and building detectors to classify adversarial and clean examples based on spectral features.

## How to Run the Code

### Setup Environment

1. Clone the repository and install the required dependencies using Conda.

    ```bash
    $ git clone https://github.com/tejanagendramolleti/AdversarialImageDetection
    $ cd SpectralAdversarialDefense
    ```




### Generate Data and Train Models

1. Train the VGG-16 on CIFAR-10:
    ```bash
    $ python train_cif10.py
    ```

2. Train the VGG-16 on CIFAR-100:
    ```bash
    $ python train_cif100.py
    ```
3. Train the VGG-16 on MNIST:
    ```bash
    $ python train_cif100.py
    ```
4. Copy the generated `.pth` files to the `/models` folder and rename them as follows:
    - CIFAR-10 model: `vgg_cif10.pth`
    - CIFAR-100 model: `vgg_cif100.pth`
    - MNIST model:`vgg_mnist.pth`

5. Extract clean CIFAR-10/100,MNIST images classified correctly by the network:
    ```bash
    $ python generate_clean_data.py --net cif10
    ```

### Generating Adversarial Examples

1. Generate adversarial examples using the attack methods. Supported attack types are `fgsm`, `bim`, `pgd`, `df`, `cw`.
    ```bash
    $ python attack.py --attack fgsm
    ```

### Build the Detector

1. Extract characteristics for the detector. Available detectors include `InputMFS`, `InputPFS`, `LayerMFS`, `LayerPFS`, `LID`, `Mahalanobis`:
    ```bash
    $ python extract_characteristics.py --attack fgsm --detector InputMFS
    ```

2. Train the classifier on the extracted characteristics to detect adversarial examples:
    ```bash
    $ python detect_adversarials.py --attack fgsm --detector InputMFS
    ```

