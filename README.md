# Overview

In this repository we remake the results of [this paper](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0177544) on breast image classification.

*T. Araújo et al., “Classification of breast cancer histology images using Convolutional Neural Networks,” PLOS ONE, vol. 12, no. 6, p. e0177544, Jun. 2017.*

The dataset used in this paper has been extended and is available as part of the [ICIAR 2018 challenge](https://iciar2018-challenge.grand-challenge.org/). The challenge contains two tasks. We are concerned with the first task - classification of breast tissue images (2048x1536 pixels @ 20X). Once the dataset has been downloaded these images can be found in the subfolder called 'Photos'.

There are four classes (100 images per class):

- Benign
- InSitu Carcinoma
- Invasive Carcinoma
- Normal

*image (below) from the challenge website*
<a href="https://imgur.com/UbxSaBC"><img src="https://i.imgur.com/UbxSaBC.png" title="source: imgur.com" /></a>

The general idea is to build a patch based classifier (CNN) and then aggregate the results (majority vote) to give an overall image class. The patch size used is 512x512.

The order in which to do things is:

1. Normalize images (normalize_data.py, requires [stain normalization code](https://github.com/Peter554/Stain-Normalization-))
2. Get training and validation patches and test images (get_patches.py)
3. Train model (CNN_keras.py)

You will need to go into the files and change the paths to data (one liner).

We can then evaluate performance on the test images. We aim to match the performance from the paper (78% image wise classification).

4. Run notebook (performance.ipynb)

Per class we use 60 images for training and keep 20 for validation and 20 for testing. The CNN is implemented in [Keras](https://keras.io/) using [Tensorflow](https://www.tensorflow.org/) as a backend.

After training for 50 epochs I achieve a test accuracy per patch of 0.68 and per image of 0.83. As the partitioning of images into train/val/test is random and so is the CNN training procedure you will not get exactly these numbers on a re-run. One of the main things noticed is that the model tends to overfit quite a lot (which one can see for example by looking at the generated [tensorboard logs](https://www.tensorflow.org/get_started/summaries_and_tensorboard)). This could perhaps be reduced by adding dropout, L2 regularization on weights or reducing model complexity.
