- Overview
- Results

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

The general idea is to build a patch based classifier (CNN) and then aggregate the results to give an overall image class. The patch size used is 512x512.

The order in which to do things is:

1. Normalize images(normalize_data.py, requires [stain normalization code](https://github.com/Peter554/Stain-Normalization-))
2. Get training and validation patches and test images (get_patches.py)
3. Train model (CNN_keras.py)

A pre-trained model is included in this repository (best_model.h5). We can then evaluate performance on the test images:

4. Run notebook (performance.ipynb)

# Results
