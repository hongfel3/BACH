# Overview

In this repository we remake the results of [this paper](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0177544) on breast image classification.

*T. Araújo et al., “Classification of breast cancer histology images using Convolutional Neural Networks,” PLOS ONE, vol. 12, no. 6, p. e0177544, Jun. 2017.*

This dataset used in this paper has been extended and is available as part of the [ICIAR 2018 challenge](https://iciar2018-challenge.grand-challenge.org/). The challenge contains two tasks. We are concerned with the first task - classification of breast tissue images (2048x1536 pixels @ 20X). Once the dataset is downloaded these images are in a subfolder called 'Photos'.



General process:

- Get dataset:
- Stain normalize (normalize_data.py)
- Get patches (get_patches.py)
- Run CNN (CNN.py)
-

