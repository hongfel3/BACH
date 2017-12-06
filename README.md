In this repository we remake the results of [this paper](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0177544) on breast image classification.

T. Araújo et al., “Classification of breast cancer histology images using Convolutional Neural Networks,” PLOS ONE, vol. 12, no. 6, p. e0177544, Jun. 2017.


General process:

- Get dataset: https://iciar2018-challenge.grand-challenge.org/
- Stain normalize (normalize_data.py)
- Get patches (get_patches.py)
- Run CNN (CNN.py)

