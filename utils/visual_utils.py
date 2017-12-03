import matplotlib.pyplot as plt
import numpy as np


def show_colors(C):
    """
    Shows rows of C as colors (RGB)
    :param C:
    :return:
    """
    n = C.shape[0]
    for i in range(n-1,0,-1):
        if C[i].max()>1.0:
            plt.plot([0, 1], [i, i], c=C[i]/255, linewidth=20)
        else:
            plt.plot([0, 1], [i, i], c=C[i], linewidth=20)
        plt.axis('off')
        plt.axis([0, 1, -1, n])


def show(image, now=True, fig_size=(10, 10)):
    """
    Show an image (np.array). Can be of shape HxWxC or flattened (if square).
    Caution! Rescales image to be in range [0,1]."
    :param image:
    :param now:
    :param fig_size:
    :return:
    """
    image = image.astype(np.float32)
    if len(image.shape) == 1:
        wh = np.sqrt(image.shape[0] / 3).astype(np.uint16)
        image = image.reshape((wh, wh, 3))
    m, M = image.min(), image.max()
    if fig_size != None:
        plt.rcParams['figure.figsize'] = (fig_size[0], fig_size[1])
    plt.imshow((image - m) / (M - m), cmap='gray')
    plt.axis('off')
    if now == True:
        plt.show()


def patch_grid(ims, width=5, sub_sample=None, rand=False):
    """
    Display a grid of patches
    :param ims:
    :param width:
    :param sub_sample:
    :param rand:
    :return:
    """
    N0 = np.shape(ims)[0]
    if sub_sample == None:
        N = N0
        stack = ims
    elif sub_sample != None and rand == True:
        N = sub_sample
        idx = np.random.choice(range(N), sub_sample, replace=False)
        stack = ims[idx]
    elif sub_sample != None and rand == False:
        N = sub_sample
        stack = ims[:N]
    height = np.ceil(float(N) / width).astype(np.uint16)
    plt.rcParams['figure.figsize'] = (18, (18 / width) * height)
    plt.figure()
    for i in range(N):
        plt.subplot(height, width, i + 1)
        im = stack[i]
        show(im, now=False, fig_size=None)
    plt.show()
