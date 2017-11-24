import random
import numpy as np
from torchvision import transforms

mean = np.load('mean.npy')
trans=transforms.ToTensor()
mean=trans(mean)

class RandomRot(object):
    def __call__(self, image):
        rand = random.choice(range(4))
        image = image.rotate(90 * rand)
        return image


class Mean_subtract(object):
    def __call__(self, image):
        image -= mean
        return image
