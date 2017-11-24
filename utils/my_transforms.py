import random


class RandomRot(object):
    def __call__(self, image):
        rand = random.choice(range(4))
        image = image.rotate(90 * rand)
        return image


class Mean_subtract(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image):
        image -= self.mean
        return image
