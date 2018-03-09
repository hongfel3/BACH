import random


class RandomRot(object):
    """
    Random rotation
    """
    def __call__(self, image):
        rand = random.choice(range(4))
        image = image.rotate(90 * rand)
        return image
