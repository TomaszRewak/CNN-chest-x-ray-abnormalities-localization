from scipy.misc import imread, imresize, imsave
import numpy as np
import os


def prepare_image(image_path):
    return np.array(imresize(
        imread(image_path, mode='RGB'),
        (224, 224)
    ))


def prepare_images(images_path, names_chunk):
    return np.array([
        prepare_image(os.path.join(images_path, image_file))
        for image_file in names_chunk
    ])


def store_result(image_path, image):
    imsave(image_path, image)
