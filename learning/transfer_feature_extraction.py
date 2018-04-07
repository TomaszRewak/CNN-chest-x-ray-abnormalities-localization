import cv2
import tensorflow as tf
import numpy as np
import os
import pickle
import sys

from vgg import prepare_vgg_model
from data_processing import prepare_images


def get_features(images, session):
    return session.run(
        'vgg/fc6/Reshape:0',
        feed_dict={
            'vgg/images:0': images
        })


def main(images_path, features_path, model_path):
    imege_file_names = os.listdir(images_path)
    result = {}

    graph = tf.Graph()
    prepare_vgg_model(graph, model_path)

    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())

        for image_file_names_chunk in np.array_split(imege_file_names, 100):
            images = prepare_images(images_path, image_file_names_chunk)

            print('Images Shape: {}'.format(images.shape))

            features = get_features(images, session)

            print('Features Shape: {}'.format(features.shape))

            for (image_name, image_features) in zip(image_file_names_chunk, features):
                result[os.path.splitext(image_name)[0]] = image_features

            print('Extracted: {}'.format(len(result)))

    with open(features_path, 'wb') as file_stream:
        pickle.dump(result, file_stream)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])

# python learning/transfer_feature_extraction.py data/images data/transfer_features.pickle data/vgg16.tfmodel