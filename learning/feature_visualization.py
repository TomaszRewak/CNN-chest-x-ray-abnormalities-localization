import sys
import tensorflow as tf
import numpy as np
import json
import os
import tf_cnnvis


from deepexplain.tensorflow import DeepExplain
from model_structure import prepare_fully_connected_layers
from vgg import prepare_vgg_model
from data_processing import prepare_image, store_result


def get_examples_list(examples_list_path):
    with open(examples_list_path, 'r') as file_stream:
        return json.load(file_stream)


def prepare_session(session, graph, convolution_model_path, full_connected_model_path):
    prepare_vgg_model(graph, convolution_model_path)
    prepare_fully_connected_layers(graph, 'vgg/fc6/Reshape:0', 25088, 2048, 2)

    with graph.as_default():
        saver = tf.train.Saver()

    session.run(tf.global_variables_initializer())
    saver.restore(session, full_connected_model_path)


def load_example(images_path, example):
    return np.array([prepare_image(os.path.join(images_path, example + '.png'))])


def visualize_features(graph, session, example, input_images_path, output_images_path, log_dir):
    image = load_example(input_images_path, example)
    placeholder = graph.get_tensor_by_name('vgg/images:0')

    feed_dictionary = {
        placeholder: image
    }

    tf_cnnvis.deconv_visualization(
        session, feed_dictionary, input_tensor=None, layers='r', path_logdir=log_dir, path_outdir=output_images_path)


def main(convolution_model_path, full_connected_model_path, examples_list_path, input_images_path, output_images_path, log_dir):
    examples_list = get_examples_list(examples_list_path)

    graph = tf.Graph()

    with graph.as_default():
        with tf.Session(graph=graph) as session:
            prepare_session(
                session,
                graph,
                convolution_model_path,
                full_connected_model_path)

            for example in examples_list:
                if example != 'CXR1358_IM-0232-3001':
                    continue

                visualize_features(
                    graph,
                    session,
                    example,
                    input_images_path,
                    os.path.join(output_images_path, example),
                    log_dir)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         sys.argv[4], sys.argv[5], sys.argv[6])

# python learning/feature_visualization.py data/vgg16.tfmodel data/model/model.ckpt data/examples.json data/images data/feature_results data/logs
