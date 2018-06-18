import sys
import tensorflow as tf
import numpy as np
import json
import os

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


def save_result(images_path, example, case, attribution):
    store_result(
        os.path.join(images_path, example + '_' + case + '.png'),
        attribution)


def to_gray_scale(image):
    return np.mean(image, axis=2)


def explain(deep_explain, input_tensor, output_tensor, image, labels, attribution_method):
    explenation = deep_explain.explain(
        attribution_method,
        output_tensor * [labels],
        input_tensor,
        image
        #, window_shape=(24, 24, 3), step=10
        )[0]

    return to_gray_scale(explenation)


def postprocess_attribution(attribution_normal, attribution_abnormal):
    threshold = np.maximum(
        attribution_normal,
        np.zeros((224, 224)))

    return np.maximum(
        attribution_abnormal - threshold,
        np.zeros((224, 224)))


def visualize_example(deep_explain, session, graph, example, input_images_path, output_images_path, attribution_method, only_abnormal):
    image = load_example(input_images_path, example)

    output = session.run(
        ['output:0'],
        feed_dict={
            'vgg/images:0': image
        })

    print(output)

    input_tensor = graph.get_tensor_by_name('vgg/images:0')
    output_tensor = graph.get_tensor_by_name('output:0')

    name = '{}[{:.2f},{:.2f}]'.format(
            example,
            output[0][0][0],
            output[0][0][1])

    attribution_abnormal = explain(
        deep_explain,
        input_tensor,
        output_tensor,
        image,
        [0., 1.],
        attribution_method)

    if only_abnormal:
        save_result(output_images_path, name, attribution_method, attribution_abnormal)
    else:
        attribution_normal = explain(
            deep_explain,
            input_tensor,
            output_tensor,
            image,
            [1., 0.],
            attribution_method)

        attribution = postprocess_attribution(
            attribution_normal,
            attribution_abnormal)

        save_result(output_images_path, name, '_base', image[0])
        #save_result(output_images_path, name, 'x_0', attribution_normal)
        #save_result(output_images_path, name, 'x_1', attribution_abnormal)
        save_result(output_images_path, name, attribution_method, attribution)


def main(convolution_model_path, full_connected_model_path, examples_list_path, input_images_path, output_images_path, attribution_method):
    examples_list = get_examples_list(examples_list_path)

    graph = tf.Graph()

    with tf.Session(graph=graph) as session:
        with DeepExplain(session=session) as deep_explain:
            prepare_session(
                session,
                graph,
                convolution_model_path,
                full_connected_model_path)

            for example in examples_list:
                visualize_example(
                    deep_explain,
                    session,
                    graph,
                    example,
                    input_images_path,
                    output_images_path,
                    attribution_method,
                    attribution_method == 'occlusion')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])

# python learning/visualization.py data/vgg16.tfmodel data/model/model.ckpt data/examples.json data/images data/results deeplift
