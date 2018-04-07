import tensorflow as tf
import numpy as np
import os
import pickle
import time
import sys
from sklearn.metrics import confusion_matrix, classification_report

from model_structure import prepare_fully_connected_layers


def prepare_inputs(graph, n_inputs):
    with graph.as_default():
        l_input = tf.placeholder(
            name='input',
            shape=[None, n_inputs],
            dtype=tf.float32)


def prepare_training(graph, n_output):
    with graph.as_default():
        last_layer = graph.get_tensor_by_name('fully_connected_2_output:0')

        labels = tf.placeholder(
            name='labels',
            shape=[None, n_output],
            dtype=tf.float32)

        error = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=last_layer,
                labels=labels),
            name='error')

        optimizer = tf.train.AdamOptimizer(
            0.00001,
            name='optimizer')

        training = optimizer.minimize(
            error,
            name='training')


def test(session, test_set):
    output, labels = session.run(
        ['output:0', 'labels:0'],
        feed_dict={
            'input:0': test_set['features'],
            'labels:0': test_set['decisions']
        })

    print(classification_report(
        tf.argmax(labels, 1).eval(session=session), 
        tf.argmax(output, 1).eval(session=session), 
        target_names=['normal', 'abnormal']))


def train(session, batches, test_set, epochs):
    session.run(tf.global_variables_initializer())

    test(session, test_set)

    for _ in range(epochs):
        for batch in batches:
            _ = session.run(
                ['training'],
                feed_dict={
                    'input:0': batch['features'],
                    'labels:0': batch['decisions']
                })

        test(session, test_set)


def main(training_set_path, testing_set_path, model_save_path):
    graph = tf.Graph()

    prepare_inputs(graph, 25088)
    prepare_fully_connected_layers(graph, 'input:0', 25088, 2048, 2)
    prepare_training(graph, 2)

    with open(training_set_path, 'rb') as file_stream:
        training_set = pickle.load(file_stream)

    with open(testing_set_path, 'rb') as file_stream:
        testing_set = pickle.load(file_stream)

    with graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as session:
        train(session, training_set, testing_set, 10)

        saver.save(session, model_save_path)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])

# python learning/fully_connected_layers_training.py data/training.pickle data/testing.pickle data/model/model.ckpt
