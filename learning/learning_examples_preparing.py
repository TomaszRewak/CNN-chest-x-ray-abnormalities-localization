import sys
import json
import pickle
import os
import numpy as np
import math

from random import shuffle


def load_description(descriptions_file_path):
    with open(descriptions_file_path, 'r') as file_stream:
        descriptions = json.load(file_stream)

    return {
        os.path.splitext(description['name'])[0]: description
        for description in descriptions
    }


def load_features(features_file_path):
    with open(features_file_path, 'rb') as file_stream:
        return pickle.load(file_stream)


def prepare_learning_data(examples):
    return {
        'names': [
            name
            for (name, _, _) in examples
        ],
        'decisions': np.array([
            decision
            for (_, decision, _) in examples
        ]),
        'features': np.array([
            features
            for (_, _, features) in examples
        ])
    }


def split_into_batches(learning_set, batches):
    length = len(learning_set)
    batch_length = int(length / batches)
    for batch in range(0, batches):
        yield learning_set[batch * batch_length:(batch + 1) * batch_length]


def main(descriptions_file_path, features_file_path, training_set_file_path, testing_set_file_path, examples_path):
    descriptions = load_description(descriptions_file_path)
    features = load_features(features_file_path)

    normal_examples = [
        (name, [1., 0.], features[name])
        for name, description in descriptions.items()
        if 'normal' in description['items']
    ]

    abnormal_examples = [
        (name, [0., 1.], features[name])
        for name, description in descriptions.items()
        if 'normal' not in description['items']
    ]

    examples = normal_examples + abnormal_examples

    shuffle(examples)

    examples_length = len(examples)
    training_set_length = int(examples_length * 0.9)

    training_set = examples[0: training_set_length]
    testing_set = examples[training_set_length: examples_length]

    training_data = [
        prepare_learning_data(batch)
        for batch in split_into_batches(training_set, 10)
    ]
    testing_data = prepare_learning_data(testing_set)

    print('Training set length: {}'.format(len(training_set)))
    print('Testing set length: {}'.format(len(testing_set)))
    print('Batches length: {}'.format([len(batch['names']) for batch in training_data]))

    with open(training_set_file_path, 'wb') as file_stream:
        pickle.dump(training_data, file_stream)

    with open(testing_set_file_path, 'wb') as file_stream:
        pickle.dump(testing_data, file_stream)

    with open(examples_path, 'w') as file_stream:
        json.dump(testing_data['names'], file_stream)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

# python learning/learning_examples_preparing.py data/images-description.json data/transfer_features.pickle data/training.pickle data/testing.pickle data/examples.json
