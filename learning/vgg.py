import tensorflow as tf


def get_vgg_model(model_path):
    with open(model_path, mode='rb') as f:
        graph_def = tf.GraphDef()
        try:
            graph_def.ParseFromString(f.read())
        except:
            print('try adding PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ' +
                  'to environment.  e.g.:\n' +
                  'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ipython\n' +
                  'See here for info: ' +
                  'https://github.com/tensorflow/tensorflow/issues/582')

    # print([n.name for n in graph_def.node])

    return graph_def


def prepare_vgg_model(graph, model_path):
    with graph.as_default():
        tf.import_graph_def(get_vgg_model(model_path), name='vgg')
