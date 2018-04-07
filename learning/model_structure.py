import tensorflow as tf


def prepare_fully_connected_layer(name, n_input, n_output, l_input):
	weights = tf.get_variable(
		name=name+'_weights',
		shape=[n_input, n_output],
		dtype=tf.float32,
		initializer=tf.variance_scaling_initializer(0.01))

	biases = tf.get_variable(
		name=name+'_biases',
		shape=[n_output],
		dtype=tf.float32,
		initializer=tf.constant_initializer(0.0))

	mul = tf.matmul(l_input, weights)
	add = tf.nn.bias_add(mul, biases)

	return tf.nn.tanh(
		add,
		name=name+'_output')


def prepare_fully_connected_layers(graph, input_name, n_input, n_hidden, n_output):
	with graph.as_default():
		l_input = graph.get_tensor_by_name(input_name)
		
		tf.placeholder(
			name='input',
			shape=[n_input],
			dtype=tf.float32)

		l_hidden_1 = prepare_fully_connected_layer(
			'fully_connected_1',
			n_input,
			n_hidden,
			l_input / 1000)

		l_hidden_2 = prepare_fully_connected_layer(
			'fully_connected_2',
			n_hidden,
			n_output,
			l_hidden_1)

		l_output = tf.nn.softmax(
			l_hidden_2,
			name='output')

	return graph