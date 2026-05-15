import tensorflow as tf
import numpy as np


class KSparse(tf.keras.layers.Layer):
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def call(self, inputs):
        k = tf.minimum(self.k, tf.shape(inputs)[-1])
        values, _ = tf.math.top_k(tf.abs(inputs), k=k)
        kth = values[..., -1:]
        mask = tf.cast(tf.greater_equal(tf.abs(inputs), kth), inputs.dtype)
        return inputs * mask


def build_deep_fully_connected_network(params):
	params = dict(params)
	input_size = params.get("input_size", 4000)
	desired_latent_size = params.get("desired_latent_size", 15)
	drop_rate = params.get("drop_rate", 0.1)
	l2_reg = params.get("l2_reg", 1e-4)
	k_sparse = params.get("k_sparse", 10)
	hidden_layer_size1 = params["hidden_layer_size1"]
	hidden_layer_size2 = params["hidden_layer_size2"]
	hidden_layer_size3 = params["hidden_layer_size3"]
	hidden_layer_size4 = params["hidden_layer_size4"]

	he = tf.keras.initializers.HeNormal()
	narrow = tf.keras.initializers.TruncatedNormal(stddev=0.05)

	def encoder_decoder(inp_name):
		inp = tf.keras.Input(shape=(input_size,), name=inp_name)
		reg = tf.keras.regularizers.l2(l2_reg)
		x = tf.keras.layers.Dense(hidden_layer_size1, kernel_initializer=he, bias_initializer="zeros", kernel_regularizer=reg, name=f"fc_1_{inp_name}")(inp)
		x = tf.keras.layers.ELU(name=f"elu_1_{inp_name}")(x)
		x = tf.keras.layers.BatchNormalization(name=f"bn_1_{inp_name}")(x)
		x = tf.keras.layers.Dropout(drop_rate, name=f"dropout_1_{inp_name}")(x)

		x = tf.keras.layers.Dense(hidden_layer_size2, kernel_initializer=he, bias_initializer="zeros", kernel_regularizer=reg, name=f"fc_2_{inp_name}")(x)
		x = tf.keras.layers.ELU(name=f"elu_2_{inp_name}")(x)
		x = tf.keras.layers.BatchNormalization(name=f"bn_2_{inp_name}")(x)
		x = tf.keras.layers.Dropout(drop_rate, name=f"dropout_2_{inp_name}")(x)

		x = tf.keras.layers.Dense(hidden_layer_size3, kernel_initializer=he, bias_initializer="zeros", kernel_regularizer=reg, name=f"fc_3_{inp_name}")(x)
		x = tf.keras.layers.ELU(name=f"elu_3_{inp_name}")(x)
		x = tf.keras.layers.BatchNormalization(name=f"bn_3_{inp_name}")(x)
		x = tf.keras.layers.Dropout(drop_rate, name=f"dropout_3_{inp_name}")(x)

		x = tf.keras.layers.Dense(hidden_layer_size4, kernel_initializer=he, bias_initializer="zeros", kernel_regularizer=reg, name=f"fc_4_{inp_name}")(x)
		x = tf.keras.layers.ELU(name=f"elu_4_{inp_name}")(x)
		x = tf.keras.layers.BatchNormalization(name=f"bn_4_{inp_name}")(x)
		x = tf.keras.layers.Dropout(drop_rate, name=f"dropout_4_{inp_name}")(x)

		latent = tf.keras.layers.Dense(desired_latent_size, kernel_initializer=narrow, bias_initializer="zeros", kernel_regularizer=reg, name=f"big_latent_{inp_name}")(x)
		x = KSparse(k_sparse, name=f"k_sparse_{inp_name}")(latent)
		y = tf.keras.layers.Dense(1, activation=None, kernel_initializer=narrow, bias_initializer="zeros", kernel_regularizer=reg, name=f"fc_latent_{inp_name}_preact")(x)
		y = tf.keras.layers.Activation("sigmoid", name=f"fc_latent_{inp_name}_sigmoid")(y)
		y = tf.keras.layers.Rescaling(1.2, name=f"fc_latent_{inp_name}")(y)

		x = tf.keras.layers.Dense(hidden_layer_size4, kernel_initializer=he, bias_initializer="zeros", kernel_regularizer=reg, name=f"fc_dec_4_{inp_name}")(latent)
		x = tf.keras.layers.ELU(name=f"elu_dec_4_{inp_name}")(x)
		x = tf.keras.layers.BatchNormalization(name=f"bn_dec_4_{inp_name}")(x)
		x = tf.keras.layers.Dropout(drop_rate, name=f"dropout_dec_4_{inp_name}")(x)

		x = tf.keras.layers.Dense(hidden_layer_size3, kernel_initializer=he, bias_initializer="zeros", kernel_regularizer=reg, name=f"fc_dec_3_{inp_name}")(x)
		x = tf.keras.layers.ELU(name=f"elu_dec_3_{inp_name}")(x)
		x = tf.keras.layers.BatchNormalization(name=f"bn_dec_3_{inp_name}")(x)
		x = tf.keras.layers.Dropout(drop_rate, name=f"dropout_dec_3_{inp_name}")(x)

		x = tf.keras.layers.Dense(hidden_layer_size2, kernel_initializer=he, bias_initializer="zeros", kernel_regularizer=reg, name=f"fc_dec_2_{inp_name}")(x)
		x = tf.keras.layers.ELU(name=f"elu_dec_2_{inp_name}")(x)
		x = tf.keras.layers.BatchNormalization(name=f"bn_dec_2_{inp_name}")(x)
		x = tf.keras.layers.Dropout(drop_rate, name=f"dropout_dec_2_{inp_name}")(x)

		x = tf.keras.layers.Dense(hidden_layer_size1, kernel_initializer=he, bias_initializer="zeros", kernel_regularizer=reg, name=f"fc_dec_1_{inp_name}")(x)
		x = tf.keras.layers.ELU(name=f"elu_dec_1_{inp_name}")(x)
		x = tf.keras.layers.BatchNormalization(name=f"bn_dec_1_{inp_name}")(x)
		x = tf.keras.layers.Dropout(drop_rate, name=f"dropout_dec_1_{inp_name}")(x)

		out = tf.keras.layers.Dense(input_size, kernel_initializer=narrow, bias_initializer="zeros", kernel_regularizer=reg, name=f"fc_output_{inp_name}")(x)
		return inp, out, y

	inputs = []
	outputs = []

	inp, out, lat = encoder_decoder("1")
	inputs.append(inp)
	outputs.append(out)
	outputs.append(lat)

	model = tf.keras.Model(inputs=inputs, outputs=outputs, name="deep_fc_autoencoder")

	return model

def build_deep_CNN_network(params):
	params = dict(params)
	# Keras Conv2D expects (H, W, C) per-sample (batch dimension is implicit)
	input_size = tuple(params.get("input_size", (4000, 2, 1)))
	desired_latent_size = params.get("desired_latent_size", 15)
	drop_rate = params.get("drop_rate", 0.1)
	k_sparse = params.get("k_sparse", 10)
	filter1 = params.get("filter1", 8)
	filter2 = params.get("filter2", 6)
	filter3 = params.get("filter3", 4)
	filter4 = params.get("filter4", 1)
	kernel_size1 = params.get("kernel_size1", 20)
	kernel_size2 = params.get("kernel_size2", 10)
	kernel_size3 = params.get("kernel_size3", 5)
	kernel_size4 = params.get("kernel_size4", 3)
	stride1 = params.get("stride1", kernel_size1 // 2)
	stride2 = params.get("stride2", kernel_size2 // 2)
	stride3 = params.get("stride3", kernel_size3 // 2)
	stride4 = params.get("stride4", kernel_size4 // 2)

	sizes = []

	def encoder_decoder(inp_name):
		inp = tf.keras.Input(shape=input_size, name=inp_name)
		x = tf.keras.layers.Conv2D(
			filter1,
			[kernel_size1, 2],
			strides=[stride1, 2],
			padding="same",
			activation="relu",
			name=f"conv1_{inp_name}",
		)(inp)
		x = tf.keras.layers.BatchNormalization(name=f"bn_1_{inp_name}")(x)
		x = tf.keras.layers.Dropout(drop_rate, name=f"dropout_1_{inp_name}")(x)
		sizes.append(x.shape)

		x = tf.keras.layers.Conv2D(
			filter2,
			[kernel_size2, 1],
			strides=[stride2, 1],
			padding="same",
			activation="relu",
			name=f"conv2_{inp_name}",
		)(x)
		x = tf.keras.layers.BatchNormalization(name=f"bn_2_{inp_name}")(x)
		x = tf.keras.layers.Dropout(drop_rate, name=f"dropout_2_{inp_name}")(x)
		sizes.append(x.shape)

		x = tf.keras.layers.Conv2D(
			filter3,
			[kernel_size3, 1],
			strides=[stride3, 1],
			padding="same",
			activation="relu",
			name=f"conv3_{inp_name}",
		)(x)
		x = tf.keras.layers.BatchNormalization(name=f"bn_3_{inp_name}")(x)
		x = tf.keras.layers.Dropout(drop_rate, name=f"dropout_3_{inp_name}")(x)
		sizes.append(x.shape)

		x = tf.keras.layers.Conv2D(
			filter4,
			[kernel_size4, 1],
			strides=[stride4, 1],
			padding="same",
			activation="relu",
			name=f"conv4_{inp_name}",
		)(x)
		x = tf.keras.layers.BatchNormalization(name=f"bn_4_{inp_name}")(x)
		x = tf.keras.layers.Dropout(drop_rate, name=f"dropout_4_{inp_name}")(x)
		sizes.append(x.shape)

		x = tf.keras.layers.Flatten(name=f"flatten_{inp_name}")(x)
		latent = tf.keras.layers.Dense(
			desired_latent_size,
			kernel_initializer="he_normal",
			bias_initializer="zeros",
			name=f"big_latent_{inp_name}",
		)(x)
		x = KSparse(k_sparse, name=f"k_sparse_{inp_name}")(latent)
		y = tf.keras.layers.Dense(
			1,
			activation=None,
			kernel_initializer="he_normal",
			bias_initializer="zeros",
			name=f"fc_latent_{inp_name}_preact",
		)(x)
		y = tf.keras.layers.Activation("sigmoid", name=f"fc_latent_{inp_name}_sigmoid")(y)
		y = tf.keras.layers.Rescaling(1.2, name=f"fc_latent_{inp_name}")(y)

		# Decoder
		x = tf.keras.layers.Dense(
			sizes[-1][1] * sizes[-1][2] * sizes[-1][3],
			kernel_initializer="he_normal",
			bias_initializer="zeros",
			name=f"fc_dec_4_{inp_name}",
		)(x)
		x = tf.keras.layers.Reshape(sizes[-1][1:], name=f"reshape_dec_4_{inp_name}")(x)
		x = tf.keras.layers.Conv2DTranspose(
			filter3,
			[kernel_size4, 1],
			strides=[stride4, 1],
			padding="same",
			activation="relu",
			name=f"deconv_dec_4_{inp_name}",
		)(x)
		x = tf.keras.layers.BatchNormalization(name=f"bn_dec_4_{inp_name}")(x)
		x = tf.keras.layers.Dropout(drop_rate, name=f"dropout_dec_4_{inp_name}")(x)

		x = tf.keras.layers.Conv2DTranspose(
			filter2,
			[kernel_size3, 1],
			strides=[stride3, 1],
			padding="same",
			activation="relu",
			name=f"deconv_dec_3_{inp_name}",
		)(x)
		x = tf.keras.layers.BatchNormalization(name=f"bn_dec_3_{inp_name}")(x)
		x = tf.keras.layers.Dropout(drop_rate, name=f"dropout_dec_3_{inp_name}")(x)

		x = tf.keras.layers.Conv2DTranspose(
			filter1,
			[kernel_size2, 1],
			strides=[stride2, 1],
			padding="same",
			activation="relu",
			name=f"deconv_dec_2_{inp_name}",
		)(x)
		x = tf.keras.layers.BatchNormalization(name=f"bn_dec_2_{inp_name}")(x)
		x = tf.keras.layers.Dropout(drop_rate, name=f"dropout_dec_2_{inp_name}")(x)

		# Final reconstruction: number of filters must equal input channels (C), not width (W)
		x = tf.keras.layers.Conv2DTranspose(
			input_size[2],
			[kernel_size1, 2],
			strides=[stride1, 2],
			padding="same",
			activation=None,
			name=f"fc_output_{inp_name}",
		)(x)

		current_h = x.shape[1]
		target_h = input_size[0]
		diff_h = current_h - target_h

		if diff_h > 0:
			# We need to crop. Let's crop from the end (easier) 
			# or split it between top and bottom.
			x = tf.keras.layers.Cropping2D(
				cropping=((0, diff_h), (0, 0)), 
				name=f"final_{inp_name}"
			)(x)
		elif diff_h < 0:
			# In some weird stride cases, it might be smaller (though rare with 'same')
			# You'd use ZeroPadding2D here
			padding_needed = abs(diff_h)
			x = tf.keras.layers.ZeroPadding2D(
				padding=((0, padding_needed), (0, 0)),
				name=f"final_{inp_name}"
			)(x)
		else:
			# No crop/pad needed, but keep a consistent public output name.
			x = tf.keras.layers.Lambda(lambda t: t, name=f"final_{inp_name}")(x)
			

		out = x
		return inp, out, y

	inputs = []
	outputs = []

	inp, out, lat = encoder_decoder("1")
	inputs.append(inp)
	outputs.append(out)
	outputs.append(lat)

	model = tf.keras.Model(inputs=inputs, outputs=outputs, name="deep_CNN_autoencoder")

	return model

def build_CNN_variable_block(params):
	params = dict(params)
	# Keras Conv2D expects (H, W, C) per-sample (batch dimension is implicit)
	input_size = tuple(params.get("input_size", (4000, 2, 1)))
	desired_latent_size = params.get("desired_latent_size", 15)
	drop_rate = params.get("drop_rate", 0.1)
	l2_reg = params.get("l2_reg", 1e-4)
	k_sparse = params.get("k_sparse", 10)
	n_blocks = params.get("n_blocks", 1)

	sizes = []
	def encoder_decoder(inp_name):
		inp = tf.keras.Input(shape=input_size, name=inp_name)
		# x = tf.keras.layers.GaussianNoise(
		# 	stddev=params.get("noise_std", 0.05),
		# 	name=f"input_noise_{inp_name}",
		# )(inp)
		for i in range(n_blocks): 
			
			filter = params.get(f"filter{i+1}", 8)
			kernel_size = params.get(f"kernel_size{i+1}", 20)
			stride = params.get(f"stride_{i+1}", 1)
			pool_size = params.get(f"pool_size{i+1}", 2)

			x = inp if i == 0 else x
			second_dim = 2 if i == 0 else 1  # First layer should convolve across both channels, later layers only across features
			x = tf.keras.layers.Conv2D(
				filter,
				[kernel_size, second_dim],
				strides=[stride, second_dim],
				padding="same",
				activation=None,
				name=f"conv_{i+1}_{inp_name}",
			)(x)
			#x = tf.keras.layers.BatchNormalization(name=f"bn_{i+1}_{inp_name}")(x)
			x = tf.keras.layers.Activation("elu", name=f"relu_{i+1}_{inp_name}")(x)
			x = tf.keras.layers.MaxPooling2D(pool_size=[pool_size, 1], name=f"pool_{i+1}_{inp_name}")(x)
			x = tf.keras.layers.Dropout(drop_rate, name=f"dropout_{i+1}_{inp_name}")(x)
			sizes.append(x.shape)
		
		# fc part of the AE (bottle neck)
		x = tf.keras.layers.Flatten(name=f"flatten_{inp_name}")(x)

		x_flat = x
		reg = tf.keras.regularizers.l2(l2_reg)
		latent = tf.keras.layers.Dense(
			desired_latent_size,
			kernel_initializer="he_normal",
			bias_initializer="zeros",
			kernel_regularizer=reg,
			name=f"big_latent_{inp_name}",
		)(x)
		x = KSparse(k_sparse, name=f"k_sparse_{inp_name}")(latent)

		y = tf.keras.layers.Dense(
			1,
			activation=None,
			kernel_initializer="he_normal",
			bias_initializer="zeros",
			kernel_regularizer=reg,
			#name=f"fc_latent_{inp_name}_preact",
			name=f"fc_latent_{inp_name}"
		)(x)
		#y = tf.keras.layers.Activation("relu", name=f"fc_latent_{inp_name}")(y)# I am not sure if sigmoid is the best choice here
		#y = tf.keras.layers.Rescaling(1.2, name=f"fc_latent_{inp_name}")(y)


		#Decoder
		x = tf.keras.layers.Dense(
			sizes[-1][1] * sizes[-1][2] * sizes[-1][3],
			kernel_initializer="he_normal",
			bias_initializer="zeros",
			kernel_regularizer=reg,
			name=f"fc_dec_{n_blocks}_{inp_name}",
		)(x)
		x = tf.keras.layers.Reshape(sizes[-1][1:], name=f"reshape_dec_{n_blocks}_{inp_name}")(x)
		for i in reversed(range(n_blocks)):
			is_last = (i == 0)
			pool_size  = params.get(f"pool_size{i+1}", 2)
			kernel     = params.get(f"kernel_size{i+1}", 20)
			stride     = params.get(f"stride{i+1}", 1)
			out_filter = 1 if is_last else params.get(f"filter{i}", 8)  # mirror encoder
			sec_dim    = 2 if is_last else 1

			x = tf.keras.layers.UpSampling2D(size=[pool_size, 1],
											name=f"unpool_dec_{i}_{inp_name}")(x)
			x = tf.keras.layers.Conv2DTranspose(
					out_filter,
					[kernel, sec_dim],
					strides=[stride, sec_dim],
					padding="same",
					activation=None if is_last else "elu",
					name=f"deconv_dec_{i}_{inp_name}")(x)
			if not is_last:
				continue
				#x = tf.keras.layers.BatchNormalization(name=f"bn_dec_{i}_{inp_name}")(x)

		current_h = x.shape[1]
		target_h = input_size[0]
		diff_h = current_h - target_h

		if diff_h > 0:
			# We need to crop. Let's crop from the end (easier) 
			# or split it between top and bottom.
			x = tf.keras.layers.Cropping2D(
				cropping=((0, diff_h), (0, 0)), 
				name=f"final_{inp_name}"
			)(x)
		elif diff_h < 0:
			# In some weird stride cases, it might be smaller (though rare with 'same')
			# You'd use ZeroPadding2D here
			padding_needed = abs(diff_h)
			x = tf.keras.layers.ZeroPadding2D(
				padding=((0, padding_needed), (0, 0)),
				name=f"final_{inp_name}"
			)(x)
		else:
			# No crop/pad needed, but keep a consistent public output name.
			x = tf.keras.layers.Lambda(lambda t: t, name=f"final_{inp_name}")(x)
			

		out = x

		return inp, out, y

	inputs = []
	outputs = []
	inp, out, lat = encoder_decoder("1")
	inputs.append(inp)
	outputs.append(out)
	outputs.append(lat)

	model = tf.keras.Model(inputs=inputs, outputs=outputs, name="variable_block_CNN_autoencoder")

	return model



if __name__ == "__main__":

	# Check shapes and that the model can run a forward pass
	params = params = {
            "drop_rate": 0.1,
            "k_sparse": 10,
            "filter1": 10,
            "filter2": 8,
            "filter3": 6,
            "filter4": 4,
            "kernel_size1": 20,
            "kernel_size2": 10,
            "kernel_size3": 5,
            "kernel_size4": 3,
			"pool_size1": 20
        }

	#model = build_deep_CNN_network(params)
	model = build_CNN_variable_block(params)
	model.summary()

