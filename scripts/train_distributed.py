# Copyright (c) Microsoft. All rights reserved.
# Authors: Mary Wahl, Kolya Malkin, Nebojsa Jojic
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import pandas as pd
import os, argparse, cntk, tifffile, model_mini_pub, warnings
import cntk.train.distributed as distributed
from cntk.train.training_session import CheckpointConfig, training_session


def load_image_pair(tile_name):
	''' Load the corresponding NAIP and LandCover images '''
	#with warnings.filterwarnings('ignore'):
	# With the currently-available training data, the tifffile package
	# generates these RuntimeWarnings and UserWarnings under normal
	# operating conditions:
	# - RuntimeWarning: py_decodelzw encountered unexpected end of stream
	# - UserWarning: unpack: string size must be a multiple of element size
	# - UserWarning: invalid tile data
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		naip_image = np.transpose(tifffile.imread(
			'{}_NAIP.tif'.format(tile_name)))  / 256.0
		landcover_image = np.transpose(tifffile.imread(
			'{}_LandCover.tif'.format(tile_name)))
	landcover_image[landcover_image > 4] = 4
	return (naip_image, landcover_image)


def get_cropped_data(image, bounds, rescale=False):
	''' Crop out a subsection of an NAIP or LandCover image. Note that NAIP
	    images have an extra axis (for color), use rescale=True. '''
	a, b, c, d = bounds
	if rescale:
		return(image[:, a : (a + c), b : (b + d)].astype(np.float32))
	else:
		return(image[a : (a + c), b : (b + d)].astype(np.int32))


def interesting_patch(label_slice):
	''' Upsample less common labels '''
	w, h = label_slice.shape
	return ((label_slice == 1).sum() + \
			(label_slice == 4).sum() > 0.003 * w * h) \
			or (np.random.random_sample() > 0.5)


class MyDataSource(cntk.io.UserMinibatchSource):
	''' A minibatch source for NAIP and label data '''
	def __init__(self, f_dim, l_dim, number_of_workers, input_dir,
		minibatches_per_image):
		''' Divvy up images between workers at initialization '''
		# Record the image dimensions for later
		self.f_dim, self.l_dim = f_dim, l_dim
		self.minibatches_per_image = minibatches_per_image
		self.num_color_channels, self.block_size, _ = self.f_dim
		self.num_landcover_classes, _, _ = self.l_dim

		# Record the stream information
		self.fsi = cntk.io.StreamInformation(
			'features', 0, 'dense', np.float32, self.f_dim)
		self.lsi = cntk.io.StreamInformation(
			'labels', 1, 'dense', np.float32, self.l_dim)

		# Create a transform for converting labels to one-hot
		self.x = cntk.input_variable((self.block_size, self.block_size))
		self.oh_tf = cntk.one_hot(self.x, self.num_landcover_classes, False,
								  axis=0)

		# Decide which tiles each worker will process
		self.tile_names = {}
		all_tiles = np.sort(
			[os.path.join(input_dir, i.replace('_NAIP.tif', '')) \
			 for i in os.listdir(input_dir) if i.endswith('_NAIP.tif')])
		if number_of_workers > len(all_tiles):
			for i in range(number_of_workers):
				self.tile_names[i] = all_tiles[np.random.randint(
					len(all_tiles), size=(2))]
		else:
			for i, tile_subset in enumerate(np.array_split(np.sort(all_tiles),
														   number_of_workers)):
				if len(tile_subset) > 5:
					tile_subset = tile_subset[:5]
				self.tile_names[i] = tile_subset


		self.current_mb_indices = dict(zip(range(number_of_workers),
										   [0] * number_of_workers))
		self.current_image_indices = dict(zip(range(number_of_workers),
										      [0] * number_of_workers))
		self.naip_images = [[]] * number_of_workers
		self.landcover_images = [[]] * number_of_workers
		self.already_loaded_images = [False] * number_of_workers

		super(MyDataSource, self).__init__()

	def stream_infos(self):
		return [self.fsi, self.lsi]

	def next_minibatch(self, mb_size_in_samples, number_of_workers=1, worker_rank=0,
		device=None):
		''' Worker loads TIF images and extracts samples from them '''

		if not self.already_loaded_images[worker_rank]:
			# It's time to load all images into memory. This can take time, so
			# we log our progress to stdout
			self.already_loaded_images[worker_rank] = True
			for i, tile_name in enumerate(self.tile_names[worker_rank]):
				try:
					naip_image, landcover_image = load_image_pair(tile_name)
					self.naip_images[worker_rank].append(naip_image)
					self.landcover_images[worker_rank].append(landcover_image)
					print('Worker {} loaded its {}th image'.format(
						worker_rank, i))
				except ValueError:
					print('Failed to load TIF pair: {}'.format(tile_name))
					pass
			print('Worker {} completed image loading'.format(worker_rank))

		if self.current_mb_indices[worker_rank] == 0:
			# It's time to advance the image index
			self.current_image_indices[worker_rank] = (
				self.current_image_indices[worker_rank] + 1) % len(
				self.naip_images[worker_rank])
		idx = self.current_image_indices[worker_rank]

		# Feature data have dimensions: num_color_channels x block size 
		#								x block size
		# Label data have dimensions: block_size x block_size
		features = np.zeros((mb_size_in_samples, self.num_color_channels,
							 self.block_size, self.block_size),
							dtype=np.float32)
		labels = np.zeros((mb_size_in_samples, self.block_size,
						   self.block_size), dtype=np.float32)

		# Randomly select subsets of the image for training
		w, h = self.naip_images[worker_rank][idx].shape[1:]
		samples_retained = 0
		while samples_retained < mb_size_in_samples:
			i = np.random.randint(0, w - self.block_size)
			j = np.random.randint(0, h - self.block_size)            
			bounds = (i, j, self.block_size, self.block_size)
			label_slice = get_cropped_data(
				self.landcover_images[worker_rank][idx], bounds, False)
			if interesting_patch(label_slice):
				features[samples_retained, :, :, :] = get_cropped_data(
					self.naip_images[worker_rank][idx], bounds, True)
				labels[samples_retained, :, :] = label_slice
				samples_retained += 1

		# Convert the label data to one-hot, then convert arrays to Values
		f_data = cntk.Value(batch=features)
		l_data = cntk.Value(batch=self.oh_tf.eval({self.x: labels}))

		result = {self.fsi: cntk.io.MinibatchData(
						f_data, mb_size_in_samples, mb_size_in_samples, False),
				  self.lsi: cntk.io.MinibatchData(
				  		l_data, mb_size_in_samples, mb_size_in_samples, False)}
		
		# Minibatch collection complete: update minibatch index so we know
		# how many more minibatches to collect using this TIFF pair
		self.current_mb_indices[worker_rank] = (1 + 
			self.current_mb_indices[worker_rank]) % self.minibatches_per_image
		return(result)


def center_square(output, block_size, padding):
	return(cntk.slice(cntk.slice(output, 1, padding, block_size - padding),
					  2, padding, block_size - padding))


def criteria(label, output, block_size, c_classes, weights):
	''' Define the loss function and metric '''
	probs = cntk.softmax(output, axis=0)
	log_probs = cntk.log(probs)
	ce = cntk.times(weights, -cntk.element_times(log_probs, label),
					output_rank=2)
	mean_ce = cntk.reduce_mean(ce)
	_, w, h = label.shape
	pe = cntk.classification_error(probs, label, axis=0) - \
		cntk.reduce_sum(cntk.slice(label, 0, 0, 1)) / cntk.reduce_sum(label)
	return(mean_ce, pe)


def train(input_dir, output_dir, num_epochs):
	''' Coordinates model creation and training; minibatch creation '''
	num_landcover_classes = 5
	num_color_channels = 4
	block_size = 256
	padding = int(block_size / 4)

	my_rank = distributed.Communicator.rank()
	number_of_workers = distributed.Communicator.num_workers()
	os.makedirs(output_dir, exist_ok=True)

	# We extract 160 sample regions from an input image before moving along to
	# the next image file. Our epoch size is 16,000 samples.
	minibatch_size = 10
	minibatches_per_image = 160
	minibatches_per_epoch = 1600
	epoch_size = minibatch_size * minibatches_per_epoch

	# Define the input variables
	f_dim = (num_color_channels, block_size, block_size)
	l_dim = (num_landcover_classes, block_size, block_size)
	feature = cntk.input_variable(f_dim, np.float32)
	label = cntk.input_variable(l_dim, np.float32)

	# Define the minibatch source
	minibatch_source = MyDataSource(f_dim, l_dim, number_of_workers, input_dir,
									minibatches_per_image)
	input_map = {feature: minibatch_source.streams.features,
				 label: minibatch_source.streams.labels}

	# Define the model
	model = model_mini_pub.model(num_landcover_classes, block_size,
								 2, [64, 32, 32, 32])(feature)

	# Define the loss function and metric. Note that loss is not computed
	# directly on the model's output; the edges are first dropped.
	output = center_square(cntk.reshape(model,
						   				(num_landcover_classes, block_size,
						   				 block_size)),
						   block_size, padding)
	label_center = center_square(label, block_size, padding)
	mean_ce, pe = criteria(label_center, output, block_size,
						   num_landcover_classes, [0.0, 1.0, 1.0, 1.0, 1.0])

	# Create the progress writer, learner, and trainer (which will be a
	# distributed trainer if number_of_workers > 1)
	progress_writers = [cntk.logging.progress_print.ProgressPrinter(
		tag='Training',
		num_epochs=num_epochs,
		freq=epoch_size,
		rank=my_rank)]

	lr_per_mb = [0.0001] * 30 + [0.00001] * 30 + [0.000001]
	lr_per_sample = [lr / minibatch_size for lr in lr_per_mb]
	lr_schedule = cntk.learning_rate_schedule(lr_per_sample,
											  epoch_size=epoch_size,
											  unit=cntk.UnitType.sample)
	learner = cntk.rmsprop(model.parameters, lr_schedule, 0.95, 1.1, 0.9, 1.1,
						   0.9, l2_regularization_weight=0.00001)

	if number_of_workers > 1:
		parameter_learner = distributed.data_parallel_distributed_learner(
			learner, num_quantization_bits=32)
		trainer = cntk.Trainer(output, (mean_ce, pe), parameter_learner,
							   progress_writers)
	else:
		trainer = cntk.Trainer(output, (mean_ce, pe), learner, progress_writers)

	# Perform the training! Note that some progress output will be generated by
	# each of the workers.
	if my_rank == 0:
		print('Retraining model for {} epochs.'.format(num_epochs))
		print('Found {} workers'.format(number_of_workers))
		print('Printing progress every {} minibatches'.format(
			minibatches_per_epoch))
		cntk.logging.progress_print.log_number_of_parameters(model)
	training_session(
		trainer=trainer,
		max_samples=num_epochs * epoch_size,
		mb_source=minibatch_source, 
		mb_size=minibatch_size,
		model_inputs_to_streams=input_map,
		checkpoint_config=CheckpointConfig(
			frequency=epoch_size,
			filename=os.path.join(output_dir, 'trained_checkpoint.model'),
			preserve_all=True),
		progress_frequency=epoch_size
	).train()

	distributed.Communicator.finalize() 
	if my_rank == 0:
		trainer.model.save(os.path.join(output_dir,
										'trained.model'))
	return

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='''
Trains a model to segment NAIP images according to land use in five categories:
- 0: No data
- 1: Water
- 2: Trees and shrubs
- 3: Herbaceous vegetation
- 4: Barren or Impervious (roads and other)
Expects an input directory containing pairs of images with naming convention
"[filename_base]_NAIP.tif" and "[filename_base]_LandCover.tif". Outputs the
trained model and training checkpoints to a specified model directory (will
load a checkpoint from this directory if a checkpoint is found there).
''')
	parser.add_argument('-i', '--input_dir', type=str, required=True,
						help='Directory containing all training image files.')
	parser.add_argument('-o', '--model_dir', type=str, required=True,
						help='Directory where model outputs will be stored.')
	parser.add_argument('-n', '--num_epochs', type=int, required=False,
						default=1,
						help='Specifies the number of epochs of training to ' +
						'be performed.')
	args = parser.parse_args()

	assert os.path.exists(args.input_dir), \
		'Input directory {} could not be accessed.'.format(args.input_dir)
	os.makedirs(args.model_dir, exist_ok=True)
	assert args.num_epochs > 0, \
		'The number of epochs must be greater than zero'

	train(args.input_dir, args.model_dir, args.num_epochs)