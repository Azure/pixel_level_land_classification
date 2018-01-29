# Copyright (c) Microsoft. All rights reserved.
# Authors: Mary Wahl, Kolya Malkin, Nebojsa Jojic
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import pandas as pd
import os, argparse, cntk, tifffile, warnings, osr
from osgeo import gdal
from gdalconst import *
from mpl_toolkits.basemap import Basemap
from collections import namedtuple
from PIL import Image


# Maps land use labels to colors
color_map = np.asarray([[0,0,0],
						[0,0,1],
						[0,0.5,0],
						[0.5,1,0.5],
						[0.5,0.375,0.375]], dtype=np.float32)


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


def find_pixel_from_latlon(img_filename, lat, lon):
	''' Find the indices for a point of interest '''
	img = gdal.Open(img_filename, GA_ReadOnly)
	img_proj = osr.SpatialReference()
	img_proj.ImportFromWkt(img.GetProjection())
	ulcrnrx, xstep, _, ulcrnry, _, ystep = img.GetGeoTransform()

	world_map = Basemap(lat_0=0,
						lon_0=0,
						llcrnrlat=-90, urcrnrlat=90,
						llcrnrlon=-180, urcrnrlon=180,
						resolution='c', projection='stere')
	world_proj = osr.SpatialReference()
	world_proj.ImportFromProj4(world_map.proj4string)
	ct_to_img = osr.CoordinateTransformation(world_proj, img_proj)
	
	xpos, ypos = world_map(lon, lat, inverse=False)
	xpos, ypos, _ = ct_to_img.TransformPoint(xpos, ypos)
	x = int((xpos - ulcrnrx) / xstep)
	y = int((ypos - ulcrnry) / ystep)

	return(x,y)


def save_naip_image(input_image, output_filename):
	#color_last = np.round(np.transpose(input_image, (1, 2, 0)) * 255, 0) \
	#		.astype(np.uint8)
	color_last = np.transpose(input_image)
	tifffile.imsave(output_filename, color_last)
	return


def save_label_image(input_image, output_filename, hard=True):
	num_labels, height, width = input_image.shape
	label_image = np.zeros((3, height, width))
	if hard:
		my_label_indices = input_image.argmax(axis=0)
		for label_idx in range(num_labels):
			for rgb_idx in range(3):
				label_image[rgb_idx, :, :] += (my_label_indices == label_idx) *\
					color_map[label_idx, rgb_idx]
	else:
		input_image = np.exp(input_image) / np.sum(np.exp(input_image), axis=0)
		for label_idx in range(num_labels):
			for rgb_idx in range(3):
				label_image[rgb_idx, :, :] += input_image[label_idx, :, :] * \
					color_map[label_idx, rgb_idx]
	label_image = np.transpose(label_image).astype(np.float32)
	tifffile.imsave(output_filename, label_image)
	return


def eval(input_filename, model_filename, output_dir, center_lat, center_lon,
	region_dim):
	''' Coordinates model evaluation '''
	model = cntk.load_model(model_filename)
	naip_image, true_lc_image = load_image_pair(
		input_filename.replace('_NAIP.tif', ''))
	
	# Crop the input image and its true labels to the ROI. Include padding on
	# the NAIP image so that we have enough info to label the whole ROI.
	delta = int(region_dim / 2)
	padding = 64
	
	center_x, center_y = find_pixel_from_latlon(input_filename, center_lat,
		center_lon)
	true_lc_image = true_lc_image[center_x - delta:center_x + delta,
		center_y - delta:center_y + delta].astype(np.float32)
	naip_image = naip_image[:,
		center_x - (delta + padding):center_x + delta + padding,
		center_y - (delta + padding):center_y + delta + padding].astype(
			np.float32)

	# Iterate over the squares
	n_rows = int(region_dim / 128) # = n_cols, since the region is square
	pred_lc_image = np.zeros((5, true_lc_image.shape[0],
		true_lc_image.shape[1]))
	for row_idx in range(n_rows):
		for col_idx in range(n_rows):
			# Extract a 256 x 256 region from the NAIP image, to feed into the
			# model.
			sq_naip = naip_image[:,
								 row_idx * 128:(row_idx * 128) + 256,
								 col_idx * 128:(col_idx * 128) + 256]
			sq_pred_lc = np.squeeze(model.eval({model.arguments[0]: [sq_naip]}))
			pred_lc_image[:,
						  row_idx * 128:(row_idx * 128) + 128,
						  col_idx * 128:(col_idx * 128) + 128] = sq_pred_lc
	
	# Save the extracted images in human-viewable form. Will drop the near-
	# infrared channel from the NAIP imagery so that it won't wind up being
	# rendered as transparency. Note that the true labels must be expanded up
	# to one-hot before using the same function to save them.
	save_naip_image(naip_image[:3, padding:-padding, padding:-padding],
                    os.path.join(output_dir, 'NAIP.tif'))
	save_label_image(pred_lc_image, os.path.join(output_dir, 'pred_labels.tif'),
		hard=True)

	temp = np.transpose(np.eye(5)[true_lc_image.astype(np.int32)], [2, 0, 1])
	save_label_image(temp, os.path.join(output_dir, 'true_labels.tif'),
		hard=True)
	return


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='''
Applies a trained model to segment a subregion of an input NAIP image
according to land use in five categories:
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
	parser.add_argument('-i', '--input_filename', type=str, required=True,
						help='Filepath to the input NAIP image')
	parser.add_argument('-m', '--model_filename', type=str, required=True,
						help='Filepath to the trained model')
	parser.add_argument('-o', '--output_dir', type=str, required=True,
						help='Directory where output will be written')
	parser.add_argument('-t', '--center_lat', type=float, required=True,
						help='The latitude at the center of the ROI')
	parser.add_argument('-n', '--center_lon', type=float, required=True,
						help='The longitude at the center of the ROI')
	parser.add_argument('-r', '--region_dim', type=int, required=False,
						default=1024,
						help='The side length of the ROI in pixels (meters)')
	args = parser.parse_args()

	assert os.path.exists(args.input_filename), \
		'Input file {} could not be accessed.'.format(args.input_filename)
	assert os.path.exists(args.model_filename), \
		'Model file {} could not be accessed.'.format(args.model_filename)
	assert args.region_dim % 128 == 0, \
		'Region dimension must be divisible by 128.'
	os.makedirs(args.output_dir, exist_ok=True)

	eval(args.input_filename, args.model_filename, args.output_dir,
		args.center_lat, args.center_lon, args.region_dim)
