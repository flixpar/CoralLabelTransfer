import cv2
import numpy as np

from sklearn import OneVsRestClassifier
from sklearn import SVC

import multiprocessing as mp
import itertools as it

import os
import yaml
import joblib
from tqdm import tqdm

from SuperPixel import SuperPixel
from Preprocessor import Preprocessor, Reducers
from labeling_utils import *

def main():

	# Setup
	config = init()

	# Read images:
	print("Reading images...")
	mosaic, mosaic_annotated = get_mosaic(config.mosaic)
	images, filenames = get_images(config.img_dir)

	# Segment mosaic
	mosaic_features, mosaic_labels, avg_size, label_db = get_mosaic_features(mosaic, mosaic_annotated, config.mosaic_superpixel, config.processors)

	# Setup preprocessor
	preprocessor = get_preprocessor(config.preprocessor, mosaic_features)
	mosaic_features = preprocessor.process(mosaic_features)

	# Train SVM
	classifier = train_svm(mosaic_features, mosaic_labels, config.svm, config.processors)
	save_classifier(classifier, config.svm, config.svm_file_compression)

	# Classify points
	masks = []
	for img in images:
		mask = classify(img, classifier, avg_size, label_db, config.superpixel, config.processors)
		masks.append(mask)

	# Write out masks
	for mask, base_fn in zip(masks, filenames):
		fn = base_fn.replace(".png", "_mask.png")
		cv2.imwrite(fn, mask)

#####################
## HELPER METHODS: ##
#####################

def get_mosaic_features(img, mask, spixel_config, processors):

	## OVERSEGMENT: ##
	print("Segmenting...")
	spixel_args = (spixel_config["approx_num_superpixels"], spixel_config["num_levels"], spixel_config["iterations"])
	segment_mask, num_spixels = oversegment(img, spixel_args)

	## GET AVG SUPERPIXEL SIZE: ##
	print("Calculating average superpixel shape...")
	avg_size = calc_avg_size(segment_mask, int(num_spixels/4))

	## EXTRACT SUPERPIXELS: ##
	print("Computing features and creating SuperPixels...")

	threadpool = mp.Pool(processors)
	args = zip(range(num_spixels), it.repeat(img), it.repeat(mask), it.repeat(segment_mask), it.repeat(avg_size))
	args = tqdm(args, total=num_spixels)
	superpixels = threadpool.starmap(create_spixel, args)
	threadpool.close()

	## FORMAT DATA: ##
	print("Formatting data...")
	features = [pixel.features for pixel in spixels if pixel is not None]
	labels = [pixel.id for pixel in spixels if pixel is not None]
	id_label_db = {pixel.id: pixel.label for pixel in spixels}

	features = np.array(features)
	labels = np.array(labels)

	return features, labels, avg_size, id_label_db

def classify(img, classifier, avg_size, label_db, spixel_config, processors):

	## OVERSEGMENT: ##
	spixel_args = (spixel_config["approx_num_superpixels"], spixel_config["num_levels"], spixel_config["iterations"])
	segment_mask, num_spixels = oversegment(img, spixel_args)

	## EXTRACT SUPERPIXELS: ##
	threadpool = mp.Pool(processors)
	args = zip(range(num_spixels), it.repeat(img), it.repeat(None), it.repeat(segment_mask), it.repeat(avg_size))
	args = tqdm(args, total=num_spixels)
	superpixels = threadpool.starmap(create_spixel, args)
	threadpool.close()

	## FORMAT FEATURES: ##
	features = [pixel.features for pixel in superpixels] # if pixel is not None]
	features = np.array(features)

	## PREDICT CLASSES FOR FEATURES: ##
	pred = classifier.predict(features)
	predictions = [label_db[p] for p in pred]

	## LABEL THE IMAGE: ##
	mask = np.zeros(segment_mask.shape, dtype=np.uint8)
	for i in range(num_spixels):
		if predictions[i] != 0:
			mask[np.where(segment_mask == i)] = predictions[i]

	return mask

def create_spixel(*args):
	try:
		pixel = SuperPixel(*args)
		return pixel
	except ValueError as err:
		tqdm.write("Skipping SuperPixel. " + str(err))

def get_mosaic(image_config):
	mosaic_fn = image_config["image"]
	mosaic = cv2.imread(mosaic_fn)

	mask_fn = image_config["annotation"]
	mask = cv2.imread(mask_fn, 0)

	return mosaic, mask

def get_images(image_dir):
	filelist = glob.glob(os.path.join(image_dir, "*.png"))
	images = []
	for fn in filelist:
		img = cv2.imread(fn)
		images.append(img)
	return zip(images, filelist)

#####################
## CONFIG METHODS: ##
#####################

class Namespace:
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)

def init_config():

	VERSION = 1
	PROCESSORS = 7

	# image files
	images = dict(
		image = "imgs/mosaic.png",
		annotation = "imgs/mosaic_mask.png"
	)

	# superpixels
	mosaic_superpixels = dict(
		approx_num_superpixels = 30000,
		num_levels = 5,
		iterations = 100
	)
	src_superpixels = dict(
		approx_num_superpixels = 5000,
		num_levels = 5,
		iterations = 100
	)

	# preprocessor
	preprocessor = dict(
		normalize = True,
		reduce_features = True,
		reducer_type = Reducers.pca,
		explained_variance = 0.985
	)

	# SVM parameter grid
	svm_params = dict(
		kernel = "rbf",
		cache_size = 20000,
		class_weight = "balanced",
		C = 0.1
	)

	# saving
	save = dict(
		config = "results/v{0:d}/config.yml".format(VERSION),
		log = "results/v{0:d}/log.txt".format(VERSION),
		svm = "results/v{0:d}/svm.pkl".format(VERSION),
		svm_compression = 3
	)

	params = Namespace(
		version = VERSION,
		processors = PROCESSORS,
		mosaic_images = images,
		mosaic_superpixels = mosaic_superpixels,
		source_superpixels = src_superpixels,
		preprocessor = preprocessor,
		svm_params = svm_params,
		save = save
	)

	return params
