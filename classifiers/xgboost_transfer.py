import os
import sys
import gc
import time
from enum import Enum

import cv2
import numpy as np

import yaml
import pickle
from tqdm import tqdm

import itertools as it
import multiprocessing as mp

import xgboost as xgb
from sklearn import metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from SuperPixel import SuperPixel
from Preprocessor import Preprocessor, Reducers
from labeling_utils import *

def main():

	config = init()
	print("XGBoostClassifier")
	print(config.hyperparams)

	train_img, train_lbls, test_img, test_lbls = read_images(config.image_paths)

	if config.regenerate_features:

		print("Generating features...")

		train_features, train_labels, avg_size, label_db = get_features(train_img, train_lbls, config.segment, config.processors, mode="reference")
		test_features, test_labels, test_classlabels = get_features(test_img, test_lbls, config.segment, config.processors, mode="search", avg_size=avg_size)

		preprocessor = get_preprocessor(config.preprocess, train_features)

		train_features = preprocessor.process(train_features)
		test_features = preprocessor.process(test_features)

		print("Saving features...")
		train_file = open(config.save_path["train_features"], 'wb')
		test_file = open(config.save_path["test_features"], 'wb')

		train_save = (train_features, train_labels, label_db)
		test_save = (test_features, test_labels, test_classlabels)

		pickle.dump(train_save, train_file)
		pickle.dump(test_save, test_file)

		train_file.close()
		test_file.close()

	else:

		print("Loading features...")
		train_file = open(config.save_path["train_features"], 'rb')
		test_file = open(config.save_path["test_features"], 'rb')

		train_features, train_labels, label_db = pickle.load(train_file)
		test_features, test_labels, test_classlabels = pickle.load(test_file)

		train_file.close()
		test_file.close()

	# Make some testing data:
	# X_test = train_features[100:200, :]
	# y_test = train_labels[100:200]
	# eval_set = [(X_test, y_test)]

	# Setup data:
	xg_train = xgb.DMatrix(train_features, label=train_labels)
	xg_test = xgb.DMatrix(test_features)
	xgb_params = {"max_depth": 2, "eta": 0.4, "silent": 0, "objective":"multi:softmax", "num_class": len(train_features), "nthread": 6}

	start_time = time.time()

	print("\tTraining...")
	classifier = xgb.train(xgb_params, xg_train, num_boost_round=3)

	print("\tPredicting...")
	pred = classifier.predict(test_features)
	pred = [label_db[p] for p in pred]

	elapsed_time = time.time() - start_time

	report, acc, iou, precision, confusion = evaluate(test_classlabels, pred)
	save_results(report, acc, iou, precision, confusion, config.save_path["results"])

	print(report)
	print()
	print("Accuracy: {0:.4f}".format(acc))
	print("Precision: {0:.4f}".format(precision))
	print("IOU: {0:.4f}".format(iou))
	print("Took {0:.2f} seconds".format(elapsed_time))
	print()

#####################
## HELPER METHODS: ##
#####################

def read_images(filenames):

	src_img = cv2.imread(filenames["train_raw"])
	src_labels = cv2.imread(filenames["train_lbl"])

	search_img = cv2.imread(filenames["test_raw"])
	search_labels = cv2.imread(filenames["test_lbl"])

	return src_img, src_labels, search_img, search_labels

def get_features(img, mask, config, processors, mode="search", avg_size=None):

	## PRINT MODE: ##
	print()
	if mode == "reference":
		print("Reference Image:")
	else:
		print("Search Image:")

	## OVERSEGMENT: ##
	print("Segmenting...")
	spixel_args = (config["approx_num_superpixels"], config["num_levels"], config["iterations"])
	segment_mask, num_spixels = oversegment(img, spixel_args)

	if mode == "reference":
		print("Calculating average superpixel shape...")
		avg_size = calc_avg_size(segment_mask, int(num_spixels/5))

	## EXTRACT SUPERPIXELS: ##
	print("Computing features and creating SuperPixels...")
	spixels = extract_superpixels(img, mask, segment_mask, avg_size, num_spixels, processors)

	## FORMAT DATA: ##
	print("Formatting data...")
	features = [pixel.features for pixel in spixels if pixel is not None]
	labels = [pixel.id for pixel in spixels if pixel is not None]

	features = np.array(features)
	labels = np.array(labels)

	print()
	## RETURN RESULTS: ##
	if mode == "reference":
		id_label_db = {pixel.id: pixel.label for pixel in spixels}
		return features, labels, avg_size, id_label_db
	else:
		class_labels = [pixel.label for pixel in spixels if pixel is not None]
		return features, labels, class_labels

def extract_superpixels(src_img, lbl_img, mask_img, avg_size, num_superpixels, processors):

	threadpool = mp.Pool(processors)

	src_img_shared = src_img.copy()
	lbl_img_shared = lbl_img.copy()
	mask_img_shared = mask_img.copy()

	args = zip(range(num_superpixels), it.repeat(src_img_shared), it.repeat(lbl_img_shared), it.repeat(mask_img_shared), it.repeat(avg_size))
	args = tqdm(args, total=num_superpixels)
	superpixels = threadpool.starmap(create_spixel, args)

	threadpool.close()
	del(src_img_shared)
	del(lbl_img_shared)
	del(mask_img_shared)

	return superpixels

def create_spixel(*args):
	try:
		pixel = SuperPixel(*args)
		return pixel
	except ValueError as err:
		tqdm.write("Skipping SuperPixel. " + str(err))

def get_preprocessor(config, features):
	print("Fitting preprocessor...")
	preprocessor = Preprocessor(normalize=config["normalize"],
								reduce_features=config["reduce_features"],
								reducer_type=config["reducer_type"],
								explained_variance=config["explained_variance"])
	preprocessor.train(features)
	return preprocessor

def evaluate(truth, pred):
	print("Evaluating...")
	report = metrics.classification_report(truth, pred)
	acc = metrics.accuracy_score(truth, pred)
	iou = metrics.jaccard_similarity_score(truth, pred)
	precision = metrics.precision_score(truth, pred, average="weighted")
	confusion = metrics.confusion_matrix(truth, pred)
	return report, acc, iou, precision, confusion

def save_results(report, acc, iou, precision, confusion, filepath):
	print("Saving results...")
	results_file = open(filepath, 'w')
	results_file.write(report)
	results_file.write("\nAccuracy: {0:.4f}".format(acc))
	results_file.write("IOU: {0:.4f}".format(iou))
	results_file.write("Precision: {0:.4f}\n".format(precision))
	results_file.write(np.array2string(confusion))
	results_file.close()

####################
## CONFIGURATION: ##
####################

def init():
	global saved_stdout
	saved_stdout = sys.stdout

	config = init_config()

	log_file = open(config.save_path["log"], 'w')
	sys.stdout = writer(saved_stdout, log_file)

	return config

def init_config():

	VERSION = 1
	PROCESSORS = 12
	CLASSES = 9

	# hyper parameters:
	hyperparams = dict(
		n_jobs = 10,
		silent = False,
		n_estimators = 2,
	)

	# image files
	images = dict(
		train_raw = "../imgs/region_raw.png",
		train_lbl = "../imgs/region_mask.png",
		test_raw = "../imgs/search_raw.png",
		test_lbl = "../imgs/search_mask.png"
	)

	# superpixels
	segment = dict(
		approx_num_superpixels = 6000,
		num_levels = 5,
		iterations = 100
	)

	# preprocessor
	preprocessor = dict(
		normalize = True,
		reduce_features = True,
		reducer_type = Reducers.pca,
		explained_variance = 0.95
	)

	# saving
	regenerate_features = False
	save_path = dict(
		log = "results/xgb_v{}_log.txt".format(VERSION),
		results = "results/xgb_v{}_results.txt".format(VERSION),
		train_features = "saves/train_features.pkl",
		test_features = "saves/test_features.pkl",
	)

	params = Namespace(
		processors = PROCESSORS,
		classes = CLASSES,
		image_paths = images,
		segment = segment,
		preprocess = preprocessor,
		save_path = save_path,
		hyperparams = hyperparams,
		regenerate_features = regenerate_features,
	)

	return params

##########
## RUN: ##
##########
if __name__ == '__main__':
	main()
