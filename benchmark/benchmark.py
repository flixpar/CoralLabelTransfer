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

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from SuperPixel import SuperPixel
from Preprocessor import Preprocessor, Reducers
from labeling_utils import *

def main():

	# Do all processing to get features:
	config = init()

	train_img, train_lbls, test_img, test_lbls = read_images(config.image_paths)

	train_features, train_labels, avg_size, label_db = get_features(train_img, train_lbls, config.segment, config.processors, mode="reference")
	test_features, test_labels, test_classlabels = get_features(test_img, test_lbls, config.segment, config.processors, mode="search", avg_size=avg_size)

	preprocessor = get_preprocessor(config.preprocess, train_features)

	train_features = preprocessor.process(train_features)
	test_features = preprocessor.process(test_features)

	print(train_features.min())
	print(train_features.max())
	print(test_features.min())
	print(test_features.max())

	# Get the classifiers:
	classifiers = sorted(get_classifiers())

	# Try classifiers:
	for name, classifier in classifiers:
		init_run(config, name)
		print("\tRunning {}...".format(name))

		print("\tTraining...")
		classifier.fit(train_features, train_labels)

		print("\Predicting...")
		pred = classifier.predict(test_features)
		pred = [label_db[p] for p in pred]

		report, acc, iou, precision, confusion = evaluate(test_classlabels, pred)
		save_results(report, acc, iou, precision, confusion, config.save_path, name)
		plot_confusion_matrix(confusion, config.classes, config.save_path, name)

		print(report)
		print("Accuracy: {0:.4f}".format(acc))
		print("Precision: {0:.4f}".format(precision))
		print("IOU: {0:.4f}".format(iou))
		print()

		del(classifier)

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

def save_results(report, acc, iou, precision, confusion, path, name):
	print("Saving results...")
	filename = os.path.join(path, name + "_results.txt")
	results_file = open(filename, 'w')
	results_file.write(report)
	results_file.write("\nAccuracy: {0:.4f}".format(acc))
	results_file.write("IOU: {0:.4f}".format(iou))
	results_file.write("Precision: {0:.4f}\n".format(precision))
	results_file.write(np.array2string(confusion))
	results_file.close()

def plot_confusion_matrix(cfm, num_classes, path, name):

	plt.figure()

	# normalize and display
	cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cfm, interpolation='nearest', cmap=plt.cm.Blues)

	# setup the title and axes
	plt.title("Normalized Confusion Matrix")
	plt.colorbar()
	tick_marks = np.arange(num_classes)
	plt.xticks(tick_marks, range(num_classes), rotation=45)
	plt.yticks(tick_marks, range(num_classes))

	# label the axes
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	# save image and display
	out_fn = os.path.join(path, name + "_confusion.png")
	plt.savefig(out_fn, dpi=100)
	plt.show()

####################
## CONFIGURATION: ##
####################

def init():
	global saved_stdout
	saved_stdout = sys.stdout

	config = init_config()
	return config

def init_run(config, name):

	filename = os.path.join(config.save_path, name + "_log.txt")
	log_file = open(filename, 'w')
	sys.stdout = writer(saved_stdout, log_file)
	print(name)

def get_classifiers():
	enabled_classifier_names = ["GradientBoosting", "RandomForest", "GaussianNB", "RadiusNeighbors", "MLP", "SGD", "LinearSVC", "LINSVC", "RBFSVC", "DecisionTree"]
	params = {
		"GradientBoosting": {},
		"RandomForest": {"n_estimators":50},
		"GaussianNB": {},
		"RadiusNeighbors": {},
		"MLP": {},
		"SGD": {},
		"LinearSVC": {},
		"LINSVC": {"kernel":"linear"},
		"RBFSVC": {"kernel":"rbf"},
		"DecisionTree": {}
	}
	classifiers = {
		"GradientBoosting": GradientBoostingClassifier(**params["GradientBoosting"]),
		"RandomForest": RandomForestClassifier(**params["RandomForest"]),
		"GaussianNB": GaussianNB(**params["GaussianNB"]),
		"RadiusNeighbors": RadiusNeighborsClassifier(**params["RadiusNeighbors"]),
		"MLP": MLPClassifier(**params["MLP"]),
		"SGD": SGDClassifier(**params["SGD"]),
		"LinearSVC": LinearSVC(**params["LinearSVC"]),
		"RBFSVC": SVC(**params["RBFSVC"]),
		"DecisionTree": DecisionTreeClassifier(**params["DecisionTree"])
	}
	enabled_classifiers = [(name, classifier) for name, classifier in classifiers.items() if name in enabled_classifier_names]
	return enabled_classifiers

def init_config():

	PROCESSORS = 12
	CLASSES = 9

	# image files
	images = dict(
		train_raw = "../imgs/region_raw.png",
		train_lbl = "../imgs/region_mask.png",
		test_raw = "../imgs/search_raw.png",
		test_lbl = "../imgs/search_mask.png"
	)

	# superpixels
	segment = dict(
		approx_num_superpixels = 5000,
		num_levels = 4,
		iterations = 100
	)

	# preprocessor
	preprocessor = dict(
		normalize = True,
		reduce_features = True,
		reducer_type = Reducers.pca,
		explained_variance = 0.96
	)

	# saving
	save = "results/"

	params = Namespace(
		processors = PROCESSORS,
		classes = CLASSES,
		image_paths = images,
		segment = segment,
		preprocess = preprocessor,
		save_path = save
	)

	return params

##########
## RUN: ##
##########
if __name__ == '__main__':
	main()
