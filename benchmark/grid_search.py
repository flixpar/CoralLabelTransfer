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

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from sklearn import metrics
from sklearn.externals import joblib

from SuperPixel import SuperPixel
from Preprocessor import Preprocessor, Reducers
from labeling_utils import *

def main():
	
	config = init()

	src_img, src_labels, search_img, search_labels = read_images(config["images"])

	ref_features, ref_labels, avg_size, label_db = get_features(src_img, src_labels, config["superpixels"], config["PROCESSORS"], mode="reference")
	search_features, search_labels, search_class_labels = get_features(search_img, search_labels, config["superpixels"], config["PROCESSORS"], mode="search", avg_size=avg_size)

	preprocessor = get_preprocessor(config["preprocessor"], ref_features)

	ref_features = preprocessor.process(ref_features)
	search_features = preprocessor.process(search_features)

	best_acc = 0.0
	best_config = {}
	for svm_params in config["svm_param_grid"]:
		init_run(config, svm_params)

		classifier = train_svm(ref_features, ref_labels, svm_params, config["PROCESSORS"])
		# save_classifier(classifier, config["save"]["svm"], config["save"]["svm_compression"])

		pred, pred_labels = predict(classifier, search_features, label_db)
		# save_predictions(pred, config["save"]["predictions"])

		report, acc, iou, precision, confusion = evaluate(search_class_labels, pred_labels)
		saveConfusionMatrix(confusion, config["classes"], config["save"]["confusion"])
		save_results(report, acc, iou, precision, confusion, config["save"]["results"])

		print(report)
		print("Accuracy: {0:.4f}".format(acc))
		print("Precision: {0:.4f}".format(precision))
		print("IOU: {0:.4f}".format(iou))
		print()

		if acc > best_acc:
			best_acc = acc
			best_config = svm_params

		del(classifier)
		del(pred)
		del(pred_labels)
		gc.collect()

	print("\n\nBest SVM Params:")
	print(best_config)
	print("Accuracy: {}\n".format(best_acc))


def init():
	global saved_stdout
	saved_stdout = sys.stdout

	config = init_config()
	return config

def init_run(config, svm_params):

	config["VERSION"] += 1
	VERSION = config["VERSION"]
	config["save"] = get_save_paths(VERSION)

	os.mkdir("results/v%d/" % VERSION)

	with open(config["save"]["config"], 'w') as config_file:
		yaml.dump(config, config_file)

	print("")

	log_file = open(config["save"]["log"], 'w')
	sys.stdout = writer(saved_stdout, log_file)

	print("Version %d" % config["VERSION"])
	print(svm_params)

def read_images(filenames):

	src_img = cv2.imread(filenames["src_img_raw"])
	src_labels = cv2.imread(filenames["src_img_lbl"])

	search_img = cv2.imread(filenames["search_img_raw"])
	search_labels = cv2.imread(filenames["search_img_lbl"])

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
		avg_size = calc_avg_size(segment_mask, int(num_spixels/4))

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

def train_svm(features, labels, params, n_processors):

	print("Training SVM...")
	start_time = time.time()

	svm = SVC(**params)
	classifier = OneVsRestClassifier(svm, n_jobs=n_processors)
	classifier.fit(features, labels)

	end_time = time.time()
	elapsed_time = end_time - start_time
	print("Training took %.1f seconds" % elapsed_time)

	return classifier

def save_classifier(classifier, filename, compression):
	print("Saving SVM...")
	start_time = time.time()

	svm_file = open(filename, 'wb')
	joblib.dump(classifier, svm_file, compress=compression)
	svm_file.close()

	end_time = time.time()
	elapsed_time = end_time - start_time
	print("Saving SVM took %.1f seconds" % elapsed_time)

def predict(classifier, features, id_label_db):
	print("Predicting...")
	start_time = time.time()

	pred = classifier.predict(features)
	pred_labels = [id_label_db[p] for p in pred]

	end_time = time.time()
	elapsed_time = end_time - start_time
	print("Prediction took %.1f seconds" % elapsed_time)

	return pred, pred_labels

def save_predictions(predictions, filename):
	print("Saving predictions...")
	pred_file = open(filename, 'wb')
	pickle.dump(predictions, pred_file)
	pred_file.close()

def evaluate(truth, pred):
	print("Evaluating...")
	report = metrics.classification_report(truth, pred)
	acc = metrics.accuracy_score(truth, pred)
	iou = metrics.jaccard_similarity_score(truth, pred)
	precision = metrics.precision_score(truth, pred, average="weighted")
	confusion = metrics.confusion_matrix(truth, pred)	
	return report, acc, iou, precision, confusion

def save_results(report, acc, iou, precision, confusion, filename):
	print("Saving results...")
	results_file = open(filename, 'w')
	results_file.write(report)
	results_file.write("\nAccuracy: {0:.4f}".format(acc))
	results_file.write("IOU: {0:.4f}".format(iou))
	results_file.write("Precision: {0:.4f}\n".format(precision))
	results_file.write(np.array2string(confusion))
	results_file.close()	

####################
## SETUP METHODS: ##
####################

def init_config():

	VERSION = 22
	PROCESSORS = 12
	CLASSES = 9

	# image files
	images = dict(
		src_img_raw = "imgs/region_raw.png",
		src_img_lbl = "imgs/region_mask.png",
		search_img_raw = "imgs/search_raw.png",
		search_img_lbl = "imgs/search_mask.png"
	)

	# superpixels
	superpixels = dict(
		approx_num_superpixels = 8000,
		num_levels = 5,
		iterations = 100
	)

	# preprocessor
	preprocessor = dict(
		normalize = True,
		reduce_features = True,
		reducer_type = Reducers.pca,
		explained_variance = 0.99
	)

	# SVM parameter grid
	parameter_grid = dict(
		kernel = ["rbf"],
		cache_size = [2000],	
		class_weight = ["balanced"],
		gamma = [0.01, 0.001, 0.0005, 0.0001, "auto"],
		C = [0.05, 0.1, 0.15, 0.2]
	)

	# create list of SVM parameters
	'''
	parameter_grid = dict(
		kernel = ["linear", "rbf"],
		cache_size = [2000],
		verbose = [True],
		class_weight = ["balanced"],
		C = [0.01, 0.05, 0.1, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]
	)
	param_grid_len = max([len(v) for v in parameter_grid.values()])
	svm_param_grid = []
	for i in range(param_grid_len):
		param = {}
		for key in parameter_grid.keys():
			index = i % len(parameter_grid[key])
			param[key] = parameter_grid[key][index]
		svm_param_grid.append(param)
	'''
	svm_param_grid = [dict()]
	for key in parameter_grid.keys():
		temp = []
		for e in svm_param_grid:
			for v in parameter_grid[key]:
				a = e.copy()
				a[key] = v
				temp.append(a)
		svm_param_grid.extend(temp)

	svm_param_grid = [l for l in svm_param_grid if len(l) == len(parameter_grid)]

	for l in svm_param_grid:
		print(l)

	# saving
	save = get_save_paths(VERSION)

	params = dict(
		VERSION = VERSION,
		PROCESSORS = PROCESSORS,
		classes = CLASSES,
		images = images,
		superpixels = superpixels,
		preprocessor = preprocessor,
		svm_param_grid = svm_param_grid,
		save = save
	)

	return params

def get_save_paths(VERSION):
	save = dict(
		config = "results/v{0:d}/config.yml".format(VERSION),
		log = "results/v{0:d}/log.txt".format(VERSION),
		predictions = "results/v{0:d}/predictions.pkl".format(VERSION),
		confusion = "results/v{0:d}/confusion.png".format(VERSION),
		results = "results/v{0:d}/results.txt".format(VERSION),
		svm = "results/v{0:d}/svm.pkl".format(VERSION),
		svm_compression = 3
	)
	return save



##########
## RUN: ##
##########
if __name__ == '__main__':
	main()

