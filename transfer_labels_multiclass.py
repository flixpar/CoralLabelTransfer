import os
import sys
import gc
import time
import shutil

import cv2
import numpy as np
import scipy.stats as stats

import yaml
import pickle
import random
from enum import Enum
from tqdm import tqdm

import itertools as it
import multiprocessing as mp
import multiprocessing.sharedctypes
import ctypes

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from sklearn.externals import joblib

from SuperPixel import SuperPixel
from Preprocessor import Preprocessor, Reducers
from labeling_utils import *

#############
## CONFIG: ##
#############

VERSION = 13
PROCESSORS = 7
CLASSIFIER_JOBS = 4

# image files
image_config = dict(
	src_img_raw = "imgs/region_raw.png",
	src_img_lbl = "imgs/region_mask.png",
	search_img_raw = "imgs/search_raw.png",
	search_img_lbl = "imgs/search_mask.png"
)

# superpixels
superpixel_config = dict(
	approx_num_superpixels = 10000,
	num_levels = 5,
	iterations = 100
)

# preprocessor
preprocessor_config = dict(
	normalize = True,
	reduce_features = True,
	reducer_type = Reducers.pca,
	explained_variance = 0.98
)

# SVM config
SVM_TYPE = "LIBSVM"
libsvm_config = dict(
	kernel = "rbf",
	C = 0.1,
	probability = False,
	cache_size = 2000,
	class_weight = "balanced"
)
liblinear_config = dict(
	C = 0.1,
	loss = "hinge"
)

# saving
class filemode(Enum):
	READ = 0
	WRITE = 1
saveconfig = dict(
	superpixels = filemode.READ,
	training = filemode.WRITE,
	prediction = filemode.WRITE,
	svm_compression = 3
)
savefiles = dict(
	src_superpixels = "saves/src_superpixels.pkl",
	search_superpixels = "saves/search_superpixels.pkl",
	svm = "results/v{0:d}/svm.pkl".format(VERSION),
	predictions = "results/v{0:d}/predictions.pkl".format(VERSION),
	confusion = "results/v{0:d}/confusion.png".format(VERSION)
)

# save config
config = ["MULTI", VERSION, PROCESSORS, image_config, superpixel_config, preprocessor_config, SVM_TYPE, libsvm_config, liblinear_config, saveconfig, savefiles]
try:
	os.mkdir("results/v%d/" % VERSION)
except FileExistsError:
	print("Error. Version %d already exists." % VERSION)
	exit()
with open("results/v%d/config.yml"%VERSION, 'w') as config_file:
	yaml.dump(config, config_file)

# setup logging
saved_stdout = sys.stdout
log_file = open("results/v%d/log.txt"%VERSION, 'w')
sys.stdout = writer(sys.stdout, log_file)

################
##    MAIN:   ##
################
def main():

	############
	## SETUP: ##
	############

	print("Version %d" % VERSION)

	src_superpixels = []
	search_superpixels = []

	##############################
	## READ IMAGES AND CONVERT: ##
	##############################

	print("Reading images...")

	src_img = cv2.imread(image_config["src_img_raw"])
	src_labels = cv2.imread(image_config["src_img_lbl"])

	search_img = cv2.imread(image_config["search_img_raw"])
	search_labels = cv2.imread(image_config["search_img_lbl"])

	#######################
	## LOAD SUPERPIXELS: ##
	#######################

	if saveconfig["superpixels"] == filemode.READ:

		print("Reading SuperPixel data...")

		src_in_file = open(savefiles["src_superpixels"], 'rb')
		src_superpixels = pickle.load(src_in_file)
		assert(len(src_superpixels) != 0)
		avg_size = src_superpixels[0].size

		search_in_file = open(savefiles["search_superpixels"], 'rb')
		search_superpixels = pickle.load(search_in_file)
		assert(len(search_superpixels) != 0)

	else:

		##################
		## OVERSEGMENT: ##
		##################

		print("Segmenting...")
		superpixel_args = (superpixel_config["approx_num_superpixels"], superpixel_config["num_levels"], superpixel_config["iterations"])
		src_segment_mask, src_num_superpixels = oversegment(src_img, superpixel_args)
		search_segment_mask, search_num_superpixels = oversegment(search_img, superpixel_args)

		print("Calculating average superpixel shape...")
		avg_size = calcAvgSize(src_segment_mask, int(src_num_superpixels/4))
		avg_size_search = calcAvgSize(search_segment_mask, int(search_num_superpixels/4))

		print("average size: " + str(avg_size))
		print("search average size: " + str(avg_size_search))

		##########################
		## EXTRACT SUPERPIXELS: ##
		##########################

		print("Computing features and creating SuperPixels...")
		src_superpixels = extract_superpixels(src_img, src_labels, src_segment_mask, avg_size, src_num_superpixels)
		search_superpixels = extract_superpixels(search_img, search_labels, search_segment_mask, avg_size, search_num_superpixels)

		####################
		## WRITE RESULTS: ##
		####################

		print("Writing results...")

		src_out_file = open(savefiles["src_superpixels"], 'wb')
		pickle.dump(src_superpixels, src_out_file)
		src_out_file.close()

		search_out_file = open(savefiles["search_superpixels"], 'wb')
		pickle.dump(search_superpixels, search_out_file)
		search_out_file.close()

	########################
	## GET TRAINING DATA: ##
	########################

	print("Formatting training data...")

	features = [pixel.features for pixel in src_superpixels if pixel is not None]
	labels = [pixel.id for pixel in src_superpixels if pixel is not None]

	features = np.array(features)
	labels = np.array(labels)

	id_label_db = {sup.id: sup.label for sup in src_superpixels}

	#################
	## PREPROCESS: ##
	#################

	print("Preprocessing data for training...")
	preprocessor = Preprocessor(normalize=preprocessor_config["normalize"],
								reduce_features=preprocessor_config["reduce_features"],
								reducer_type=preprocessor_config["reducer_type"],
								explained_variance=preprocessor_config["explained_variance"])
	preprocessor.train(features)
	features = preprocessor.process(features)

	###############
	## TRAINING: ##
	###############

	if saveconfig["training"] == filemode.READ:

		print("Reading SVM models...")
		svm_file = open(savefiles["svm"], 'rb')
		svm = joblib.load(svm_file)
		svm_file.close()

	else:

		print("Training...")

		num_src_superpixels = len(src_superpixels)
		svm = trainSVM(features, labels, num_src_superpixels)

		###############
		## SAVE SVM: ##
		###############

		print("Saving SVM...")
		start_time = time.time()

		svm_file = open(savefiles["svm"], 'wb')
		joblib.dump(svm, svm_file, compress=saveconfig["svm_compression"])
		svm_file.close()

		end_time = time.time()
		elapsed_time = end_time - start_time
		print("Saving SVM took %.1f seconds" % elapsed_time)

	######################
	## GET SEARCH DATA: ##
	######################

	print("Formatting search data...")

	search_features = [pixel.features for pixel in search_superpixels if pixel is not None]
	search_features = np.array(search_features)
	search_features = preprocessor.process(search_features)

	search_labels = [pixel.label for pixel in search_superpixels if pixel is not None]
	search_labels = np.array(search_labels)

	##############
	## PREDICT: ##
	##############

	if saveconfig["prediction"] == filemode.READ:

		print("Reading predictions...")
		pred_file = open(savefiles["predictions"], 'rb')
		pred = pickle.load(pred_file)
		pred_labels = [id_label_db[p] for p in pred]
		pred_file.close()

	else:

		print("Predicting...")
		start_time = time.time()

		pred = svm.predict(search_features)
		pred_labels = [id_label_db[p] for p in pred]

		end_time = time.time()
		elapsed_time = end_time - start_time
		print("Prediction took %.1f seconds" % elapsed_time)

		#######################
		## SAVE PREDICTIONS: ##
		#######################

		print("Saving predictions...")
		pred_file = open(savefiles["predictions"], 'wb')
		pickle.dump(pred, pred_file)
		pred_file.close()

	###############
	## EVALUATE: ##
	###############

	report = metrics.classification_report(search_labels, pred_labels)
	acc = metrics.accuracy_score(search_labels, pred_labels)
	confusion = metrics.confusion_matrix(search_labels, pred_labels)

	print(report)
	print("Accuracy: %.4f" % acc)
	print()

	plotConfusionMatrix(confusion, 9, savefiles["confusion"])



#####################
## HELPER METHODS: ##
#####################

# takes the mask produced by oversegmenting and extracts each
# of the superpixels as a SuperPixel object, returns the list
# of all superpixels
def extract_superpixels(src_img, lbl_img, mask_img, avg_size, num_superpixels):

	threadpool = mp.Pool(PROCESSORS)

	src_img_shared = src_img.copy()
	lbl_img_shared = lbl_img.copy()
	mask_img_shared = mask_img.copy()

	args = zip(range(num_superpixels), it.repeat(src_img_shared), it.repeat(lbl_img_shared), it.repeat(mask_img_shared), it.repeat(avg_size))
	args = tqdm(args, total=num_superpixels)
	superpixels = threadpool.starmap(createSuperPixel, args)

	threadpool.close()
	gc.collect()
	return superpixels

# helper for creating superpixels
def createSuperPixel(*args):
	try:
		pixel = SuperPixel(*args)
		return pixel
	except ValueError as err:
		tqdm.write("Skipping SuperPixel. " + str(err))

# Train a OneVsRestClassifier
def trainSVM(features, labels, total_features):

	start_time = time.time()

	if SVM_TYPE == "LIBLINEAR":
		classifier = LinearSVC(C=liblinear_config["C"], loss=liblinear_config["loss"])
	elif SVM_TYPE == "LIBSVM":
		classifier = SVC(kernel=libsvm_config["kernel"], C=libsvm_config["C"], probability=libsvm_config["probability"])
	else:
		print("Error. Invalid SVM type.")
		exit(-1)

	svm = OneVsRestClassifier(classifier, n_jobs=CLASSIFIER_JOBS)
	svm.fit(features, labels)

	end_time = time.time()
	elapsed_time = end_time - start_time
	print("Training took %.2f seconds" % elapsed_time)

	return svm

def close():
	sys.stdout = saved_stdout
	log_file.close()


##########
## RUN: ##
##########
if __name__ == '__main__':
	main()
	close()
