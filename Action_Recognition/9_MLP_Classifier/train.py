import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import struct
import subprocess
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'							#  Suppress TensorFlow barf.

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from build_mlp_model import build_model

epoch_start_time = None												#  time.process_time() when an epoch begins.
epoch_end_time = None												#  Make these global so everybody can reach them.

class BookmarkCallback(keras.callbacks.Callback):
	def on_train_begin(self, logs=None):
		return

	def on_train_end(self, logs=None):
		return

	def on_epoch_begin(self, epoch, logs=None):
		global epoch_start_time
		epoch_start_time = time.process_time()
		return

	def on_epoch_end(self, epoch, logs=None):
		global epoch_start_time
		global epoch_end_time
		epoch_end_time = time.process_time()						#  Save time when epoch finishes.

		if not os.path.exists('bookmark.txt'):						#  If there is no bookmark, make a bookmark
			fh = open('bookmark.txt', 'w')
			fh.write('Epoch\tLoss\tVal.Loss\tTime\n')
			fh.write('0\tinf\tinf\t' + str(epoch_end_time - epoch_start_time))
			fh.close()

		if not os.path.exists('bookmark.log'):						#  If there is no log, make a log
			fh = open('bookmark.log')
			fh.write('Epoch\tLoss\tVal.Loss\tTime\n')
			fh.close()

		fh = open('bookmark.txt', 'r')								#  Read the existing bookmark
		lines = fh.readlines()
		fh.close()

		shutil.copy('bookmark.txt', 'bookmark.backup.txt')			#  Write a backup of the existing bookmark

		fh = open('bookmark.txt', 'w')								#  Write the new (current at epoch's end) bookmark
		fh.write('Epoch\tLoss\tVal.Loss\tTime\n')
		fh.write(str(epoch) + '\t' + str(logs['loss']) + '\t' + str(logs['val_loss']) + '\t' + str(epoch_end_time - epoch_start_time))
		fh.close()

		loss = []													#  Prepare to collect history for intermediate graph
		val_loss = []

		fh = open('bookmark.log', 'r')								#  Read the existing log
		lines = fh.readlines()
		fh.close()

		shutil.copy('bookmark.log', 'bookmark.backup.log')			#  Make a copy of the existing log

		fh = open('bookmark.log', 'w')								#  Write a new log
		linectr = 0
		for line in lines:											#  Copy the existing contents
			fh.write(line)
			if linectr > 0:
				arr = line.strip().split('\t')
				loss.append(float(arr[1]))							#  At the same time, read them into our graphable lists
				val_loss.append(float(arr[2]))
			linectr += 1
																	#  Write the new (current at epoch's end) losses
		fh.write(str(epoch) + '\t' + str(logs['loss']) + '\t' + str(logs['val_loss']) + '\t' + str(epoch_end_time - epoch_start_time) + '\n')
		loss.append(logs['loss'])									#  Add these to the graphable lists
		val_loss.append(logs['val_loss'])
		fh.close()
																	#  Graph the intermediate loss
		plt.plot(range(len(loss)), loss, 'b', label='Training loss')
		plt.plot(range(len(loss)), val_loss, 'r', label='Validation loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig('latest.png')
		plt.clf()													#  Clear the graph--or it'll fill up with old plots

		return

	def on_test_begin(self, logs=None):
		return

	def on_test_end(self, logs=None):
		return

	def on_predict_begin(self, logs=None):
		return

	def on_predict_end(self, logs=None):
		return

	def on_train_batch_begin(self, batch, logs=None):
		return

	def on_train_batch_end(self, batch, logs=None):
		return

	def on_test_batch_begin(self, batch, logs=None):
		return

	def on_test_batch_end(self, batch, logs=None):
		return

	def on_predict_batch_begin(self, batch, logs=None):
		return

	def on_predict_batch_end(self, batch, logs=None):
		return

def main():
	params = get_command_line_params()								#  Collect parameters.
	if params['helpme'] or params['source'] is None:
		usage()
		return

	labels, label_weights, train_X, train_y, valid_X, valid_y = load_data(params)

	data_x = np.array([x[0] for x in train_X])						#  Save distances.
	data_y = np.array(train_y)										#  Save distributions.

	data_x_valid = np.array([x[0] for x in valid_X])				#  Save distances.
	data_y_valid = np.array(valid_y)								#  Save distributions.

	sample_weights_valid = []										#  There is no class_weight equivalent for validation,
	for sample_y in valid_y:										#  but we can use sample_weights.
		label_index = np.argmax(sample_y)
		sample_weights_valid.append( label_weights[label_index] )
	sample_weights_valid = np.array(sample_weights_valid, dtype=np.float32)

	if params['verbose']:
		print('Training set:   ' + str(len(train_y)) + ' samples')
		print('                ' + str(data_x.shape) + ' --> ' + str(data_y.shape))
		print('Validation set: ' + str(len(valid_y)) + ' samples')
		print('                ' + str(data_x_valid.shape) + ' --> ' + \
		                           str(data_y_valid.shape) + ' @ ' + str(sample_weights_valid.shape))
		print('')
		for i in range(0, len(labels)):
			print('[' + str(i) + '] ' + labels[i] + ': ' + str(label_weights[i]))
		print('[' + str(len(labels)) + '] ' + '*' + ': ' + str(label_weights[len(labels)]))
		print('')

		counts = {}
		for i in range(0, len(labels) + 1):
			counts[i] = 0
		for y in train_y:
			counts[ np.argmax(y) ] += 1

		print('Training Set:')
		for i in range(0, len(labels)):
			print('  [' + str(i) + '] ' + labels[i] + ': ' + str(counts[i]))
		print('  [' + str(len(labels)) + '] ' + '*' + ': ' + str(counts[ len(labels) ]))
		print('')

		counts = {}
		for i in range(0, len(labels) + 1):
			counts[i] = 0
		for y in valid_y:
			counts[ np.argmax(y) ] += 1

		print('Validation Set:')
		for i in range(0, len(labels)):
			print('  [' + str(i) + '] ' + labels[i] + ': ' + str(counts[i]))
		print('  [' + str(len(labels)) + '] ' + '*' + ': ' + str(counts[ len(labels) ]))
		print('')

	if not os.path.exists('bookmark.txt'):							#  I want to be able to check in and see
		fh = open('bookmark.txt', 'w')								#  how far training has gone
		fh.write('Epoch\tLoss\tVal.Loss\n')
		fh.write('0\tinf\tinf')
		fh.close()

	if not os.path.exists('bookmark.log'):
		fh = open('bookmark.log', 'w')
		fh.write('Epoch\tLoss\tVal.Loss\n')
		fh.close()

	model = build_model(params)										#  Build the model.

	latest = float('-inf')											#  Each is mlp *DOT* <iteration> *DOT* pb
	latest_modelpath = None
	for modelpath in [x for x in os.listdir('.') if x.endswith('.pb')]:
		arr = modelpath.split('.')
		if arr[0] == params['model-name']:
			iteration = int(arr[1])
			if iteration > latest:
				latest = iteration
				latest_modelpath = modelpath[:]
	if latest > float('-inf'):
		model.load_weights(latest_modelpath)
		epochs = params['target-epochs'] - latest
		print('>>> Resuming training from epoch ' + str(latest))
	else:
		epochs = params['target-epochs']
		model.summary()												#  Print the details

	filepath = params['model-name'] + '.{epoch:02d}.pb'
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	bookmark = BookmarkCallback()
	if params['use_early-stopping']:
		earlystop = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, patience=5)
		callbacksList = [checkpoint, bookmark, earlystop]
	else:
		callbacksList = [checkpoint, bookmark]
	history = model.fit( [data_x], data_y, \
	                     validation_data=( [data_x_valid], data_y_valid, sample_weights_valid ), \
	                     epochs=epochs, batch_size=params['batch-size'], \
	                     class_weight=label_weights, \
	                     callbacks=callbacksList)

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	plt.plot(range(len(loss)), loss, 'b', label='Training loss')
	plt.plot(range(len(loss)), val_loss, 'r', label='Validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()

	plt.savefig('MLP_batch' + str(params['batch-size']) + '_' + \
	               'epochs' + str(params['target-epochs']) + '_' + \
	                   'lr' + str(params['learning-rate']) + '.png')

	if params['verbose']:
		print('>>> Training complete. Making predictions on validation set.')
	y_hat = model.predict( [ data_x_valid ] )						#  Training is over. Predict.

	if params['verbose']:
		print('>>> Building confusion matrix.')
																	#  Build a confusion matrix:
																	#  Rows are predictions;
																	#  Columns are ground-truth.
	conf_mat = np.zeros((len(labels) + 1, len(labels) + 1), dtype=np.uint16)
	for i in range(0, y_hat.shape[0]):
		prediction = np.argmax( y_hat[i] )
		ground_truth_label = np.argmax( data_y_valid[i] )
		conf_mat[prediction, ground_truth_label] += 1

	fh = open('confusion-matrix.txt', 'w')
	fh.write('#  MLP confusion matrix made at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
	fh.write('\t' + '\t'.join(labels + ['*']) + '\n')				#  Write the column headers.
	for i in range(0, len(labels)):
		fh.write(labels[i] + '\t' + '\t'.join([str(x) for x in conf_mat[i]]) + '\n')
	fh.write('*' + '\t' + '\t'.join([str(x) for x in conf_mat[ len(labels) ]]) + '\n')
	fh.close()

	return

def load_data(params):
	fh = open(params['source'], 'r')
	lines = fh.readlines()
	fh.close()

	header = [x for x in lines if x[0] == '#']
	lines = [x for x in lines if x[0] != '#']

	reading_labels = False
	pre_allocated = False
	for line in header:
		if 'RECOGNIZABLE ACTIONS:' in line:
			reading_labels = True
		elif 'PRE-ALLOCATED' in line:
			pre_allocated = True
		elif reading_labels:
			labels = line[1:].strip().split('\t')
			reading_labels = False

	label_lkup_samples = {}											#  key:label ==> val:[ (sample), (sample), ... ]
	label_weights = {}
	for i in range(0, len(labels)):
		label_lkup_samples[i] = []
		label_weights[i] = 0.0
	label_weights[len(labels)] = 0.0
	label_lkup_samples[len(labels)] = []

	train_X = []
	train_y = []

	valid_X = []
	valid_y = []

	for line in lines:
		arr = line.strip().split('\t')								#  Split each line by tabs.
		if pre_allocated:
			target_set = arr[0]
			time_start = arr[1]
			time_stop = arr[2]
			source_enactment = arr[3]
			distances = [float(x) for x in arr[4:-2]]
			ground_truth_label = arr[-2]
			fair = (arr[-1] == 'fair')
		else:
			time_start = arr[0]
			time_stop = arr[1]
			source_enactment = arr[2]
			distances = [float(x) for x in arr[3:-2]]
			ground_truth_label = arr[-2]
			fair = (arr[-1] == 'fair')

		if fair:													#  Only design fair tests.
			x = np.array(distances)
			y = np.zeros(len(labels) + 1, dtype=np.uint8)

			if ground_truth_label == '*':
				y[ len(labels) ] = 1								#  Nothing-labels become a K+1-th hot.
				label_weights[ len(labels) ] += 1.0
				if not pre_allocated:
					label_lkup_samples[ len(labels) ].append( (x, y, time_start, time_stop, source_enactment, distances, ground_truth_label, fair) )
				else:
					if target_set == 'train':
						train_X.append( [x] )
						train_y.append( y )
					else:
						valid_X.append( [x] )
						valid_y.append( y )
			else:
				y[ labels.index(ground_truth_label) ] = 1			#  Something-labels become an K > i-th hot.
				label_weights[ labels.index(ground_truth_label) ] += 1.0
				if not pre_allocated:
					label_lkup_samples[ labels.index(ground_truth_label) ].append( (x, y, time_start, time_stop, source_enactment, distances, ground_truth_label, fair) )
				else:
					if target_set == 'train':
						train_X.append( [x] )
						train_y.append( y )
					else:
						valid_X.append( [x] )
						valid_y.append( y )

	if not pre_allocated:
		if params['shuffle']:
			for key in label_lkup_samples.keys():
				np.random.shuffle( label_lkup_samples[key] )

		fh = open('dataset.txt', 'w')
		fh.write('#  PRE-ALLOCATED\n')
		fh.write('#  RECOGNIZABLE ACTIONS:\n')
		fh.write('#    ' + '\t'.join(labels) + '\n')
		fh.write('#  Each line is:\n')
		fh.write('#    {train,test}    First-Timestamp    Final-Timestamp    Source-Enactment    ' + '    '.join(['Cost_' + label for label in labels]) + '    Ground-Truth-Label    {fair,unfair}\n')

		total = []
		for key in label_lkup_samples.keys():
			lim = int(np.floor(float(len(label_lkup_samples[key])) * params['train-portion']))
			for i in range(0, lim):
				total.append( tuple(['train'] + list(label_lkup_samples[key][i])) )
			for i in range(lim, len(label_lkup_samples[key])):
				total.append( tuple(['test'] + list(label_lkup_samples[key][i])) )
		np.random.shuffle( total )

		for i in range(0, len(total)):
			target_set = total[i][0]
			x = total[i][1]
			y = total[i][2]
			time_start = total[i][3]
			time_stop  = total[i][4]
			source_enactment = total[i][5]
			distances = total[i][6]
			ground_truth_label = total[i][7]
			fair = total[i][8]

			if target_set == 'train':
				train_X.append( [ x ] )
				train_y.append( y )
			else:
				valid_X.append( [ x ] )
				valid_y.append( y )

			fh.write(target_set + '\t' + time_start + '\t' + time_stop + '\t' + source_enactment + '\t' + \
			         '\t'.join([str(x) for x in distances]) + '\t' + ground_truth_label + '\t')
			if fair:
				fh.write('fair\n')
			else:
				fh.write('unfair\n')

		fh.close()

	if params['weight-lookup'] is None:
		max_label = 0
		max_count = 0
		for label, ctr in label_weights.items():
			if ctr > max_count:
				max_count = ctr
				max_label = label

		for label in label_weights.keys():
			if label_weights[label] > 0:
				label_weights[label] = float(max_count) / float(label_weights[label])
	else:
		fh = open(params['weight-lookup'], 'r')
		for line in fh.readlines():
			if line[0] != '#':
				arr = line.strip().split('\t')
				class_index = int(arr[0])
				class_weight = float(arr[1])
				label_weights[class_index] = class_weight
		fh.close()

	return labels, label_weights, train_X, train_y, valid_X, valid_y

def get_command_line_params():
	params = {}
	params['source'] = None

	params['shuffle'] = True										#  Default.
	params['learning-rate'] = 0.001									#  Default.
	params['momentum'] = 0.0										#  Default.
	params['target-epochs'] = 50									#  Default.
	params['weight-lookup'] = None
	params['batch-size'] = 32										#  Default.

	params['model-name'] = 'mlp'									#  Give all models the same root name.
	params['train-portion'] = 0.8									#  Default.
	params['validation-portion'] = 0.2								#  Default.
	params['use_early-stopping'] = False

	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-src', \
	         '-shuffle', '-lr', '-m', '-e', '-b', \
	         '-train', '-valid', '-W', '-early', \
	         '-v', '-?', '-help', '--help']

	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-v':
				params['verbose'] = True
			elif sys.argv[i] == '-shuffle':
				params['shuffle'] = True
			elif sys.argv[i] == '-early':
				params['use_early-stopping'] = True
			elif sys.argv[i] == '-?' or sys.argv[i] == '-help' or sys.argv[i] == '--help':
				params['helpme'] = True
			else:
				argtarget = sys.argv[i]
		else:
			argval = sys.argv[i]

			if argtarget is not None:
				if argtarget == '-src':
					params['source'] = argval
				elif argtarget == '-lr':
					params['learning-rate'] = float(argval)
				elif argtarget == '-m':
					params['momentum'] = float(argval)
				elif argtarget == '-e':
					params['target-epochs'] = int(argval)
				elif argtarget == '-b':
					params['batch-size'] = int(argval)
				elif argtarget == '-train':
					params['train-portion'] = max(0.0, min(float(argval), 1.0))
					params['validation-portion'] = 1.0 - params['train-portion']
				elif argtarget == '-valid':
					params['validation-portion'] = max(0.0, min(float(argval), 1.0))
					params['train-portion'] = 1.0 - params['validation-portion']
				elif argtarget == '-W':
					params['weight-lookup'] = argval

	if params['train-portion'] == 0.0 or params['validation-portion'] == 0.0 or params['train-portion'] + params['validation-portion'] != 1.0:
		print('>>> ERROR: Invalid values received for training-set portion and/or validation-set portion.')
		print('           Resorting to defaults.')
		params['train-portion'] = 0.8
		params['validation-portion'] = 0.2

	return params

def usage():
	print('Build and train an MLP using DTW distance data.')
	print('')
	print('Usage:  python3 train.py <parameters, preceded by flags>')
	print(' e.g.:  python3 train.py -src matching_costs-split3.txt -e 50 -shuffle -v -W class_weights.txt')
	print(' e.g.:  python3 train.py -src dataset.txt -e 50 -v -W class_weights.txt')
	print('')
	print('Flags:  -src    REQUIRED.')
	print('        -lr     Learning rate. Default is 0.001.')
	print('        -m      Momentum. Default is 0.0.')
	print('        -e      Number of epochs. Default is 50.')
	print('        -b      Batch size. Default is 32.')
	print('        -early  Apply early stopping.')
	print('')
	print('        -v      Enable verbosity')
	print('        -?      Display this message')
	return

if __name__ == '__main__':
	main()
