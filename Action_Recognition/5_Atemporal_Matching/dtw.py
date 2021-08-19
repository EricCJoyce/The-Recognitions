#  $ sudo -i R														Run R
#  > install.packages("dtw")                                        Install Dynamic Time-Warping library so that rpy2 can call upon it
#  > q()															Quit R

import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
import rpy2.robjects as robj
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KDTree
from sklearn.neighbors import (KNeighborsClassifier, NeighborhoodComponentsAnalysis)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sys
import time

def main():
	params = getCommandLineParams()
	if params['helpme'] or (len(params['train']) == 0 and len(params['divided']) == 0) or \
	                       (len(params['valid']) == 0 and len(params['divided']) == 0):
		usage()
		return

	if params['verbose']:
		print('TRAINING SET SOURCES:')
		for filename in params['train']:
			print('\t' + filename)
		print('VALIDATION SET SOURCES:')
		for filename in params['valid']:
			print('\t' + filename)
		print('DIVIDED SET SOURCES:')
		for filename in params['divided']:
			print('\t' + filename)
		print('')

	#################################################################  Load, merge, clean, redistribute...

	all_actions, validation_actions, divide_actions = load_raw_actions(params)

	all_actions, validation_actions, divide_actions, seqs, validation_seqs, divide_seqs, vector_len = join_sequences(all_actions, validation_actions, divide_actions, params)
	params['vector_len'] = vector_len								#  Pack the vector-length into params

	all_actions, validation_actions, divide_actions, seqs, validation_seqs, divide_seqs = clean_sequences(all_actions, validation_actions, divide_actions, seqs, validation_seqs, divide_seqs, params)

	all_actions, validation_actions, seqs, validation_seqs = redistribute_divide_set(all_actions, validation_actions, divide_actions, seqs, validation_seqs, divide_seqs, params)

	#################################################################  Bookkeeping. Be able to look up: Validation set's sequence # 69
	if params['logging']:											#  comes from Enactment4, frames 669_66.9.png to 690_69.png, inclusive
		fh = open('validation-set-lookup.txt', 'w')
		for i in range(0, len(validation_seqs)):
			fh.write(str(i))
			fh.write('\t' + validation_seqs[i][0]['file'])
			fh.write('\t' + validation_seqs[i][0]['label'])
			fh.write('\t' + validation_seqs[i][0]['frame'])
			fh.write('\t' + validation_seqs[i][-1]['frame'] + '\n')
		fh.close()

	#################################################################  Build the training and validation sets we will actually use

	X = []															#  To contain time series
	y = []															#  To contain labels (indices into 'labels')

	X_train = []													#  To contain time series
	y_train = []													#  To contain labels (indices into 'labels')

	X_test = []														#  To contain time series
	y_test = []														#  To contain labels (indices into 'labels')

	X_test_lookup = []												#  Store tuples: (index-into-'validation_seqs', start-frame-inclusinv, end-frame-inclusive)

	#################################################################  Build sets of frames to fit and transform

	X_train_learn = []												#  To contain vectors/frames
	y_train_learn = []												#  To contain labels (indices into 'labels')

	X_test_learn = []												#  To contain vectors/frames
	y_test_learn = []												#  To contain labels (indices into 'labels')

	#  X_learn and y_learn are for illustration only: that's why they receive frames from both training and validation sets.
	X_learn = []													#  To contain vectors/frames
	y_learn = []													#  To contain labels (indices into 'labels')

	if params['verbose']:
		print('>>> Building training and validation sets...')
		print('    Training Set window length:   ' + str(params['window-length-T']))
		print('    Training Set window stride:   ' + str(params['window-stride-T']))
		print('    Validation Set window length: ' + str(params['window-length-V']))
		print('    Validation Set window stride: ' + str(params['window-stride-V']))
		print('    Vector length:                ' + str(params['vector_len']))

	#  NOTE: applying the sliding window AFTER actions have been allocated to either training or validation sets
	#        ensures that no redundant frames in sub-sequences will cross between training and validation sets!!!

	for action, seqdata in all_actions.items():						#  key is an action label;
		indices = seqdata['seq_indices'][:]							#  val is a dictionary: seq_indices, frame_ctr

		for i in range(0, len(indices)):							#  For every training-set sequence, seqs[indices[i]]...
			for frame in seqs[indices[i]]:							#  Anticipating that we will be doing metric learning or plotting,
				X_learn.append( frame['vec'] )						#  build a set of all the vectors--NOT time-series!!
				X_train_learn.append( frame['vec'] )
				y_learn.append(action)
				y_train_learn.append(action)

			if params['window-length-T'] < float('inf'):			#  Use window and stride
				if params['window-stride-T'] < float('inf'):		#  Finite stride
					for fr_head_index in range(0, len(seqs[indices[i]]) - params['window-length-T'], params['window-stride-T']):
						seq = []
						for fr_ctr in range(0, params['window-length-T']):
							seq.append( seqs[indices[i]][fr_head_index + fr_ctr]['vec'] )
						X.append( seq )
						X_train.append( seq )
						y.append(action)
						y_train.append(action)
				else:												#  Infinite stride: only read the window once
					seq = []
					for fr_ctr in range(0, min(len(seqs[indices[i]]), params['window-length-T'])):
						seq.append( seqs[indices[i]][fr_ctr]['vec'] )
					X.append( seq )
					X_train.append( seq )
					y.append(action)
					y_train.append(action)
			else:													#  Use the whole sequence
				seq = []
				for frame in seqs[indices[i]]:
					seq.append( frame['vec'] )						#  Build the sequence
				X.append( seq )										#  Append the sequence
				X_train.append( seq )								#  Append the sequence
				y.append(action)
				y_train.append(action)

	for action, seqdata in validation_actions.items():				#  key is an action label;
		indices = seqdata['seq_indices'][:]							#  val is a dictionary: seq_indices, frame_ctr

		for i in range(0, len(indices)):							#  For every validation-set sequence, validation_seqs[indices[i]]...
			for frame in validation_seqs[indices[i]]:				#  Anticipating that we will be doing metric learning or plotting,
				X_learn.append( frame['vec'] )						#  build a set of all the vectors--NOT time-series!!
				X_test_learn.append( frame['vec'] )
				y_learn.append(action)
				y_test_learn.append(action)

			if params['window-length-V'] < float('inf'):			#  Use window and stride
				if params['window-stride-V'] < float('inf'):		#  Finite stride
					for fr_head_index in range(0, len(validation_seqs[indices[i]]) - params['window-length-V'], params['window-stride-V']):
																	#  Append index into 'validation_seqs' of test-set sample
						X_test_lookup.append( (indices[i], fr_head_index, fr_head_index + params['window-length-V'] - 1) )
						seq = []
						for fr_ctr in range(0, params['window-length-V']):
							seq.append( validation_seqs[indices[i]][fr_head_index + fr_ctr]['vec'] )
						X.append( seq )
						X_test.append( seq )
						y.append(action)
						y_test.append(action)
				else:												#  Infinite stride: only read the window once
																	#  Append index into 'validation_seqs' of test-set sample
					X_test_lookup.append( (indices[i], 0, min(len(validation_seqs[indices[i]]), params['window-length-V']) - 1) )
					seq = []
					for fr_ctr in range(0, min(len(validation_seqs[indices[i]]), params['window-length-V'])):
						seq.append( validation_seqs[indices[i]][fr_ctr]['vec'] )
					X.append( seq )
					X_test.append( seq )
					y.append(action)
					y_test.append(action)
			else:													#  Use the whole sequence
																	#  Append index into 'validation_seqs' of test-set sample
				X_test_lookup.append( (indices[i], 0, len(validation_seqs[indices[i]]) - 1) )
				seq = []
				for frame in validation_seqs[indices[i]]:
					seq.append( frame['vec'] )						#  Build the sequence
				X.append( seq )										#  Append the sequence
				X_test.append( seq )								#  Append the sequence
				y.append(action)
				y_test.append(action)

	labels = sorted(np.unique(y))									#  Make an enumerable list of labels
	num_classes = len(labels)										#  Save the number of (unique) class labels

	knn = KNeighborsClassifier(n_neighbors=params['k'])				#  Create a nearest neighbor classifier
																	#  (Really, just for show here. DTW does the real classification)

	for i in range(0, len(y)):										#  Convert everything to indices into 'labels'
		y[i] = labels.index( y[i] )
	for i in range(0, len(y_train)):								#  Convert training set to indices into 'labels'
		y_train[i] = labels.index( y_train[i] )
	for i in range(0, len(y_test)):									#  Convert test set into indices into 'labels'
		y_test[i] = labels.index( y_test[i] )

	for i in range(0, len(y_learn)):								#  Convert everything to indices into 'labels'
		if y_learn[i] in labels:
			y_learn[i] = labels.index( y_learn[i] )
	for i in range(0, len(y_train_learn)):							#  Convert training set to indices into 'labels'
		if y_train_learn[i] in labels:
			y_train_learn[i] = labels.index( y_train_learn[i] )
	for i in range(0, len(y_test_learn)):							#  Convert test set into indices into 'labels'
		if y_test_learn[i] in labels:
			y_test_learn[i] = labels.index( y_test_learn[i] )

	if params['verbose']:
		print('\tX       = ' + str(len(X))       + ' time series with y       = ' + str(len(y)) + ' labels')
		print('\tX_train = ' + str(len(X_train)) + ' time series with y_train = ' + str(len(y_train)) + ' labels')
		print('\tX_test  = ' + str(len(X_test))  + ' time series with y_test  = ' + str(len(y_test)) + ' labels')
		print('')
		print('    ' + str(num_classes) + ' unique labels in data set:')
		for label in labels:
			print('\t' + label)
		print('')

	if params['metric'] != 'euclidean':								#  Are we applying metric learning?
		random_state = 0											#  Remove stochasticity
		if params['reduce'] is not None:							#  Are we REDUCING DIMENSIONALITY?
																	#  Dimensionality reduction must be AT MOST the number of classes
			target_dimensions = min(num_classes - 1, params['reduce'])

			if params['verbose']:
				if target_dimensions != params['reduce']:
					print('*\tCannot reduce dimensionality to requested ' + str(params['reduce']))
				print('>>> Reducing dimensionality from ' + str(vector_len) + ' to ' + str(target_dimensions) + '...')

			if params['metric'] == 'pca':
				if params['verbose']:
					print('>>> Applying PCA to reduced-dimension training set...')
																	#  Reduce dimension to 'target_dimensions' with PCA
				pca = make_pipeline(StandardScaler(), PCA(n_components=target_dimensions, random_state=random_state))
				pca.fit(X_train_learn, y_train_learn)
				if params['graph']:									#  Scatter plot?
					knn.fit(pca.transform(X_train_learn), y_train_learn)
																	#  Get a per-frame accuracy score
					acc_knn = knn.score(pca.transform(X_test_learn), y_test_learn)
					X_embedded = pca.transform(X_learn)				#  Embed the data set in 2 dimensions using the fitted model
					plt.figure()									#  Plot the projected points and show the evaluation score
					plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_learn, s=30, cmap='Set1')
					plt.title("PCA, KNN (k={})\nTest accuracy = {:.2f}".format(params['k'], acc_knn))
					plt.savefig('PCA_reduced-' + str(params['reduce']) + '.png')
			elif params['metric'] == 'lda':
				if params['verbose']:
					print('>>> Applying LDA to reduced-dimension training set...')
																	#  Reduce dimension to 'target_dimensions' with LinearDiscriminantAnalysis
				lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=target_dimensions))
				lda.fit(X_train_learn, y_train_learn)
				if params['graph']:									#  Scatter plot?
					knn.fit(lda.transform(X_train_learn), y_train_learn)
																	#  Get a per-frame accuracy score
					acc_knn = knn.score(lda.transform(X_test_learn), y_test_learn)
					X_embedded = lda.transform(X_learn)				#  Embed the data set in 2 dimensions using the fitted model
					plt.figure()									#  Plot the projected points and show the evaluation score
					plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_learn, s=30, cmap='Set1')
					plt.title("LDA, KNN (k={})\nTest accuracy = {:.2f}".format(params['k'], acc_knn))
					plt.savefig('LDA_reduced-' + str(params['reduce']) + '.png')
			elif params['metric'] == 'nca':
				if params['verbose']:
					print('>>> Applying NCA to reduced-dimension training set...')
																	#  Reduce dimension to 'target_dimensions' with NeighborhoodComponentAnalysis
				nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=target_dimensions, random_state=random_state))
				nca.fit(X_train_learn, y_train_learn)
				if params['graph']:									#  Scatter plot?
					knn.fit(nca.transform(X_train_learn), y_train_learn)
																	#  Get a per-frame accuracy score
					acc_knn = knn.score(nca.transform(X_test_learn), y_test_learn)
					X_embedded = nca.transform(X_learn)				#  Embed the data set in 2 dimensions using the fitted model
					plt.figure()									#  Plot the projected points and show the evaluation score
					plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_learn, s=30, cmap='Set1')
					plt.title("NCA, KNN (k={})\nTest accuracy = {:.2f}".format(params['k'], acc_knn))
					plt.savefig('NCA_reduced-' + str(params['reduce']) + '.png')
		else:
			if params['metric'] == 'pca':
				if params['verbose']:
					print('>>> Applying PCA to full-dimension training set...')
				pca = PCA(random_state=random_state)				#  Reduce dimension to 'target_dimensions' with PCA
				pca.fit(X_train_learn, y_train_learn)
				if params['graph']:									#  Scatter plot?
					knn.fit(pca.transform(X_train_learn), y_train_learn)
																	#  Get a per-frame accuracy score
					acc_knn = knn.score(pca.transform(X_test_learn), y_test_learn)
					X_embedded = pca.transform(X_learn)				#  Embed the data set in 2 dimensions using the fitted model
					plt.figure()									#  Plot the projected points and show the evaluation score
					plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_learn, s=30, cmap='Set1')
					plt.title("PCA, KNN (k={})\nTest accuracy = {:.2f}".format(params['k'], acc_knn))
					plt.savefig('PCA.png')
			elif params['metric'] == 'lda':
				if params['verbose']:
					print('>>> Applying LDA to full-dimension training set...')
				lda = LinearDiscriminantAnalysis()					#  Reduce dimension to 'target_dimensions' with LinearDiscriminantAnalysis
				lda.fit(X_train_learn, y_train_learn)
				if params['graph']:									#  Scatter plot?
					knn.fit(lda.transform(X_train_learn), y_train_learn)
																	#  Get a per-frame accuracy score
					acc_knn = knn.score(lda.transform(X_test_learn), y_test_learn)
					X_embedded = lda.transform(X_learn)				#  Embed the data set in 2 dimensions using the fitted model
					plt.figure()									#  Plot the projected points and show the evaluation score
					plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_learn, s=30, cmap='Set1')
					plt.title("LDA, KNN (k={})\nTest accuracy = {:.2f}".format(params['k'], acc_knn))
					plt.savefig('LDA.png')
			elif params['metric'] == 'nca':
				if params['verbose']:
					print('>>> Applying NCA to full-dimension training set...')
																	#  Reduce dimension to 'target_dimensions' with NeighborhoodComponentAnalysis
				nca = NeighborhoodComponentsAnalysis(random_state=random_state)
				nca.fit(X_train_learn, y_train_learn)
				if params['graph']:									#  Scatter plot?
					knn.fit(nca.transform(X_train_learn), y_train_learn)
																	#  Get a per-frame accuracy score
					acc_knn = knn.score(nca.transform(X_test_learn), y_test_learn)
					X_embedded = nca.transform(X_learn)				#  Embed the data set in 2 dimensions using the fitted model
					plt.figure()									#  Plot the projected points and show the evaluation score
					plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_learn, s=30, cmap='Set1')
					plt.title("NCA, KNN (k={})\nTest accuracy = {:.2f}".format(params['k'], acc_knn))
					plt.savefig('NCA.png')

		X = []														#  Rebuild, transformed
		X_train = []												#  Rebuild, transformed
		X_test = []													#  Rebuild, transformed

		for action, seqdata in all_actions.items():					#  key is an action label, including '*';
			indices = seqdata['seq_indices'][:]						#  val is a dictionary: seq_indices, frame_ctr

			for i in range(0, len(indices)):
				seq = []
				for frame in seqs[indices[i]]:
					if params['metric'] == 'pca':					#  Apply PCA
						seq.append( pca.transform([frame['vec']])[0] )
					elif params['metric'] == 'lda':					#  Apply LDA
						seq.append( lda.transform([frame['vec']])[0] )
					elif params['metric'] == 'nca':					#  Apply NCA
						seq.append( nca.transform([frame['vec']])[0] )

				X.append( seq )										#  Append the sequence
				X_train.append( seq )								#  Append the sequence

		for action, seqdata in validation_actions.items():				#  key is an action label, including '*';
			indices = seqdata['seq_indices'][:]							#  val is a dictionary: seq_indices, frame_ctr

			for i in range(0, len(indices)):
				seq = []
				for frame in validation_seqs[indices[i]]:
					if params['metric'] == 'pca':					#  Apply PCA
						seq.append( pca.transform([frame['vec']])[0] )
					elif params['metric'] == 'lda':					#  Apply LDA
						seq.append( lda.transform([frame['vec']])[0] )
					elif params['metric'] == 'nca':					#  Apply NCA
						seq.append( nca.transform([frame['vec']])[0] )

				X.append( seq )										#  Append the sequence
				X_test.append( seq )								#  Append the sequence

		if params['verbose']:
			print('')

	elif params['graph']:
		knn.fit(X_train_learn, y_train_learn)
		acc_knn = knn.score(X_test_learn, y_test_learn)				#  Get the accuracy score
		if params['graph']:											#  Scatter plot?
			X_embedded = np.array(X_learn)
			plt.figure()											#  Plot the projected points and show the evaluation score
			plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_learn, s=30, cmap='Set1')
			plt.title("Euclidean, KNN (k={})\nTest accuracy = {:.2f}".format(params['k'], acc_knn))
			plt.savefig('Euclidean.png')
			plt.clf()												#  Clear

	#################################################################  Set up R back-end
	R = rpy2.robjects.r												#  Shortcut to the R backend
	DTW = importr('dtw')											#  Shortcut to the R DTW library
	if params['verbose']:
		print('>>> Set up R back-end')
		print('')

	#################################################################  If we were given an isotonic mapping file, then load it here
	if params['iso-map'] is not None:
		if params['verbose']:
			print('>>> Loading isotonic mapping from file "' + params['iso-map'] + '"')

		fh = open(params['iso-map'], 'r')
		lines = fh.readlines()
		fh.close()

		params['iso-map'] = {}										#  Redefine this as a dictionary: key:(lower-bound, upper-bound) ==> val:probability

		for line in lines:
			if line[0] != '#':
				arr = line.strip().split('\t')

				if arr[0] == '*':
					lb = float('-inf')
				else:
					lb = float(arr[0])

				if arr[1] == '*':
					ub = float('inf')
				else:
					ub = float(arr[1])

				p = float(arr[2])

				params['iso-map'][ (lb, ub) ] = p

	#################################################################  Classify each in test set by evaluating warps to each in training set
	if params['verbose']:
		print('>>> Beginning predictions...')
																	#  Used when rendering videos of sequence matches:
	successctr = 0													#    Unique index for every successful classification
	errctr = 0														#    Unique index for every mismatch

	predictions = []												#  To become a list of indices into 'labels'
	confidences = []												#  Track (the winning labels') confidence measures for each prediction
	confidences_all = []											#  Track ALL labels' confidence measures for each prediction

	conf_pred_gt = []												#  Store tuples: (confidence, prediction, ground_truth) for WINNERS ONLY
	conf_pred_gt_all = []											#  Store tuples: (confidence, prediction, ground_truth) for ALL CLASSES

	test_ctr = 0
	current_q_seq = None
	for q_seq in X_test:											#  For every test-set sample, 'query'...
		query = np.array(q_seq)										#  Apply trim and convert to numpy

		query_seq_index = X_test_lookup[test_ctr][0]				#  Identify which sequence in 'validation_seqs' this is
		head_frame = X_test_lookup[test_ctr][1]
		tail_frame = X_test_lookup[test_ctr][2]

		if params['logging']:
			if query_seq_index != current_q_seq:					#  Track sub-sequence predictions per action
				if current_q_seq is not None:
					fh.close()
				fh = open('query-seq-' + str(query_seq_index) + '.log', 'w')
				fh.write('GROUND-TRUTH\t' + labels[ y_test[test_ctr] ] + '\n')
				current_q_seq = query_seq_index

			fh.write(str(head_frame) + '\t' + str(tail_frame) + '\n')

		if params['verbose']:
			print('')
			print('    Q: [' + ' '.join(["{:.2f}".format(x) for x in q_seq[0][:3]]) + ' ... ' + \
			                   ' '.join(["{:.2f}".format(x) for x in q_seq[0][-3:]]) + '] . . . [' + \
			                   ' '.join(["{:.2f}".format(x) for x in q_seq[-1][:3]]) + ' ... ' + \
			                   ' '.join(["{:.2f}".format(x) for x in q_seq[-1][-3:]]) + ']')
			print('    y_gt = ' + labels[ y_test[test_ctr] ] + ', query-index ' + str(query_seq_index) + \
			                                                   ', frames [' + str(head_frame) + ', ' + str(tail_frame) + ']')

		rq, cq = query.shape										#  Save numRows (height) and numCols (width)
		queryR = R.matrix(query, nrow=rq, ncol=cq)					#  Convert to R matrix

		distances = {}												#  Build distance-table: key=label;
		for label in labels:										#                        val=[list of 4-tuples for T-set seqs with this label]
			distances[label] = []									#                        Each is (distance, index into 'seqs', T_alignIdx, Q_alignIdx)

		i = 0
		for t_seq in X_train:										#  For every training-set sample, 'template'...
			template = np.array(t_seq)								#  Convert to numpy
			rt, ct = template.shape									#  Save numRows (height) and numCols (width)
			templateR = R.matrix(template, nrow=rt, ncol=ct)		#  Convert to R matrix

																	#  What is the cost of aligning this t_seq with q_seq?
			alignment = R.dtw(templateR, queryR, open_begin=False, open_end=False)

			dist = alignment.rx('normalizedDistance')[0][0]			#  (Normalized) cost of matching this query to this template
																	#  Save sequences of aligned frames (we might render them side by side)
			templateIndices = list(np.array(alignment.rx('index1s'), dtype=np.uint64)[0])
			queryIndices = list(np.array(alignment.rx('index2s'), dtype=np.uint64)[0])

			if len(distances[ labels[ y_train[i] ] ]) == 0:			#  Add this (dist, index, T_Idx, Q_Idx) tuple to list of distances for this label
				distances[ labels[ y_train[i] ] ].append( (dist, i, templateIndices, queryIndices) )
			else:													#  INSERT IN (ASC) ORDER so that each list is already sorted by the end of this loop
				j = 0
				while j < len(distances[ labels[ y_train[i] ] ]) and dist > distances[ labels[ y_train[i] ] ][j][0]:
					j += 1
				distances[ labels[ y_train[i] ] ].insert( j, (dist, i, templateIndices, queryIndices) )

			i += 1
																	#  Convert to [ (dist, label, index, T-align, Q-align),
																	#               (dist, label, index, T-align, Q-align), ... ]
		dists_sorted = sorted([(v[0][0], k, v[0][1], v[0][2], v[0][3]) for k, v in distances.items()])
																	#  Only send over (dist, label)
																	#  Get back a list of confidences
		c = confidences_from_distances([x[:2] for x in dists_sorted], params)
		confidences.append(c[0])									#  Store the winner's confidence
		confidences_all += list(zip(c, \
		                            [x[1] for x in dists_sorted], \
		                            [labels[ y_test[test_ctr] ] for x in range(0, len(c))]))

		if params['iso-map'] is not None:							#  We have been given an isotonic mapping file to apply
			brackets = sorted(params['iso-map'].keys())
			i = 0
			while i < len(brackets) and not (c[0] >= brackets[i][0] and c[0] < brackets[i][1]):
				i += 1
			p = params['iso-map'][ brackets[i] ]
			if p > params['threshold']:
				prediction = dists_sorted[0][1][:]					#  Save predicted label
				nearest_neighbor_index = dists_sorted[0][2]			#  Save the index of the best-matching sequence
				alignment_T = dists_sorted[0][3][:]					#  Save the frame indices of the training-set alignment
				alignment_Q = dists_sorted[0][4][:]					#  Save the frame indices of the query alignment

				predictions.append( labels.index(prediction) )		#  Convert label back to number
			else:
				prediction = None
				predictions.append(None)
		else:														#  No mapping: pick winners, maybe apply a threshold
			if c[0] > params['threshold']:

				prediction = dists_sorted[0][1][:]					#  Save predicted label
				nearest_neighbor_index = dists_sorted[0][2]			#  Save the index of the best-matching sequence
				alignment_T = dists_sorted[0][3][:]					#  Save the frame indices of the training-set alignment
				alignment_Q = dists_sorted[0][4][:]					#  Save the frame indices of the query alignment

				predictions.append( labels.index(prediction) )		#  Convert label back to number
			else:
				prediction = None
				predictions.append(None)

		if params['logging']:
			for label in sorted(distances.keys()):					#  Write confidence for this snippet/sequence to the current log file
				index = 0
				while index < len(dists_sorted) and dists_sorted[index][1] != label:
					index += 1
				fh.write('\t' + label + '\t' + str(c[index]) + '\n')

		if params['verbose']:
			maxlen = 0
			for label in distances.keys():
				if len(label) > maxlen:
					maxlen = len(label)
			for label in sorted(distances.keys()):
				index = 0
				while index < len(dists_sorted) and dists_sorted[index][1] != label:
					index += 1
				if prediction is not None and label == prediction:
					print('\t  * T:' + label + ' '*(maxlen - len(label)) + ' conf: ' + str(c[index]))
				else:
					print('\t    T:' + label + ' '*(maxlen - len(label)) + ' conf: ' + str(c[index]))

		if params['render']:										#  We are rendering

			if prediction != labels[ y_test[test_ctr] ]:			#  Rendering a misclassification
																	#  Init video object
				vid = cv2.VideoWriter('mismatch_' + str(errctr + 1) + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (params['renderw'], params['renderh']) )
				errctr += 1
			else:
				vid = cv2.VideoWriter('success_' + str(successctr + 1) + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (params['renderw'], params['renderh']) )
				successctr += 1

			#  validation_seqs[query_seq_index]  <--matched with-->  seqs[nearest_neighbor_index]
			#  alignment_Q                                           alignment_T
			#  (1-indexed)                                           (1-indexed)

			if prediction != '*':									#  A prediction was made. Something valid exists for seqs[nearest_neighbor_index], alignment_Q, and alignment_T
				for i in range(0, len(alignment_Q)):				#  For every frame of the alignment
																	#  Make a base-frame (black)
					frame = np.zeros((params['renderh'], params['renderw'], 3), dtype='uint8')

					#################################################  Open the source image from the query sequence
					img_q_fn  = validation_seqs[query_seq_index][ int(alignment_Q[i] - 1) ]['file']
					img_q_fn += '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/'
					img_q_fn += validation_seqs[query_seq_index][ int(alignment_Q[i] - 1) ]['frame']
																	#  Open the query image frame
					img_q = cv2.imread(img_q_fn, cv2.IMREAD_UNCHANGED)
					if img_q.shape[2] > 3:							#  Drop alpha channel
						img_q = cv2.cvtColor(img_q, cv2.COLOR_RGBA2RGB)
																	#  Add mask overlays
					maskcanvas = np.zeros((img_q.shape[0], img_q.shape[1], 3), dtype='uint8')
					maskfiles_for_frame = fetch_masks_for_enactment_frame(validation_seqs[query_seq_index][ int(alignment_Q[i] - 1) ]['file'], \
					                                                      validation_seqs[query_seq_index][ int(alignment_Q[i] - 1) ]['frame'], \
					                                                      params)
					for maskdata in maskfiles_for_frame:			#  Each is (object-class-name, BBox, mask-filename)
						g = validation_seqs[query_seq_index][ int(alignment_Q[i] - 1) ]['vec'][params['classes'].index(maskdata[0]) + 12]
																	#  Open the mask file
						mask = cv2.imread(maskdata[2], cv2.IMREAD_UNCHANGED)
						mask[mask > 1] = 1							#  All things greater than 1 become 1
																	#  Extrude to three channels
						mask = mask[:, :, None] * np.ones(3, dtype='uint8')[None, None, :]
																	#  Convert this to a graphical overlay
						mask[:, :, 0] *= int(round(params['colors'][ maskdata[0] ][2] * g))
						mask[:, :, 1] *= int(round(params['colors'][ maskdata[0] ][1] * g))
						mask[:, :, 2] *= int(round(params['colors'][ maskdata[0] ][0] * g))

						cv2.rectangle(mask, (maskdata[1][0], maskdata[1][1]), \
						                    (maskdata[1][2], maskdata[1][3]), (params['colors'][ maskdata[0] ][0], \
						                                                       params['colors'][ maskdata[0] ][1], \
						                                                       params['colors'][ maskdata[0] ][2]), 1)

						maskcanvas += mask							#  Add mask to mask accumulator
						maskcanvas[maskcanvas > 255] = 255			#  Clip accumulator to 255
																	#  Add mask accumulator to source frame
					img_q = cv2.addWeighted(img_q, 1.0, maskcanvas, 0.7, 0)
					img_q = cv2.cvtColor(img_q, cv2.COLOR_RGBA2RGB)	#  Flatten alpha
																	#  Resize the composite image
					img_q = cv2.resize(img_q, (int(round(params['renderw'] * 0.5)), int(round(params['renderh'] * 0.5))), interpolation=cv2.INTER_AREA)

					#################################################  Open the source image from the template sequence
					img_t_fn  = seqs[nearest_neighbor_index][ int(alignment_T[i] - 1) ]['file']
					img_t_fn += '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/'
					img_t_fn += seqs[nearest_neighbor_index][ int(alignment_T[i] - 1) ]['frame']
																	#  Open the template image frame
					img_t = cv2.imread(img_t_fn, cv2.IMREAD_UNCHANGED)
					if img_t.shape[2] > 3:							#  Drop alpha channel
						img_t = cv2.cvtColor(img_t, cv2.COLOR_RGBA2RGB)
																	#  Add mask overlays
					maskcanvas = np.zeros((img_t.shape[0], img_t.shape[1], 3), dtype='uint8')
					maskfiles_for_frame = fetch_masks_for_enactment_frame(seqs[nearest_neighbor_index][ int(alignment_T[i] - 1) ]['file'], \
					                                                      seqs[nearest_neighbor_index][ int(alignment_T[i] - 1) ]['frame'], \
					                                                      params)
					for maskdata in maskfiles_for_frame:			#  Each is (object-class-name, BBox, mask-filename)
						g = seqs[nearest_neighbor_index][ int(alignment_T[i] - 1) ]['vec'][params['classes'].index(maskdata[0]) + 12]
																	#  Open the mask file
						mask = cv2.imread(maskdata[2], cv2.IMREAD_UNCHANGED)
						mask[mask > 1] = 1							#  All things greater than 1 become 1
																	#  Extrude to three channels
						mask = mask[:, :, None] * np.ones(3, dtype='uint8')[None, None, :]
																	#  Convert this to a graphical overlay
						mask[:, :, 0] *= int(round(params['colors'][ maskdata[0] ][2] * g))
						mask[:, :, 1] *= int(round(params['colors'][ maskdata[0] ][1] * g))
						mask[:, :, 2] *= int(round(params['colors'][ maskdata[0] ][0] * g))

						cv2.rectangle(mask, (maskdata[1][0], maskdata[1][1]), \
						                    (maskdata[1][2], maskdata[1][3]), (params['colors'][ maskdata[0] ][0], \
						                                                       params['colors'][ maskdata[0] ][1], \
						                                                       params['colors'][ maskdata[0] ][2]), 1)

						maskcanvas += mask							#  Add mask to mask accumulator
						maskcanvas[maskcanvas > 255] = 255			#  Clip accumulator to 255
																	#  Add mask accumulator to source frame
					img_t = cv2.addWeighted(img_t, 1.0, maskcanvas, 0.7, 0)
					img_t = cv2.cvtColor(img_t, cv2.COLOR_RGBA2RGB)	#  Flatten alpha
																	#  Resize the composite image
					img_t = cv2.resize(img_t, (int(round(params['renderw'] * 0.5)), int(round(params['renderh'] * 0.5))), interpolation=cv2.INTER_AREA)

					#################################################  Insert query frame and template frame into base frame
					frame[params['v_offset']:params['v_offset']+(int(round(params['renderh']*0.5))), :int(round(params['renderw']*0.5))] = img_q
					frame[params['v_offset']:params['v_offset']+(int(round(params['renderh']*0.5))), int(round(params['renderw']*0.5)):] = img_t

					cv2.putText(frame, 'Query: ' + validation_seqs[query_seq_index][0]['label'], \
					    (10, params['v_label_offset']), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (255, 255, 255, 255), 3)
					cv2.putText(frame, 'Train: ' + seqs[nearest_neighbor_index][0]['label'], \
					    (970, params['v_label_offset']), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (255, 255, 255, 255), 3)

					vid.write(frame)
			else:													#  No prediction was made; only validation_seqs[query_seq_index] exists
																	#  For every frame of the query sequence
				for i in range(0, len(validation_seqs[query_seq_index])):
																	#  Make a base-frame (black)
					frame = np.zeros((params['renderh'], params['renderw'], 3), dtype='uint8')

					#################################################  Open the source image from the query sequence
					img_q_fn  = validation_seqs[query_seq_index][ i ]['file']
					img_q_fn += '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/'
					img_q_fn += validation_seqs[query_seq_index][ i ]['frame']
																	#  Open the query image frame
					img_q = cv2.imread(img_q_fn, cv2.IMREAD_UNCHANGED)
					if img_q.shape[2] > 3:							#  Drop alpha channel
						img_q = cv2.cvtColor(img_q, cv2.COLOR_RGBA2RGB)
																	#  Add mask overlays
					maskcanvas = np.zeros((img_q.shape[0], img_q.shape[1], 3), dtype='uint8')
					maskfiles_for_frame = fetch_masks_for_enactment_frame(validation_seqs[query_seq_index][ i ]['file'], \
					                                                      validation_seqs[query_seq_index][ i ]['frame'], \
					                                                      params)
					for maskdata in maskfiles_for_frame:			#  Each is (object-class-name, BBox, mask-filename)
						g = validation_seqs[query_seq_index][ int(alignment_Q[i] - 1) ]['vec'][params['classes'].index(maskdata[0]) + 12]
																	#  Open the mask file
						mask = cv2.imread(maskdata[2], cv2.IMREAD_UNCHANGED)
						mask[mask > 1] = 1							#  All things greater than 1 become 1
																	#  Extrude to three channels
						mask = mask[:, :, None] * np.ones(3, dtype='uint8')[None, None, :]
																	#  Convert this to a graphical overlay
						mask[:, :, 0] *= int(round(params['colors'][ maskdata[0] ][2] * g))
						mask[:, :, 1] *= int(round(params['colors'][ maskdata[0] ][1] * g))
						mask[:, :, 2] *= int(round(params['colors'][ maskdata[0] ][0] * g))

						cv2.rectangle(mask, (maskdata[1][0], maskdata[1][1]), \
						                    (maskdata[1][2], maskdata[1][3]), (params['colors'][ maskdata[0] ][0], \
						                                                       params['colors'][ maskdata[0] ][1], \
						                                                       params['colors'][ maskdata[0] ][2]), 1)

						maskcanvas += mask							#  Add mask to mask accumulator
						maskcanvas[maskcanvas > 255] = 255			#  Clip accumulator to 255
																	#  Add mask accumulator to source frame
					img_q = cv2.addWeighted(img_q, 1.0, maskcanvas, 0.7, 0)
					img_q = cv2.cvtColor(img_q, cv2.COLOR_RGBA2RGB)	#  Flatten alpha
																	#  Resize the composite image
					img_q = cv2.resize(img_q, (int(round(params['renderw'] * 0.5)), int(round(params['renderh'] * 0.5))), interpolation=cv2.INTER_AREA)

					#################################################  Insert query frame and template frame into base frame
					frame[params['v_offset']:params['v_offset']+(int(round(params['renderh']*0.5))), :int(round(params['renderw']*0.5))] = img_q

					cv2.putText(frame, 'Query: ' + validation_seqs[query_seq_index][0]['label'], \
					    (10, params['v_label_offset']), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (255, 255, 255, 255), 3)
					cv2.putText(frame, 'Train: ', \
					    (970, params['v_label_offset']), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (255, 255, 255, 255), 3)

					vid.write(frame)

			vid.release()											#  Close video object

		test_ctr += 1

	#################################################################  Graph & report
	class_stats = {}												#  key: class label (string);
																	#  val: {TP, FP, FN, support=# in training set, support_v=# in test set}
	conf_correct = []
	conf_incorrect = []

	decision_ctr = 0
	no_decision_ctr = 0

	for classname in labels:
		class_stats[classname] = {'TP':0, 'FP':0, 'FN':0, \
		                          'support':len([x for x in y_train if x == labels.index(classname)]), \
		                          'support_v':len([x for x in y_test if x == labels.index(classname)])}

	for i in range(0, len(predictions)):
		true_label = labels[ y_test[i] ]							#  True label always exists FOR THIS DATA SET

		if predictions[i] is not None:
			conf_pred_gt.append( (confidences[i], labels[ predictions[i] ], true_label) )
		else:
			conf_pred_gt.append( (confidences[i], None, true_label) )

		if predictions[i] is not None:								#  ...but we may NOT have been confident enough to make a prediction
			decision_ctr += 1
			predicted_label = labels[ predictions[i] ]

			if predictions[i] == y_test[i]:
				class_stats[true_label]['TP'] += 1
				conf_correct.append(confidences[i])
			else:
				class_stats[true_label]['FN'] += 1
				class_stats[predicted_label]['FP'] += 1
				conf_incorrect.append(confidences[i])
		else:
			no_decision_ctr += 1
			class_stats[true_label]['FN'] += 1
			conf_incorrect.append(confidences[i])					#  We happen to know that, for this dataset, no decision is ever a correct no-decision

	conf_pred_gt = sorted(conf_pred_gt, reverse=True)				#  High confidence scores first

	now = datetime.datetime.now()									#  Build a distinct substring so I don't accidentally overwrite results.
	file_timestamp = now.strftime("%d") + now.strftime("%m") + now.strftime("%Y")[-2:] + 'T' + now.strftime("%H:%M:%S").replace(':', '')

	fh = open('conf-pred-gt_' + file_timestamp + '.txt', 'w')
	fh.write('#  DTW-Classifier predictions made at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
	fh.write('#  python3.5 ' + ' '.join(sys.argv) + '\n')
	fh.write('#  Confidence function is "' + params['conf-func'] + '"\n')
	fh.write('#  Confidence    Predicted-Label    Ground-Truth-Label\n')
	for tup in conf_pred_gt:
		if tup[1] is not None:
			fh.write(str(tup[0]) + '\t' + tup[1] + '\t' + tup[2] + '\n')
		else:
			fh.write(str(tup[0]) + '\t' + 'NO-DECISION' + '\t' + tup[2] + '\n')
	fh.close()

	confidences_all = sorted(confidences_all, reverse=True)			#  High confidence scores first
	fh = open('conf-pred-gt-all_' + file_timestamp + '.txt', 'w')
	fh.write('#  DTW-Classifier predictions made at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
	fh.write('#  python3.5 ' + ' '.join(sys.argv) + '\n')
	fh.write('#  Confidence function is "' + params['conf-func'] + '"\n')
	fh.write('#  Confidence    Label    Ground-Truth-Label\n')
	for tup in confidences_all:
		if tup[1] is not None:
			fh.write(str(tup[0]) + '\t' + tup[1] + '\t' + tup[2] + '\n')
		else:
			fh.write(str(tup[0]) + '\t' + 'NO-DECISION' + '\t' + tup[2] + '\n')
	fh.close()

	avg_conf_correct = np.mean(conf_correct)
	avg_conf_incorrect = np.mean(conf_incorrect)

	stddev_conf_correct = np.std(conf_correct)
	stddev_conf_incorrect = np.std(conf_incorrect)

	print('')
	print('Classification:')										#  Report
	print('===============')
	print('\tAccuracy\tPrecision\tRecall\tF1-score\tSupport\tV.Instances')
	meanAcc = []
	for k, v in class_stats.items():

		if v['support'] > 0 and v['support_v'] > 0:					#  ONLY ATTEMPT TO CLASSIFY IF THIS IS A "FAIR" QUESTION

			if v['TP'] + v['FP'] + v['FN'] == 0:
				acc    = 0.0
			else:
				acc    = float(v['TP']) / float(v['TP'] + v['FP'] + v['FN'])
			meanAcc.append(acc)

			if v['TP'] + v['FP'] == 0:
				prec   = 0.0
			else:
				prec   = float(v['TP']) / float(v['TP'] + v['FP'])

			if v['TP'] + v['FN'] == 0:
				recall = 0.0
			else:
				recall = float(v['TP']) / float(v['TP'] + v['FN'])

			if prec + recall == 0:
				f1     = 0.0
			else:
				f1     = float(2 * prec * recall) / float(prec + recall)

			support = class_stats[ k ]['support']
			support_v = class_stats[ k ]['support_v']

			print(k + '\t' + str(acc) + '\t' + str(prec) + '\t' + str(recall) + '\t' + str(f1) + '\t' + str(support) + '\t' + str(support_v))

	print('')
	print('Mean Avg. Accuracy:')
	print('===================')
	if len(meanAcc) > 0:
		print('\t' + str(np.mean(meanAcc)))
	else:
		print('N/A')

	print('')
	print('Confusion Matrix:')
	print('=================')
	print('\t[Ground-truth class]')
	print('\t' + '\t'.join(labels))									#  Print (ground-truth) column header

	ConfMat = np.zeros((num_classes, num_classes), dtype='uint8')
	for i in range(0, len(predictions)):
		if predictions[i] is not None:
			ConfMat[ predictions[i] ][ y_test[i] ] += 1

	for i in range(0, num_classes):
		print(labels[i] + '\t' + ','.join(str(x) for x in ConfMat[i]))
	print('')
	print('Diagonal         = ' + str(int(np.trace(ConfMat))))
	print('Total            = ' + str(int(np.sum(ConfMat))))
	print('Diagonal / Total = ' + str(np.trace(ConfMat) / np.sum(ConfMat)))
	print('')

	print('Total decisions made = ' + str(decision_ctr))
	print('Total non-decisions made = ' + str(no_decision_ctr))
	print('')

	print('Avg. Confidence when correct = ' + str(avg_conf_correct))
	print('Avg. Confidence when incorrect = ' + str(avg_conf_incorrect))
	print('')
	print('Std.Dev. Confidence when correct = ' + str(stddev_conf_correct))
	print('Std.Dev. Confidence when incorrect = ' + str(stddev_conf_incorrect))
	print('')

	return

#  Given a list of tuples (dist, label) that has already been sorted ascending according to dist.
def confidences_from_distances(dists, params):
	confidences = []

	if params['conf-func'] == 'sum2':								#  (Minimum distance + 2nd-minimal distance) / my distance
		min_d = dists[0][0]
		min_d_2 = dists[1][0]

		for i in range(0, len(dists)):								#  Now use it to compute putative confidences
			confidences.append( (min_d + min_d_2) / dists[i][0] )

	elif params['conf-func'] == 'sum3':								#  (Sum of three minimal distances) / my distance
		s = sum([x[0] for x in dists[:3]])

		for i in range(0, len(dists)):								#  Now use it to compute putative confidences
			confidences.append( s / dists[i][0] )

	elif params['conf-func'] == 'sum4':								#  (Sum of four minimal distances) / my distance
		s = sum([x[0] for x in dists[:4]])

		for i in range(0, len(dists)):								#  Now use it to compute putative confidences
			confidences.append( s / dists[i][0] )

	elif params['conf-func'] == 'sum5':								#  (Sum of five minimal distances) / my distance
		s = sum([x[0] for x in dists[:5]])

		for i in range(0, len(dists)):								#  Now use it to compute putative confidences
			confidences.append( s / dists[i][0] )

	elif params['conf-func'] == 'n-min-obsv':						#  Normalized minimum distance observed over your distance
		min_d = dists[0][0]
		for i in range(0, len(dists)):								#  Now use it to compute putative confidences
			confidences.append( min_d / dists[i][0] )
		s = sum(confidences)
		for i in range(0, len(dists)):								#  Normalize confidences
			confidences[i] /= s

	#################################################################  Do not actually use this one: it's an illustrative fail-case.
	elif params['conf-func'] == 'min-obsv':							#  Minimum distance observed over your distance.
		min_d = dists[0][0]
		for i in range(0, len(dists)):
			confidences.append( min_d / dists[i][0] )

	elif params['conf-func'] == 'max-marg':							#  'max-marg': second-best distance minus best distance. Worst match gets zero.
		for i in range(0, len(dists)):
			if i < len(dists) - 1:
				confidences.append( dists[i + 1][0] - dists[i][0] )
			else:
				confidences.append( 0.0 )

	else:															#  '2over1': second-best distance over best distance. Worst match gets zero.
		for i in range(0, len(dists)):
			if i < len(dists) - 1:
				confidences.append( dists[i + 1][0] / (dists[i][0] + params['epsilon']) )
			else:
				confidences.append( 0.0 )

	return confidences

#  Given an enactment like 'ArcSuit1', a frame like '21_2.1.png', and params['detsrc'] in {groundtruth, mask_rcnn_factual_0799}
#  return a list of all 3-tuples (object-class-name, BBox, mask-filename) where filename == 'frame'
#  Bbox is itself a 4-tuple
def fetch_masks_for_enactment_frame(enactment, frame, params):
	maskfiles_for_frame = []
	if params['detsrc'] == 'groundtruth':
		fh = open(enactment + '_groundtruth.txt', 'r')
	else:
		fh = open(enactment + '_' + params['detsrc'] + '_detections.txt', 'r')
	for line in fh.readlines():
		if line[0] != '#':
			arr = line.strip().split('\t')
			timestamp = arr[0]
			filename = arr[1]
			label = arr[2]
			object_class_name = arr[3]
			score = arr[4]
			bboxstr = arr[5]
			mask_filename = arr[6]
			if filename == frame and object_class_name != 'LeftHand' and object_class_name != 'RightHand':
				bbox = bboxstr.split(';')							#  Yields "x1,y1", "x2,y2"
				bbox = bbox[0].split(',') + bbox[1].split(',')		#  Yields ["x1", "y1"] + ["x2", "y2"] = ["x1", "y1", "x2", "y2"]
				bbox = tuple([int(x) for x in bbox])				#  Yields (x1, y1, x2, y2)
				maskfiles_for_frame.append( (object_class_name, bbox, mask_filename) )
	fh.close()
	return maskfiles_for_frame

#  Distribute everything in the "Divide" Set
def redistribute_divide_set(all_actions, validation_actions, divide_actions, seqs, validation_seqs, divide_seqs, params):
	if len(divide_seqs) > 0:
		if params['verbose']:
			print('$\tRe-distributing sequences of the divide set')

		for action, seqdata in divide_actions.items():				#  'action' is a label like Grab("Flashlight")
																	#  'seqdata' is {seq_indices:[int, int, int], frame_ctr:int}
			if len(seqdata['seq_indices']) == 1:					#  Only one sequence? Give it to the TRAINING SET
				if params['verbose']:
					print('$\t1 sequence, ' + action + ', added to TRAINING SET')


				if action not in all_actions:						#  Update all_actions, which is...
					all_actions[action] = {}						#  key: string ==> val: {seq_indices:[int, int, int], frame_ctr:int}
					all_actions[action]['seq_indices'] = []
					all_actions[action]['frame_ctr'] = 0
				all_actions[action]['seq_indices'].append( len(seqs) )
				all_actions[action]['frame_ctr'] += len(divide_seqs[ seqdata['seq_indices'][0] ])

				seqs.append( [] )									#  Append to seqs
				for frame in divide_seqs[ seqdata['seq_indices'][0] ]:
					seqs[-1].append( {} )
					seqs[-1][-1]['timestamp'] = frame['timestamp']
					seqs[-1][-1]['file'] = frame['file']
					seqs[-1][-1]['frame'] = frame['frame']
					seqs[-1][-1]['label'] = frame['label']
					seqs[-1][-1]['vec'] = frame['vec'][:]
			else:													#  More than one sequence? Divide it up among TRAINING and VALIDATION set
				lim = int(round(float(len(seqdata['seq_indices'])) * params['t-portion']))

				if params['verbose']:
					print('$\t' + str(lim) + ' sequences, ' + action + ', added to TRAINING SET')
					print('$\t' + str(len(seqdata['seq_indices']) - lim) + ' sequences, ' + action + ', added to VALIDATION SET')

				for i in range(0, lim):								#  Add to training set

					if action not in all_actions:					#  Update all_actions, which is...
						all_actions[action] = {}					#  key: string ==> val: {seq_indices:[int, int, int], frame_ctr:int}
						all_actions[action]['seq_indices'] = []
						all_actions[action]['frame_ctr'] = 0
					all_actions[action]['seq_indices'].append( len(seqs) )
					all_actions[action]['frame_ctr'] += len(divide_seqs[ seqdata['seq_indices'][i] ])

					seqs.append( [] )								#  Append to seqs
					for frame in divide_seqs[ seqdata['seq_indices'][i] ]:
						seqs[-1].append( {} )
						seqs[-1][-1]['timestamp'] = frame['timestamp']
						seqs[-1][-1]['file'] = frame['file']
						seqs[-1][-1]['frame'] = frame['frame']
						seqs[-1][-1]['label'] = frame['label']
						seqs[-1][-1]['vec'] = frame['vec'][:]

				for i in range(lim, len(seqdata['seq_indices'])):	#  Add to validation set

					if action not in validation_actions:			#  Update all_actions, which is...
						validation_actions[action] = {}				#  key: string ==> val: {seq_indices:[int, int, int], frame_ctr:int}
						validation_actions[action]['seq_indices'] = []
						validation_actions[action]['frame_ctr'] = 0
					validation_actions[action]['seq_indices'].append( len(validation_seqs) )
					validation_actions[action]['frame_ctr'] += len(divide_seqs[ seqdata['seq_indices'][i] ])

					validation_seqs.append( [] )					#  Append to validation_seqs
					for frame in divide_seqs[ seqdata['seq_indices'][i] ]:
						validation_seqs[-1].append( {} )
						validation_seqs[-1][-1]['timestamp'] = frame['timestamp']
						validation_seqs[-1][-1]['file'] = frame['file']
						validation_seqs[-1][-1]['frame'] = frame['frame']
						validation_seqs[-1][-1]['label'] = frame['label']
						validation_seqs[-1][-1]['vec'] = frame['vec'][:]

		if params['verbose']:
			print('')
			print('REDISTRIBUTED ACTIONS for TRAINING SET:')
			for k, v in sorted(all_actions.items()):
				if k == '*':
					print('\t<NEURTAL>: ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
				else:
					print('\t' + k + ': ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
			print('')

			print('REDISTRIBUTED ACTIONS for VALIDATION SET:')
			for k, v in sorted(validation_actions.items()):
				if k == '*':
					print('\t<NEURTAL>: ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
				else:
					print('\t' + k + ': ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
			print('')

	return all_actions, validation_actions, seqs, validation_seqs

#  Perform drops and filter out any sequences that are too short
def clean_sequences(all_actions, validation_actions, divide_actions, seqs, validation_seqs, divide_seqs, params):
	clean_all_actions = {}											#  Reset. This still counts AND tracks:
	clean_validation_actions = {}									#  key: string ==> val: {seq_indices:[int, int, int], frame_ctr:int}
	clean_divide_actions = {}

	if params['verbose']:											#  Tell user which labels will be dropped
		print('*\tSequences with fewer than ' + str(params['minlen']) + ' frames will be dropped')
		print('*\t<NEURTAL> will be dropped')
		for drop in params['drops']:
			print('*\t' + drop + ' will be dropped')				#  This is the RE-LABELED label to be dropped
		print('')

	clean_seqs = []													#  List of Lists of Dicts(key: timestamp ==> val: {filename, (re)label, vector})
																	#  [ [{}, {}, {}], [{}, {}], [{}, {}, ..., {}], ..., [{}, {}, {}] ]
																	#    sequence      sequence  sequence       ^         sequence
																	#                                           |
																	#                                        frame
	for i in range(0, len(seqs)):
		if len(seqs[i]) >= params['minlen']:						#  If this is longer than or equal to the minimum, keep it.
			action = seqs[i][0]['label']
			if action != '*' and action not in params['drops']:
				if action not in clean_all_actions:
					clean_all_actions[action] = {}
					clean_all_actions[action]['seq_indices'] = []
					clean_all_actions[action]['frame_ctr'] = 0
				clean_all_actions[action]['seq_indices'].append( len(clean_seqs) )
				clean_seqs.append( [] )								#  Add a sequence
				for frame in seqs[i]:								#  Add all frames
					clean_seqs[-1].append( {} )
					clean_seqs[-1][-1]['timestamp'] = frame['timestamp']
					clean_seqs[-1][-1]['file'] = frame['file']
					clean_seqs[-1][-1]['frame'] = frame['frame']
					clean_seqs[-1][-1]['label'] = frame['label']
					clean_seqs[-1][-1]['vec'] = frame['vec'][:]

				clean_all_actions[action]['frame_ctr'] += len(seqs[i])

	clean_validation_seqs = []										#  List of Lists of Dicts(key: timestamp ==> val: {filename, (re)label, vector})
																	#  [ [{}, {}, {}], [{}, {}], [{}, {}, ..., {}], ..., [{}, {}, {}] ]
																	#    sequence      sequence  sequence       ^         sequence
																	#                                           |
																	#                                        frame
	for i in range(0, len(validation_seqs)):
		if len(validation_seqs[i]) >= params['minlen']:				#  If this is longer than or equal to the minimum, keep it.
			action = validation_seqs[i][0]['label']
			if action != '*' and action not in params['drops']:
				if action not in clean_validation_actions:
					clean_validation_actions[action] = {}
					clean_validation_actions[action]['seq_indices'] = []
					clean_validation_actions[action]['frame_ctr'] = 0
				clean_validation_actions[action]['seq_indices'].append( len(clean_validation_seqs) )
				clean_validation_seqs.append( [] )					#  Add a sequence
				for frame in validation_seqs[i]:
					clean_validation_seqs[-1].append( {} )
					clean_validation_seqs[-1][-1]['timestamp'] = frame['timestamp']
					clean_validation_seqs[-1][-1]['file'] = frame['file']
					clean_validation_seqs[-1][-1]['frame'] = frame['frame']
					clean_validation_seqs[-1][-1]['label'] = frame['label']
					clean_validation_seqs[-1][-1]['vec'] = frame['vec'][:]

				clean_validation_actions[action]['frame_ctr'] += len(validation_seqs[i])


	clean_divide_seqs = []											#  List of Lists of Dicts(key: timestamp ==> val: {filename, (re)label, vector})
																	#  [ [{}, {}, {}], [{}, {}], [{}, {}, ..., {}], ..., [{}, {}, {}] ]
																	#    sequence      sequence  sequence       ^         sequence
																	#                                           |
																	#                                        frame
	for i in range(0, len(divide_seqs)):
		if len(divide_seqs[i]) >= params['minlen']:					#  If this is longer than or equal to the minimum, keep it.
			action = divide_seqs[i][0]['label']
			if action != '*' and action not in params['drops']:
				if action not in clean_divide_actions:
					clean_divide_actions[action] = {}
					clean_divide_actions[action]['seq_indices'] = []
					clean_divide_actions[action]['frame_ctr'] = 0
				clean_divide_actions[action]['seq_indices'].append( len(clean_divide_seqs) )
				clean_divide_seqs.append( [] )						#  Add a sequence
				for frame in divide_seqs[i]:						#  Add all frames
					clean_divide_seqs[-1].append( {} )
					clean_divide_seqs[-1][-1]['timestamp'] = frame['timestamp']
					clean_divide_seqs[-1][-1]['file'] = frame['file']
					clean_divide_seqs[-1][-1]['frame'] = frame['frame']
					clean_divide_seqs[-1][-1]['label'] = frame['label']
					clean_divide_seqs[-1][-1]['vec'] = frame['vec'][:]

				clean_divide_actions[action]['frame_ctr'] += len(divide_seqs[i])

	seqs = clean_seqs												#  Copy back
	validation_seqs = clean_validation_seqs							#  Copy back
	divide_seqs = clean_divide_seqs									#  Copy back
	del clean_seqs													#  Destroy the temp
	del clean_validation_seqs										#  Destroy the temp
	del clean_divide_seqs											#  Destroy the temp

	all_actions = dict(clean_all_actions.items())					#  Rebuild dictionaries
	validation_actions = dict(clean_validation_actions.items())
	divide_actions = dict(clean_divide_actions.items())

	if params['verbose']:
		print('FILTERED ACTIONS for TRAINING SET:')
		for k, v in sorted(all_actions.items()):
			if k == '*':
				print('\t<NEURTAL>: ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
			else:
				print('\t' + k + ': ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
		print('')

		print('FILTERED ACTIONS for VALIDATION SET:')
		for k, v in sorted(validation_actions.items()):
			if k == '*':
				print('\t<NEURTAL>: ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
			else:
				print('\t' + k + ': ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
		print('')

		print('FILTERED ACTIONS for DIVIDE SET:')
		for k, v in sorted(divide_actions.items()):
			if k == '*':
				print('\t<NEURTAL>: ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
			else:
				print('\t' + k + ': ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
		print('')

	return all_actions, validation_actions, divide_actions, seqs, validation_seqs, divide_seqs

#  Read all sequences. Perform any joins/relabelings.
def join_sequences(all_actions, validation_actions, divide_actions, params):
	reverse_joins = {}
	for k, v in params['joins'].items():
		for vv in v:
			if vv not in reverse_joins:
				reverse_joins[vv] = k
	if params['verbose']:											#  Tell user which labels will be joined/merged
		for k, v in params['joins'].items():
			for vv in v:
				print('$\t"' + vv + '" will be re-labeled as "' + k + '"')
		print('')

	seqs = []														#  List of Lists of Dicts(key: timestamp ==> val: {filename, (re)label, vector})
	validation_seqs = []											#  [ [{}, {}, {}], [{}, {}], [{}, {}, ..., {}], ..., [{}, {}, {}] ]
	divide_seqs = []												#    sequence      sequence  sequence       ^         sequence
																	#                                           |
																	#                                        frame
	all_actions = {}												#  Reset. This will now count AND track:
	validation_actions = {}											#  key: string ==> val: {seq_indices:[int, int, int], frame_ctr:int}
	divide_actions = {}
	vector_len = 0													#  I'd also like to know how long a vector is

	for filename in params['train']:
		fh = open(filename + '.enactment', 'r')
		lines = fh.readlines()
		fh.close()
		currentLabel = None
		for line in lines:
			if line[0] != '#':
				arr = line.strip().split('\t')
				timestamp = float(arr[0])
				srcfile = arr[1]									#  Frame file name (old data set does not have these)
				label = arr[2]
				vec = [float(x) for x in arr[3:]]
				if vector_len == 0:									#  Save vector length
					vector_len = len(vec)

				if label in reverse_joins:
					label = reverse_joins[label]

				if label not in all_actions:
					all_actions[label] = {}
					all_actions[label]['seq_indices'] = []
					all_actions[label]['frame_ctr'] = 0

				if label != currentLabel:
					currentLabel = label
					all_actions[currentLabel]['seq_indices'].append( len(seqs) )
					seqs.append( [] )

				all_actions[currentLabel]['frame_ctr'] += 1

				seqs[-1].append( {} )
				seqs[-1][-1]['timestamp'] = timestamp
				seqs[-1][-1]['file'] = filename
				seqs[-1][-1]['frame'] = srcfile
				seqs[-1][-1]['label'] = currentLabel
				seqs[-1][-1]['vec'] = vec[:]

	for filename in params['valid']:
		fh = open(filename + '.enactment', 'r')
		lines = fh.readlines()
		fh.close()
		currentLabel = None
		for line in lines:
			if line[0] != '#':
				arr = line.strip().split('\t')
				timestamp = float(arr[0])
				srcfile = arr[1]									#  Frame file name (old data set does not have these)
				label = arr[2]
				vec = [float(x) for x in arr[3:]]
				if vector_len == 0:									#  Save vector length
					vector_len = len(vec)

				if label in reverse_joins:
					label = reverse_joins[label]

				if label not in validation_actions:
					validation_actions[label] = {}
					validation_actions[label]['seq_indices'] = []
					validation_actions[label]['frame_ctr'] = 0

				if label != currentLabel:
					currentLabel = label
					validation_actions[currentLabel]['seq_indices'].append( len(validation_seqs) )
					validation_seqs.append( [] )

				validation_actions[currentLabel]['frame_ctr'] += 1

				validation_seqs[-1].append( {} )
				validation_seqs[-1][-1]['timestamp'] = timestamp
				validation_seqs[-1][-1]['file'] = filename
				validation_seqs[-1][-1]['frame'] = srcfile
				validation_seqs[-1][-1]['label'] = currentLabel
				validation_seqs[-1][-1]['vec'] = vec[:]

	for filename in params['divided']:
		fh = open(filename + '.enactment', 'r')
		lines = fh.readlines()
		fh.close()
		currentLabel = None
		for line in lines:
			if line[0] != '#':
				arr = line.strip().split('\t')
				timestamp = float(arr[0])
				srcfile = arr[1]									#  Frame file name (old data set does not have these)
				label = arr[2]
				vec = [float(x) for x in arr[3:]]
				if vector_len == 0:									#  Save vector length
					vector_len = len(vec)

				if label in reverse_joins:
					label = reverse_joins[label]

				if label not in divide_actions:
					divide_actions[label] = {}
					divide_actions[label]['seq_indices'] = []
					divide_actions[label]['frame_ctr'] = 0

				if label != currentLabel:
					currentLabel = label
					divide_actions[currentLabel]['seq_indices'].append( len(divide_seqs) )
					divide_seqs.append( [] )

				divide_actions[currentLabel]['frame_ctr'] += 1

				divide_seqs[-1].append( {} )
				divide_seqs[-1][-1]['timestamp'] = timestamp
				divide_seqs[-1][-1]['file'] = filename
				divide_seqs[-1][-1]['frame'] = srcfile
				divide_seqs[-1][-1]['label'] = currentLabel
				divide_seqs[-1][-1]['vec'] = vec[:]

	if params['verbose']:
		print('MERGED ACTIONS for TRAINING SET:')
		for k, v in sorted(all_actions.items()):
			if k == '*':
				print('\t<NEURTAL>: ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
			else:
				print('\t' + k + ': ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
		print('')

		print('MERGED ACTIONS for VALIDATION SET:')
		for k, v in sorted(validation_actions.items()):
			if k == '*':
				print('\t<NEURTAL>: ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
			else:
				print('\t' + k + ': ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
		print('')

		print('MERGED ACTIONS for DIVIDED SET:')
		for k, v in sorted(divide_actions.items()):
			if k == '*':
				print('\t<NEURTAL>: ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
			else:
				print('\t' + k + ': ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
		print('')

	return all_actions, validation_actions, divide_actions, seqs, validation_seqs, divide_seqs, vector_len

#  Survey the set of all actions in the training set only.
#  Actions/labels that occur in the validation set only should
#  be "recognized" as "nothing" or "no action."
def load_raw_actions(params):
	all_actions = {}												#  key: string ==> val: {seq_ctr:int, frame_ctr:int}
	validation_actions = {}											#  key: string ==> val: {seq_ctr:int, frame_ctr:int}
	divide_actions = {}												#  key: string ==> val: {seq_ctr:int, frame_ctr:int}

	for filename in params['train']:
		fh = open(filename + '.enactment', 'r')
		lines = fh.readlines()
		fh.close()
		currentLabel = None
		for line in lines:
			if line[0] != '#':
				arr = line.strip().split('\t')
				label = arr[2]

				if label not in all_actions:
					all_actions[label] = {}
					all_actions[label]['seq_ctr'] = 0
					all_actions[label]['frame_ctr'] = 0

				if label != currentLabel:
					currentLabel = label
					all_actions[currentLabel]['seq_ctr'] += 1

				all_actions[currentLabel]['frame_ctr'] += 1

	for filename in params['valid']:
		fh = open(filename + '.enactment', 'r')
		lines = fh.readlines()
		fh.close()
		currentLabel = None
		for line in lines:
			if line[0] != '#':
				arr = line.strip().split('\t')
				label = arr[2]

				if label not in validation_actions:
					validation_actions[label] = {}
					validation_actions[label]['seq_ctr'] = 0
					validation_actions[label]['frame_ctr'] = 0

				if label != currentLabel:
					currentLabel = label
					validation_actions[currentLabel]['seq_ctr'] += 1

				validation_actions[currentLabel]['frame_ctr'] += 1

	for filename in params['divided']:
		fh = open(filename + '.enactment', 'r')
		lines = fh.readlines()
		fh.close()
		currentLabel = None
		for line in lines:
			if line[0] != '#':
				arr = line.strip().split('\t')
				label = arr[2]

				if label not in divide_actions:
					divide_actions[label] = {}
					divide_actions[label]['seq_ctr'] = 0
					divide_actions[label]['frame_ctr'] = 0

				if label != currentLabel:
					currentLabel = label
					divide_actions[currentLabel]['seq_ctr'] += 1

				divide_actions[currentLabel]['frame_ctr'] += 1

	if params['verbose']:
		print('RAW DEFINED ACTIONS in TRAINING SET:')
		for k, v in sorted(all_actions.items()):
			if k == '*':
				print('\t<NEURTAL>: ' + str(v['frame_ctr']) + ' total frames, ' + str(v['seq_ctr']) + ' sequences')
			else:
				print('\t' + k + ': ' + str(v['frame_ctr']) + ' total frames, ' + str(v['seq_ctr']) + ' sequences')
		print('')

		print('RAW DEFINED ACTIONS in VALIDATION SET:')
		for k, v in sorted(validation_actions.items()):
			if k == '*':
				print('\t<NEURTAL>: ' + str(v['frame_ctr']) + ' total frames, ' + str(v['seq_ctr']) + ' sequences')
			else:
				print('\t' + k + ': ' + str(v['frame_ctr']) + ' total frames, ' + str(v['seq_ctr']) + ' sequences')
		print('')

		print('RAW DEFINED ACTIONS in DIVIDED SET:')
		for k, v in sorted(divide_actions.items()):
			if k == '*':
				print('\t<NEURTAL>: ' + str(v['frame_ctr']) + ' total frames, ' + str(v['seq_ctr']) + ' sequences')
			else:
				print('\t' + k + ': ' + str(v['frame_ctr']) + ' total frames, ' + str(v['seq_ctr']) + ' sequences')
		print('')

	return all_actions, validation_actions, divide_actions

def getCommandLineParams():
	params = {}
	params['train'] = []											#  List of filenames
	params['valid'] = []											#  List of filenames
	params['divided'] = []											#  List of filenames
	params['t-portion'] = 0.8										#  Among "divide" set sequences, allocate this portion to the training set
	params['v-portion'] = 1.0 - params['t-portion']					#  and this portion to the validation set

	params['drops'] = []											#  List of (RE-LABELED) lables to be dropped (Use underscores rather than spaces)

	params['window-length-T'] = float('inf')						#  By default, "infinite" windows cover entire (Training Set) sequences
	params['window-stride-T'] = float('inf')						#  By default, advance the window to infinity (past the end of the sequence)

	params['window-length-V'] = float('inf')						#  By default, "infinite" windows cover entire (Validation Set) sequences
	params['window-stride-V'] = float('inf')						#  By default, advance the window to infinity (past the end of the sequence)

	params['iso-map'] = None										#  By default, no isotonic mapping is applied
	params['conf-func'] = '2over1'									#  Default is distance of 2nd-best match over best match (plus epsilon)
	params['threshold'] = 0.0										#  Default is no threshold; just attempt to classify everything

	params['epsilon'] = 0.00001

	params['metric'] = 'euclidean'									#  Metric learning? (Default to none)
	params['minlen'] = 2											#  Smallest length of a sequence deemed acceptable for the dataset
	params['k'] = 1													#  Really just for show in the per-frame scatter-plot
	params['reduce'] = None											#  Reducing dimensionality in metric learning?
	params['graph'] = False											#  Scatter plot?

	params['render'] = False										#  Render video?
	params['renderw'] = 1920
	params['renderh'] = 1080
	params['v_label_offset'] = 150
	params['v_offset'] = 400
	params['fontsize'] = 1											#  For rendering text to images and videos
	params['User'] = 'vr1'											#  It used to be "admin", and I don't want to change a bunch of file paths when it changes again
	params['detsrc'] = 'groundtruth'

	params['joins'] = {}											#  key: new label; val: [old labels]
	params['verbose'] = False
	params['logging'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-t', '-v', '-d', '-tPor', '-vPor', '-x', \
	         '-Tw', '-Ts', '-Vw', '-Vs', \
	         '-iso', '-C', '-th', '-metric', \
	         '-minlen', '-k', '-reduce', '-joinfile', '-graph', \
	         '-render', '-renderw', '-renderh', '-fontsize', '-User', '-classes', '-colors', \
	         '-V', '-?', '-help', '--help']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-graph':
				params['graph'] = True
			elif sys.argv[i] == '-render':
				params['render'] = True
			elif sys.argv[i] == '-V':
				params['verbose'] = True
			elif sys.argv[i] == '-?' or sys.argv[i] == '-help' or sys.argv[i] == '--help':
				params['helpme'] = True
			else:
				argtarget = sys.argv[i]
		else:
			argval = sys.argv[i]

			if argtarget is not None:
				if argtarget == '-t':
					params['train'].append(argval)
				elif argtarget == '-v':
					params['valid'].append(argval)
				elif argtarget == '-d':
					params['divided'].append(argval)
				elif argtarget == '-tPor':
					params['t-portion'] = max(0.0, min(1.0, float(argval)))
					params['v-portion'] = 1.0 - params['t-portion']
				elif argtarget == '-vPor':
					params['v-portion'] = max(0.0, min(1.0, float(argval)))
					params['t-portion'] = 1.0 - params['v-portion']

				elif argtarget == '-x':
					params['drops'].append(argval.replace('_', ' '))

				elif argtarget == '-Tw':
					params['window-length-T'] = int(argval)
				elif argtarget == '-Ts':
					params['window-stride-T'] = int(argval)
				elif argtarget == '-Vw':
					params['window-length-V'] = int(argval)
				elif argtarget == '-Vs':
					params['window-stride-V'] = int(argval)

				elif argtarget == '-iso':
					params['iso-map'] = argval
				elif argtarget == '-C':
					params['conf-func'] = argval
				elif argtarget == '-th':
					params['threshold'] = float(argval)
				elif argtarget == '-metric':
					params['metric'] = argval

				elif argtarget == '-joinfile':
					fh = open(argval, 'r')
					lines = fh.readlines()
					fh.close()
					for line in lines:
						if line[0] != '#':
							arr = line.strip().split('\t')
							newlabel = arr[0]
							oldlabel = arr[1]
							if newlabel not in params['joins']:
								params['joins'][newlabel] = []
							if oldlabel not in params['joins'][newlabel]:
								params['joins'][newlabel].append(oldlabel)
				elif argtarget == '-minlen':
					params['minlen'] = int(argval)
				elif argtarget == '-k':
					params['k'] = int(argval)
				elif argtarget == '-reduce':
					params['reduce'] = int(argval)

				elif argtarget == '-renderw':
					params['renderw'] = int(argval)
				elif argtarget == '-renderh':
					params['renderh'] = int(argval)
				elif argtarget == '-User':
					params['User'] = argval
				elif argtarget == '-fontsize':
					params['fontsize'] = float(argval)

	if len(params['divided']) > 0 and (params['t-portion'] != 1.0 - params['v-portion'] or params['t-portion'] == 1.0 \
	                                                                                    or params['t-portion'] == 0.0 \
	                                                                                    or params['v-portion'] == 1.0 \
	                                                                                    or params['v-portion'] == 0.0):
		print('WARNING: Invalid values received for distribution to training and validation sets.')
		print('         Resorting to defaults.')
		params['t-portion'] = 0.8
		params['v-portion'] = 0.2

	if params['window-length-T'] <= 0 or params['window-stride-T'] < 1:
		print('WARNING: Invalid values received for training-set window size and/or stride.')
		print('         Resorting to defaults.')
		params['window-length-T'] = float('inf')
		params['window-stride-T'] = float('inf')

	if params['window-length-V'] <= 0 or params['window-stride-V'] < 1:
		print('WARNING: Invalid values received for validation-set window size and/or stride.')
		print('         Resorting to defaults.')
		params['window-length-V'] = float('inf')
		params['window-stride-V'] = float('inf')

	if params['metric'] not in ['euclidean', 'pca', 'lda', 'nca']:
		print('WARNING: Invalid value received for metric.')
		print('         Resorting to default.')
		params['metric'] = 'euclidean'

	if params['k'] <= 0:
		print('WARNING: Invalid value received for k.')
		print('         Resorting to default.')
		params['k'] = 1

	if params['reduce'] is not None and params['reduce'] <= 0:
		print('WARNING: Invalid value received for dimensionality reduction.')
		print('         Resorting to default.')
		params['reduce'] = None

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('Classify actions using Dynamic Time Warping.')
	print('Use full enactments or snippets, optionally apply metric learning, and attempt to classify using a confidence threshold.')
	print('You can use this script to prepare for isotonic regression. This DTW classifier is atemporal, meaning it "knows" where')
	print('sequences begin and end and simply tries to match them. Running DTW saves results to conf-pred-gt_*.txt files that can')
	print('be used to compute isotonic regression. The buckets thus determined map confidence scores to probabilities in [0.0, 1.0].')
	print('')
	print('Usage:  python3.5 dtw.py <parameters, preceded by flags>')
	print('')
	print('e.g.')
	print('       python3.5 dtw.py -d BackBreaker1 -d Enactment1 -d Enactment2 -d Enactment3 -d Enactment4 -d Enactment5 -d Enactment6 -d Enactment7 -d Enactment9 -d Enactment10 -d MainFeederBox1 -d Regulator1 -d Regulator2 -V -joinfile joins.txt -tPor 0.8 -C sum2 -Tw 10 -Ts 2 -Vw 10 -Vs 2')
	print('')
	print('Flags:  -t         Following argument is a file to be added to the TRAINING set.')
	print('        -v         Following argument is a file to be added to the VALIDATION set.')
	print('        -d         Following argument is a file to be divided among TRAINING and VALIDATION sets.')
	print('        -tPor      Following real in (1.0, 0.0) is the portion of divided actions to allocate to the TRAINING set.')
	print('                   Default is 0.8, and this only matters if there are enactments to be divided.')
	print('        -vPor      Following real in (1.0, 0.0) is the portion of divided actions to allocate to the VALIDATION set.')
	print('                   Default is 0.2, and this only matters if there are enactments to be divided.')
	print('        -x         Following argument is a class label to be dropped from all sets.')
	print('                   Target labels according to their RE-labeled names (if applicable) and use underscores instead of spaces.')
	print('')
	print('        However they are partitioned, neither the training set nor the validation set can be empty.')
	print('')
	print('        -Tw        Following integer > 0 sets the width of the training set subsequence window.')
	print('                   Default is infinity, signifying no limit on the window and using whole sequences.')
	print('        -Ts        Following integer > 0 sets the stride of the training set subsequence window.')
	print('                   Default is infinity, signifying that we only look at the window once.')
	print('        -Vw        Following integer > 0 sets the width of the validation set subsequence window.')
	print('                   Default is infinity, signifying no limit on the window and using whole sequences.')
	print('        -Vs        Following integer > 0 sets the stride of the validation set subsequence window.')
	print('                   Default is infinity, signifying that we only look at the window once.')
	print('')
	print('        -th        Following argument in [0.0, 1.0) determines the threshold applied to predictions.')
	print('                   Default is 0.0, meaning no threshold is applied; the best match wins.')
	print('        -C         Following string in {2over1, sum2, sum3, sum4, sum5, n-min-obsv, max-marg}')
	print('                   is the confidence function to use.')
	print('                   Default is "2over1", meaning the distance of the second-best match over the best match.')
	print('        -iso       Following argument is the path for an Isotonic Regression Mapping file.')
	print('                   By default, no isotonic mapping is applied, and winners are picked according to distance only.')
	print('        -metric    Following argument in {euclidean, pca, lda, nca} applies that metric learning method.')
	print('                   Default is "euclidean".')
	print('')
	print('        -minlen    Following argument is the minimum acceptable length for a sequence. Default is 2.')
	print('        -k         Following argument is the number of nearest neighbors to collect for each per-frame prediction.')
	print('                   This is not a significant program detail, since classification is done by Dynamic Time Warping.')
	print('                   Setting k will only affect the per-frame classification as it appears in the optional scatter plot.')
	print('                   Default is 1.')
	print('        -joinfile  Following argument is the path to a relabeling guide file.')
	print('        -reduce    Following argument is the dimensionality to which data will be reduced.')
	print('                   The default is to not reduce dimensionality.')
	print('        -graph     Produce a scatter plot of the training set')
	print('        -render    Render videos of predictions showing labels and ground-truth labels.')
	print('')
	print('        -V         Enable verbosity')
	print('        -?         Display this message')
	return

if __name__ == '__main__':
	main()