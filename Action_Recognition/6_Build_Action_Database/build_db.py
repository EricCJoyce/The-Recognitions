#  sudo /home/eric/.local/bin/pip3.5 install rpy2==2.9.5

#  $ sudo -i R														Run R
#  > install.packages("dtw")                                        Install Dynamic Time-Warping library so that rpy2 can call upon it
#  > q()															Quit R

import matplotlib.pyplot as plt
import numpy as np
import os
import rpy2.robjects.numpy2ri										#  We will need R here if we're asked to do intelligent DB down-sampling
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
	if params['helpme'] or len(params['enactments']) == 0:
		usage()
		return

	if params['verbose']:
		print('SOURCES:')
		for filename in params['enactments']:
			print('\t' + filename)
		print('')

	#################################################################  Determine the list of objects
	object_labels = {}												#  Begin as a dictionary, end as a sorted list
	for enactment in params['enactments']:
		reading_classes = False

		fh = open(enactment + '.enactment', 'r')
		lines = fh.readlines()
		fh.close()

		for line in lines:
			if line[0] == '#':
				arr = line.strip().split()
				if len(arr) > 1 and arr[1] == 'CLASSES:':
					reading_classes = True
				elif reading_classes and '    ' in line:
					for classname in arr[1:]:
						object_labels[ classname ] = True
				elif reading_classes:
					reading_classes = False

	object_labels = sorted(object_labels.keys())

	#################################################################  Load, merge, clean.

	survey_raw_actions(params)

	seqs, vector_len = join_sequences(params)
	params['vector_len'] = vector_len								#  Pack the vector-length into params

	actions, seqs = clean_sequences(seqs, params)

	#################################################################  Build the sets we will actually use

	X, y = build(actions, seqs, params)

	labels = sorted(np.unique(y))									#  Make an enumerable list of labels
	num_classes = len(labels)										#  Save the number of (unique) class labels

	for i in range(0, len(y)):										#  Convert everything to indices into 'labels'
		y[i] = labels.index( y[i] )

	if params['verbose']:
		if params['thin'] is not None:
			print('    ORIGINAL:')
		print('\tX = ' + str(len(X)) + ' time series with y = ' + str(len(y)) + ' labels')
		print('')
		print('    ' + str(num_classes) + ' unique labels in data set:')
		maxlabellen = 0
		for label in labels:
			if len(label) > maxlabellen:
				maxlabellen = len(label)
		for label in labels:
			print('\t' + label + ' '*(maxlabellen - len(label)) + ' ' + str(len([x for x in y if x == labels.index(label)])))
		print('')

	#################################################################  Thin the database
	if params['thin-method'] is not None:
		class_indices = {}											#  key:int ==> val:[int, int, int, ... ]
		for i in range(0, len(y)):
			if y[i] not in class_indices:
				class_indices[ y[i] ] = []
			class_indices[ y[i] ].append( i )						#  Index i is an instance of class y[i]

		if params['thin-method'] == 'nn':							#  NEAREST-NEIGHBOR DOWN-SAMPLING: delete down to target size
			R = rpy2.robjects.r										#  Shortcut to the R backend
			DTW = importr('dtw')									#  Shortcut to the R DTW library

			X_tmp = X[:]											#  Clone X
			y_tmp = y[:]											#  Clone y

			X = []													#  Blank out original X
			y = []													#  Blank out original y

			for k, v in class_indices.items():
				if params['verbose']:
					print('    >>> NN-Thinning ' + labels[k])

				D = {}												#  key:(min(i, j), max(i, j)) ==> val:distance
																	#  We *could* build a distance matrix, but it would be wastefully symmetrical.
																	#  Plus, we'll need to flatten it into a 1D descending list anyway.
				for i in v:
					query = np.array(X_tmp[i])						#  Convert to NumPy array
					rq, cq = query.shape							#  Save number of rows (height) and number of columns (width)
					queryR = R.matrix(query, nrow=rq, ncol=cq)		#  Convert to R matrix

					for j in v:
						if i != j and (min(i, j), max(i, j)) not in D:
							template = np.array(X_tmp[j])			#  Convert to NumPy array
							rt, ct = template.shape					#  Save number of rows (height) and number of columns (width)
																	#  Convert to R matrix
							templateR = R.matrix(template, nrow=rt, ncol=ct)
																	#  What is the cost of aligning this t_seq with q_seq?
							alignment = R.dtw(templateR, queryR, open_begin=False, open_end=False)
																	#  (Normalized) cost of matching this query to this template
							D[ (min(i, j), max(i, j)) ] = alignment.rx('normalizedDistance')[0][0]

				target_size = max(1, int(round(float(len(v)) / float(params['thin']))))
																	#  List of tuples ((i, j), distance) DESCENDING by distance
				dists = sorted([x for x in D.items()], reverse=True, key=lambda x: x[1])
				addition = {}
				for tup in dists[:target_size]:
					addition[ tup[0][0] ] = True					#  Save the src index
					addition[ tup[0][1] ] = True					#  Save the dst index: together, these make a great distance
				X += [X_tmp[i] for i in addition.keys()]
				y += [k for i in range(0, len(addition))]

		elif params['thin-method'] == 'fps':						#  FURTHEST-POINT DOWN-SAMPLING: add up to target size
			R = rpy2.robjects.r										#  Shortcut to the R backend
			DTW = importr('dtw')									#  Shortcut to the R DTW library

			X_tmp = []
			y_tmp = []

			for k, v in class_indices.items():
				if params['verbose']:
					print('    >>> FPS-Thinning ' + labels[k])

				target_size = max(1, int(round(float(len(v)) / float(params['thin']))))
				D = np.zeros((len(v), len(v)))						#  Here it makes sense to build a matrix because
																	#  we will consult and combine several distances
				for i in range(0, len(v)):
					query = np.array(X[i])							#  Convert to NumPy array
					rq, cq = query.shape							#  Save number of rows (height) and number of columns (width)
					queryR = R.matrix(query, nrow=rq, ncol=cq)		#  Convert to R matrix

					for j in range(0, len(v)):
						if i != j:
							template = np.array(X[j])				#  Convert to NumPy array
							rt, ct = template.shape					#  Save number of rows (height) and number of columns (width)
																	#  Convert to R matrix
							templateR = R.matrix(template, nrow=rt, ncol=ct)
																	#  What is the cost of aligning this t_seq with q_seq?
							alignment = R.dtw(templateR, queryR, open_begin=False, open_end=False)
																	#  (Normalized) cost of matching this query to this template
							D[i, j] = alignment.rx('normalizedDistance')[0][0]

				index = np.random.randint(0, len(v))
				addition = [ index ]								#  Initial random keep (index into 'v')
				while len(addition) < target_size:
					for used_index in addition:						#  Blank out elements for everything already included in 'addition'
						D[index, used_index] = 0.0
					new_index = np.argmax(D[index, :])				#  Find the next index (greatest distance from current index)
					addition.append( new_index )
					D[new_index, :] += D[index, :]
					index = new_index

				for index in addition:
					X_tmp.append( X[ v[index] ] )
					y_tmp.append(k)

			X = X_tmp[:]
			y = y_tmp[:]

		else:														#  Randomly downsample to target size
			X_tmp = []
			y_tmp = []

			for k, v in class_indices.items():
				if params['verbose']:
					print('    >>> Randomly thinning ' + labels[k])

				target_size = max(1, int(round(float(len(v)) / float(params['thin']))))
				keep_indices_in_class = np.random.choice(len(v), size=target_size, replace=False)
				X_tmp += [X[i][:] for i in keep_indices_in_class]
				y_tmp += [k for i in keep_indices_in_class]

			X = X_tmp[:]
			y = y_tmp[:]

		del X_tmp													#  Save memory; clean up
		del y_tmp

		if params['verbose']:
			print('    THINNED:')
			print('\tX = ' + str(len(X)) + ' time series with y = ' + str(len(y)) + ' labels')
			print('')
			maxlabellen = 0
			for label in labels:
				if len(label) > maxlabellen:
					maxlabellen = len(label)
			for label in labels:
				print('\t' + label + ' '*(maxlabellen - len(label)) + ' ' + str(len([x for x in y if x == labels.index(label)])))
			print('')

	#################################################################  Write database to file
	formatstring = '{:.' + str(params['precision']) + 'f}'

	fh = open(params['outfile'] + '.db', 'w')
	fh.write('#  Action recognition database made at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
	fh.write('#  ENACTMENTS:\n')
	for enactment in params['enactments']:
		fh.write('#      ' + enactment + '\n')
	fh.write('#  ACTIONS:\n')
	for label in labels:
		fh.write('#      ' + label + '\n')
	fh.write('#  Vector length: ' + str(params['vector_len']) + '\n')
	fh.write('#  OBJECT SUB-VECTOR STRUCTURE:\n')
	for object_label in object_labels:
		fh.write('#      ' + object_label + '\n')
	fh.write('#  Snippet window size: ' + str(params['window-length']) + '\n')
	fh.write('#  Snippet window stride: ' + str(params['window-stride']) + '\n')
	fh.write('#  Vector metric: ' + params['metric'] + '\n')
	fh.write('#  Database contains ' + str(len(X)) + ' sequences.\n')
	if params['thin-method'] is not None:
		if params['thin-method'] == 'nn':							#  Delete down to target size
			fh.write('#  Database has been thinned using Nearest-Neighbor reduction, ' + str(params['thin']) + '.\n')
		elif params['thin-method'] == 'fps':						#  Add up to target size
			fh.write('#  Database has been thinned using Furthest Point Sampling reduction, ' + str(params['thin']) + '.\n')
		else:														#  Randomly downsample to target size
			fh.write('#  Database has been thinned using random downsampling, ' + str(params['thin']) + '.\n')
	else:
		fh.write('#  Database has NOT been thinned.\n')

	for i in range(0, len(X)):
		fh.write(labels[y[i]] + '\n')
		for vec in X[i]:
			fh.write('\t' + ' '.join([formatstring.format(x) for x in vec]) + '\n')

	fh.close()

	return

#  Build the database, applying metric learning if requested.
#  Return X, a list of sequences, and y, a list of labels (strings)
def build(actions, seqs, params):
	X = []															#  To contain time series
	y = []															#  To contain labels (indices into 'labels')

	X_metriclearn_whole = []										#  To contain frames, used for fitting
	y_metriclearn_whole = []										#  To contain labels

	X_metriclearn_train = []										#  To contain frames, used for graphing and evaluating
	y_metriclearn_train = []										#  To contain labels

	X_metriclearn_test = []											#  To contain frames, used for graphing and evaluating
	y_metriclearn_test = []											#  To contain labels

	if params['verbose']:
		print('>>> Building databse...')
		print('    Vector length = ' + str(params['vector_len']))
		print('')

	for action, seqdata in actions.items():							#  key is an action label;
		indices = seqdata['seq_indices'][:]							#  val is a dictionary: seq_indices, frame_ctr

		if (len(params['exclude']) > 0 and action not in params['exclude']) or \
		   (len(params['admit']) > 0 and action in params['admit']) or \
		   (len(params['exclude']) == 0 and len(params['admit']) == 0):

			print(action)

			X_metriclearn = []										#  To contain frames
			y_metriclearn = []										#  To contain labels

			for i in range(0, len(indices)):						#  For every sequence, seqs[indices[i]]...
				for frame in seqs[indices[i]]:						#  Anticipating that we will be doing metric learning or plotting,
					X_metriclearn.append( frame['vec'] )			#  build a set of all the vectors--NOT time-series!!
					y_metriclearn.append( action )

					X_metriclearn_whole.append( frame['vec'] )
					y_metriclearn_whole.append( action )

				if params['window-length'] < float('inf'):			#  Use window and stride
					if params['window-stride'] < float('inf'):		#  Finite stride
																	#  What to do with actions shorter than the window length?
						if len(seqs[indices[i]]) >= params['window-length']:
							for fr_head_index in range(0, len(seqs[indices[i]]) - params['window-length'], params['window-stride']):
								seq = []
								for fr_ctr in range(0, params['window-length']):
									seq.append( seqs[indices[i]][fr_head_index + fr_ctr]['vec'] )
								X.append( seq )
								y.append(action)
						elif params['include-shorts']:
							seq = []
							for fr_ctr in range(0, len(seqs[indices[i]])):
								seq.append( seqs[indices[i]][fr_ctr]['vec'] )
							X.append( seq )
							y.append(action)
					else:											#  Infinite stride: only read the window once
						seq = []
						for fr_ctr in range(0, min(len(seqs[indices[i]]), params['window-length'])):
							seq.append( seqs[indices[i]][fr_ctr]['vec'] )
						X.append( seq )
						y.append(action)
				else:												#  Use the whole sequence
					seq = []
					for frame in seqs[indices[i]]:
						seq.append( frame['vec'] )					#  Build the sequence
					X.append( seq )									#  Append the sequence
					y.append(action)
																	#  This work is done here within an outer loop that visits every action
																	#  so that the train-test split represents the total distribution of actions.
			lim = int(round(params['metric-learn-train'] * len(seqdata['seq_indices'])))

			if params['metric-shuffle']:
				Xy_metriclearn = list(zip(X_metriclearn, y_metriclearn))
				np.random.shuffle(Xy_metriclearn)
				X_metriclearn = [x[0] for x in Xy_metriclearn]
				y_metriclearn = [x[1] for x in Xy_metriclearn]

			X_metriclearn_train += X_metriclearn[:lim]
			y_metriclearn_train += y_metriclearn[:lim]

			X_metriclearn_test += X_metriclearn[lim:]
			y_metriclearn_test += y_metriclearn[lim:]

	labels = sorted(np.unique(y))									#  Make an enumerable list of labels
	num_classes = len(labels)										#  Save the number of (unique) class labels

	for i in range(0, len(y_metriclearn_train)):					#  Convert everything to indices into 'labels'
		if y_metriclearn_train[i] in labels:
			y_metriclearn_train[i] = labels.index( y_metriclearn_train[i] )
	for i in range(0, len(y_metriclearn_test)):						#  Convert everything to indices into 'labels'
		if y_metriclearn_test[i] in labels:
			y_metriclearn_test[i] = labels.index( y_metriclearn_test[i] )
	for i in range(0, len(y_metriclearn_whole)):					#  Convert everything to indices into 'labels'
		if y_metriclearn_whole[i] in labels:
			y_metriclearn_whole[i] = labels.index( y_metriclearn_whole[i] )

	knn = KNeighborsClassifier(n_neighbors=params['k'])				#  Create a nearest neighbor classifier

	if params['metric'] != 'euclidean':								#  Are we applying metric learning?
		random_state = 0											#  Remove stochasticity

		if params['reduce'] is not None:							#  Are we REDUCING DIMENSIONALITY?
																	#  Dimensionality reduction must be AT MOST the number of classes
			target_dimensions = min(num_classes - 1, params['reduce'])

			if params['verbose']:
				if target_dimensions != params['reduce']:
					print('*\tCannot reduce dimensionality to requested ' + str(params['reduce']))
				print('>>> Reducing dimensionality from ' + str(params['vector_len']) + ' to ' + str(target_dimensions) + '...')

			if params['metric'] == 'pca':
				if params['verbose']:
					print('>>> Applying PCA to reduced-dimension set...')
																	#  Reduce dimension to 'target_dimensions' with PCA
				pca = make_pipeline(StandardScaler(), PCA(n_components=target_dimensions, random_state=random_state))

				if params['graph']:									#  Scatter plot?
					pca.fit(X_metriclearn_train, y_metriclearn_train)
					knn.fit(pca.transform(X_metriclearn_train), y_metriclearn_train)
																	#  Get a per-frame accuracy score
					acc_knn = knn.score(pca.transform(X_metriclearn_test), y_metriclearn_test)
					X_embedded = pca.transform(X_metriclearn_train)	#  Embed the data set in 2 dimensions using the fitted model
					plt.figure()									#  Plot the projected points and show the evaluation score
					plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_metriclearn_train, s=30, cmap='Set1')
					plt.title("PCA, KNN (k={})\nTest accuracy = {:.2f}".format(params['k'], acc_knn))
					plt.savefig('PCA_reduced-' + str(params['reduce']) + '.png')

				pca.fit(X_metriclearn_whole, y_metriclearn_whole)	#  Take the benefit of the whole

			elif params['metric'] == 'lda':
				if params['verbose']:
					print('>>> Applying LDA to reduced-dimension set...')
																	#  Reduce dimension to 'target_dimensions' with LinearDiscriminantAnalysis
				lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=target_dimensions))

				if params['graph']:									#  Scatter plot?
					lda.fit(X_metriclearn_train, y_metriclearn_train)
					knn.fit(lda.transform(X_metriclearn_train), y_metriclearn_train)
																	#  Get a per-frame accuracy score
					acc_knn = knn.score(lda.transform(X_metriclearn_test), y_metriclearn_test)
					X_embedded = lda.transform(X_metriclearn_train)	#  Embed the data set in 2 dimensions using the fitted model
					plt.figure()									#  Plot the projected points and show the evaluation score
					plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_metriclearn_train, s=30, cmap='Set1')
					plt.title("LDA, KNN (k={})\nTest accuracy = {:.2f}".format(params['k'], acc_knn))
					plt.savefig('LDA_reduced-' + str(params['reduce']) + '.png')

				lda.fit(X_metriclearn_whole, y_metriclearn_whole)	#  Take the benefit of the whole

			elif params['metric'] == 'nca':
				if params['verbose']:
					print('>>> Applying NCA to reduced-dimension set...')
																	#  Reduce dimension to 'target_dimensions' with NeighborhoodComponentAnalysis
				nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=target_dimensions, random_state=random_state))

				if params['graph']:									#  Scatter plot?
					nca.fit(X_metriclearn_train, y_metriclearn_train)
					knn.fit(nca.transform(X_metriclearn_train), y_metriclearn_train)
																	#  Get a per-frame accuracy score
					acc_knn = knn.score(nca.transform(X_metriclearn_test), y_metriclearn_test)
					X_embedded = nca.transform(X_metriclearn_train)	#  Embed the data set in 2 dimensions using the fitted model
					plt.figure()									#  Plot the projected points and show the evaluation score
					plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_metriclearn_train, s=30, cmap='Set1')
					plt.title("NCA, KNN (k={})\nTest accuracy = {:.2f}".format(params['k'], acc_knn))
					plt.savefig('NCA_reduced-' + str(params['reduce']) + '.png')

				nca.fit(X_metriclearn_whole, y_metriclearn_whole)	#  Take the benefit of the whole

		else:														#  We are NOT reducing dimensions
			if params['metric'] == 'pca':
				if params['verbose']:
					print('>>> Applying PCA to full-dimension set...')

				pca = PCA(random_state=random_state)

				if params['graph']:									#  Scatter plot?
					pca.fit(X_metriclearn_train, y_metriclearn_train)
					knn.fit(pca.transform(X_metriclearn_train), y_metriclearn_train)
																	#  Get a per-frame accuracy score
					acc_knn = knn.score(pca.transform(X_metriclearn_test), y_metriclearn_test)
					X_embedded = pca.transform(X_metriclearn_train)	#  Embed the data set in 2 dimensions using the fitted model
					plt.figure()									#  Plot the projected points and show the evaluation score
					plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_metriclearn_train, s=30, cmap='Set1')
					plt.title("PCA, KNN (k={})\nTest accuracy = {:.2f}".format(params['k'], acc_knn))
					plt.savefig('PCA.png')

				pca.fit(X_metriclearn_whole, y_metriclearn_whole)	#  Take the benefit of the whole

			elif params['metric'] == 'lda':
				if params['verbose']:
					print('>>> Applying LDA to full-dimension set...')

				lda = LinearDiscriminantAnalysis()

				if params['graph']:									#  Scatter plot?
					lda.fit(X_metriclearn_train, y_metriclearn_train)
					knn.fit(lda.transform(X_metriclearn_train), y_metriclearn_train)
																	#  Get a per-frame accuracy score
					acc_knn = knn.score(lda.transform(X_metriclearn_test), y_metriclearn_test)
					X_embedded = lda.transform(X_metriclearn_train)	#  Embed the data set in 2 dimensions using the fitted model
					plt.figure()									#  Plot the projected points and show the evaluation score
					plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_metriclearn_train, s=30, cmap='Set1')
					plt.title("LDA, KNN (k={})\nTest accuracy = {:.2f}".format(params['k'], acc_knn))
					plt.savefig('LDA.png')

				lda.fit(X_metriclearn_whole, y_metriclearn_whole)	#  Take the benefit of the whole

			elif params['metric'] == 'nca':
				if params['verbose']:
					print('>>> Applying NCA to full-dimension set...')

				nca = NeighborhoodComponentsAnalysis(random_state=random_state)

				if params['graph']:									#  Scatter plot?
					nca.fit(X_metriclearn_train, y_metriclearn_train)
					knn.fit(nca.transform(X_metriclearn_train), y_metriclearn_train)
																	#  Get a per-frame accuracy score
					acc_knn = knn.score(nca.transform(X_metriclearn_test), y_metriclearn_test)
					X_embedded = nca.transform(X_metriclearn_train)	#  Embed the data set in 2 dimensions using the fitted model
					plt.figure()									#  Plot the projected points and show the evaluation score
					plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_metriclearn_train, s=30, cmap='Set1')
					plt.title("NCA, KNN (k={})\nTest accuracy = {:.2f}".format(params['k'], acc_knn))
					plt.savefig('NCA.png')

				nca.fit(X_metriclearn_whole, y_metriclearn_whole)	#  Take the benefit of the whole

		X = []														#  Rebuild, transformed
		y = []

		for action, seqdata in actions.items():						#  key is an action label, including '*';
			indices = seqdata['seq_indices'][:]						#  val is a dictionary: seq_indices, frame_ctr

			for i in range(0, len(indices)):						#  For every sequence, seqs[indices[i]]...

				if params['window-length'] < float('inf'):			#  Use window and stride
					if params['window-stride'] < float('inf'):		#  Finite stride
						for fr_head_index in range(0, len(seqs[indices[i]]) - params['window-length'], params['window-stride']):
							seq = []
							for fr_ctr in range(0, params['window-length']):
								if params['metric'] == 'pca':		#  Apply PCA
									seq.append( pca.transform([seqs[indices[i]][fr_head_index + fr_ctr]['vec']])[0] )
								elif params['metric'] == 'lda':		#  Apply LDA
									seq.append( lda.transform([seqs[indices[i]][fr_head_index + fr_ctr]['vec']])[0] )
								elif params['metric'] == 'nca':		#  Apply NCA
									seq.append( nca.transform([seqs[indices[i]][fr_head_index + fr_ctr]['vec']])[0] )
								else:								#  Apply nothing (Euclidean)
									seq.append( seqs[indices[i]][fr_head_index + fr_ctr]['vec'] )
							X.append( seq )
							y.append(action)
					else:											#  Infinite stride: only read the window once
						seq = []
						for fr_ctr in range(0, min(len(seqs[indices[i]]), params['window-length'])):
							if params['metric'] == 'pca':			#  Apply PCA
								seq.append( pca.transform([seqs[indices[i]][fr_ctr]['vec']])[0] )
							elif params['metric'] == 'lda':			#  Apply LDA
								seq.append( lda.transform([seqs[indices[i]][fr_ctr]['vec']])[0] )
							elif params['metric'] == 'nca':			#  Apply NCA
								seq.append( nca.transform([seqs[indices[i]][fr_ctr]['vec']])[0] )
							else:									#  Apply nothing (Euclidean)
								seq.append( seqs[indices[i]][fr_ctr]['vec'] )
						X.append( seq )
						y.append(action)
				else:												#  Use the whole sequence
					seq = []
					for frame in seqs[indices[i]]:
						if params['metric'] == 'pca':				#  Apply PCA
							seq.append( pca.transform([frame['vec']])[0] )
						elif params['metric'] == 'lda':				#  Apply LDA
							seq.append( lda.transform([frame['vec']])[0] )
						elif params['metric'] == 'nca':				#  Apply NCA
							seq.append( nca.transform([frame['vec']])[0] )
						else:										#  Apply nothing (Euclidean)
							seq.append( frame['vec'] )
					X.append( seq )									#  Append the sequence
					y.append(action)

		if params['verbose']:
			print('')

	elif params['graph']:
		knn.fit(X_metriclearn_train, y_metriclearn_train)
		acc_knn = knn.score(X_metriclearn_test, y_metriclearn_test)	#  Get the accuracy score
		if params['graph']:											#  Scatter plot?
			X_embedded = np.array(X_metriclearn_train)
			plt.figure()											#  Plot the projected points and show the evaluation score
			plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_metriclearn_train, s=30, cmap='Set1')
			plt.title("Euclidean, KNN (k={})\nTest accuracy = {:.2f}".format(params['k'], acc_knn))
			plt.savefig('Euclidean.png')
			plt.clf()												#  Clear

	return X, y

#  Perform drops and filter out any sequences that are too short
def clean_sequences(seqs, params):
	clean_actions = {}												#  Reset. This still counts AND tracks:
																	#  key: string ==> val: {seq_indices:[int, int, int], frame_ctr:int}

	if params['verbose']:											#  Tell user which labels will be dropped
		print('*\tSequences with fewer than ' + str(params['minlen']) + ' frames will be dropped')
		print('*\t<NEURTAL> will be dropped')
		print('')

	clean_seqs = []													#  List of Lists of Dicts(key: timestamp ==> val: {filename, (re)label, vector})
																	#  [ [{}, {}, {}], [{}, {}], [{}, {}, ..., {}], ..., [{}, {}, {}] ]
																	#    sequence      sequence  sequence       ^         sequence
																	#                                           |
																	#                                        frame
	for i in range(0, len(seqs)):
		if len(seqs[i]) >= params['minlen']:						#  If this is longer than or equal to the minimum, keep it.
			action = seqs[i][0]['label']
			if action != '*':
				if action not in clean_actions:
					clean_actions[action] = {}
					clean_actions[action]['seq_indices'] = []
					clean_actions[action]['frame_ctr'] = 0
				clean_actions[action]['seq_indices'].append( len(clean_seqs) )
				clean_seqs.append( [] )								#  Add a sequence
				for frame in seqs[i]:								#  Add all frames
					clean_seqs[-1].append( {} )
					clean_seqs[-1][-1]['timestamp'] = frame['timestamp']
					clean_seqs[-1][-1]['file'] = frame['file']
					clean_seqs[-1][-1]['frame'] = frame['frame']
					clean_seqs[-1][-1]['label'] = frame['label']
					clean_seqs[-1][-1]['vec'] = frame['vec'][:]

				clean_actions[action]['frame_ctr'] += len(seqs[i])

	seqs = clean_seqs												#  Copy back
	del clean_seqs													#  Destroy the temp

	actions = dict(clean_actions.items())							#  Rebuild dictionary

	if params['verbose']:
		print('FILTERED ACTIONS:')
		for k, v in sorted(actions.items()):
			if k == '*':
				print('\t<NEURTAL>: ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
			else:
				print('\t' + k + ': ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
		print('')

	return actions, seqs

#  Read all sequences. Perform any joins/relabelings.
def join_sequences(params):
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
																	#  [ [{}, {}, {}], [{}, {}], [{}, {}, ..., {}], ..., [{}, {}, {}] ]
																	#    sequence      sequence  sequence       ^         sequence
																	#                                           |
																	#                                        frame
	actions = {}													#  This will count AND track:
																	#  key: string ==> val: {seq_indices:[int, int, int], frame_ctr:int}
	vector_len = 0													#  I'd also like to know how long a vector is

	for filename in params['enactments']:
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

				if label not in actions:
					actions[label] = {}
					actions[label]['seq_indices'] = []
					actions[label]['frame_ctr'] = 0

				if label != currentLabel:
					currentLabel = label
					actions[currentLabel]['seq_indices'].append( len(seqs) )
					seqs.append( [] )

				actions[currentLabel]['frame_ctr'] += 1

				seqs[-1].append( {} )
				seqs[-1][-1]['timestamp'] = timestamp
				seqs[-1][-1]['file'] = filename
				seqs[-1][-1]['frame'] = srcfile
				seqs[-1][-1]['label'] = currentLabel
				seqs[-1][-1]['vec'] = vec[:]

	if params['verbose']:
		print('MERGED ACTIONS:')
		for k, v in sorted(actions.items()):
			if k == '*':
				print('\t<NEURTAL>: ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
			else:
				print('\t' + k + ': ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
		print('')

	return seqs, vector_len

#  Survey the set of all actions in the given enactments.
def survey_raw_actions(params):
	actions = {}													#  key: string ==> val: {seq_ctr:int, frame_ctr:int}

	for filename in params['enactments']:
		fh = open(filename + '.enactment', 'r')
		lines = fh.readlines()
		fh.close()
		currentLabel = None
		for line in lines:
			if line[0] != '#':
				arr = line.strip().split('\t')
				label = arr[2]

				if label not in actions:
					actions[label] = {}
					actions[label]['seq_ctr'] = 0
					actions[label]['frame_ctr'] = 0

				if label != currentLabel:
					currentLabel = label
					actions[currentLabel]['seq_ctr'] += 1

				actions[currentLabel]['frame_ctr'] += 1

	if params['verbose']:
		print('RAW DEFINED ACTIONS:')
		for k, v in sorted(actions.items()):
			if k == '*':
				print('\t<NEURTAL>: ' + str(v['frame_ctr']) + ' total frames, ' + str(v['seq_ctr']) + ' sequences')
			else:
				print('\t' + k + ': ' + str(v['frame_ctr']) + ' total frames, ' + str(v['seq_ctr']) + ' sequences')
		print('')

	return

def getCommandLineParams():
	params = {}
	params['enactments'] = []										#  List of filenames
	params['outfile'] = 'recognitions'
	params['thin-method'] = None									#  A string, indicating the method used to thin the DB
	params['thin'] = None											#  Reduction factor; we always respect cluster size.
																	#  Do not kill any classes.

	params['window-length'] = float('inf')							#  By default, "infinite" windows cover entire (Training Set) sequences
	params['window-stride'] = float('inf')							#  By default, advance the window to infinity (past the end of the sequence)

	params['include-shorts'] = False								#  Whether to include, whole, sequences shorter than the given window size
	params['admit'] = []											#  List of labels to be admitted TO THE EXCLUSION OF ALL OTHERS
	params['exclude'] = []											#  List of labels to be excluded

	params['metric'] = 'euclidean'									#  Metric learning? (Default to none)
	params['minlen'] = 2											#  Smallest length of a sequence deemed acceptable for the dataset
	params['k'] = 1													#  Really just for show in the per-frame scatter-plot
	params['reduce'] = None											#  Reducing dimensionality in metric learning?
	params['metric-shuffle'] = False
	params['metric-learn-train'] = 0.8
	params['metric-learn-test'] = 1.0 - params['metric-learn-train']
	params['graph'] = False											#  Scatter plot?

	params['precision'] = 8
	params['joins'] = {}											#  key: new label; val: [old labels]
	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-e', '-o', '-joinfile', '-thinmeth', '-thin', '-shuffle', \
	         '-window', '-stride', '-shorts', '-i', '-x', \
	         '-metric', \
	         '-minlen', '-k', '-reduce', '-graph', '-precision', \
	         '-v', '-?', '-help', '--help']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-shorts' or sys.argv[i] == '-short':
				params['include-shorts'] = True
			elif sys.argv[i] == '-shuffle':
				params['metric-shuffle'] = True
			elif sys.argv[i] == '-graph':
				params['graph'] = True
			elif sys.argv[i] == '-v':
				params['verbose'] = True
			elif sys.argv[i] == '-?' or sys.argv[i] == '-help' or sys.argv[i] == '--help':
				params['helpme'] = True
			else:
				argtarget = sys.argv[i]
		else:
			argval = sys.argv[i]

			if argtarget is not None:
				if argtarget == '-e':
					params['enactments'].append(argval)
				elif argtarget == '-o':
					params['outfile'] = argval

				elif argtarget == '-window':
					params['window-length'] = int(argval)
				elif argtarget == '-stride':
					params['window-stride'] = int(argval)

				elif argtarget == '-thinmeth':
					params['thin-method'] = argval
				elif argtarget == '-thin':
					params['thin'] = int(argval)

				elif argtarget == '-i':
					params['admit'].append( argval.replace('_', ' ') )
				elif argtarget == '-x':
					params['exclude'].append( argval.replace('_', ' ') )

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
				elif argtarget == '-precision':
					params['precision'] = int(argval)

	if params['window-length'] <= 0 or params['window-stride'] < 1:
		print('WARNING: Invalid values received for window size and/or stride.')
		print('         Resorting to defaults.')
		params['window-length'] = float('inf')
		params['window-stride'] = float('inf')

	if params['metric'] not in ['euclidean', 'pca', 'lda', 'nca']:
		print('WARNING: Invalid value received for metric.')
		print('         Resorting to default.')
		params['metric'] = 'euclidean'

	if params['thin-method'] is not None and params['thin-method'] not in ['rnd', 'nn', 'fps']:
		print('WARNING: Invalid value received for thinmeth.')
		print('         Defaulting to no down-sampling.')
		params['thin-method'] = None

	if params['thin'] is not None:
		if params['thin-method'] is None:
			print('WARNING: Parameter thin was specified without specifying a method.')
			print('         Defaulting to no down-sampling.')
			params['thin'] = None
		elif params['thin'] < 2:
			print('WARNING: Invalid value received for thin.')
			print('         Defaulting to no down-sampling.')
			params['thin'] = None

	if params['k'] <= 0:
		print('WARNING: Invalid value received for k.')
		print('         Resorting to default.')
		params['k'] = 1

	if params['reduce'] is not None and params['reduce'] <= 0:
		print('WARNING: Invalid value received for dimensionality reduction.')
		print('         Resorting to default.')
		params['reduce'] = None

	if params['precision'] < 1:
		print('WARNING: Invalid value received for output float precision.')
		print('         Resorting to default.')
		params['precision'] = 8

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('Build an action-recognition database from Factual enactments.')
	print('Use full enactments or snippets and optionally apply metric learning.')
	print('')
	print('Usage:  python3 build_db.py <parameters, preceded by flags>')
	print('')
	print('e.g.:  Use Ground-Truth with intermediate states. Thin using FPS. Omit Enactment11 and Enactment12; use them for testing. DB to include ALL actions.')
	print('       python3.5 build_db.py -o databases/gt-intermediates/10f_fps -e GT-intermediates/BackBreaker1 -e GT-intermediates/Enactment1 -e GT-intermediates/Enactment2 -e GT-intermediates/Enactment3 -e GT-intermediates/Enactment4 -e GT-intermediates/Enactment5 -e GT-intermediates/Enactment6 -e GT-intermediates/Enactment7 -e GT-intermediates/Enactment9 -e GT-intermediates/Enactment10 -e GT-intermediates/MainFeederBox1 -e GT-intermediates/Regulator1 -e GT-intermediates/Regulator2 -v -joinfile joins.txt -window 10 -stride 2 -shorts -thinmeth fps -thin 5')
	print('')
	print('e.g.:  Use Ground-Truth without intermediate states. Thin using FPS. Omit Enactment11 and Enactment12; use them for testing. DB to include ALL actions.')
	print('       python3.5 build_db.py -o databases/gt-no-intermediates/10f_fps -e GT-NO-intermediates/BackBreaker1 -e GT-NO-intermediates/Enactment1 -e GT-NO-intermediates/Enactment2 -e GT-NO-intermediates/Enactment3 -e GT-NO-intermediates/Enactment4 -e GT-NO-intermediates/Enactment5 -e GT-NO-intermediates/Enactment6 -e GT-NO-intermediates/Enactment7 -e GT-NO-intermediates/Enactment9 -e GT-NO-intermediates/Enactment10 -e GT-NO-intermediates/MainFeederBox1 -e GT-NO-intermediates/Regulator1 -e GT-NO-intermediates/Regulator2 -v -joinfile joins.txt -window 10 -stride 2 -shorts -thinmeth fps -thin 5')
	print('')
	print('e.g.:  Use Mask-RCNN 0028. Thin using FPS. Omit Enactment11 and Enactment12; use them for testing. DB to include ALL actions.')
	print('       python3.5 build_db.py -o databases/mrcnn0028/10f_fps -e MRCNN28/BackBreaker1 -e MRCNN28/Enactment1 -e MRCNN28/Enactment2 -e MRCNN28/Enactment3 -e MRCNN28/Enactment4 -e MRCNN28/Enactment5 -e MRCNN28/Enactment6 -e MRCNN28/Enactment7 -e MRCNN28/Enactment9 -e MRCNN28/Enactment10 -e MRCNN28/MainFeederBox1 -e MRCNN28/Regulator1 -e MRCNN28/Regulator2 -v -joinfile joins.txt -window 10 -stride 2 -shorts -thinmeth fps -thin 5')
	print('\n')
	print('Flags:  -e         Following argument is an enactment file from which actions will be added to the database.')
	print('        -o         Following string will be the output file name. (.db is added automatically).')
	print('                   Default is to write to "recognitions.db"')
	print('')
	print('        -joinfile  Following argument is the path to a relabeling guide file.')
	print('')
	print('        -thinmeth  Following argument is a string in {\'rnd\', \'nn\', \'fps\'} standing for random down-sampling,')
	print('                   nearest-neighbor down-sampling, and furthest point down-sampling respectively. This argument specifies')
	print('                   how the database should be down-sampled. If this parameter is given, then you must also specify the')
	print('                   down-sampling factor using \'-thin\'. (See below.)')
	print('        -thin      Following argument is an integer > 1 to indicate that databse clusters are to be thinning out by a')
	print('                   factor of the integer. The default behavior (without this flag at all) is to leave the database intact.')
	print('')
	print('        -window    Following integer > 0 sets the width of the subsequence window.')
	print('                   Default is infinity, signifying no limit on the window and using whole sequences.')
	print('        -stride    Following integer > 0 sets the stride of the subsequence window.')
	print('                   Default is infinity, signifying that we only look at the window once.')
	print('        -shorts    Include actions that are shorter than the given window size. By default we do not do this.')
	print('        -i         Following string with underscores in place of spaces is an action label (post join-file) to be')
	print('                   included to the exclusion of all others. By default the list of exclusive inclusions is empty.')
	print('        -x         Following string with underscores in place of spaces is an action label (post join-file) to be')
	print('                   excluded from the database. By default, there are no exclusions unless determined by other conditions.')
	print('')
	print('        -metric    Following argument in {euclidean, pca, lda, nca} applies that metric learning method.')
	print('                   Default is "euclidean".')
	print('        -shuffle   Shuffle training and test sets when scoring metric learning')
	print('')
	print('        -minlen    Following argument is the minimum acceptable length for a sequence. Default is 2.')
	print('        -k         Following argument is the number of nearest neighbors to collect for each per-frame prediction.')
	print('                   This is not a significant program detail, since this script does not perform classification.')
	print('                   Setting k will only affect the per-frame classification as it appears in the optional scatter plot.')
	print('                   Default is 1.')
	print('        -reduce    Following argument is the dimensionality to which data will be reduced.')
	print('                   The default is to not reduce dimensionality.')
	print('        -graph     Produce a scatter plot of the database')
	print('')
	print('        -v         Enable verbosity')
	print('        -?         Display this message')
	return

if __name__ == '__main__':
	main()