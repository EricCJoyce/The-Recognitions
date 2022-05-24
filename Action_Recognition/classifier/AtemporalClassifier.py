import numpy as np
import os
import sys
import time

from classifier.Classifier import Classifier
from enactment import ProcessedEnactment

'''
Why "atemporal"? Because sequence boundaries are given; they need not be discovered frame by frame.
Give this classifier enactment files, and it will divvy them up, trying to be fair, turn them into a dataset,
and perform "atemporal" classification.

In the interpreter:
  atemporal = AtemporalClassifier(window_size=10, stride=2, train=['BackBreaker1', 'Enactment1', 'Enactment2', 'Enactment3', 'Enactment4', 'Enactment5', 'Enactment6', 'Enactment9', 'Enactment10', 'MainFeederBox1', 'Regulator1', 'Regulator2'], test=['Enactment11', 'Enactment12'], verbose=True)
  atemporal = AtemporalClassifier(window_size=10, stride=2, divide=['BackBreaker1', 'Enactment1', 'Enactment2', 'Enactment3', 'Enactment4', 'Enactment5', 'Enactment6', 'Enactment9', 'Enactment10', 'MainFeederBox1', 'Regulator1', 'Regulator2', 'Enactment11', 'Enactment12'], verbose=True)

Alternatively, you can give this class only a test set, not training set, and load a database file like the TemporalClassifier uses.

In the interpreter:
  atemporal = AtemporalClassifier(window_size=10, stride=2, test=['Enactment11', 'Enactment12'], verbose=True)
  atemporal.relabel_allocations_from_file('relabels.txt')
  atemporal.commit()
  atemporal.load_db('10f.db')

Or this:
  atemporal = AtemporalClassifier(window_size=10, stride=2, db_file='10f.db', test=['Enactment11', 'Enactment12'], hand_schema='strong-hand', props_coeff=27.0, hands_one_hot_coeff=17.0, verbose=True)
  atemporal.relabel_allocations_from_file('relabels.txt')
  atemporal.commit()

'''
class AtemporalClassifier(Classifier):
	def __init__(self, **kwargs):
		super(AtemporalClassifier, self).__init__(**kwargs)

		train_portion = 0.8											#  Set defaults. If they get overridden, then they get overridden.
		test_portion = 0.2

		if 'window_size' in kwargs:									#  Were we given a window size?
			assert isinstance(kwargs['window_size'], int) and kwargs['window_size'] > 0, \
			       'Argument \'window_size\' passed to AtemporalClassifier must be an int > 0.'
			self.window_size = kwargs['window_size']
		else:
			self.window_size = 10

		if 'stride' in kwargs:										#  Were we given a stride?
			assert isinstance(kwargs['stride'], int) and kwargs['stride'] > 0, \
			       'Argument \'stride\' passed to AtemporalClassifier must be an int > 0.'
			self.stride = kwargs['stride']
		else:
			self.stride = 2

		if 'train' in kwargs:										#  Were we given a list of files to put in the training set?
			assert isinstance(kwargs['train'], list), 'Argument \'train\' passed to AtemporalClassifier must be a list of strings.'
			train_list = kwargs['train']
		else:
			train_list = []

		if 'divide' in kwargs:										#  Were we given a list of files to divide evenly among training and test sets?
			assert isinstance(kwargs['divide'], list), 'Argument \'divide\' passed to AtemporalClassifier must be a list of strings.'
			divide_list = kwargs['divide']
		else:
			divide_list = []

		if 'test' in kwargs:										#  Were we given a list of files to put in the test set?
			assert isinstance(kwargs['test'], list), 'Argument \'test\' passed to AtemporalClassifier must be a list of strings.'
			test_list = kwargs['test']
		else:
			test_list = []

		if 'train_portion' in kwargs:								#  Were we given a portion of divided enactments to allocate to the training set?
			assert isinstance(kwargs['train_portion'], float) and kwargs['train_portion'] > 0.0 and kwargs['train_portion'] < 1.0, \
			       'Argument \'train_portion\' passed to AtemporalClassifier must be a float in (0.0, 1.0).'
			train_portion = kwargs['train_portion']
			test_portion = 1.0 - train_portion

		if 'test_portion' in kwargs:								#  Were we given a portion of divided enactments to allocate to the test set?
			assert isinstance(kwargs['test_portion'], float) and kwargs['test_portion'] > 0.0 and kwargs['test_portion'] < 1.0, \
			       'Argument \'test_portion\' passed to AtemporalClassifier must be a float in (0.0, 1.0).'
			test_portion = kwargs['test_portion']
			train_portion = 1.0 - test_portion

		if 'minimum_length' in kwargs:								#  Were we given a minimum sequence length?
			assert isinstance(kwargs['minimum_length'], int) and kwargs['minimum_length'] > 0, \
			       'Argument \'minimum_length\' passed to AtemporalClassifier must be an int > 0.'
			self.minimum_length = kwargs['minimum_length']
		else:
			self.minimum_length = self.window_size					#  Default to the window size.

		if 'shuffle' in kwargs:										#  Were we given an explicit order to shuffle the divided set?
			assert isinstance(kwargs['shuffle'], bool), 'Argument \'shuffle\' passed to AtemporalClassifier must be a Boolean.'
			shuffle = kwargs['shuffle']
		else:
			shuffle = False

		#############################################################
		#  The data sets and some added attributes to track them.   #
		#############################################################
		self.X_train = []											#  To become a list of lists of vectors
		self.y_train = []											#  To become a list of lables (strings)

		self.X_test = []											#  To become a list of lists of vectors
		self.y_test = []											#  To become a list of lables (strings)

		self.allocation = {}										#  key: (enactment, action index, action label) ==> val: string in {test, train}

		self.train_sample_lookup = {}								#  Allow ourselves the possibility of looking up which snippets were matched.
		self.test_sample_lookup = {}								#  key: index into X_train ==> val: (enactment, start time, start frame,
																	#                                               end time,   end frame)
																	#  key: index into X_test  ==> val: (enactment, start time, start frame,
																	#                                               end time,   end frame)
		#############################################################
		#  Load ProcessedEnactments from the given enactment names. #
		#############################################################
		if self.verbose and len(train_list) > 0:
			print('>>> Loading atemporal training set.')
		for enactment in train_list:
			pe = ProcessedEnactment(enactment, verbose=self.verbose)
			for i in range(0, pe.num_actions()):
				action_len = pe.action_len(i)						#  Get the length of this action in frames.
				if action_len >= self.minimum_length:
																	#  Mark this action in this enactment for the training set.
					self.allocation[ (enactment, i, pe.actions[i][0]) ] = 'train'
		if self.verbose and len(train_list) > 0:
			print('')

		if self.verbose and len(test_list) > 0:
			print('>>> Loading atemporal test set.')
		for enactment in test_list:
			pe = ProcessedEnactment(enactment, verbose=self.verbose)
			for i in range(0, pe.num_actions()):
				action_len = pe.action_len(i)						#  Get the length of this action in frames.
				if action_len >= self.minimum_length:
																	#  Mark this action in this enactment for the test set.
					self.allocation[ (enactment, i, pe.actions[i][0]) ] = 'test'
		if self.verbose and len(test_list) > 0:
			print('')

		if len(divide_list) > 0:
			if self.verbose:
				print('>>> Loading atemporal divided set.')
			label_accumulator = {}									#  key:label ==> val:[ (enactment-name, action-index),
																	#                      (enactment-name, action-index),
																	#                                    ...
			for enactment in divide_list:							#                      (enactment-name, action-index) ]
				pe = ProcessedEnactment(enactment, verbose=self.verbose)
				for i in range(0, pe.num_actions()):
					action_len = pe.action_len(i)					#  Get the length of this action in frames.
					if action_len >= self.minimum_length:
						if pe.actions[i][0] not in label_accumulator:
							label_accumulator[ pe.actions[i][0] ] = []
						label_accumulator[ pe.actions[i][0] ].append( (enactment, i) )

			if shuffle:
				for label in label_accumulator.keys():
					np.random.shuffle(label_accumulator[label])		#  Shuffle in place.

			for label, actions in label_accumulator.items():
				lim = min(int(round(float(len(actions)) * train_portion)), len(actions) - 1)
				for i in range(0, lim):								#  Mark this action in this enactment for the training set.
					self.allocation[ (actions[i][0], actions[i][1], label) ] = 'train'
				for i in range(lim, len(actions)):					#  Mark this action in this enactment for the test set.
					self.allocation[ (actions[i][0], actions[i][1], label) ] = 'test'
			if self.verbose:
				print('')

		if 'db_file' in kwargs:										#  Were we given a database file?
			assert isinstance(kwargs['db_file'], str), 'Argument \'db_file\' passed to AtemporalClassifier must be a string.'
			self.load_db(kwargs['db_file'])

	#################################################################
	#  Atemporal classification.                                    #
	#################################################################

	#  After you have allocated actions to the training and test sets and then called AtemporalClassifier.commit() to
	#  actually fill (self.X_train, self.y_train) and (self.X_test, self.y_test) with vectors and labels, then call
	#  this method to perform classification using the parent class's core engine.
	#  Results and statistics are collected into a dictionary and returned.
	def classify(self):
		if self.verbose:
			print('>>> Performing atemporal classifications on test set.')

		classification_stats = self.initialize_stats()				#  Init.

		self.initialize_timers()									#  (Re)set.

		num_labels = len(self.y_test)
		prev_ctr = 0
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.
		success_ctr = 0
		mismatch_ctr = 0

		t0_start = time.process_time()								#  Start timer.
		for i in range(0, len(self.y_test)):
			query = self.X_test[i]									#  Bookmark the query and ground-truth label
			ground_truth_label = self.y_test[i]
			if ground_truth_label in self.labels('train'):			#  Is this label "fair" to the classifier?
				fair = True
			else:
				fair = False
			t1_start = time.process_time()							#  Start timer.
																	#  Call the parent class's core matching engine.
			tentative_prediction, matching_costs, confidences, probabilities, metadata = super(AtemporalClassifier, self).classify(query)
			t1_stop = time.process_time()							#  Stop timer.
			self.timing['dtw-classification'].append(t1_stop - t1_start)

			#########################################################
			#  tentative_prediction:                  label or None #
			#  matching_costs: key: label ==> val: cost             #
			#  confidences:    key: label ==> val: score            #
			#  probabilities:  key: label ==> val: probability      #
			#  metadata:       key: label ==> val: {query-indices,  #
			#                                       tmplate-indices,#
			#                                       db-index}       #
			#########################################################
																	#  Save all costs for all labels.
			classification_stats['_costs'].append( tuple([0, self.window_size - 1, 'Test-snippet'] + \
			                                             [matching_costs[label] for label in self.labels('train')] + \
			                                             [ground_truth_label]) )
																	#  Save confidence scores for all labels, regardless of what the system picks.
			for label in self.labels('train'):						#  We use these values downstream in the pipeline for isotonic regression.
				classification_stats['_conf'].append( (confidences[label], label, ground_truth_label, \
				                                       'Test-snippet', 0, self.window_size - 1) )

			for label in self.labels('train'):						#  Save probabilities for all labels, regardless of what the system picks.
				classification_stats['_prob'].append( (probabilities[label], label, ground_truth_label, \
				                                       'Test-snippet', 0, self.window_size - 1) )

			t1_start = time.process_time()							#  Start timer.
			prediction = None
			if tentative_prediction is not None:
																	#  Is it above the threshold?
				if probabilities[tentative_prediction] >= self.threshold:
					prediction = tentative_prediction
					if prediction in self.hidden_labels:			#  Is this a hidden label? Then dummy up.
						prediction = None
			t1_stop = time.process_time()							#  Stop timer.
			self.timing['make-decision'].append(t1_stop - t1_start)

																	#  For the Atemporal Classifier, ground_truth_label will never be None.
			classification_stats = self.update_stats(prediction, ground_truth_label, fair, classification_stats)
			if tentative_prediction is not None:
				classification_stats['_tests'].append( (prediction, ground_truth_label, \
				                                        confidences[tentative_prediction], probabilities[tentative_prediction], \
				                                        'Test-snippet', i, \
				                                        metadata[tentative_prediction]['db-index'], \
				                                        fair) )
			else:													#  The tentative prediction is None if applied conditions make ALL possibilities impossible.
				classification_stats['_tests'].append( (prediction, ground_truth_label, \
				                                        0.0, 0.0, \
				                                        'Test-snippet', i, \
				                                        -1, \
				                                        fair) )
																	#  First-Timestamp    Final-Timestamp    Source-Enactment    Confs...    Ground-Truth-Label    {fair,unfair}
			classification_stats['_test-conf'].append( tuple([0, self.window_size - 1, 'Test-snippet'] + \
			                                                 [confidences[x] for x in self.labels('train')] + \
			                                                 [ground_truth_label, fair]) )
																	#  First-Timestamp    Final-Timestamp    Source-Enactment    Probs...    Ground-Truth-Label    {fair,unfair}
			classification_stats['_test-prob'].append( tuple([0, self.window_size - 1, 'Test-snippet'] + \
			                                                 [probabilities[x] for x in self.labels('train')] + \
			                                                 [ground_truth_label, fair]) )

			if self.render:											#  Put the query and the template side by side.
				if prediction is not None:							#  If there's no prediction, there is nothing to render.
					if 'template-indices' in metadata[prediction] and 'query-indices' in metadata[prediction]:
						t1_start = time.process_time()				#  Start timer.
						if prediction == ground_truth_label:
							success_ctr += 1
							vid_file_name = 'atemporal-success_' + str(success_ctr) + '.avi'
						else:
							mismatch_ctr += 1
							vid_file_name = 'atemporal-mismatch_' + str(mismatch_ctr) + '.avi'

						self.render_side_by_side(vid_file_name, prediction, ground_truth_label, i, metadata)
						t1_stop = time.process_time()
						self.timing['render-side-by-side'].append(t1_stop - t1_start)

			if self.verbose:
				if int(round(float(i) / float(num_labels - 1) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(i) / float(num_labels - 1) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(i) / float(num_labels - 1) * 100.0))) + '%]')
					sys.stdout.flush()
		if self.verbose:
			print('')
		t0_stop = time.process_time()								#  Stop timer.
		self.timing['total'] = t0_stop - t0_start

		return classification_stats

	#################################################################
	#  Lock settings down; get ready to classify.                   #
	#################################################################

	#  Take the present allocation of ProcessedEnactments and turn them into snippets that fill the training and/or the test sets.
	def commit(self, sets='both'):
		#############################################################
		#  (Re)set the data set attributes.                         #
		#############################################################
		if sets == 'train' or sets == 'both':
			self.X_train = []										#  To become a list of lists of vectors.
			self.y_train = []										#  To become a list of lables (strings).
			self.train_sample_lookup = {}							#  Be able to look up which training-set sample matched.

		if sets == 'test' or sets == 'both':
			self.X_test = []										#  To become a list of lists of vectors.
			self.y_test = []										#  To become a list of lables (strings).
			self.test_sample_lookup = {}							#  Be able to look up which test-set sample matched.

		#############################################################
		#  Make sure that all ProcessedEnactments align.            #
		#############################################################
																	#  Use this to make sure that all enactments recognize the same objects.
		recognizable_object_alignment = {}							#  key:(object-name, object-name, ..., object-name) ==> val:True
		image_dimension_alignment = {}								#  key:(width, height) ==> val:True
		fps_alignment = {}
		object_detection_source_alignment = {}

		#############################################################
		#  Reverse lookup.                                          #
		#############################################################

		rev_allocation = {}											#  key:string in {train, test} ==> val:[ (enactment-name, action-index, action-label),
		if sets == 'train' or sets == 'both':						#                                        (enactment-name, action-index, action-label),
			rev_allocation['train'] = []							#                                                              ...
		if sets == 'test' or sets == 'both':						#                                        (enactment-name, action-index, action-label) ]
			rev_allocation['test'] = []

		for enactment_action, set_name in self.allocation.items():	#  Build the reverse lookup.
			if set_name in rev_allocation:
				rev_allocation[set_name].append(enactment_action)

		#############################################################
		#  Create snippets.                                         #
		#############################################################

		if 'train' in rev_allocation and len(rev_allocation['train']) > 0:
			if self.verbose:
				print('>>> Collecting snippets of length ' + str(self.window_size) + ', stride ' + str(self.stride) + ' from training set enactments.')
			num = len(rev_allocation['train'])
			prev_ctr = 0
			max_ctr = os.get_terminal_size().columns - 7			#  Leave enough space for the brackets, space, and percentage.
			ctr = 0													#  Count through allocations.
			sample_ctr = 0											#  Count through snippets.

			for enactment_action in rev_allocation['train']:		#  For each tuple (enactment-name, action-index, action-label)...
				enactment_name = enactment_action[0]
				action_index = enactment_action[1]
				action_label = enactment_action[2]

				pe = ProcessedEnactment(enactment_name, verbose=False)
																	#  List of tuples: ( float(timestamp), {key:'file'               ==> val:file-path,
				enactment_frames = pe.get_frames()					#                                       key:'ground-truth-label' ==> val:label (incl. "*"),
																	#                                       key:'vector'             ==> val:vector} ), ...
																	#  Separate frame file-paths and use these to index into 'enactment_frames'.
																	#  (Avoids float-->str-->float errors.)
				video_frames = [x[1]['file'] for x in enactment_frames]

				snippets = pe.snippets_from_action(self.window_size, self.stride, action_index)
				for snippet in snippets:							#  Each 'snippet' = (label, start-time, start-frame, end-time, end-frame).
					seq = []
					for i in range(0, self.window_size):			#  Build the snippet sequence.
						vec = enactment_frames[ video_frames.index(snippet[2]) + i ][1]['vector'][:]

						if self.hand_schema == 'strong-hand':		#  Re-arrange for "strong-hand-first" encoding?
							vec = self.strong_hand_encode(vec)
																	#  Apply subvector coefficients.
						seq.append( self.apply_vector_coefficients(vec) )

					self.X_train.append( seq )						#  Append the snippet sequence.
					self.y_train.append( action_label )				#  Append ground-truth-label.

					self.train_sample_lookup[sample_ctr] = (enactment_name, snippet[1], snippet[2], snippet[3], snippet[4])
					sample_ctr += 1

				recognizable_object_alignment[ tuple(pe.recognizable_objects) ] = True
				image_dimension_alignment[ (pe.width, pe.height) ] = True
				fps_alignment[ pe.fps ] = True
				object_detection_source_alignment[ pe.object_detection_source] = True

				if self.verbose:
					if int(round(float(ctr) / float(num - 1) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
						prev_ctr = int(round(float(ctr) / float(num - 1) * float(max_ctr)))
						sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(ctr) / float(num - 1) * 100.0))) + '%]')
						sys.stdout.flush()
				ctr += 1

			if self.verbose:
				print('')

		if 'test' in rev_allocation and len(rev_allocation['test']) > 0:
			if self.verbose:
				print('>>> Collecting snippets of length ' + str(self.window_size) + ', stride ' + str(self.stride) + ' from test set enactments.')
			num = len(rev_allocation['test'])
			prev_ctr = 0
			max_ctr = os.get_terminal_size().columns - 7			#  Leave enough space for the brackets, space, and percentage.
			ctr = 0													#  Count through allocations.
			sample_ctr = 0											#  Count through snippets.

			for enactment_action in sorted(rev_allocation['test']):
				enactment_name = enactment_action[0]
				action_index = enactment_action[1]
				action_label = enactment_action[2]

				pe = ProcessedEnactment(enactment_name, verbose=False)
																	#  List of tuples: ( float(timestamp), {key:'file'               ==> val:file-path,
				enactment_frames = pe.get_frames()					#                                       key:'ground-truth-label' ==> val:label (incl. "*"),
																	#                                       key:'vector'             ==> val:vector} ), ...
																	#  Separate frame file-paths and use these to index into 'enactment_frames'.
																	#  (Avoids float-->str-->float errors.)
				video_frames = [x[1]['file'] for x in enactment_frames]

				snippets = pe.snippets_from_action(self.window_size, self.stride, action_index)
				for snippet in snippets:							#  Each 'snippet' = (label, start time, start frame, end time, end frame).
					seq = []
					for i in range(0, self.window_size):			#  Build the snippet sequence.
						vec = enactment_frames[ video_frames.index(snippet[2]) + i ][1]['vector'][:]

						if self.hand_schema == 'strong-hand':		#  Re-arrange for "strong-hand-first" encoding?
							vec = self.strong_hand_encode(vec)
																	#  Apply subvector coefficients.
						seq.append( self.apply_vector_coefficients(vec) )

					self.X_test.append( seq )						#  Append the snippet sequence.
					self.y_test.append( action_label )				#  Append ground-truth-label.

					self.test_sample_lookup[sample_ctr] = (enactment_name, snippet[1], snippet[2], snippet[3], snippet[4])
					sample_ctr += 1

				recognizable_object_alignment[ tuple(pe.recognizable_objects) ] = True
				image_dimension_alignment[ (pe.width, pe.height) ] = True
				fps_alignment[ pe.fps ] = True
				object_detection_source_alignment[ pe.object_detection_source] = True

				if self.verbose:
					if int(round(float(ctr) / float(num - 1) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
						prev_ctr = int(round(float(ctr) / float(num - 1) * float(max_ctr)))
						sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(ctr) / float(num - 1) * 100.0))) + '%]')
						sys.stdout.flush()
				ctr += 1

			if self.verbose:
				print('')

		#############################################################
		#  Check alignments.                                        #
		#############################################################

		assert len(recognizable_object_alignment.keys()) == 1, \
		       'ERROR: The objects recognizable in the enactments given to AtemporalClassifier() do not align.'
		for key in recognizable_object_alignment.keys():
			for recognizable_object in key:
																	#  Initialize with random colors
				self.robject_colors[recognizable_object] = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
				if recognizable_object not in self.recognizable_objects:
					self.recognizable_objects.append(recognizable_object)

		assert len(image_dimension_alignment.keys()) == 1, \
		       'ERROR: The image dimensions in the enactments given to AtemporalClassifier() do not align.'
		for key in image_dimension_alignment.keys():
			self.width = key[0]
			self.height = key[1]

		assert len(fps_alignment.keys()) == 1, \
		       'ERROR: The frames-per-second in the enactments given to AtemporalClassifier() do not align.'
		for key in fps_alignment.keys():
			self.fps = key

		assert len(object_detection_source_alignment.keys()) == 1, \
		       'ERROR: The object-detection sources for the enactments given to AtemporalClassifier() do not align.'
		for key in object_detection_source_alignment.keys():
			self.object_detection_source = key

		return

	#  This method in the parent class only handles X_train.
	#  This method calls the parent method and then separately handles clearing columns from X_test.
	def drop_vector_element(self, index):
		super(AtemporalClassifier, self).drop_vector_element(index)	#  The parent class's method treats the training set.
																	#  This child class treats the test set itself.
		assert isinstance(index, int) or isinstance(index, list), \
		  'Argument \'index\' passed to AtemporalClassifier.drop_vector_element() must be either a single integer or a list of integers.'

		if isinstance(index, int):									#  Cut a single index from everything in self.X_train.
			assert index < len(self.recognizable_objects), \
			  'Argument \'index\' passed to AtemporalClassifier.drop_vector_element() must be an integer less than the number of recognizable objects.'

			X_test = []
			for sequence in self.X_test:
				seq = []
				for vector in sequence:
					vec = list(vector[:12])							#  Save the intact hands-subvector.
					vec += [vector[i + 12] for i in range(0, len(self.recognizable_objects)) if i != index]
					seq.append( tuple(vec) )						#  Return to tuple.
				X_test.append( seq )								#  Return mutilated snippet to training set.

			self.vector_length -= 1									#  Decrement the vector length.
			self.vector_drop_map[index] = False						#  Mark the index-th element for omission in the test set, too!
			self.X_test = X_test

		elif isinstance(index, list):								#  Cut all given indices from everything in self.X_train.
																	#  Accept all or nothing.
			assert len([x for x in index if x < len(self.recognizable_objects)]) == len(index), \
			  'Argument \'index\' passed to AtemporalClassifier.drop_vector_element() must be a list of integers, all less than the number of recognizable objects.'

			X_test = []
			for sequence in self.X_test:
				seq = []
				for vector in sequence:
					vec = list(vector[:12])							#  Save the intact hands-subvector.
					vec += [vector[i + 12] for i in range(0, len(self.recognizable_objects)) if i not in index]
					seq.append( tuple(vec) )						#  Return to tuple.
				X_test.append( seq )								#  Return mutilated snippet to training set.

			self.vector_length -= len(index)						#  Shorten the vector length.
			for i in index:
				self.vector_drop_map[i] = False						#  Mark all indices for omission in the test set, too!
			self.X_test = X_test

		return

	#################################################################
	#  (Re)labeling.                                                #
	#################################################################

	#  Return a list of unique action labels as evidences by the snippet labels in 'y_train' and/or 'y_test.'
	#  It is possible that the training and test sets contain different action labels.
	def labels(self, sets='both'):
		if sets == 'train':
			return list(np.unique(self.y_train))
		if sets == 'test':
			return list(np.unique(self.y_test))
		return list(np.unique(self.y_train + self.y_test))

	#  Return a list of unique action labels as evidences by the snippet labels in 'y_train' and/or 'y_test.'
	#  It is possible that the training and test sets contain different action labels.
	def num_labels(self, sets='both'):
		labels = self.labels(sets)
		return len(labels)

	#  "Snippets" are already in X_train and/or X_test.
	def relabel_snippets(self, old_label, new_label):
		self.relabelings[old_label] = new_label						#  Save internally.
		for i in range(0, len(self.y_train)):
			if self.y_train[i] == old_label:
				self.y_train[i] = new_label
		for i in range(0, len(self.y_test)):
			if self.y_test[i] == old_label:
				self.y_test[i] = new_label
		return

	#  The expected format for this file is that it has ignored comment lines beginning with #
	#  and each live line has the format: old_label <tab> new_label <carriage return>
	def relabel_snippets_from_file(self, relabel_file):
		fh = open(relabel_file, 'r')
		for line in fh.readlines():
			if line[0] != '#':
				arr = line.strip().split('\t')
				old_label = arr[0]
				new_label = arr[1]
				self.relabel_snippets(old_label, new_label)
		fh.close()
		return

	#  This applies to allocations--NOT to snippets.
	#  Replace all old labels with new labels in both training and test sets.
	def relabel_allocations(self, old_label, new_label):
		self.relabelings[old_label] = new_label						#  Save internally.

		reverse_allocation = {}										#  key:string in {train, test} ==> [ (enactment-name, action-index, action-label),
		reverse_allocation['train'] = []							#                                    (enactment-name, action-index, action-label),
		reverse_allocation['test'] = []								#                                                          ...
																	#                                    (enactment-name, action-index, action-label) ]
		for k, v in self.allocation.items():
			if v == 'train':
				reverse_allocation['train'].append(k)
			elif v == 'test':
				reverse_allocation['test'].append(k)

		for i in range(0, len(reverse_allocation['train'])):
			if reverse_allocation['train'][i][2] == old_label:
				reverse_allocation['train'][i] = (reverse_allocation['train'][i][0], reverse_allocation['train'][i][1], new_label)

		for i in range(0, len(reverse_allocation['test'])):
			if reverse_allocation['test'][i][2] == old_label:
				reverse_allocation['test'][i] = (reverse_allocation['test'][i][0], reverse_allocation['test'][i][1], new_label)

		self.allocation = {}										#  key:(enactment-name, action-index, action-label) ==> val:string in {train, test}

		for i in range(0, len(reverse_allocation['train'])):
			self.allocation[ reverse_allocation['train'][i] ] = 'train'

		for i in range(0, len(reverse_allocation['test'])):
			self.allocation[ reverse_allocation['test'][i] ] = 'test'

		return

	#  Use a text file to replace all old labels with new labels in allocations (NOT snippets) for both training and test sets.
	#  The expected format for this file is that it has ignored comment lines beginning with #
	#  and each live line has the format: old_label <tab> new_label <carriage return>
	def relabel_allocations_from_file(self, relabel_file):
		fh = open(relabel_file, 'r')
		for line in fh.readlines():
			if line[0] != '#':
				arr = line.strip().split('\t')
				old_label = arr[0]
				new_label = arr[1]
				self.relabel_allocations(old_label, new_label)
		fh.close()
		return

	#################################################################
	#  Itemize set samples.                                         #
	#################################################################

	#  How many snippets are in the test set?
	def itemize(self, skip_unfair=True):
		labels = self.labels('train')
		maxindexlen = 0
		maxlabellen = 0
		for i in range(0, len(self.X_test)):
			label = self.y_test[i]
			if (skip_unfair and label in labels) or not skip_unfair:
				if len(label) > maxlabellen:
					maxlabellen = len(label)
		maxindexlen = len(str(len(self.X_test)))

		ctr = 0
		for i in range(0, len(self.X_test)):
			label = self.y_test[i]
			if (skip_unfair and label in labels) or not skip_unfair:
				print_str = '    [' + str(i) + ']:' + ' '*(maxindexlen - len(str(i))) + \
				                 label + ' '*(maxlabellen - len(label)) + '\t' + \
				                 str(self.test_sample_lookup[i][1]) + '\t-->\t' + str(self.test_sample_lookup[i][3])
				print(print_str)
				ctr += 1
		print('')
		if skip_unfair:
			print('>>> Total (fair) snippets in test set: ' + str(ctr))
		else:
			print('>>> Total snippets in test set: ' + str(ctr))
		return

	#################################################################
	#  Editing: snippets.                                           #
	#################################################################

	#  Identify labels that are in y_test but not in y_train.
	def bogie_snippets(self):
		training_labels = {}
		test_labels = {}
		for label in self.y_train:
			training_labels[ label ] = True
		for label in self.y_test:
			test_labels[ label ] = True
		return sorted([x for x in test_labels.keys() if x not in training_labels])

	#  Drop from the test set snippets that have no representation in the training set.
	def drop_bogie_snippets(self):
		bogies = self.bogie_snippets()
		for bogie in bogies:
			self.drop_snippets_from_test(bogie)
		return

	#  Drop snippets with the given 'label' from both sets.
	def drop_snippets(self, label):
		self.drop_snippets_from_train(label)
		self.drop_snippets_from_test(label)
		return

	#  Drop snippets with the given 'label' from the training set.
	def drop_snippets_from_train(self, label):
		X_tmp = []
		y_tmp = []
		lookup_tmp = {}
		ctr = 0
		for i in range(0, len(self.y_train)):
			if self.y_train[i] != label:
				X_tmp.append( self.X_train[i] )
				y_tmp.append( self.y_train[i] )
				lookup_tmp[ctr] = self.train_sample_lookup[i]
				ctr += 1
		self.X_train = X_tmp
		self.y_train = y_tmp
		self.train_sample_lookup = lookup_tmp
		return

	#  Drop snippets with the given 'label' from the test set.
	def drop_snippets_from_test(self, label):
		X_tmp = []
		y_tmp = []
		lookup_tmp = {}
		ctr = 0
		for i in range(0, len(self.y_test)):
			if self.y_test[i] != label:
				X_tmp.append( self.X_test[i] )
				y_tmp.append( self.y_test[i] )
				lookup_tmp[ctr] = self.test_sample_lookup[i]
				ctr += 1
		self.X_test = X_tmp
		self.y_test = y_tmp
		self.test_sample_lookup = lookup_tmp
		return

	#  Print information about the snippets in the training and/or test sets.
	def itemize_snippets(self, sets='both'):
		maxlabellen = 0
		maxcountlen = 0
		training_counts = {}
		testing_counts = {}
		for label in list(np.unique(self.y_train)):
			if len(label) > maxlabellen:
				maxlabellen = len(label)
			training_counts[label] = len([x for x in self.y_train if x == label])
			if len(str(training_counts[label])) > maxcountlen:
				maxcountlen = len(str(training_counts[label]))
		for label in list(np.unique(self.y_test)):
			if len(label) > maxlabellen:
				maxlabellen = len(label)
			testing_counts[label] = len([x for x in self.y_test if x == label])
			if len(str(testing_counts[label])) > maxcountlen:
				maxcountlen = len(str(testing_counts[label]))

		if (sets == 'train' or sets == 'both') and len(self.X_train) > 0:
			print('    Training set: ' + str(len(self.labels('train'))) + ' unique labels, ' + str(len(self.X_train)) + ' total ' + str(self.window_size) + '-frame snippets: ' + \
			           "{:.2f}".format(float(len(self.X_train)) / float(len(self.X_train) + len(self.X_test)) * 100.0) + ' %.')

			for label in self.labels('train'):
				print('        ' + label + ': ' + ' '*(maxlabellen - len(label)) + \
				                   str(training_counts[label]) + ' '*(maxcountlen - len(str(training_counts[label]))) + \
				   ' snippets, ' + "{:.3f}".format(float(training_counts[label]) / float(len(self.y_train)) * 100.0) + ' % of train.')

		if (sets == 'test' or sets == 'both') and len(self.X_test) > 0:
			print('    Test set:     ' + str(len(self.labels('test'))) + ' unique labels, ' + str(len(self.X_test))  + ' total ' + str(self.window_size) + '-frame snippets: ' + \
			           "{:.2f}".format(float(len(self.X_test)) / float(len(self.X_train) + len(self.X_test)) * 100.0) + ' %.')

			for label in self.labels('test'):
				if label not in list(np.unique(self.y_train)):			#  Let users know that a label in the test set is not in the training set.
					print('     !! ' + label + ': ' + ' '*(maxlabellen - len(label)) + \
					                   str(testing_counts[label]) + ' '*(maxcountlen - len(str(testing_counts[label]))) + \
					   ' snippets, ' + "{:.3f}".format(float(testing_counts[label]) / float(len(self.y_test)) * 100.0) + ' % of test.')
				else:
					print('        ' + label + ': ' + ' '*(maxlabellen - len(label)) + \
					                   str(testing_counts[label]) + ' '*(maxcountlen - len(str(testing_counts[label]))) + \
					   ' snippets, ' + "{:.3f}".format(float(testing_counts[label]) / float(len(self.y_test)) * 100.0) + ' % of test.')
		return

	#  Print information about the snippets in the test set which have no representation in the training set.
	def itemize_bogie_snippets(self):
		maxlabellen = 0
		maxcountlen = 0
		training_counts = {}
		testing_counts = {}
		for label in list(np.unique(self.y_train)):
			if len(label) > maxlabellen:
				maxlabellen = len(label)
			training_counts[label] = len([x for x in self.y_train if x == label])
			if len(str(training_counts[label])) > maxcountlen:
				maxcountlen = len(str(training_counts[label]))
		for label in list(np.unique(self.y_test)):
			if len(label) > maxlabellen:
				maxlabellen = len(label)
			testing_counts[label] = len([x for x in self.y_test if x == label])
			if len(str(testing_counts[label])) > maxcountlen:
				maxcountlen = len(str(testing_counts[label]))

		if len(self.X_test) > 0:
			print('    Test set:     ' + str(len(self.labels('test'))) + ' unique labels, ' + str(len(self.X_test))  + ' total ' + str(self.window_size) + '-frame snippets: ' + \
			           "{:.2f}".format(float(len(self.X_test)) / float(len(self.X_train) + len(self.X_test)) * 100.0) + ' %.')

			for label in self.labels('test'):
				if label not in list(np.unique(self.y_train)):			#  Let users know that a label in the test set is not in the training set.
					print('     !! ' + label + ': ' + ' '*(maxlabellen - len(label)) + \
					                   str(testing_counts[label]) + ' '*(maxcountlen - len(str(testing_counts[label]))) + \
					   ' snippets, ' + "{:.3f}".format(float(testing_counts[label]) / float(len(self.y_test)) * 100.0) + ' % of test.')
		return

	#################################################################
	#  Editing: allocations.                                        #
	#################################################################

	#  Return a list of all labels that are in allocations belonging to the test set, but which are not represented in the training set.
	def bogie_allocations(self):
		training_labels = {}
		test_labels = {}
		for enactment_action, set_name in self.allocation.items():
			if set_name == 'train':
				training_labels[ enactment_action[2] ] = True
			elif set_name == 'test':
				test_labels[ enactment_action[2] ] = True
		return sorted([x for x in test_labels.keys() if x not in training_labels])

	#  Drop all bogies (instances that are slated for the test set, but which have no representation in the training set.)
	def drop_bogie_allocations(self):
		bogies = self.bogie_allocations()
		for bogie in bogies:
			self.drop_allocation_from_test(bogie)
		return

	#  Drop all instances of the given label from both the training and the test sets.
	def drop_allocations(self, label):
		self.drop_allocations_from_train(label)
		self.drop_allocation_from_test(label)
		return

	#  Drop a specific action allocation (from whichever set contains it).
	#  (Some action performances are just no good: usually when the important elements are obscured or occur off-screen.)
	#  The argument 'index' is the index into the enactment. So to drop the sixth (zero-indexed) action performed in Enactment7, call
	#  drop_allocation('Enactment7', 5)
	def drop_allocation(self, enactment_name, index):
		keys = list(self.allocation.keys())
		i = 0
		while i < len(keys) and not (keys[i][0] == enactment_name and keys[i][1] == index):
			i += 1
		if i < len(keys):
			del self.allocation[ keys[i] ]
		return

	#  Drop all instances of the given label from allocations to the training set.
	def drop_allocations_from_train(self, label):
		marked_for_death = []
		for enactment_action, set_name in self.allocation.items():
			if set_name == 'train' and enactment_action[2] == label:
				marked_for_death.append(enactment_action)
		for dead_man in marked_for_death:
			del self.allocation[dead_man]
		return

	#  Drop all instances of the given label from allocations to the test set.
	def drop_allocation_from_test(self, label):
		marked_for_death = []
		for enactment_action, set_name in self.allocation.items():
			if set_name == 'test' and enactment_action[2] == label:
				marked_for_death.append(enactment_action)
		for dead_man in marked_for_death:
			del self.allocation[dead_man]
		return

	#  Print information about the allocations for the training and test sets.
	def itemize_allocations(self, sets='both'):
		train_labels_combos = {}									#  key:label ==> [ (enactment-name, action-index),
		test_labels_combos = {}										#                  (enactment-name, action-index),
																	#                                ...
																	#                  (enactment-name, action-index) ]
		maxlabellen = 0
		maxenactmentlen = 0

		for enactment_action, set_name in self.allocation.items():	#  key:(enactment-name, action-index, action-label) ==> val:string in {train, test}
			if set_name == 'train':
				enactment_name = enactment_action[0]
				action_index = enactment_action[1]
				action_label = enactment_action[2]
				if action_label not in train_labels_combos:
					train_labels_combos[action_label] = []
				train_labels_combos[action_label].append( (enactment_name, action_index) )

				if len(action_label) > maxlabellen:
					maxlabellen = len(action_label)
				if len(enactment_name) > maxenactmentlen:
					maxenactmentlen = len(enactment_name)

		for enactment_action, set_name in self.allocation.items():	#  key:(enactment-name, action-index, action-label) ==> val:string in {train, test}
			if set_name == 'test':
				enactment_name = enactment_action[0]
				action_index = enactment_action[1]
				action_label = enactment_action[2]
				if action_label not in test_labels_combos:
					test_labels_combos[action_label] = []
				test_labels_combos[action_label].append( (enactment_name, action_index) )

				if len(action_label) > maxlabellen:
					maxlabellen = len(action_label)
				if len(enactment_name) > maxenactmentlen:
					maxenactmentlen = len(enactment_name)

		if (sets == 'train' or sets == 'both') and len(train_labels_combos) > 0:
			print('    Training set: ' + str(len(train_labels_combos)) + ' unique labels.')
			for label, combos in sorted(train_labels_combos.items()):
				print('        ' + label + ': ' + ' '*(maxlabellen - len(label)) + str(len(combos)) + ' actions.')
				for action in sorted(combos):
					print('            ' + action[0] + ' '*(maxenactmentlen - len(action[0])) + ' action ' + str(action[1]))
			print('')

		if (sets == 'test' or sets == 'both') and len(test_labels_combos) > 0:
			print('    Test set: ' + str(len(test_labels_combos)) + ' unique labels.')
			for label, combos in sorted(test_labels_combos.items()):
				if label not in train_labels_combos:
					print('    !!  ' + label + ': ' + ' '*(maxlabellen - len(label)) + str(len(combos)) + ' actions.')
				else:
					print('        ' + label + ': ' + ' '*(maxlabellen - len(label)) + str(len(combos)) + ' actions.')
				for action in sorted(combos):
					print('            ' + action[0] + ' '*(maxenactmentlen - len(action[0])) + ' action ' + str(action[1]))
			print('')

		return

	#  Print information about allocations for the test set that are not in the training set.
	def itemize_bogie_allocations(self):
		train_labels_combos = {}									#  key:label ==> [ (enactment-name, action-index),
		test_labels_combos = {}										#                  (enactment-name, action-index),
																	#                                ...
																	#                  (enactment-name, action-index) ]
		maxlabellen = 0
		maxenactmentlen = 0

		for enactment_action, set_name in self.allocation.items():	#  key:(enactment-name, action-index, action-label) ==> val:string in {train, test}
			if set_name == 'train':
				enactment_name = enactment_action[0]
				action_index = enactment_action[1]
				action_label = enactment_action[2]
				if action_label not in train_labels_combos:
					train_labels_combos[action_label] = []
				train_labels_combos[action_label].append( (enactment_name, action_index) )

		for enactment_action, set_name in self.allocation.items():	#  key:(enactment-name, action-index, action-label) ==> val:string in {train, test}
			if set_name == 'test':
				enactment_name = enactment_action[0]
				action_index = enactment_action[1]
				action_label = enactment_action[2]
				if action_label not in test_labels_combos:
					test_labels_combos[action_label] = []
				test_labels_combos[action_label].append( (enactment_name, action_index) )

				if len(action_label) > maxlabellen:
					maxlabellen = len(action_label)
				if len(enactment_name) > maxenactmentlen:
					maxenactmentlen = len(enactment_name)

		if len(test_labels_combos) > 0:
			print('    Test set: ' + str(len(test_labels_combos)) + ' unique labels.')
			for label, combos in sorted(test_labels_combos.items()):
				if label not in train_labels_combos:
					print('    !!  ' + label + ': ' + ' '*(maxlabellen - len(label)) + str(len(combos)) + ' actions.')
					for action in sorted(combos):
						print('            ' + action[0] + ' '*(maxenactmentlen - len(action[0])) + ' action ' + str(action[1]))
			print('')

		return

	#################################################################
	#  Editing: reallocate and manipulate the data sets.            #
	#################################################################

	#  Write the current data set allocation to file.
	def export_data_split(self, file_name='data-split.txt'):
		fh = open(file_name, 'w')
		fh.write('#  Data split export from AtemporalClassifier.export_data_split(), run at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		fh.write('#  Enactment name  <tab>  Action number  <tab>  Action label  <tab>  Set\n')
		for enactment_action, set_name in sorted(self.allocation.items()):
			enactment_name = enactment_action[0]
			action_index = enactment_action[1]
			action_label = enactment_action[2]
			fh.write(enactment_name + '\t' + str(action_index) + '\t' + action_label + '\t' + set_name + '\n')
		fh.close()
		return

	#  Write the current data set allocation to file.
	#  Call this BEFORE calling .commit().
	def load_data_split(self, file_name='data-split.txt'):
		if self.verbose:
			print('>>> Loading data split from "' + file_name + '".')

		fh = open(file_name, 'r')
		self.allocation = {}

		for line in fh.readlines():
			if line[0] != '#':
				arr = line.strip().split('\t')
				enactment_name = arr[0]
				action_index = int(arr[1])
				action_label = arr[2]
				set_name = arr[3]

				self.allocation[ (enactment_name, action_index, action_label) ] = set_name

		fh.close()
		return

	#  Reassign the 'src_index'-th action in 'src_enactment' to the set ('train' or 'test') 'dst_set'.
	def reassign(self, src_enactment, src_index, dst_set):
		assert isinstance(dst_set, str) and dst_set in ['train', 'test'], \
		       'Argument \'dst_set\' passed to AtemporalClassifier.reassign() must be a string in {train, test}.'
		i = 0
		keys = [x for x in self.allocation.keys()]
		while i < len(keys) and not (keys[i][0] == src_enactment and keys[i][1] == src_index):
			i += 1
		if i < len(keys):
			self.allocation[ keys[i] ] = dst_set
		return

	#  Load a database file to the test set.
	def load_db_to_test(self, db_file):
		self.X_test = []											#  Reset.
		self.y_test = []

		reading_recognizable_objects = False
		reading_vector_length = False
		reading_snippet_size = False
		reading_stride = False

		fh = open(db_file, 'r')
		lines = fh.readlines()
		fh.close()

		sample_ctr = 0
		for line in lines:
			if line[0] == '#':
				if 'RECOGNIZABLE OBJECTS:' in line:
					reading_recognizable_objects = True
				elif reading_recognizable_objects:
					recognizable_objects = line[1:].strip().split('\t')
					assert recognizable_objects == self.recognizable_objects, 'ERROR: the recognizable objects for the test-set database do not match this classifier\'s recognizable objects.'
					reading_recognizable_objects = False

				elif 'VECTOR LENGTH:' in line:
					reading_vector_length = True
				elif reading_vector_length:
					vector_length = int(line[1:].strip())
					assert vector_length == self.vector_length, 'ERROR: the vector length for the test-set database does not match this classifier\'s vector length.'
					reading_vector_length = False

				elif 'SNIPPET SIZE:' in line:
					reading_snippet_size = True
				elif reading_snippet_size:
					db_snippet_size = int(line[1:].strip())
																	#  Atemporal: mind the self.window_size.
					assert db_snippet_size == self.window_size, 'ERROR: the snippet size for the test-set database does not match this classifier\'s window size.'
					reading_snippet_size = False

				elif 'STRIDE:' in line:
					reading_stride = True
				elif reading_stride:
					db_stride = int(line[1:].strip())
																	#  Atemporal: mind the self.stride.
					assert db_stride == self.stride, 'ERROR: the stride for the test-set database does not match this classifier\'s stride.'
					reading_stride = False
			else:
				if line[0] == '\t':
					vector = [float(x) for x in line.strip().split('\t')]
					if self.hand_schema == 'strong-hand':
						vector = self.strong_hand_encode(vector)
					self.X_test[-1].append( self.apply_vector_coefficients(vector) )
				else:
					action_arr = line.strip().split('\t')
					label                     = action_arr[0]
					db_index_str              = action_arr[1]		#  For human reference only; ignored upon loading. 'sample_ctr' handles lookup tracking.
					db_entry_enactment_source = action_arr[2]
					db_entry_start_time       = float(action_arr[3])
					db_entry_start_frame      = action_arr[4]
					db_entry_end_time         = float(action_arr[5])
					db_entry_end_frame        = action_arr[6]
					self.y_test.append( label )
					self.X_test.append( [] )
																	#  Be able to lookup the frames of a matched database sample.
					self.test_sample_lookup[sample_ctr] = (db_entry_enactment_source, db_entry_start_time, db_entry_start_frame, \
					                                                                  db_entry_end_time,   db_entry_end_frame)
					sample_ctr += 1
		return

	#################################################################
	#  Timing.                                                      #
	#################################################################

	#  Set up 'self.timing' to start collecting measurements.
	def initialize_timers(self):
		self.timing = {}											#  (Re)set.
		self.timing['total'] = 0									#  Measure total time taken
		self.timing['dtw-classification'] = []						#  This is a coarser grain: time each classification process.
		self.timing['make-decision'] = []							#  Prepare to collect final decision-making runtimes.
		if self.render:
			self.timing['render-side-by-side'] = []					#  Prepare to collect rendering times.
		return

	#################################################################
	#  Rendering                                                    #
	#################################################################

	#  Create a video with the given 'vid_file_name' illustrating the query snippet on the left and the best-matching template snippet on the right.
	#  'query_number' is an index into 'self.X_test.'
	#  Find the matched snippet in 'self.X_train' through metadata[prediction]['db-index'].
	def render_side_by_side(self, vid_file_name, prediction, ground_truth_label, query_number, metadata):
		half_width = int(round(float(self.width) * 0.5))
		half_height = int(round(float(self.height) * 0.5))

		vid = cv2.VideoWriter( vid_file_name, \
		                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
		                       self.fps, \
		                      (self.width, self.height) )

		q_enactment = self.test_sample_lookup[ query_number ][0]	#  Enactment names.
		t_enactment = self.train_sample_lookup[ metadata[prediction]['db-index'] ][0]

		q_pe = ProcessedEnactment(q_enactment, verbose=False)		#  Enactments cited (query and template).
		t_pe = ProcessedEnactment(t_enactment, verbose=False)
																	#  Indices into frames.
		q_alignment_length = len(metadata[prediction]['query-indices'])
		t_alignment_length = len(metadata[prediction]['template-indices'])
																	#  Frames of snippets cited.
		q_frames = [x[1]['file'] for x in q_pe.get_frames() if x[0] >= self.test_sample_lookup[query_number][1] and \
		                                                       x[0] < self.test_sample_lookup[query_number][3]]
		t_frames = [x[1]['file'] for x in t_pe.get_frames() if x[0] >= self.train_sample_lookup[metadata[prediction]['db-index']][1] and \
		                                                       x[0] < self.train_sample_lookup[metadata[prediction]['db-index']][3]]
																	#  Iterate over the two alignments.
		for j in range(0, max(q_alignment_length, t_alignment_length)):
			if j < q_alignment_length:
				q_index = metadata[prediction]['query-indices'][j]
			else:
				q_index = None
			if j < t_alignment_length:
				t_index = metadata[prediction]['template-indices'][j]
			else:
				t_index = None
																	#  Allocate a blank frame.
			canvas = np.zeros((self.height, self.width, 3), dtype='uint8')
																	#  Open source images.
			if q_index is not None:
				q_img = cv2.imread(q_frames[q_index], cv2.IMREAD_UNCHANGED)
			else:
				q_img = np.zeros((self.height, self.width, 3), dtype='uint8')
			if t_index is not None:
				t_img = cv2.imread(t_frames[t_index], cv2.IMREAD_UNCHANGED)
			else:
				t_img = np.zeros((self.height, self.width, 3), dtype='uint8')
																	#  Mask overlay accumulators
			q_mask_canvas = np.zeros((self.height, self.width, 3), dtype='uint8')
			t_mask_canvas = np.zeros((self.height, self.width, 3), dtype='uint8')

			if self.object_detection_source == 'GT':				#  If detections come from Ground-Truth, then look to *_props.txt for overlays.
				fh = open(q_enactment.split('/')[-1] + '_props.txt', 'r')
				lines = fh.readlines()
				fh.close()
				for line in lines:									#  Find all lines itemizing masks for the current query frame.
					if line[0] != '#':
						arr = line.strip().split('\t')				#  In this pass, we are only interested in the masks;
																	#  which are affected by transparency.
						if q_index is not None and arr[1] == q_frames[q_index]:
							object_name = arr[3]
							if object_name not in ['LeftHand', 'RightHand']:
								mask_path = arr[7]
								mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
								mask[mask > 1] = 1					#  Knock all values down to 1
																	#  Extrude to three channels.
								mask = mask[:, :, None] * np.ones(3, dtype='uint8')[None, None, :]
																	#  Convert this to a graphical overlay.
								mask[:, :, 0] *= int(round(self.robject_colors[ object_name ][2]))
								mask[:, :, 1] *= int(round(self.robject_colors[ object_name ][1]))
								mask[:, :, 2] *= int(round(self.robject_colors[ object_name ][0]))
								q_mask_canvas += mask				#  Add mask to mask accumulator.
								q_mask_canvas[q_mask_canvas > 255] = 255

				fh = open(t_enactment.split('/')[-1] + '_props.txt', 'r')
				lines = fh.readlines()
				fh.close()
				for line in lines:									#  Find all lines itemizing masks for the current template frame.
					if line[0] != '#':
						arr = line.strip().split('\t')				#  In this pass, we are only interested in the masks;
																	#  which are affected by transparency.
						if t_index is not None and arr[1] == t_frames[t_index]:
							object_name = arr[3]
							if object_name not in ['LeftHand', 'RightHand']:
								mask_path = arr[7]
								mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
								mask[mask > 1] = 1					#  Knock all values down to 1
																	#  Extrude to three channels.
								mask = mask[:, :, None] * np.ones(3, dtype='uint8')[None, None, :]
																	#  Convert this to a graphical overlay.
								mask[:, :, 0] *= int(round(self.robject_colors[ object_name ][2]))
								mask[:, :, 1] *= int(round(self.robject_colors[ object_name ][1]))
								mask[:, :, 2] *= int(round(self.robject_colors[ object_name ][0]))
								t_mask_canvas += mask				#  Add mask to mask accumulator.
								t_mask_canvas[t_mask_canvas > 255] = 255
			else:													#  If detections come from elsewhere, then look to *_detections.txt for overlays.
				fh = open(q_enactment.split('/')[-1] + '_' + self.object_detection_source.split('/')[-1] + '_detections.txt', 'r')
				lines = fh.readlines()
				fh.close()
				for line in lines:									#  Find all lines itemizing masks for the current query frame.
					if line[0] != '#':
						arr = line.strip().split('\t')				#  In this pass, we are only interested in the masks;
																	#  which are affected by transparency.
						if q_index is not None and arr[1] == q_frames[q_index]:
							object_name = arr[3]
							if object_name not in ['LeftHand', 'RightHand']:
								mask_path = arr[7]
								mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
								mask[mask > 1] = 1					#  Knock all values down to 1
																	#  Extrude to three channels.
								mask = mask[:, :, None] * np.ones(3, dtype='uint8')[None, None, :]
																	#  Convert this to a graphical overlay.
								mask[:, :, 0] *= int(round(self.robject_colors[ object_name ][2]))
								mask[:, :, 1] *= int(round(self.robject_colors[ object_name ][1]))
								mask[:, :, 2] *= int(round(self.robject_colors[ object_name ][0]))
								q_mask_canvas += mask				#  Add mask to mask accumulator.
								q_mask_canvas[q_mask_canvas > 255] = 255

				fh = open(t_enactment.split('/')[-1] + '_' + self.object_detection_source.split('/')[-1] + '_detections.txt', 'r')
				lines = fh.readlines()
				fh.close()
				for line in lines:									#  Find all lines itemizing masks for the current template frame.
					if line[0] != '#':
						arr = line.strip().split('\t')				#  In this pass, we are only interested in the masks;
																	#  which are affected by transparency.
						if t_index is not None and arr[1] == t_frames[t_index]:
							object_name = arr[3]
							if object_name not in ['LeftHand', 'RightHand']:
								mask_path = arr[7]
								mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
								mask[mask > 1] = 1					#  Knock all values down to 1
																	#  Extrude to three channels.
								mask = mask[:, :, None] * np.ones(3, dtype='uint8')[None, None, :]
																	#  Convert this to a graphical overlay.
								mask[:, :, 0] *= int(round(self.robject_colors[ object_name ][2]))
								mask[:, :, 1] *= int(round(self.robject_colors[ object_name ][1]))
								mask[:, :, 2] *= int(round(self.robject_colors[ object_name ][0]))
								t_mask_canvas += mask				#  Add mask to mask accumulator.
								t_mask_canvas[t_mask_canvas > 255] = 255

			q_img = cv2.addWeighted(q_img, 1.0, q_mask_canvas, 0.7, 0)
			q_img = cv2.cvtColor(q_img, cv2.COLOR_RGBA2RGB)			#  Flatten alpha

			t_img = cv2.addWeighted(t_img, 1.0, t_mask_canvas, 0.7, 0)
			t_img = cv2.cvtColor(t_img, cv2.COLOR_RGBA2RGB)			#  Flatten alpha

			if self.object_detection_source == 'GT':				#  If detections come from Ground-Truth, then look to *_props.txt for overlays.
				fh = open(q_enactment.split('/')[-1] + '_props.txt', 'r')
				lines = fh.readlines()
				fh.close()
				for line in lines:									#  Find those same lines again for the query frame.
					if line[0] != '#':
						arr = line.strip().split('\t')				#  In this pass, we are interested in the bounding boxes and centroids.
						if q_index is not None and arr[1] == q_frames[q_index]:
							object_name = arr[3]
							if object_name not in ['LeftHand', 'RightHand']:
								bbox_arr = arr[6].split(';')
								bbox = tuple([int(x) for x in bbox_arr[0].split(',')] + [int(x) for x in bbox_arr[1].split(',')])
								center_bbox = (int(round(float(bbox[0] + bbox[2]) * 0.5)), \
								               int(round(float(bbox[1] + bbox[3]) * 0.5)))
								cv2.rectangle(q_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (self.robject_colors[ object_name ][2], \
								                                                              self.robject_colors[ object_name ][1], \
								                                                              self.robject_colors[ object_name ][0]), 1)
								cv2.circle(q_img, (center_bbox[0], center_bbox[1]), 5, (self.robject_colors[ object_name ][2], \
								                                                        self.robject_colors[ object_name ][1], \
								                                                        self.robject_colors[ object_name ][0]), 3)

				fh = open(t_enactment.split('/')[-1] + '_props.txt', 'r')
				lines = fh.readlines()
				fh.close()
				for line in lines:									#  Find those same lines again for the template frame.
					if line[0] != '#':
						arr = line.strip().split('\t')				#  In this pass, we are interested in the bounding boxes and centroids.
						if t_index is not None and arr[1] == t_frames[t_index]:
							object_name = arr[3]
							if object_name not in ['LeftHand', 'RightHand']:
								bbox_arr = arr[6].split(';')
								bbox = tuple([int(x) for x in bbox_arr[0].split(',')] + [int(x) for x in bbox_arr[1].split(',')])
								center_bbox = (int(round(float(bbox[0] + bbox[2]) * 0.5)), \
								               int(round(float(bbox[1] + bbox[3]) * 0.5)))
								cv2.rectangle(t_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (self.robject_colors[ object_name ][2], \
								                                                              self.robject_colors[ object_name ][1], \
								                                                              self.robject_colors[ object_name ][0]), 1)
								cv2.circle(t_img, (center_bbox[0], center_bbox[1]), 5, (self.robject_colors[ object_name ][2], \
								                                                        self.robject_colors[ object_name ][1], \
								                                                        self.robject_colors[ object_name ][0]), 3)
			else:													#  If detections come from elsewhere, then look to *_detections.txt for overlays.
				fh = open(q_enactment.split('/')[-1] + '_' + self.object_detection_source.split('/')[-1] + '_detections.txt', 'r')
				lines = fh.readlines()
				fh.close()
				for line in lines:									#  Find those same lines again for the query frame.
					if line[0] != '#':
						arr = line.strip().split('\t')				#  In this pass, we are interested in the bounding boxes and centroids.
						if q_index is not None and arr[1] == q_frames[q_index]:
							object_name = arr[3]
							if object_name not in ['LeftHand', 'RightHand']:
								bbox_arr = arr[6].split(';')
								bbox = tuple([int(x) for x in bbox_arr[0].split(',')] + [int(x) for x in bbox_arr[1].split(',')])
								center_bbox = (int(round(float(bbox[0] + bbox[2]) * 0.5)), \
								               int(round(float(bbox[1] + bbox[3]) * 0.5)))
								cv2.rectangle(q_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (self.robject_colors[ object_name ][2], \
								                                                              self.robject_colors[ object_name ][1], \
								                                                              self.robject_colors[ object_name ][0]), 1)
								cv2.circle(q_img, (center_bbox[0], center_bbox[1]), 5, (self.robject_colors[ object_name ][2], \
								                                                        self.robject_colors[ object_name ][1], \
								                                                        self.robject_colors[ object_name ][0]), 3)

				fh = open(t_enactment.split('/')[-1] + '_' + self.object_detection_source.split('/')[-1] + '_detections.txt', 'r')
				lines = fh.readlines()
				fh.close()
				for line in lines:									#  Find those same lines again for the template frame.
					if line[0] != '#':
						arr = line.strip().split('\t')				#  In this pass, we are interested in the bounding boxes and centroids.
						if t_index is not None and arr[1] == t_frames[t_index]:
							object_name = arr[3]
							if object_name not in ['LeftHand', 'RightHand']:
								bbox_arr = arr[6].split(';')
								bbox = tuple([int(x) for x in bbox_arr[0].split(',')] + [int(x) for x in bbox_arr[1].split(',')])
								center_bbox = (int(round(float(bbox[0] + bbox[2]) * 0.5)), \
								               int(round(float(bbox[1] + bbox[3]) * 0.5)))
								cv2.rectangle(t_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (self.robject_colors[ object_name ][2], \
								                                                              self.robject_colors[ object_name ][1], \
								                                                              self.robject_colors[ object_name ][0]), 1)
								cv2.circle(t_img, (center_bbox[0], center_bbox[1]), 5, (self.robject_colors[ object_name ][2], \
								                                                        self.robject_colors[ object_name ][1], \
								                                                        self.robject_colors[ object_name ][0]), 3)

			q_img = cv2.resize(q_img, (half_width, half_height), interpolation=cv2.INTER_AREA)
			t_img = cv2.resize(t_img, (half_width, half_height), interpolation=cv2.INTER_AREA)

			canvas[self.side_by_side_layout_veritcal_offset:self.side_by_side_layout_veritcal_offset + half_height,           :half_width] = q_img
			canvas[self.side_by_side_layout_veritcal_offset:self.side_by_side_layout_veritcal_offset + half_height, half_width:          ] = t_img
																	#  Superimpose (query) source.
			cv2.putText(canvas, 'Atemporal Query', \
			                    (self.side_by_side_src_super['x'], self.side_by_side_src_super['y']), cv2.FONT_HERSHEY_SIMPLEX, \
			                    self.side_by_side_src_super['fontsize'], (255, 255, 255), 2)
																	#  Superimpose ground-truth label.
			cv2.putText(canvas, ground_truth_label, \
			                    (self.side_by_side_label_super['x'], self.side_by_side_label_super['y']), cv2.FONT_HERSHEY_SIMPLEX, \
			                    self.side_by_side_label_super['fontsize'], (0, 255, 0), 2)
			if q_index is not None:									#  Superimpose enactment source and frame file name.
				cv2.putText(canvas, 'From ' + q_enactment.split('/')[-1] + ', ' + q_frames[q_index].split('/')[-1], \
				                    (self.side_by_side_source_super['x'], self.side_by_side_source_super['y']), cv2.FONT_HERSHEY_SIMPLEX, \
				                    self.side_by_side_source_super['fontsize'], (255, 255, 255), 2)

																	#  Superimpose (template) source
			cv2.putText(canvas, 'Best Matching Template', \
			                    (self.side_by_side_src_super['x'] + half_width, self.side_by_side_src_super['y']), cv2.FONT_HERSHEY_SIMPLEX, \
			                    self.side_by_side_src_super['fontsize'], (255, 255, 255), 2)
			if prediction == ground_truth_label:					#  Superimpose predicted label
				cv2.putText(canvas, prediction, \
				                    (self.side_by_side_label_super['x'] + half_width, self.side_by_side_label_super['y']), cv2.FONT_HERSHEY_SIMPLEX, \
				                    self.side_by_side_label_super['fontsize'], (0, 255, 0), 2)
			else:
				cv2.putText(canvas, prediction, \
				                    (self.side_by_side_label_super['x'] + half_width, self.side_by_side_label_super['y']), cv2.FONT_HERSHEY_SIMPLEX, \
				                    self.side_by_side_label_super['fontsize'], (0, 0, 255), 2)
			if t_index is not None:									#  Superimpose enactment source and frame file name.
				cv2.putText(canvas, 'From ' + t_enactment.split('/')[-1] + ', ' + t_frames[t_index].split('/')[-1], \
				                    (self.side_by_side_source_super['x'] + half_width, self.side_by_side_source_super['y']), cv2.FONT_HERSHEY_SIMPLEX, \
				                    self.side_by_side_source_super['fontsize'], (255, 255, 255), 2)
			vid.write(canvas)
		vid.release()
		return
