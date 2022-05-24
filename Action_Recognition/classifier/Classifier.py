import cv2
import datetime
																	#  python3 setup.py build
from classifier import DTW											#    produces DTW.cpython-36m-x86_64-linux-gnu.so
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys
import time

sys.path.append('../enactment')										#  Be able to access enactment classes.
from enactment import Enactment, Gaussian3D, RecognizableObject, ProcessedEnactment

sys.path.append('../mlp')											#  Be able to access MLP class.
from mlp import MLP

'''
The Classifier object really serves as a home for attributes and functions used by both its derived classes:
  TemporalClassifier (simulating a real-time system)
  AtemporalClassifier (which has prior knowledge about where sequences begin and end.)
'''
class Classifier():
	def __init__(self, **kwargs):
		#############################################################
		#  Define permissible values for confidence function and    #
		#  hand schema.                                             #
		#############################################################
		self.confidence_function_names = ['sum2', 'sum3', 'sum4', 'sum5', 'n-min-obsv', 'min-obsv', 'max-marg', '2over1']
		self.hand_schema_names = ['left-right', 'strong-hand']

		#############################################################
		#  Define subvector lookup codes.                           #
		#############################################################
		self.hand_subvector_codes = {}
		self.hand_subvector_codes['LHx'] = 0
		self.hand_subvector_codes['LHy'] = 1
		self.hand_subvector_codes['LHz'] = 2
		self.hand_subvector_codes['LH0'] = 3
		self.hand_subvector_codes['LH1'] = 4
		self.hand_subvector_codes['LH2'] = 5
		self.hand_subvector_codes['RHx'] = 6
		self.hand_subvector_codes['RHy'] = 7
		self.hand_subvector_codes['RHz'] = 8
		self.hand_subvector_codes['RH0'] = 9
		self.hand_subvector_codes['RH1'] = 10
		self.hand_subvector_codes['RH2'] = 11

		#############################################################
		#  Set default pipeline components.                         #
		#############################################################
		self.matchingcost_to_probability = {}
		self.matchingcost_to_probability['mode'] = None				#  By default, do not expect an MLP to directly compute label probabilities.
		self.matchingcost_to_probability['pipeline'] = None

		self.confidence_to_probability = {}
		self.confidence_to_probability['mode'] = None				#  By default, do not expect an isotonic lookup, and do not expect an MLP.
		self.confidence_to_probability['pipeline'] = None

		if 'conf_func' in kwargs:									#  Were we given a confidence function?
			assert isinstance(kwargs['conf_func'], str) and kwargs['conf_func'] in self.confidence_function_names, \
			       'Argument \'conf_func\' passed to Classifier must be a string in {' + ', '.join(self.confidence_function_names) + '}.'
			self.confidence_function = kwargs['conf_func']
		else:
			self.confidence_function = 'sum2'						#  Default to 'sum2'

		if 'dtw_diagonal' in kwargs:								#  Were we given a cost for moving diagonally across the cost matrix?
			assert isinstance(kwargs['dtw_diagonal'], float), \
			       'Argument \'dtw_diagonal\' passed to Classifier must be a float.'
			self.diagonal_cost = kwargs['dtw_diagonal']
		else:
			self.diagonal_cost = 2.0								#  Default to 2.0.

		if 'config' in kwargs:										#  Were we given a configuration, as a file path or a dictionary?
			assert isinstance(kwargs['config'], str) or (isinstance(kwargs['config'], dict) and \
			                                             'matching-cost-to-probability' in kwargs['config'] and \
			                                             'confidence-to-probability' in kwargs['config'] and \
			                                             isinstance(kwargs['config']['matching-cost-to-probability'], dict) and \
			                                             isinstance(kwargs['config']['confidence-to-probability'], dict) and \
			                                             'mode' in kwargs['config']['matching-cost-to-probability'] and \
			                                             'pipeline' in kwargs['config']['matching-cost-to-probability'] and \
			                                             'mode' in kwargs['config']['confidence-to-probability'] and \
			                                             'pipeline' in kwargs['config']['confidence-to-probability']), \
			  'Argument \'config\' passed to Classifier must be either a filepath (string) to a config file or a dictionary with the appropriate keys.'
			if isinstance(kwargs['config'], str):
				self.load_config_file(kwargs['config'])
			else:
				self.matchingcost_to_probability['mode'] = kwargs['config']['matching-cost-to-probability']['mode']
				self.matchingcost_to_probability['pipeline'] = kwargs['config']['matching-cost-to-probability']['pipeline']

				self.confidence_to_probability['mode'] = kwargs['config']['confidence-to-probability']['mode']
				self.confidence_to_probability['pipeline'] = kwargs['config']['confidence-to-probability']['pipeline']

		if 'verbose' in kwargs:
			assert isinstance(kwargs['verbose'], bool), \
			       'Argument \'verbose\' passed to Classifier must be a boolean.'
			self.verbose = kwargs['verbose']
		else:
			self.verbose = False									#  Default to False.

		if 'threshold' in kwargs:									#  Were we given a threshold (for predictions)?
			assert isinstance(kwargs['threshold'], float), 'Argument \'threshold\' passed to Classifier must be a float.'
			self.threshold = kwargs['threshold']
		else:
			self.threshold = 0.0
																	#  Were we given an isotonic mapping file?
		if 'isotonic_file' in kwargs and kwargs['isotonic_file'] is not None:
			assert isinstance(kwargs['isotonic_file'], str), 'Argument \'isotonic_file\' passed to Classifier must be a string.'
			self.load_isotonic_map(kwargs['isotonic_file'])

		self.conditions = None										#  Were we given cut-off conditions?
		if 'conditions' in kwargs and kwargs['conditions'] is not None:
			assert isinstance(kwargs['conditions'], str) or isinstance(kwargs['conditions'], list), \
			       'Argument \'conditions\' passed to Classifier must be a string or a list of strings.'
			self.load_conditions(kwargs['conditions'])

		if 'presence_threshold' in kwargs:							#  Were we given an object presence threshold?
			assert isinstance(kwargs['presence_threshold'], float) and kwargs['presence_threshold'] > 0.0 and kwargs['presence_threshold'] <= 1.0, \
			       'Argument \'presence_threshold\' passed to Classifier must be a float in (0.0, 1.0].'
			self.object_presence_threshold = kwargs['presence_threshold']
		else:
			self.object_presence_threshold = 0.5

		if 'hands_coeff' in kwargs:									#  Were we given a hands-subvector coefficient?
			assert isinstance(kwargs['hands_coeff'], float), 'Argument \'hands_coeff\' passed to Classifier must be a float.'
			self.hands_coeff = kwargs['hands_coeff']
		else:
			self.hands_coeff = 1.0

		if 'hands_one_hot_coeff' in kwargs:							#  Were we given a coefficient for the one-hot hands-status-subvector?
			assert isinstance(kwargs['hands_one_hot_coeff'], float), 'Argument \'hands_one_hot_coeff\' passed to Classifier must be a float.'
			self.hands_one_hot_coeff = kwargs['hands_one_hot_coeff']
		else:
			self.hands_one_hot_coeff = 1.0

		if 'props_coeff' in kwargs:									#  Were we given a props-subvector coefficient?
			assert isinstance(kwargs['props_coeff'], float), 'Argument \'props_coeff\' passed to Classifier must be a float.'
			self.props_coeff = kwargs['props_coeff']
		else:
			self.props_coeff = 1.0

		if 'hand_schema' in kwargs:									#  Were we given a hand-schema?
			assert isinstance(kwargs['hand_schema'], str) and kwargs['hand_schema'] in self.hand_schema_names, \
			       'Argument \'hand_schema\' passed to Classifier must be a string in {' + ', '.join(self.hand_schema_names) + '}.'
			self.hand_schema = kwargs['hand_schema']
		else:
			self.hand_schema = 'strong-hand'						#  Default to strong-hand.

		if 'open_begin' in kwargs:									#  Were we told to permit or refuse an open beginning?
			assert isinstance(kwargs['open_begin'], bool), \
			       'Argument \'open_begin\' passed to Classifier must be a boolean.'
			self.open_begin = kwargs['open_begin']
		else:
			self.open_begin = False									#  Default to False.

		if 'open_end' in kwargs:									#  Were we told to permit or refuse an open ending?
			assert isinstance(kwargs['open_end'], bool), \
			       'Argument \'open_end\' passed to Classifier must be a boolean.'
			self.open_end = kwargs['open_end']
		else:
			self.open_end = False									#  Default to False.

		if 'render' in kwargs:
			assert isinstance(kwargs['render'], bool), \
			       'Argument \'render\' passed to Classifier must be a boolean.'
			self.render = kwargs['render']
		else:
			self.render = False										#  Default to False.

		self.X_train = []											#  To become a list of lists of vectors (lists of floats).
		self.y_train = []											#  To become a list of ground-truth labels (strings).
		self.recognizable_objects = []								#  Filled in by the derived classes,
																	#  either from Enactments (atemporal) or a database (temporal).
		self.vector_drop_map = []									#  List of Booleans will have as many entries as self.recognizable_objects.
																	#  If an object's corresponding element in this list is False,
																	#  then omit that column from ALL vectors.
																	#  This parent class only has X_train, so clearing out columns in the
																	#  training set happens here. Child classes must each handle clearing columns
																	#  from their respective X_test lists themselves.
		self.relabelings = {}										#  key: old-label ==> val: new-label
		self.hidden_labels = {}										#  key: label ==> True
		self.timing = {}											#  Really only used by the derived classes.

		self.epsilon = 0.000001										#  Prevent divisions by zero.

		#############################################################
		#  Attributes that facilitate display and visualization.    #
		#############################################################
		self.robject_colors = {}									#  key:recognizable-object(string) ==> val:(r, g, b)

		self.side_by_side_layout_veritcal_offset = 360
		self.side_by_side_src_super = {}							#  Where to locate and how to type the source of a side-by-side image
		self.side_by_side_src_super['x'] = 10
		self.side_by_side_src_super['y'] = 50
		self.side_by_side_src_super['fontsize'] = 1.0
		self.side_by_side_label_super = {}							#  Where to locate and how to type the label of a side-by-side video-frame
		self.side_by_side_label_super['x'] = 10
		self.side_by_side_label_super['y'] = 90
		self.side_by_side_label_super['fontsize'] = 1.0
		self.side_by_side_source_super = {}							#  Where to locate and how to type the source of a side-by-side video-frame
		self.side_by_side_source_super['x'] = 10
		self.side_by_side_source_super['y'] = 130
		self.side_by_side_source_super['fontsize'] = 1.0

	#################################################################
	#  Loading.                                                     #
	#################################################################

	#  Load a database from file. This loads directly into X_train and y_train. No "allocations."
	#  Rearrange according to schema and apply subvector coefficients before saving internally to the "training set."
	def load_db(self, db_file):
		self.X_train = []											#  Reset.
		self.y_train = []

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
					self.recognizable_objects = line[1:].strip().split('\t')
																	#  Initialize all recognizable objects to true--that is, not omitted.
					self.vector_drop_map = [True for x in self.recognizable_objects]
					self.robject_colors = {}						#  (Re)set.
					for robject in self.recognizable_objects:		#  Initialize with random colors as soon as we know our objects.
						self.robject_colors[robject] = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
					reading_recognizable_objects = False

				elif 'VECTOR LENGTH:' in line:
					reading_vector_length = True
				elif reading_vector_length:
					self.vector_length = int(line[1:].strip())
					reading_vector_length = False

				elif 'SNIPPET SIZE:' in line:
					reading_snippet_size = True
				elif reading_snippet_size:
					db_snippet_size = int(line[1:].strip())
																	#  Atemporal: mind the self.window_size.
					if type(self).__name__ == 'AtemporalClassifier':
						if db_snippet_size != self.window_size:
							if self.verbose:
								print('>>> WARNING: the previous window size of ' + str(self.window_size) + \
								                  ' will be reset to ' + str(db_snippet_size) + \
								                  ' to use the database "' + db_file + '".')
							self.window_size = db_snippet_size
																	#  Temporal: mind the self.rolling_buffer_length.
					elif type(self).__name__ == 'TemporalClassifier':
						if db_snippet_size != self.rolling_buffer_length:
							if self.verbose:
								print('>>> WARNING: the previous rolling-buffer size of ' + str(self.rolling_buffer_length) + \
								                  ' will be reset to ' + str(db_snippet_size) + \
								                  ' to use the database "' + db_file + '".')
							self.rolling_buffer_length = db_snippet_size
					reading_snippet_size = False

				elif 'STRIDE:' in line:
					reading_stride = True
				elif reading_stride:
					db_stride = int(line[1:].strip())
																	#  Atemporal: mind the self.stride.
					if type(self).__name__ == 'AtemporalClassifier':
						if db_stride != self.stride:
							if self.verbose:
								print('>>> WARNING: the previous stride of ' + str(self.stride) + \
								                  ' will be reset to ' + str(db_stride) + \
								                  ' to use the database "' + db_file + '".')
							self.stride = db_stride
																	#  Temporal: mind the self.rolling_buffer_stride.
					elif type(self).__name__ == 'TemporalClassifier':
						if db_stride != self.rolling_buffer_stride:
							if self.verbose:
								print('>>> WARNING: the previous rolling-buffer stride of ' + str(self.rolling_buffer_stride) + \
								                  ' will be reset to ' + str(db_stride) + \
								                  ' to use the database "' + db_file + '".')
							self.rolling_buffer_stride = db_stride
					reading_stride = False
			else:
				if line[0] == '\t':
					vector = [float(x) for x in line.strip().split('\t')]
					if self.hand_schema == 'strong-hand':
						vector = self.strong_hand_encode(vector)
					self.X_train[-1].append( self.apply_vector_coefficients(vector) )
				else:
					action_arr = line.strip().split('\t')
					label                     = action_arr[0]
					db_index_str              = action_arr[1]		#  For human reference only; ignored upon loading. 'sample_ctr' handles lookup tracking.
					db_entry_enactment_source = action_arr[2]
					db_entry_start_time       = float(action_arr[3])
					db_entry_start_frame      = action_arr[4]
					db_entry_end_time         = float(action_arr[5])
					db_entry_end_frame        = action_arr[6]

					if label in self.relabelings:					#  Apply relabelings.
						self.y_train.append( self.relabelings[label] )
					else:
						self.y_train.append( label )

					self.X_train.append( [] )
																	#  Be able to lookup the frames of a matched database sample.
					if type(self).__name__ == 'AtemporalClassifier' or type(self).__name__ == 'TemporalClassifier':
						self.train_sample_lookup[sample_ctr] = (db_entry_enactment_source, db_entry_start_time, db_entry_start_frame, \
						                                                                   db_entry_end_time,   db_entry_end_frame)
						sample_ctr += 1
		return

	def load_isotonic_map(self, isotonic_file):
		if self.verbose:
			print('>>> Loading isotonic map from "' + isotonic_file + '".')

		self.confidence_to_probability['mode'] = 'isotonic'
		self.confidence_to_probability['pipeline'] = {}				#  key:(lower-bound, upper-bound) ==> val:probability

		fh = open(isotonic_file, 'r')
		for line in fh.readlines():
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
				self.confidence_to_probability['pipeline'][ (lb, ub) ] = p
		fh.close()

		if self.verbose:
			for k, v in sorted(self.confidence_to_probability['pipeline'].items()):
				print('    [' + "{:.6f}".format(k[0]) + ', ' + "{:.6f}".format(k[1]) + '] ==> ' + "{:.6f}".format(v))

		return

	#  Read cut-off conditions from file and add them to the Classifier's 'conditions' table.
	def load_conditions(self, conditions_file):
		if self.conditions is None:									#  If the attribute doesn't already exist, create it now.
			self.conditions = {}

		if isinstance(conditions_file, str):						#  Convert singleton conditions file to list.
			conditions_file = [conditions_file]

		for cond_file in conditions_file:
			if self.verbose:
				print('>>> Loading cut-off conditions from "' + cond_file + '".')

			fh = open(cond_file, 'r')
			for line in fh.readlines():
				if line[0] != '#' and len(line) > 1:
					arr = line.strip().split('\t')
					action = arr[0]
					condition = arr[1]
					if action not in self.conditions:				#  If conditions for this action are not already in the table...
						self.conditions[action] = {}
						self.conditions[action]['and'] = []
						self.conditions[action]['or'] = []
					for i in range(2, len(arr)):
						if condition == 'AND':
							self.conditions[action]['and'].append( arr[i] )
						else:
							self.conditions[action]['or'].append( arr[i] )
			fh.close()

		if self.verbose:
			max_header_str_len = 0
			for key in sorted(self.conditions.keys()):
				header_str = '    In order to consider "' + key + '": '
				if len(header_str) > max_header_str_len:
					max_header_str_len = len(header_str)

			for key in sorted(self.conditions.keys()):
				header_str = '    In order to consider "' + key + '": '
				print(header_str + ' '*(max_header_str_len - len(header_str)) + ' * '.join(self.conditions[key]['and']))
				if len(self.conditions[key]['or']) > 0:
					print(' '*max_header_str_len + ' + '.join(self.conditions[key]['or']))
		return

	#  Load the pipeline from file.
	def load_config_file(self, filename):
		if self.verbose:
			print('>>> Loading config file "' + filename + '"')

		fh = open(filename, 'r')
		for line in fh.readlines():
			if line[0] != '#':
				arr = line.strip().split('\t')
				if arr[0] == 'matching-cost-to-probability':
					self.matchingcost_to_probability['mode'] = arr[1]
					if self.matchingcost_to_probability['mode'] == 'mlp':
						self.matchingcost_to_probability['pipeline'] = MLP(arr[2], executable=arr[3])
						if self.verbose:
							print('    MLP(' + arr[2] + ') to compute probabilities from matching costs.')

				elif arr[0] == 'confidence-to-probability':
					self.confidence_to_probability['mode'] = arr[1]
					if self.confidence_to_probability['mode'] == 'isotonic':
						self.load_isotonic_map(arr[2])

					elif self.confidence_to_probability['mode'] == 'mlp':
						self.confidence_to_probability['pipeline'] = MLP(arr[2], executable=arr[3])
						if self.verbose:
							print('    MLP(' + arr[2] + ') to compute probabilities from confidence scores.')

		fh.close()

		return

	#################################################################
	#  Classification prep.                                         #
	#################################################################

	#  Return a dictionary of the form: key: action-label ==> val: inf
	def prepare_matching_costs_table(self):
		matching_costs = {}											#  Matching cost for the nearest neighbor per class.
																	#  In defense of hashing versus building a flat list:
																	#  a database is likely to have many more samples than there are labels.
																	#  As a hash table, updating takes O(1) * |X_train| rather than
																	#  O(n) * |X_train| to find matching_costs[ matching_costs.index(template_label) ].
		if type(self).__name__ == 'AtemporalClassifier' or type(self).__name__ == 'TemporalClassifier':
			labels = self.labels('train')
		else:
			labels = sorted(np.unique(self.y_train))

		for label in labels:										#  Initialize everything to infinitely far away.
			matching_costs[label] = float('inf')

		return matching_costs

	#  Return a dictionary of the form: key: action-label ==> val: 0.0
	def prepare_probabilities_table(self, include_nothing_label=False):
		probabilities = {}											#  If we apply isotonic mapping, then this is a different measure than confidence.
		if type(self).__name__ == 'AtemporalClassifier' or type(self).__name__ == 'TemporalClassifier':
			labels = self.labels('train')
		else:
			labels = sorted(np.unique(self.y_train))

		for label in labels:
			probabilities[label] = 0.0

		if include_nothing_label:									#  Expect to explicitly predict the nothing-label.
			probabilities['*'] = 0.0

		return probabilities

	#  "metadata" includes innformation about matches: which template snippet made the best match, and how the template and query frames align.
	def prepare_metadata_table(self, include_nothing_label=False):
		metadata = {}												#  Track information about the best match per class.
		if type(self).__name__ == 'AtemporalClassifier' or type(self).__name__ == 'TemporalClassifier':
			labels = self.labels('both')							#  Include labels the classifier may not know; these will simply be empty.
		else:
			labels = sorted(np.unique(self.y_train))

		for label in labels:
			metadata[label] = {}

		if include_nothing_label:									#  Allowed here for completeness, but the DB is never expected
			metadata['*'] = {}										#  to include "nothing" snippets.

		return metadata

	#################################################################
	#  Shared classification engine.                                #
	#################################################################

	#  This is the core classification routine, usable by Atemporal and Temporal subclasses alike.
	#  Given a single query sequence, return:
	#    - nearest_neighbor_label:                      string
	#    - Matching costs over all classes:             dict = key: label ==> val: least cost for that label
	#    - Confidences over all classes:                dict = key: label ==> val: confidence
	#                                                          * May include the nothing-label
	#    - Probability distribution over all classes:   dict = key: label ==> val: probability
	#                                                          * May include the nothing-label
	#    - Metadata (nearest neighbor indices, alignment sequences) over all classes
	#                                                   dict = key: label ==> val:{key: db-index         ==> DB index
	#                                                                              key: template-indices ==> frames
	#                                                                              key: query-indices    ==> frames }
	#  Use the DTW module written in C.
	def classify(self, query):
																	#  Always begin with matching costs.
		nearest_neighbor_label, matching_costs, metadata = self.DTW_match(query)
																	#  Compute confidences according to self.confidence_function.
																	#  Get a dictionary of key:label ==> val:confidence.
		confidences = self.compute_confidences(sorted([x for x in matching_costs.items()], key=lambda x: x[1]))

		#############################################################
		#  The Classifier can pass the matching costs to a Multi-   #
		#  Layer Perceptron to estimate probabilities directly.     #
		#                                                           #
		#  In this case, classify() returns:                        #
		#    nearest_neighbor_label                                 #
		#    matching_costs in R^N                                  #
		#    confidences    in R^N                                  #
		#    probabilities  in R^{N+1}                              #
		#    metadata                                               #
		#############################################################
		if self.matchingcost_to_probability['mode'] == 'mlp':		#  Expect self.matchingcost_to_probability['pipeline'] is an MLP object.

			probabilities = self.prepare_probabilities_table(True)	#  Initialize all to zero. Explicitly predict the nothing-label.

			if type(self).__name__ == 'AtemporalClassifier' or type(self).__name__ == 'TemporalClassifier':
				labels = self.labels('train')
			else:
				labels = sorted(np.unique(self.y_train))

			labels_and_nothing = labels + ['*']						#  Append the nothing-label.

			assert isinstance(self.matchingcost_to_probability['pipeline'], MLP), \
			  'In order to estimate label probabilities directly from matching costs, the Classifier.matchingcost_to_probability[\'pipeline\'] must be an MLP object.'

			y_hat = self.matchingcost_to_probability['pipeline'].run( [matching_costs[label] for label in labels] )

			for i in range(0, len(labels_and_nothing)):
				probabilities[ labels_and_nothing[i] ] = y_hat[i]

		#############################################################
		#  The Classifier can use the matching costs to compute     #
		#  confidences according to self.confidence_function and    #
		#  then convert these to probabilities.                     #
		#                                                           #
		#  In this case, classify() returns:                        #
		#    nearest_neighbor_label                                 #
		#    matching_costs in R^N                                  #
		#    confidences    in R^N                                  #
		#    probabilities  in R^N                                  #
		#    metadata                                               #
		#############################################################
		if self.confidence_to_probability['mode'] == 'isotonic':	#  Expect self.confidence_to_probability['pipeline'] is an isotonic lookup table.

			probabilities = self.prepare_probabilities_table(False)	#  Initialize all to zero. DO NOT explicitly predict the nothing-label.

			for label, confidence in confidences.items():
				brackets = sorted(self.confidence_to_probability['pipeline'].keys())
				i = 0
				while i < len(brackets) and not (confidence > brackets[i][0] and confidence <= brackets[i][1]):
					i += 1

				probabilities[label] = self.confidence_to_probability['pipeline'][ brackets[i] ]

			prob_norm = sum( probabilities.values() )				#  Normalize probabilities.
			for k in probabilities.keys():
				if prob_norm > 0.0:
					probabilities[k] /= prob_norm
				else:
					probabilities[k] = 0.0

		#############################################################
		#  The Classifier can pass confidence scores to a Multi-    #
		#  Layer Perceptron (MLP) to estimate probabilities.        #
		#                                                           #
		#  In this case, classify() returns:                        #
		#    nearest_neighbor_label                                 #
		#    matching_costs in R^N                                  #
		#    confidences    in R^N                                  #
		#    probabilities  in R^{N+1}                              #
		#    metadata                                               #
		#############################################################
		elif self.confidence_to_probability['mode'] == 'mlp':		#  Expect self.confidence_to_probability['pipeline'] is an MLP object.

			probabilities = self.prepare_probabilities_table(True)	#  Initialize all to zero. Explicitly predict the nothing-label.

			if type(self).__name__ == 'AtemporalClassifier' or type(self).__name__ == 'TemporalClassifier':
				labels = self.labels('train')
			else:
				labels = sorted(np.unique(self.y_train))

			labels_and_nothing = labels + ['*']						#  Append the nothing-label.

			assert isinstance(self.confidence_to_probability['pipeline'], MLP), \
			  'In order to estimate label probabilities from confidence scores, the Classifier.confidence_to_probability[\'pipeline\'] must be an MLP object.'

			y_hat = self.confidence_to_probability['pipeline'].run( [confidences[label] for label in labels] )

			for i in range(0, len(labels_and_nothing)):
				probabilities[ labels_and_nothing[i] ] = y_hat[i]

		#############################################################
		#  The Classifier can simply normalize confidence scores and#
		#  consider them probabilities, though this is discouraged. #
		#                                                           #
		#  In this case, classify() returns:                        #
		#    nearest_neighbor_label                                 #
		#    matching_costs in R^N                                  #
		#    confidences    in R^N                                  #
		#    probabilities  in R^N                                  #
		#    metadata                                               #
		#############################################################
		else:														#  No isotonic map and no MLP?

			probabilities = self.prepare_probabilities_table(False)	#  Initialize all to zero. DO NOT explicitly predict the nothing-label.

			for label, confidence in confidences.items():			#  Then probability = (normalized) confidence, which is sloppy, but... meh.
				probabilities[label] = confidence

			prob_norm = sum( probabilities.values() )				#  Normalize probabilities.
			for k in probabilities.keys():
				if prob_norm > 0.0:
					probabilities[k] /= prob_norm
				else:
					probabilities[k] = 0.0

		return nearest_neighbor_label, matching_costs, confidences, probabilities, metadata

	#  Performs DTW matching on the given query snippet.
	#  If curoff conditions apply (such as "Only consider Grab(Helmet) if vector[helmet] > 0.0"), they apply here.
	#  Returns
	#    'nearest_neighbor_label', a string
	#    'matching_costs',         a dictionary:  key: action-label ==> val: matching-cost
	#    'metadata',               a dictionary:  key: action-label ==> val: {key:'db-index'         ==> val: index of best matching DB snippet
	#                                                                         key:'template-indices' ==> val: frames in T snippet
	#                                                                         key:'query-indices'    ==> val: frames in Q snippet
	#                                                                        }
	def DTW_match(self, query):
		matching_costs = self.prepare_matching_costs_table()		#  Initialize all costs for all predictable labels as +inf.
		metadata = self.prepare_metadata_table()					#  Initialize all metadata.

		least_cost = float('inf')									#  Initialize least cost found to +inf.
		nearest_neighbor_label = None								#  Initialize best label to None.

		db_index = 0												#  Index into self.X_train let us know which sample best matches the query.
		for template in self.X_train:								#  For every training-set sample, 'template'...
			template_label = self.y_train[db_index]					#  Save the true label for this template sequence.

			conditions_passed = False
			if self.conditions is not None:
				conditions_passed = self.test_cutoff_conditions(template_label, query)

			if self.conditions is None or conditions_passed:		#  Either we have no conditions, or our conditions give us reason to run DTW.
																	#  What is the distance between this query and this template?
				dist, _, query_indices, template_indices = DTW.DTW(query, template, self.diagonal_cost)

				if least_cost > dist:								#  A preferable match over all!
					least_cost = dist								#  Save the cost.
					nearest_neighbor_label = template_label[:]		#  Save the (tentative) prediction.

				if matching_costs[template_label] > dist:			#  A preferable match within class!
					matching_costs[template_label] = dist			#  Save the preferable cost.

					metadata[template_label]['db-index'] = db_index	#  Save information about the best match found so far.
					metadata[template_label]['template-indices'] = template_indices
					metadata[template_label]['query-indices'] = query_indices

			db_index += 1

		return nearest_neighbor_label, matching_costs, metadata

	#  Does the given 'query_seq' present enough support for us to even consider attempting to match this query
	#  with templates exemplifying 'candidate_label'?
	#  Return True if the classification routine should proceed with DTW matching.
	#  Return False if the given sequence provides no reason to bother running DTW matching.
	def test_cutoff_conditions(self, candidate_label, query_seq):
		if candidate_label in self.conditions:						#  This action/label is subject to condition.

			if len(self.conditions[candidate_label]['and']) > 0:	#  Do ANDs apply?
				ctr = 0
				for vector in query_seq:
					i = 0
					while i < len(self.conditions[candidate_label]['and']):
						if self.conditions[candidate_label]['and'][i] in self.recognizable_objects:
																	#  12 is the offset past the hand encodings, into the props sub-vector.
																	#  If anything required in the AND list has a zero signal, then this frame fails.
							if vector[ self.recognizable_objects.index(self.conditions[candidate_label]['and'][i]) + 12 ] == 0.0:
								break
						else:										#  Necessary object in {LHx, LHy, LHz, LH0, LH1, LH2,
																	#                       RHx, RHy, RHz, RH0, RH1, RH2}
							if vector[ self.hand_subvector_codes[ self.conditions[candidate_label]['and'][i] ] ] == 0.0:
								break
						i += 1
																	#  Did we make it all the way through the list without zero-ing out?
																	#  That means that, for this frame at least, all necessary objects are non-zero.
					if i == len(self.conditions[candidate_label]['and']):
						ctr += 1
																	#  Were all necessary objects present enough?
				if float(ctr) / float(len(query_seq)) >= self.object_presence_threshold:
					passed_and = True
				else:
					passed_and = False
			else:
				passed_and = True

			if len(self.conditions[candidate_label]['or']) > 0:		#  Do ORs apply?
				ctr = 0
				for vector in query_seq:
					i = 0
					while i < len(self.conditions[candidate_label]['or']):
						if self.conditions[candidate_label]['or'][i] in self.recognizable_objects:
																	#  12 is the offset past the hand encodings, into the props sub-vector.
																	#  If anything required in the AND list has a zero signal, then this frame fails.
							if vector[ self.recognizable_objects.index(self.conditions[candidate_label]['or'][i]) + 12 ] > 0.0:
								break
						else:										#  Necessary object in {LHx, LHy, LHz, LH0, LH1, LH2,
																	#                       RHx, RHy, RHz, RH0, RH1, RH2}
							if vector[ self.hand_subvector_codes[ self.conditions[candidate_label]['or'][i] ] ] > 0.0:
								break
						i += 1
																	#  Did we bail early because we found something--anything that was > 0.0?
																	#  That means that, for this frame at least, at least one necessary object is non-zero.
					if i < len(self.conditions[candidate_label]['or']):
						ctr += 1
																	#  Were all necessary objects present enough?
				if float(ctr) / float(len(query_seq)) >= self.object_presence_threshold:
					passed_or = True
				else:
					passed_or = False
			else:
				passed_or = True

			return passed_and and passed_or

		return True

	#  Receive a sorted list of tuples(label, cost); return a dictionary: key:label ==> val:confidence.
	def compute_confidences(self, labels_costs):
		conf = {}
		for key in labels_costs:
			conf[ key[0] ] = 0.0

		if self.confidence_function == 'sum2':						#  (Minimum distance + 2nd-minimal distance) / my distance
			s = sum([x[1] for x in labels_costs[:2]])
			for i in range(0, len(labels_costs)):
				conf[ labels_costs[i][0] ] = s / (labels_costs[i][1] + self.epsilon)
				if np.isnan(conf[ labels_costs[i][0] ]):
					conf[ labels_costs[i][0] ] = 0.0

		elif self.confidence_function == 'sum3':					#  (Sum of three minimal distances) / my distance
			s = sum([x[1] for x in labels_costs[:3]])
			for i in range(0, len(labels_costs)):
				conf[ labels_costs[i][0] ] = s / (labels_costs[i][1] + self.epsilon)
				if np.isnan(conf[ labels_costs[i][0] ]):
					conf[ labels_costs[i][0] ] = 0.0

		elif self.confidence_function == 'sum4':					#  (Sum of four minimal distances) / my distance
			s = sum([x[1] for x in labels_costs[:4]])
			for i in range(0, len(labels_costs)):
				conf[ labels_costs[i][0] ] = s / (labels_costs[i][1] + self.epsilon)
				if np.isnan(conf[ labels_costs[i][0] ]):
					conf[ labels_costs[i][0] ] = 0.0

		elif self.confidence_function == 'sum5':					#  (Sum of five minimal distances) / my distance
			s = sum([x[1] for x in labels_costs[:5]])
			for i in range(0, len(labels_costs)):
				conf[ labels_costs[i][0] ] = s / (labels_costs[i][1] + self.epsilon)
				if np.isnan(conf[ labels_costs[i][0] ]):
					conf[ labels_costs[i][0] ] = 0.0

		elif self.confidence_function == 'n-min-obsv':				#  Normalized minimum distance observed / my distance
			min_d = labels_costs[0][1]
			for i in range(0, len(labels_costs)):
				conf[ labels_costs[i][0] ] = min_d / (labels_costs[i][1] + self.epsilon)
			s = sum(conf.values())
			for i in range(0, len(labels_costs)):
				conf[ labels_costs[i][0] ] /= s
				if np.isnan(conf[ labels_costs[i][0] ]):
					conf[ labels_costs[i][0] ] = 0.0

		elif self.confidence_function == 'min-obsv':				#  Minimum distance observed / my distance
			min_d = labels_costs[0][1]								#  DO NOT ACTUALLY USE THIS FUNCTION! it's an illustrative fail-case ONLY.
			for i in range(0, len(labels_costs)):
				conf[ labels_costs[i][0] ] = min_d / (labels_costs[i][1] + self.epsilon)
				if np.isnan(conf[ labels_costs[i][0] ]):
					conf[ labels_costs[i][0] ] = 0.0

		elif self.confidence_function == 'max-marg':				#  Second-best distance minus best distance. Worst match gets zero.
			for i in range(0, len(labels_costs)):
				if i < len(labels_costs) - 1:
					conf[ labels_costs[i][0] ] = labels_costs[i + 1][1] - labels_costs[i][1]
					if np.isnan(conf[ labels_costs[i][0] ]):
						conf[ labels_costs[i][0] ] = 0.0
				else:
					conf[ labels_costs[i][0] ] = 0.0

		elif self.confidence_function == '2over1':					#  Second-best distance over best distance. Worst match gets zero.
			for i in range(0, len(labels_costs)):
				if i < len(labels_costs) - 1:
					conf[ labels_costs[i][0] ] = labels_costs[i + 1][1] / (labels_costs[i][1] + self.epsilon)
					if np.isnan(conf[ labels_costs[i][0] ]):
						conf[ labels_costs[i][0] ] = 0.0
				else:
					conf[ labels_costs[i][0] ] = 0.0

		return conf

	#  Given a prediction, the ground-truth label, and the stats-collection object (dictionary), update the counts.
	#    stats = {key:_tests  ==> val:[(prediction, ground-truth, enactment-source, timestamp, DB-index, fair),
	#                                  (prediction, ground-truth, enactment-source, timestamp, DB-index, fair),
	#                                   ... ],
	#             key:_conf   ==> val:[confidence,                        confidence,                      ... ],
	#             key:_prob   ==> val:[probability,                       probability,                     ... ],
	#             key:<label> ==> val:{key:tp      ==> val:true positive count,
	#                                  key:fp      ==> val:false positive count,
	#                                  key:tn      ==> val:true negative count,
	#                                  key:fn      ==> val:false negative count,
	#                                  key:support ==> val:instances in training set}
	#            }
	def update_stats(self, prediction, ground_truth_label, fair, stats):
		if fair:
			if prediction == ground_truth_label:					#  Hard match: TP++
				stats[ground_truth_label]['tp'] += 1

			elif prediction is not None:							#  Classifier chose something, but it's not a match.
				stats[prediction]['fp']  += 1						#  FP++ on the mis-identified action.
				if ground_truth_label != '*':						#  FN++ on the falsely negated true action.
					stats[ground_truth_label]['fn'] += 1

			elif prediction is None:								#  Classifier withheld.
				if ground_truth_label == '*':						#  Nothing is happening; this is a true negation. TN++
					stats['*']['tn'] += 1
				else:												#  FN++ on the falsely negated true action.
					stats[ground_truth_label]['fn'] += 1

		return stats

	#  When a label is 'hidden', it is still in the DB (y_train), and it can still be recognized.
	#  However, when selected as the system's prediction, a hidden label becomes a no-vote.
	def hide_label(self, label):
		self.hidden_labels[label] = True
		return

	#  Remove the given 'label' from the table of hidden labels.
	def unhide_label(self, label):
		if label in self.hidden_labels:
			del self.hidden_labels[label]
		return

	#################################################################
	#  Vector encoding.                                             #
	#################################################################

	def strong_hand_encode(self, vector):
		lh_norm = np.linalg.norm(vector[:3])
		rh_norm = np.linalg.norm(vector[6:9])
		if lh_norm > rh_norm:										#  Left hand is the strong hand; leave everything in place.
			vec = [x for x in vector]
		else:														#  Right hand is the strong hand; swap [:6] and [6:12].
			vec = [x for x in vector[6:12]] + \
			      [x for x in vector[:6]]   + \
			      [x for x in vector[12:]]
		return tuple(vec)

	#  Return a clone of the given vector with the hands and props subvectors weighted by their respective coefficients.
	def apply_vector_coefficients(self, vector):
		vec = [x for x in vector]									#  Convert to a list so we can manipulate it.

		vec[0] *= self.hands_coeff									#  Weigh LH_x.
		vec[1] *= self.hands_coeff									#  Weigh LH_y.
		vec[2] *= self.hands_coeff									#  Weigh LH_z.
		vec[3] *= self.hands_one_hot_coeff							#  Weigh one-hot encoded LH_0.
		vec[4] *= self.hands_one_hot_coeff							#  Weigh one-hot encoded LH_1.
		vec[5] *= self.hands_one_hot_coeff							#  Weigh one-hot encoded LH_2.
		vec[6] *= self.hands_coeff									#  Weigh RH_x.
		vec[7] *= self.hands_coeff									#  Weigh RH_y.
		vec[8] *= self.hands_coeff									#  Weigh RH_z.
		vec[9] *= self.hands_one_hot_coeff							#  Weigh one-hot encoded RH_0.
		vec[10] *= self.hands_one_hot_coeff							#  Weigh one-hot encoded RH_1.
		vec[11] *= self.hands_one_hot_coeff							#  Weigh one-hot encoded RH_2.
		for i in range(12, len(vector)):							#  Weigh props_i.
			vec[i] *= self.props_coeff

		return tuple(vec)

	#  We would want to do this when rendering seismographs: prevent them from being HUGE.
	def undo_vector_coefficients(self, vector):
		vec = [x for x in vector]									#  Convert to a list so we can manipulate it.

		vec[0] /= self.hands_coeff									#  Weigh LH_x.
		vec[1] /= self.hands_coeff									#  Weigh LH_y.
		vec[2] /= self.hands_coeff									#  Weigh LH_z.
		vec[3] /= self.hands_one_hot_coeff							#  Weigh one-hot encoded LH_0.
		vec[4] /= self.hands_one_hot_coeff							#  Weigh one-hot encoded LH_1.
		vec[5] /= self.hands_one_hot_coeff							#  Weigh one-hot encoded LH_2.
		vec[6] /= self.hands_coeff									#  Weigh RH_x.
		vec[7] /= self.hands_coeff									#  Weigh RH_y.
		vec[8] /= self.hands_coeff									#  Weigh RH_z.
		vec[9] /= self.hands_one_hot_coeff							#  Weigh one-hot encoded RH_0.
		vec[10] /= self.hands_one_hot_coeff							#  Weigh one-hot encoded RH_1.
		vec[11] /= self.hands_one_hot_coeff							#  Weigh one-hot encoded RH_2.
		for i in range(12, len(vector)):							#  Weigh props_i.
			vec[i] /= self.props_coeff

		return tuple(vec)

	#  Remove the index-th element from all vectors in self.X_train.
	#  Note that 'index' only treats the props-subvector.
	#  In other words, dropping index 0 will drop the first recognizable object--NOT the left-hand's X component!
	def drop_vector_element(self, index):
		assert isinstance(index, int) or isinstance(index, list), \
		  'Argument \'index\' passed to Classifier.drop_vector_element() must be either a single integer or a list of integers.'

		if isinstance(index, int):									#  Cut a single index from everything in self.X_train.
			assert index < len(self.recognizable_objects), \
			  'Argument \'index\' passed to Classifier.drop_vector_element() must be an integer less than the number of recognizable objects.'

			X_train = []
			for sequence in self.X_train:
				seq = []
				for vector in sequence:
					vec = list(vector[:12])							#  Save the intact hands-subvector.
					vec += [vector[i + 12] for i in range(0, len(self.recognizable_objects)) if i != index]
					seq.append( tuple(vec) )						#  Return to tuple.
				X_train.append( seq )								#  Return mutilated snippet to training set.

			self.vector_length -= 1									#  Decrement the vector length.
			self.vector_drop_map[index] = False						#  Mark the index-th element for omission in the test set, too!
			self.X_train = X_train

		elif isinstance(index, list):								#  Cut all given indices from everything in self.X_train.
																	#  Accept all or nothing.
			assert len([x for x in index if x < len(self.recognizable_objects)]) == len(index), \
			  'Argument \'index\' passed to Classifier.drop_vector_element() must be a list of integers, all less than the number of recognizable objects.'

			X_train = []
			for sequence in self.X_train:
				seq = []
				for vector in sequence:
					vec = list(vector[:12])							#  Save the intact hands-subvector.
					vec += [vector[i + 12] for i in range(0, len(self.recognizable_objects)) if i not in index]
					seq.append( tuple(vec) )						#  Return to tuple.
				X_train.append( seq )								#  Return mutilated snippet to training set.

			self.vector_length -= len(index)						#  Shorten the vector length.
			for i in index:
				self.vector_drop_map[i] = False						#  Mark all indices for omission in the test set, too!
			self.X_train = X_train

		return

	#################################################################
	#  Initializers                                                 #
	#################################################################

	def initialize_stats(self):
		classification_stats = {}
		classification_stats['_costs'] = []							#  key:_costs ==> val:[ (timestamp-start, timestamp-end, enactment-source,
																	#                        cost_0, cost_1, cost_2, ..., cost_N, ground-truth-label),
																	#                       (timestamp-start, timestamp-end, enactment-source,
																	#                        cost_0, cost_1, cost_2, ..., cost_N, ground-truth-label),
																	#                       ... ]
		classification_stats['_tests'] = []							#  key:_tests ==> val:[ ( prediction, ground-truth,
																	#                         confidence-of-prediction, probability-of-pred,
																	#                         enactment-source, timestamp, DB-index, fair ),
																	#                       ( prediction, ground-truth,
																	#                         confidence-of-prediction, probability-of-pred,
																	#                         enactment-source, timestamp, DB-index, fair ),
																	#                       ... ]
		classification_stats['_test-conf'] = []						#  key:_test-conf ==> val:[ (c_0, c_1, c_2, ..., c_N) for test 0,
																	#                           (c_0, c_1, c_2, ..., c_N) for test 1,
																	#                           (c_0, c_1, c_2, ..., c_N) for test 2,
																	#                           ... ]
		classification_stats['_test-prob'] = []						#  key:_test-prob ==> val:[ (p_0, p_1, p_2, ..., p_N) for test 0,
																	#                           (p_0, p_1, p_2, ..., p_N) for test 1,
																	#                           (p_0, p_1, p_2, ..., p_N) for test 2,
																	#                           ... ]
		classification_stats['_test-smooth-prob'] = []				#  key:_test-smooth-prob ==> val:[ (sp_0, sp_1, sp_2, ..., sp_N) for test 0,
																	#                                  (sp_0, sp_1, sp_2, ..., sp_N) for test 1,
																	#                                  (sp_0, sp_1, sp_2, ..., sp_N) for test 2,
																	#                                  ... ]
		classification_stats['_conf'] = []							#  key:_conf  ==> val:[ (confidence-for-label, label, ground-truth,
																	#                        source-enactment, first-snippet-timestamp, final-snippet-timestamp),
																	#                       (confidence-for-label, label, ground-truth,
																	#                        source-enactment, first-snippet-timestamp, final-snippet-timestamp),
																	#                       ... ]

		classification_stats['_prob'] = []							#  key:_prob  ==> val:[ (probability-for-label, label, ground-truth,
																	#                        source-enactment, first-snippet-timestamp, final-snippet-timestamp),
																	#                       (probability-for-label, label, ground-truth,
																	#                        source-enactment, first-snippet-timestamp, final-snippet-timestamp),
																	#                       ... ]

		for label in self.labels('both'):							#  May include "unfair" labels, but will not include the "*" nothing-label.
			classification_stats[label] = {}						#  key:label ==> val:{key:tp      ==> val:true positive count
			classification_stats[label]['tp']      = 0				#                     key:fp      ==> val:false positive count
			classification_stats[label]['fp']      = 0				#                     key:tn      ==> val:true negative count
			classification_stats[label]['tn']      = 0				#                     key:fn      ==> val:false negative count
			classification_stats[label]['fn']      = 0				#                     key:support ==> val:instances in training set}
			classification_stats[label]['support'] = len([x for x in self.y_train if x == label])

		classification_stats['*'] = {}								#  Add the nothing-label manually.
		classification_stats['*']['tn'] = 0							#  This will only have true negatives.

		return classification_stats

	#################################################################
	#  Rendering: (common to both derived classes.)                 #
	#################################################################

	#  Load an RGB 3-tuple for each string in self.recognizable_objects.
	def load_color_map(self, color_file):
		self.robject_colors = {}									#  (Re)set
		for robject in self.recognizable_objects:					#  What if the colors file is missing something? Initialize with randoms.
			self.robject_colors[robject] = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

		fh = open(color_file, 'r')
		for line in fh.readlines():
			if line[0] != '#':
				arr = line.strip().split('\t')
				self.robject_colors[ arr[0] ] = (int(arr[1]), int(arr[2]), int(arr[3]))
		fh.close()

		return

	#################################################################
	#  Reporting.                                                   #
	#################################################################

	#  Unify this object's outputs with a unique time stamp.
	def time_stamp(self):
		now = datetime.datetime.now()								#  Build a distinct substring so I don't accidentally overwrite results.
		file_timestamp = now.strftime("%d") + now.strftime("%m") + now.strftime("%Y")[-2:] + 'T' + now.strftime("%H:%M:%S").replace(':', '')
		return file_timestamp

	#  Given 'predictions_truths' is a list of tuples: ( prediction, ground-truth,
	#                                                    confidence-of-prediction, probability-of-prediction,
	#                                                    enactment-source, timestamp, DB-index, fair ).
	def confusion_matrix(self, predictions_truths, sets='both'):
		labels = self.labels(sets)
		labels.append('*')											#  Add the nothing-label, a posteriori.
		num_classes = len(labels)

		M = np.zeros((num_classes, num_classes), dtype='uint16')

		for pred_gt in predictions_truths:
			prediction = pred_gt[0]
			if prediction is None:
				prediction = '*'
			ground_truth_label = pred_gt[1]
			fair = pred_gt[-1]

			if prediction in labels and fair:
				i = labels.index(prediction)
				if ground_truth_label in labels:
					j = labels.index(ground_truth_label)
					M[i, j] += 1

		return M

	#  Write the confusion matrix to file.
	#  Given 'predictions_truths' is a list of tuples: ( prediction, ground-truth,
	#                                                    confidence-of-prediction, probability-of-prediction,
	#                                                    enactment-source, timestamp, DB-index, fair ).
	def write_confusion_matrix(self, predictions_truths, file_timestamp=None, sets='both'):
		if file_timestamp is None:
			file_timestamp = self.time_stamp()						#  Build a distinct substring so I don't accidentally overwrite results.

		labels = self.labels(sets)
		labels.append('*')											#  Add the nothing-label, a posteriori.
		num_classes = len(labels)

		M = self.confusion_matrix(predictions_truths, sets)

		fh = open('confusion-matrix-' + file_timestamp + '.txt', 'w')
		fh.write('#  Classifier confusion matrix made at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		fh.write('\t' + '\t'.join(labels) + '\n')					#  Write the column headers.
		for i in range(0, num_classes):
			fh.write(labels[i] + '\t' + '\t'.join([str(x) for x in M[i]]) + '\n')
		fh.close()

		return

	#  Writes file: "matching_costs-<time stamp>.txt".
	#  This contains all matching costs for all training-set labels for each call to DTW.
	#  Each line of this file reads:  time-start  <tab>  time-end  <tab>  enactment  <tab>  cost_0  <tab>  cost_1  <tab> ... <tab>  cost_N  <tab>  ground-truth  <tab>  {fair/unfair}.
	def write_matching_costs(self, costs, file_timestamp=None):
		if file_timestamp is None:
			file_timestamp = self.time_stamp()						#  Build a distinct substring so I don't accidentally overwrite results.

		train_labels = self.labels('train')

		fh = open('matching_costs-' + file_timestamp + '.txt', 'w')
		fh.write('#  Classifier matching costs made at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		fh.write('#  RECOGNIZABLE ACTIONS:\n')
		fh.write('#    ' + '\t'.join(train_labels) + '\n')
		fh.write('#  Each line is:\n')
		fh.write('#    First-Timestamp    Final-Timestamp    Source-Enactment    ' + \
		         '    '.join(['Cost_' + label for label in train_labels]) + '    Ground-Truth-Label    {fair,unfair}' + '\n')

		for i in range(0, len(costs)):
			fh.write(str(costs[i][0]) + '\t' + str(costs[i][1]) + '\t' + costs[i][2] + '\t')

			for j in range(0, len(train_labels)):					#  Write all costs.
				fh.write(str(costs[i][j + 3]) + '\t')
			fh.write(costs[i][-1] + '\t')							#  Write ground truth.
																	#  Anything in the training set is fair,
			if costs[i][-1] in train_labels or costs[i][-1] == '*':	#  and the nothing-label is fair.
				fh.write('fair\n')
			else:
				fh.write('unfair\n')
		fh.close()

		return

	#  Writes file: "confidences-<time stamp>.txt".
	#  This contains all confidence scores for all training-set labels.
	#  That means, for every prediction the system made, we save N scores, where N is the number of training-set labels.
	#  Each line of this file reads:  score  <tab>  label  <tab>  ground-truth  <tab>  source-enactment  <tab>  first-timestamp  <tab>  final-timestamp  <tab>  fair/unfair.
	def write_confidences(self, confidences, file_timestamp=None):
		if file_timestamp is None:
			file_timestamp = self.time_stamp()						#  Build a distinct substring so I don't accidentally overwrite results.
																	#  Sort DESCENDING by confidence.
		conf_label_gt = sorted(confidences, key=lambda x: x[0], reverse=True)
		train_labels = self.labels('train')
		fh = open('confidences-' + file_timestamp + '.txt', 'w')
		fh.write('#  Classifier predictions made at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		fh.write('#  Confidence function is "' + self.confidence_function + '"\n')
		fh.write('#  Each line is:\n')
		fh.write('#    Confidence    Label    Ground-Truth-Label    Source-Enactment    First-Timestamp   Final-Timestamp   {fair, unfair}\n')
		for i in range(0, len(conf_label_gt)):
			fh.write(str(conf_label_gt[i][0]) + '\t' + conf_label_gt[i][1] + '\t' + conf_label_gt[i][2] + '\t')
			fh.write(conf_label_gt[i][3] + '\t' + str(conf_label_gt[i][4]) + '\t' + str(conf_label_gt[i][5]) + '\t')
			if conf_label_gt[i][2] in train_labels:
				fh.write('fair\n')
			else:
				fh.write('unfair\n')
		fh.close()
		return

	#  Writes file: "probabilities-<time stamp>.txt".
	#  This contains all probabilities for all training-set labels.
	#  That means, for every prediction the system made, we save N probabilities, where N is the number of training-set labels.
	#  For each prediction, all N label probabilities sum to 1.0.
	#  Each line of this file reads:  probability  <tab>  label  <tab>  ground-truth  <tab>  source-enactment  <tab>  first-timestamp  <tab>  final-timestamp  <tab>  fair/unfair.
	def write_probabilities(self, probabilities, file_timestamp=None):
		if file_timestamp is None:
			file_timestamp = self.time_stamp()						#  Build a distinct substring so I don't accidentally overwrite results.
																	#  Sort DESCENDING by probability.
		prob_label_gt = sorted(probabilities, key=lambda x: x[0], reverse=True)
		train_labels = self.labels('train')
		fh = open('probabilities-' + file_timestamp + '.txt', 'w')
		fh.write('#  Classifier predictions made at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		fh.write('#  Confidence function is "' + self.confidence_function + '"\n')
		fh.write('#  Each line is:\n')
		fh.write('#    Probability    Label    Ground-Truth-Label    Source-Enactment    First-Timestamp   Final-Timestamp   {fair, unfair}\n')
		for i in range(0, len(prob_label_gt)):
			fh.write(str(prob_label_gt[i][0]) + '\t' + prob_label_gt[i][1] + '\t' + prob_label_gt[i][2] + '\t')
			fh.write(prob_label_gt[i][3] + '\t' + str(prob_label_gt[i][4]) + '\t' + str(prob_label_gt[i][5]) + '\t')
			if prob_label_gt[i][2] in train_labels:
				fh.write('fair\n')
			else:
				fh.write('unfair\n')
		fh.close()
		return

	#  Avoid repeating time-consuming experiments. Save results to file.
	#  'stats' is dictionary with key:'_tests' ==> val:[ ( prediction, ground-truth,
	#                                                      confidence-of-prediction, probability-of-prediction,
	#                                                      enactment-source, timestamp, DB-index, fair ),
	#                                                    ( prediction, ground-truth,
	#                                                      confidence-of-prediction, probability-of-prediction,
	#                                                      enactment-source, timestamp, DB-index, fair ),
	#                                                    ... ]
	#                             key:'_test-conf' ==> val:[ (c_0, c_1, c_2, ..., c_N) for test 0,
	#                                                        (c_0, c_1, c_2, ..., c_N) for test 1,
	#                                                        (c_0, c_1, c_2, ..., c_N) for test 2,
	#                                                        ... ]
	#                             key:'_test-prob' ==> val:[ (p_0, p_1, p_2, ..., p_N) for test 0,
	#                                                        (p_0, p_1, p_2, ..., p_N) for test 1,
	#                                                        (p_0, p_1, p_2, ..., p_N) for test 2,
	#                                                        ... ]
	#                             key:'_conf'  ==> val:[ (confidence, label, ground-truth), (confidence, label, ground-truth), ... ]
	#
	#                             key:'_prob'  ==> val:[ (probability, label, ground-truth), (probability, label, ground-truth), ... ]
	#
	#                             key:<label>  ==> val:{key:'tp'      ==> val: true positive count for <label>,
	#                                                   key:'fp'      ==> val: false positive count for <label>,
	#                                                   key:'fn'      ==> val: false negative count for <label>,
	#                                                   key:'support' ==> val: samples for <label> in training set.}
	def write_results(self, stats, file_timestamp=None):
		if file_timestamp is None:
			file_timestamp = self.time_stamp()						#  Build a distinct substring so I don't accidentally overwrite results.

		fh = open('results-' + file_timestamp + '.txt', 'w')
		fh.write('#  Classifier results completed at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		if len(sys.argv) > 1:										#  Was this called from a script? Save the command-line call.
			fh.write('#  ' + ' '.join(sys.argv) + '\n\n')

		train_labels = self.labels('train')							#  List of strings.
		train_labels.append('*')									#  Add the nothing-label, a posteriori.

		fair_test_pred = []											#  Convert all tests into indices into self.labels('train').
		fair_test_y    = []
		fair_conf      = []
		for i in range(0, len(stats['_tests'])):
			pred_label = stats['_tests'][i][0]
			if pred_label is None:
				pred_label = '*'									#  Convert None to the nothing-label.
			gt_label   = stats['_tests'][i][1]
			conf       = stats['_tests'][i][2]
			fair       = stats['_tests'][i][-1]

			#if pred_label is not None and gt_label in train_labels and fair:
			if gt_label in train_labels and fair:
				fair_test_pred.append( train_labels.index(pred_label) )
				fair_test_y.append(    train_labels.index(gt_label)   )
				fair_conf.append(              conf                   )

		acc = accuracy_score(fair_test_y, fair_test_pred)			#  Compute total accuracy.

																	#  Compute precision for EACH class.
		class_prec = precision_score(fair_test_y, fair_test_pred, average=None, labels=list(range(0, len(train_labels))), zero_division=0)

																	#  Compute recall for EACH class.
		class_recall = recall_score(fair_test_y, fair_test_pred, average=None, labels=list(range(0, len(train_labels))), zero_division=0)

																	#  Compute per-class F1 score.
		class_f1 = f1_score(fair_test_y, fair_test_pred, average=None, labels=list(range(0, len(train_labels))), zero_division=0)

		acc_str = 'Accuracy:  ' + str(acc)
		fh.write(acc_str + '\n')
		fh.write('='*len(acc_str) + '\n')
		fh.write('\n')

		mean_acc = []
		conf_mat = confusion_matrix(fair_test_y, fair_test_pred)

		fh.write('Per-Class:\n')
		fh.write('==========\n')
		fh.write('\tAccuracy\tPrecision\tRecall\tF1-score\tSupport\n')
		for k, v in sorted( [x for x in stats.items() if x[0] not in ['_tests', '_costs', '_test-conf', '_test-prob', '_conf', '_prob', '*']] ):
			if v['support'] > 0:									#  ONLY ATTEMPT TO CLASSIFY IF THIS IS A "FAIR" QUESTION
				tn = np.sum(np.delete(np.delete(conf_mat, train_labels.index(k), axis=0), train_labels.index(k), axis=1))
				tp = conf_mat[train_labels.index(k), train_labels.index(k)]
				per_class_acc = (tp + tn) / np.sum(conf_mat)
				mean_acc.append(per_class_acc)

				prec = class_prec[ train_labels.index(k) ]
				recall = class_recall[ train_labels.index(k) ]
				f1 = class_f1[ train_labels.index(k) ]

				support = v['support']
				fh.write(k + '\t' + str(per_class_acc) + '\t' + str(prec) + '\t' + str(recall) + '\t' + str(f1) + '\t' + str(support) + '\n')
		fh.write('\n')

		if len(mean_acc) == 0:
			maacc_str = 'Mean Avg. Accuracy:  N/A'
		else:
			maacc_str = 'Mean Avg. Accuracy:  ' + str(np.mean(mean_acc))
		fh.write(maacc_str + '\n')
		fh.write('='*len(maacc_str) + '\n\n')

		conf_correct = []											#  Accumulate confidences when the classifier is correct.
		conf_incorrect = []											#  Accumulate confidences when the classifier is incorrect.

		prob_correct = []											#  Accumulate probabilities when the classifier is correct.
		prob_incorrect = []											#  Accumulate probabilities when the classifier is incorrect.

		decision_ctr = 0											#  Count times the classifier made a decision.
		no_decision_ctr = 0											#  Count times the classifier abstained from making a decision.

		for i in range(0, len(stats['_tests'])):
			pred_label = stats['_tests'][i][0]
			gt_label   = stats['_tests'][i][1]
			conf       = stats['_tests'][i][2]						#  Confidence may be None
			prob       = stats['_tests'][i][3]
			fair       = stats['_tests'][i][-1]
			if pred_label is None:
				pred_label = '*'

			if gt_label in train_labels and fair:
				if pred_label == '*':
					no_decision_ctr += 1
				else:
					decision_ctr += 1

				if pred_label == gt_label:
					if conf is not None:
						conf_correct.append(conf)
					prob_correct.append(prob)
				else:
					if conf is not None:
						conf_incorrect.append(conf)
					prob_incorrect.append(prob)

		avg_conf_correct = np.mean(conf_correct)					#  Compute average confidence when correct.
		avg_conf_incorrect = np.mean(conf_incorrect)				#  Compute average confidence when incorrect.

		stddev_conf_correct = np.std(conf_correct)					#  Compute standard deviation of confidence when correct.
		stddev_conf_incorrect = np.std(conf_incorrect)				#  Compute standard deviation of confidence when incorrect.

		avg_prob_correct = np.mean(prob_correct)					#  Compute average probability when correct.
		avg_prob_incorrect = np.mean(prob_incorrect)				#  Compute average probability when incorrect.

		stddev_prob_correct = np.std(prob_correct)					#  Compute standard deviation of probability when correct.
		stddev_prob_incorrect = np.std(prob_incorrect)				#  Compute standard deviation of probability when incorrect.

		fh.write('Total decisions made = ' + str(decision_ctr) + '\n')
		fh.write('Total non-decisions made = ' + str(no_decision_ctr) + '\n')
		fh.write('\n')
		fh.write('Avg. Confidence when correct = ' + str(avg_conf_correct) + '\n')
		fh.write('Avg. Confidence when incorrect = ' + str(avg_conf_incorrect) + '\n')
		fh.write('\n')
		fh.write('Std.Dev. Confidence when correct = ' + str(stddev_conf_correct) + '\n')
		fh.write('Std.Dev. Confidence when incorrect = ' + str(stddev_conf_incorrect) + '\n')
		fh.write('\n')
		fh.write('Avg. Probability when correct = ' + str(avg_prob_correct) + '\n')
		fh.write('Avg. Probability when incorrect = ' + str(avg_prob_incorrect) + '\n')
		fh.write('\n')
		fh.write('Std.Dev. Probability when correct = ' + str(stddev_prob_correct) + '\n')
		fh.write('Std.Dev. Probability when incorrect = ' + str(stddev_prob_incorrect) + '\n')
		fh.write('\n')

		fh.write('Raw Results:\n')
		fh.write('============\n')
		fh.write('\tTP\tFP\tFN\tSupport\tTests\n')
		for k, v in sorted( [x for x in stats.items() if x[0] not in ['_tests', '_costs', '_test-conf', '_test-prob', '_conf', '_prob', '*']] ):
			fh.write(k + '\t' + str(v['tp']) + '\t' + str(v['fp']) + '\t' + str(v['fn']) + '\t' + str(v['support']) + '\t')
			fh.write(str(len( [x for x in stats['_tests'] if x[1] == k and x[-1]] )) + '\n')
		fh.write('\n')

		fh.write('Test Set Survey:\n')
		fh.write('================\n')
		for label in train_labels:
			fh.write(label + '\t' + str(len( [x for x in stats['_tests'] if x[1] == label and x[-1]] )) + '\n')
		fh.write('\n')
		fh.close()

		self.write_test_itemization(stats, file_timestamp)			#  Log test itemization.
		self.write_per_test_confidences(stats, file_timestamp)		#  Log per-test confidences.
		self.write_per_test_probabilities(stats, file_timestamp)	#  Log per-test probabilities.

		return

	#  Write an itemization of test results in the order in which they were performed.
	#  'stats' is dictionary with key:'_tests' ==> val:[ ( prediction, ground-truth,
	#                                                      confidence-of-prediction, probability-of-prediction,
	#                                                      enactment-source, timestamp, DB-index, fair ),
	#                                                    ( prediction, ground-truth,
	#                                                      confidence-of-prediction, probability-of-prediction,
	#                                                      enactment-source, timestamp, DB-index, fair ),
	#                                                    ... ]
	def write_test_itemization(self, stats, file_timestamp=None):
		if file_timestamp is None:
			file_timestamp = self.time_stamp()						#  Build a distinct substring so I don't accidentally overwrite results.

		train_labels = self.labels('train')							#  List of strings.

		fh = open('test-itemization_' + file_timestamp + '.txt', 'w')
		fh.write('#  Classifier test itemization completed at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		if len(sys.argv) > 1:										#  Was this called from a script? Save the command-line call.
			fh.write('#  COMMAND:\n')
			fh.write('#    ' + ' '.join(sys.argv) + '\n')
		fh.write('#  RECOGNIZABLE ACTIONS:\n')
		fh.write('#    ' + '\t'.join(train_labels) + '\n')
		fh.write('#  FORMAT per LINE:\n')
		fh.write('#    Pred.    G.T.    Conf.    Prob.    Enactment    Time    DB-Index    fair/unfair\n')

		train_labels.append('*')									#  Add the nothing-label, a posteriori.

		for i in range(0, len(stats['_tests'])):
			if stats['_tests'][i][0] is None:						#  [0]  Prediction
				pred = '*'
			else:
				pred = stats['_tests'][i][0]
			gt = stats['_tests'][i][1]								#  [1]  Ground-Truth
			conf = stats['_tests'][i][2]							#  [2]  Confidence (may be None)
			if conf is None:
				conf = '*'
			else:
				conf = str(conf)
			prob = stats['_tests'][i][3]							#  [3]  Probability
			enactment_src = stats['_tests'][i][4]					#  [4]  Enactment source
			timestamp = stats['_tests'][i][5]						#  [5]  Time stamp at of newest frame
			db_index = stats['_tests'][i][6]						#  [6]  Database index

			fh.write(pred + '\t' + gt + '\t' + conf + '\t' + str(prob) + '\t' + enactment_src + '\t' + str(timestamp) + '\t' + str(db_index) + '\t')
			if stats['_tests'][i][7]:								#  [7]  Fair/Unfair
				fh.write('fair\n')
			else:
				fh.write('unfair\n')
		fh.close()

		return

	#  'stats' is dictionary with key:'_test-conf' ==> val:[ (c_0, c_1, c_2, ..., c_N) for test 0,
	#                                                        (c_0, c_1, c_2, ..., c_N) for test 1,
	#                                                        (c_0, c_1, c_2, ..., c_N) for test 2,
	#                                                        ... ]
	#  Each line of this file reads:  time-start  <tab>  time-end  <tab>  enactment  <tab>  conf_0  <tab>  conf_1  <tab> ... <tab>  conf_N  <tab>  ground-truth  <tab>  {fair/unfair}.
	#  Note that we will never have a confidence score explicitly for the nothing-label.
	def write_per_test_confidences(self, stats, file_timestamp=None):
		if file_timestamp is None:
			file_timestamp = self.time_stamp()						#  Build a distinct substring so I don't accidentally overwrite results.

		train_labels = self.labels('train')							#  List of strings.

		fh = open('test-confidences_' + file_timestamp + '.txt', 'w')
		fh.write('#  Classifier test confidences per class, completed at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		if len(sys.argv) > 1:										#  Was this called from a script? Save the command-line call.
			fh.write('#  COMMAND:\n')
			fh.write('#    ' + ' '.join(sys.argv) + '\n')
		fh.write('#  RECOGNIZABLE ACTIONS:\n')
		fh.write('#    ' + '\t'.join(train_labels) + '\n')
		fh.write('#  Each line is:\n')
		fh.write('#    First-Timestamp    Final-Timestamp    Source-Enactment    ' + \
		         '    '.join(['Conf_' + label for label in train_labels]) + '    Ground-Truth-Label    {fair,unfair}' + '\n')

		for conf in stats['_test-conf']:
			fh.write(str(conf[0]) + '\t' + str(conf[1]) + '\t' + conf[2] + '\t')

			for i in range(0, len(train_labels)):					#  Write all costs.
				fh.write(str(conf[i + 3]) + '\t')
			fh.write(conf[-2] + '\t')								#  Write ground truth.

			if conf[-1]:											#  Write fairness.
				fh.write('fair\n')
			else:
				fh.write('unfair\n')

		fh.close()

		return

	#  'stats' is dictionary with key:'_test-prob' ==> val:[ (p_0, p_1, p_2, ..., p_N) for test 0,
	#                                                        (p_0, p_1, p_2, ..., p_N) for test 1,
	#                                                        (p_0, p_1, p_2, ..., p_N) for test 2,
	#                                                        ... ]
	#  Each line of this file reads:  time-start  <tab>  time-end  <tab>  enactment  <tab>  prob_0  <tab>  prob_1  <tab> ... <tab>  prob_N  <tab>  ground-truth  <tab>  {fair/unfair}.
	#  Note that we MAY have a probability explicitly for the nothing-label.
	def write_per_test_probabilities(self, stats, file_timestamp=None):
		if file_timestamp is None:
			file_timestamp = self.time_stamp()						#  Build a distinct substring so I don't accidentally overwrite results.

		train_labels = self.labels('train')							#  List of strings.

		fh = open('test-probabilities_' + file_timestamp + '.txt', 'w')
		fh.write('#  Classifier test probabilities per class, completed at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		if len(sys.argv) > 1:										#  Was this called from a script? Save the command-line call.
			fh.write('#  COMMAND:\n')
			fh.write('#    ' + ' '.join(sys.argv) + '\n')
		fh.write('#  RECOGNIZABLE ACTIONS:\n')
		fh.write('#    ' + '\t'.join(train_labels) + '\n')
		fh.write('#  Each line is:\n')

		if len(stats['_test-prob'][0]) == len(train_labels) + 5:	#  No nothing-label probability.
			include_nothing = False
			fh.write('#    First-Timestamp    Final-Timestamp    Source-Enactment    ' + \
			         '    '.join(['Prob_' + label for label in train_labels]) + '    Ground-Truth-Label    {fair,unfair}' + '\n')
		else:														#  Includes the nothing-label.
			include_nothing = True
			fh.write('#    First-Timestamp    Final-Timestamp    Source-Enactment    ' + \
			         '    '.join(['Prob_' + label for label in train_labels]) + '    Prob_*    Ground-Truth-Label    {fair,unfair}' + '\n')

		for prob in stats['_test-prob']:
			fh.write(str(prob[0]) + '\t' + str(prob[1]) + '\t' + prob[2] + '\t')

			for i in range(0, len(train_labels)):					#  Write all costs.
				fh.write(str(prob[i + 3]) + '\t')

			if include_nothing:
				fh.write(str(prob[len(train_labels) + 3]) + '\t')

			fh.write(prob[-2] + '\t')								#  Write ground truth.

			if prob[-1]:											#  Write fairness.
				fh.write('fair\n')
			else:
				fh.write('unfair\n')

		fh.close()

		return

	#  Look for:
	#    - total													Total time taken.
	#
	#    - load-enactment											Times taken to load enactments.
	#    - image-open												Times taken to read images from disk.
	#    - object-detection											Times taken to perform object detection on a single frame.
	#    - centroid-computation										Times taken to compute a 3D centroid from a detection bounding box.
	#    - dtw-classification										Times taken to make a single call to the DTW backend.
	#    - sort-confidences											Times taken to put confidence scores in label-order.
	#    - sort-probabilities										Times taken to put probabilities in label-order.
	#    - push-temporal-buffer										Times taken to update temporal-buffer.
	#    - temporal-smoothing										Times taken to perform temporal smoothing.
	#    - make-temporally-smooth-decision							Times taken to make a final decision, given temporally-smoothed probabilities.
	#
	#    - per-frame                                                Time taken per frame of video
	#
	#    - render-side-by-side										Times taken to produce a video showing the query and best-matched template.
	#    - render-annotated-source									Times taken to produce that annotated source image.
	#    - render-rolling-buffer									Times taken to produce the rolling buffer seismograph.
	#    - render-confidence										Times taken to produce the confidences seismograph.
	#    - render-probabilities										Times taken to produce the probabilities seismograph.
	#    - render-smoothed-probabilities							Times taken to produce the smoothed-probabilities seismograph.
	def write_timing(self, file_timestamp=None):
		if file_timestamp is None:
			file_timestamp = self.time_stamp()						#  Build a distinct substring so I don't accidentally overwrite results.

		bootup_time = 0.0
		detection_time = 0.0
		classification_time = 0.0
		rendering_time = 0.0
		accounted_time = 0.0										#  Separate itemized times from overhead

		fh = open('timing-' + file_timestamp + '.txt', 'w')
		fh.write('#  Times for classifier tasks, completed at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		if len(sys.argv) > 1:										#  Was this called from a script? Save the command-line call.
			fh.write('#  ' + ' '.join(sys.argv) + '\n')
		fh.write('\n')
																	#  'total' MUST be in the table.
		fh.write('TOTAL TIME\t' + str(self.timing['total']) + '\n\n')
																	#  Report per-frame time.
		if 'per-frame' in self.timing and len(self.timing['per-frame']) > 0:
			fh.write('PER-FRAME avg. time  \t' + str(np.mean(self.timing['per-frame'])) + '\n')
			fh.write('PER-FRAM std.dev time\t' + str(np.std(self.timing['per-frame'])) + '\n\n')

		#############################################################
		#                          Boot up                          #
		#############################################################
		if ('load-model' in self.timing) or \
		   ('load-enactment' in self.timing and len(self.timing['load-enactment']) > 0):
			fh.write('###  BOOT-UP TIMES:\n')

		if 'load-model' in self.timing:
			bootup_time += np.sum(self.timing['load-model'])
			accounted_time += np.sum(self.timing['load-model'])
		if 'load-enactment' in self.timing and len(self.timing['load-enactment']) > 0:
			bootup_time += np.sum(self.timing['load-enactment'])
			accounted_time += np.sum(self.timing['load-enactment'])

																	#  Report object-detection model-loading times.
		if 'load-model' in self.timing:
			fh.write('  DETECTION-MODEL-LOADING avg. time\t' + str(self.timing['load-model']) + '\t' + str(self.timing['load-model'] / bootup_time * 100.0) + '% of bootup time\n')
			fh.write('  DETECTION-MODEL-LOADING std.dev time\t' + str(np.std(self.timing['load-model'])) + '\n\n')
																	#  Report enactment-loading times.
		if 'load-enactment' in self.timing and len(self.timing['load-enactment']) > 0:
			fh.write('  ENACTMENT-LOADING avg. time\t' + str(np.mean(self.timing['load-enactment'])) + '\t' + str(np.sum(self.timing['load-enactment']) / bootup_time * 100.0) + '% of bootup time\n')
			fh.write('  ENACTMENT-LOADING std.dev time\t' + str(np.std(self.timing['load-enactment'])) + '\n\n')

		#############################################################
		#                         Detection                         #
		#############################################################
		if ('image-open' in self.timing and len(self.timing['image-open']) > 0) or \
		   ('object-detection' in self.timing and len(self.timing['object-detection']) > 0) or \
		   ('centroid-computation' in self.timing and len(self.timing['centroid-computation']) > 0):
			fh.write('###  DETECTION TIMES:\n')

		if 'image-open' in self.timing and len(self.timing['image-open']) > 0:
			detection_time += np.sum(self.timing['image-open'])
			accounted_time += np.sum(self.timing['image-open'])
		if 'object-detection' in self.timing and len(self.timing['object-detection']) > 0:
			detection_time += np.sum(self.timing['object-detection'])
			accounted_time += np.sum(self.timing['object-detection'])
		if 'centroid-computation' in self.timing and len(self.timing['centroid-computation']) > 0:
			detection_time += np.sum(self.timing['centroid-computation'])
			accounted_time += np.sum(self.timing['centroid-computation'])
																	#  Report image-opening times.
		if 'image-open' in self.timing and len(self.timing['image-open']) > 0:
			fh.write('  IMAGE-LOADING avg. time\t' + str(np.mean(self.timing['image-open'])) + '\t' + str(np.sum(self.timing['image-open']) / detection_time * 100.0) + '% of detection time\n')
			fh.write('  IMAGE-LOADING std.dev time\t' + str(np.std(self.timing['image-open'])) + '\n\n')
																	#  Report object detection times.
		if 'object-detection' in self.timing and len(self.timing['object-detection']) > 0:
			fh.write('  OBJECT-DETECTION avg. time\t' + str(np.mean(self.timing['object-detection'])) + '\t' + str(np.sum(self.timing['object-detection']) / detection_time * 100.0) + '% of detection time\n')
			fh.write('  OBJECT-DETECTION std.dev time\t' + str(np.std(self.timing['object-detection'])) + '\n\n')
																	#  Report centroid computation times.
		if 'centroid-computation' in self.timing and len(self.timing['centroid-computation']) > 0:
			fh.write('  CENTROID-COMPUTE avg. time\t' + str(np.mean(self.timing['centroid-computation'])) + '\t' + str(np.sum(self.timing['centroid-computation']) / detection_time * 100.0) + '% of detection time\n')
			fh.write('  CENTROID-COMPUTE std.dev time\t' + str(np.std(self.timing['centroid-computation'])) + '\n\n')

		#############################################################
		#                     Classification                        #
		#############################################################
		if ('dtw-classification' in self.timing and len(self.timing['dtw-classification']) > 0) or \
		   ('sort-confidences' in self.timing and len(self.timing['sort-confidences']) > 0) or \
		   ('sort-probabilities' in self.timing and len(self.timing['sort-probabilities']) > 0) or \
		   ('push-temporal-buffer' in self.timing and len(self.timing['push-temporal-buffer']) > 0) or \
		   ('temporal-smoothing' in self.timing and len(self.timing['temporal-smoothing']) > 0) or \
		   ('make-temporally-smooth-decision' in self.timing and len(self.timing['make-temporally-smooth-decision']) > 0):
			fh.write('###  CLASSIFICATION TIMES:\n')

		if 'dtw-classification' in self.timing and len(self.timing['dtw-classification']) > 0:
			classification_time += np.sum(self.timing['dtw-classification'])
			accounted_time += np.sum(self.timing['dtw-classification'])
		if 'sort-confidences' in self.timing and len(self.timing['sort-confidences']) > 0:
			classification_time += np.sum(self.timing['sort-confidences'])
			accounted_time += np.sum(self.timing['sort-confidences'])
		if 'sort-probabilities' in self.timing and len(self.timing['sort-probabilities']) > 0:
			classification_time += np.sum(self.timing['sort-probabilities'])
			accounted_time += np.sum(self.timing['sort-probabilities'])
		if 'push-temporal-buffer' in self.timing and len(self.timing['push-temporal-buffer']) > 0:
			classification_time += np.sum(self.timing['push-temporal-buffer'])
			accounted_time += np.sum(self.timing['push-temporal-buffer'])
		if 'temporal-smoothing' in self.timing and len(self.timing['temporal-smoothing']) > 0:
			classification_time += np.sum(self.timing['temporal-smoothing'])
			accounted_time += np.sum(self.timing['temporal-smoothing'])
		if 'make-temporally-smooth-decision' in self.timing and len(self.timing['make-temporally-smooth-decision']) > 0:
			classification_time += np.sum(self.timing['make-temporally-smooth-decision'])
			accounted_time += np.sum(self.timing['make-temporally-smooth-decision'])
																	#  Report DTW-classification times.
		if 'dtw-classification' in self.timing and len(self.timing['dtw-classification']) > 0:
			fh.write('  DTW-CLASSIFICATION avg. time (per query)\t' + str(np.mean(self.timing['dtw-classification'])) + '\t' + str(np.sum(self.timing['dtw-classification']) / classification_time * 100.0) + '% of classification time\n')
			fh.write('  DTW-CLASSIFICATION std.dev time (per query)\t' + str(np.std(self.timing['dtw-classification'])) + '\n\n')
																	#  Report confidence score-sorting times.
		if 'sort-confidences' in self.timing and len(self.timing['sort-confidences']) > 0:
			fh.write('  SORT-CONFIDENCE avg. time\t' + str(np.mean(self.timing['sort-confidences'])) + '\t' + str(np.sum(self.timing['sort-confidences']) / classification_time * 100.0) + '% of classification time\n')
			fh.write('  SORT-CONFIDENCE std.dev time\t' + str(np.std(self.timing['sort-confidences'])) + '\n\n')
																	#  Report probability-sorting times.
		if 'sort-probabilities' in self.timing and len(self.timing['sort-probabilities']) > 0:
			fh.write('  SORT-PROBABILITIES avg. time\t' + str(np.mean(self.timing['sort-probabilities'])) + '\t' + str(np.sum(self.timing['sort-probabilities']) / classification_time * 100.0) + '% of classification time\n')
			fh.write('  SORT-PROBABILITIES std.dev time\t' + str(np.std(self.timing['sort-probabilities'])) + '\n\n')
																	#  Report temporal-buffer update times.
		if 'push-temporal-buffer' in self.timing and len(self.timing['push-temporal-buffer']) > 0:
			fh.write('  PUSH TEMPORAL-BUFFER avg. time\t' + str(np.mean(self.timing['push-temporal-buffer'])) + '\t' + str(np.sum(self.timing['push-temporal-buffer']) / classification_time * 100.0) + '% of classification time\n')
			fh.write('  PUSH TEMPORAL-BUFFER std.dev time\t' + str(np.std(self.timing['push-temporal-buffer'])) + '\n\n')
																	#  Report temporal-smoothing times.
		if 'temporal-smoothing' in self.timing and len(self.timing['temporal-smoothing']) > 0:
			fh.write('  TEMPORAL SMOOTHING avg. time\t' + str(np.mean(self.timing['temporal-smoothing'])) + '\t' + str(np.sum(self.timing['temporal-smoothing']) / classification_time * 100.0) + '% of classification time\n')
			fh.write('  TEMPORAL SMOOTHING std.dev time\t' + str(np.std(self.timing['temporal-smoothing'])) + '\n\n')
																	#  Report final decision-making times.
		if 'make-temporally-smooth-decision' in self.timing and len(self.timing['make-temporally-smooth-decision']) > 0:
			fh.write('  TEMPORALLY-SMOOTH CLASSIFICATION avg. time\t' + str(np.mean(self.timing['make-temporally-smooth-decision'])) + '\t' + str(np.sum(self.timing['make-temporally-smooth-decision']) / classification_time * 100.0) + '% of classification time\n')
			fh.write('  TEMPORALLY-SMOOTH CLASSIFICATION std.dev time\t' + str(np.std(self.timing['make-temporally-smooth-decision'])) + '\n\n')

		#############################################################
		#                        Rendering                          #
		#############################################################
		if ('render-side-by-side' in self.timing and len(self.timing['render-side-by-side']) > 0) or \
		   ('render-annotated-source' in self.timing and len(self.timing['render-annotated-source']) > 0) or \
		   ('render-rolling-buffer' in self.timing and len(self.timing['render-rolling-buffer']) > 0) or \
		   ('render-confidence' in self.timing and len(self.timing['render-confidence']) > 0) or \
		   ('render-probabilities' in self.timing and len(self.timing['render-probabilities']) > 0) or \
		   ('render-smoothed-probabilities' in self.timing and len(self.timing['render-smoothed-probabilities']) > 0):
			fh.write('###  RENDERING TIMES:\n')

		if 'render-side-by-side' in self.timing and len(self.timing['render-side-by-side']) > 0:
			rendering_time += np.sum(self.timing['render-side-by-side'])
			accounted_time += np.sum(self.timing['render-side-by-side'])
		if 'render-annotated-source' in self.timing and len(self.timing['render-annotated-source']) > 0:
			rendering_time += np.sum(self.timing['render-annotated-source'])
			accounted_time += np.sum(self.timing['render-annotated-source'])
		if 'render-rolling-buffer' in self.timing and len(self.timing['render-rolling-buffer']) > 0:
			rendering_time += np.sum(self.timing['render-rolling-buffer'])
			accounted_time += np.sum(self.timing['render-rolling-buffer'])
		if 'render-confidence' in self.timing and len(self.timing['render-confidence']) > 0:
			rendering_time += np.sum(self.timing['render-confidence'])
			accounted_time += np.sum(self.timing['render-confidence'])
		if 'render-probabilities' in self.timing and len(self.timing['render-probabilities']) > 0:
			rendering_time += np.sum(self.timing['render-probabilities'])
			accounted_time += np.sum(self.timing['render-probabilities'])
		if 'render-smoothed-probabilities' in self.timing and len(self.timing['render-smoothed-probabilities']) > 0:
			rendering_time += np.sum(self.timing['render-smoothed-probabilities'])
			accounted_time += np.sum(self.timing['render-smoothed-probabilities'])

																	#  Report side-by-side rendering times.
		if 'render-side-by-side' in self.timing and len(self.timing['render-side-by-side']) > 0:
			fh.write('  SIDE-BY-SIDE VIDEO RENDER avg. time\t' + str(np.mean(self.timing['render-side-by-side'])) + '\t' + str(np.sum(self.timing['render-side-by-side']) / rendering_time * 100.0) + '% of rendering time\n')
			fh.write('  SIDE-BY-SIDE VIDEO RENDER std.dev time\t' + str(np.std(self.timing['render-side-by-side'])) + '\n\n')
																	#  Report annotation times.
		if 'render-annotated-source' in self.timing and len(self.timing['render-annotated-source']) > 0:
			fh.write('  VIDEO ANNOTATION avg. time\t' + str(np.mean(self.timing['render-annotated-source'])) + '\t' + str(np.sum(self.timing['render-annotated-source']) / rendering_time * 100.0) + '% of rendering time\n')
			fh.write('  VIDEO ANNOTATION std.dev time\t' + str(np.std(self.timing['render-annotated-source'])) + '\n\n')
																	#  Report rolling-buffer seismograph rendering times.
		if 'render-rolling-buffer' in self.timing and len(self.timing['render-rolling-buffer']) > 0:
			fh.write('  RENDER ROLLING-BUFFER avg. time\t' + str(np.mean(self.timing['render-rolling-buffer'])) + '\t' + str(np.sum(self.timing['render-rolling-buffer']) / rendering_time * 100.0) + '% of rendering time\n')
			fh.write('  RENDER ROLLING-BUFFER std.dev time\t' + str(np.std(self.timing['render-rolling-buffer'])) + '\n\n')
																	#  Report confidence seismograph rendering times.
		if 'render-confidence' in self.timing and len(self.timing['render-confidence']) > 0:
			fh.write('  CONFIDENCE RENDER avg. time\t' + str(np.mean(self.timing['render-confidence'])) + '\t' + str(np.sum(self.timing['render-confidence']) / rendering_time * 100.0) + '% of rendering time\n')
			fh.write('  CONFIDENCE RENDER std.dev time\t' + str(np.std(self.timing['render-confidence'])) + '\n\n')
																	#  Report probabilities seismograph rendering times.
		if 'render-probabilities' in self.timing and len(self.timing['render-probabilities']) > 0:
			fh.write('  PROBABILITY RENDER avg. time\t' + str(np.mean(self.timing['render-probabilities'])) + '\t' + str(np.sum(self.timing['render-probabilities']) / rendering_time * 100.0) + '% of rendering time\n')
			fh.write('  PROBABILITY RENDER std.dev time\t' + str(np.std(self.timing['render-probabilities'])) + '\n\n')
																	#  Report smoothed-probabilities seismograph rendering times.
		if 'render-smoothed-probabilities' in self.timing and len(self.timing['render-smoothed-probabilities']) > 0:
			fh.write('  RENDER SMOOTHED PROBABILITIES avg. time\t' + str(np.mean(self.timing['render-smoothed-probabilities'])) + '\t' + str(np.sum(self.timing['render-smoothed-probabilities']) / rendering_time * 100.0) + '% of rendering time\n')
			fh.write('  RENDER SMOOTHED PROBABILITIES std.dev time\t' + str(np.std(self.timing['render-smoothed-probabilities'])) + '\n\n')

		fh.write('Overhead\t' + str(self.timing['total'] - accounted_time) + '\t' + str((self.timing['total'] - accounted_time) / self.timing['total'] * 100.0) + '% of total time\n')

		fh.close()
		return
