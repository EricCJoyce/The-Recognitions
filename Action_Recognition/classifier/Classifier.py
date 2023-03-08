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

sys.path.append('../cnn')											#  Be able to access CNN class.
from cnn import CNN

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
		self.prediction_source = 'nearest-neighbor'					#  By default, system prediction is nearest-neighbor label (subject to threshold.)
																	#  This might also be determined by probability (as determined by MLP, for example.)
		self.knn = 5												#  If 'prediction_source' == 'knn', then this applies.
		self.query_to_probability = {}
		self.query_to_probability['mode'] = None					#  By default, do not expect an MLP to directly compute label probabilities from the query.
		self.query_to_probability['explicit-nothing'] = False
		self.query_to_probability['pipeline'] = None

		self.matchingcost_to_probability = {}
		self.matchingcost_to_probability['mode'] = None				#  By default, do not expect an MLP to directly compute label probabilities from costs.
		self.matchingcost_to_probability['explicit-nothing'] = False
		self.matchingcost_to_probability['pipeline'] = None

		self.confidence_to_probability = {}
		self.confidence_to_probability['mode'] = None				#  By default, do not expect an isotonic lookup, and do not expect an MLP.
		self.confidence_to_probability['explicit-nothing'] = False
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

		if 'dtw_l' in kwargs:										#  Were we given a norm to use?
			assert isinstance(kwargs['dtw_l'], int) and kwargs['dtw_l'] >= 0 and kwargs['dtw_l'] < 256, \
			       'Argument \'dtw_l\' passed to Classifier must be an integer in [0, 255].'
			self.L_type = kwargs['dtw_l']
		else:
			self.L_type = 2											#  Default to L2

		if 'config' in kwargs:										#  Were we given a configuration, as a file path or a dictionary?
			assert isinstance(kwargs['config'], str) or isinstance(kwargs['config'], dict), \
			  'Argument \'config\' passed to Classifier must be either a filepath (string) to a config file or a dictionary with the appropriate keys.'
			if isinstance(kwargs['config'], str):
				self.load_config_file(kwargs['config'])
			else:
				self.matchingcost_to_probability['mode'] = kwargs['config']['matching-cost-to-probability']['mode']
				self.matchingcost_to_probability['explicit-nothing'] = kwargs['config']['matching-cost-to-probability']['explicit-nothing']
				self.matchingcost_to_probability['pipeline'] = kwargs['config']['matching-cost-to-probability']['pipeline']

				self.confidence_to_probability['mode'] = kwargs['config']['confidence-to-probability']['mode']
				self.confidence_to_probability['explicit-nothing'] = kwargs['config']['confidence-to-probability']['explicit-nothing']
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
		self.w_train = []											#  To become a list of sample weights (floats in [0.0, 1.0].
		self.recognizable_objects = []								#  Filled in by the derived classes,
																	#  either from Enactments (atemporal) or a database (temporal).
		self.vector_drop_map = []									#  List of Booleans will have as many entries as self.recognizable_objects.
																	#  If an object's corresponding element in this list is False,
																	#  then omit that column from ALL vectors.
																	#  This parent class only has X_train, so clearing out columns in the
																	#  training set happens here. Child classes must each handle clearing columns
																	#  from their respective X_test lists themselves.
		self.relabelings = {}										#  key: old-label ==> val: new-label.
		self.hidden_labels = {}										#  key: label ==> True.
		self.proxy_labels = {}										#  key: label ==> Another label.
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
		self.w_train = []

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
					else:											#  Sometimes the Classifier is just a Classifier.
						self.snippet_size = db_snippet_size

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
					else:											#  Sometimes the Classifier is just a Classifier.
						self.stride = db_stride
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
					db_sample_weight          = float(action_arr[2])
					db_entry_enactment_source = action_arr[3]
					db_entry_start_time       = float(action_arr[4])
					db_entry_start_frame      = action_arr[5]
					if action_arr[6] == '*':						#  If an action lasts all the way up to the end of an enactment, then there is no
						db_entry_end_time = '*'						#  then there is no exclusive ending timecode: [start, end).
					else:
						db_entry_end_time         = float(action_arr[6])
					db_entry_end_frame        = action_arr[7]

					self.w_train.append( db_sample_weight )			#  Save this sample's weight.

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

				if arr[0] == 'rolling-buffer-to-probability':		#  We will compute probabilities directly from the query buffer.
					self.query_to_probability['mode'] = arr[1]		#  Save mode.
					if self.query_to_probability['mode'] == 'mlp':	#  An MLP will perform query-buffer ---> probabilities.
						if arr[2] == '*':							#  The MLP will explicitly predict the nothing class.
							self.query_to_probability['explicit-nothing'] = True
							if self.verbose:
								print('    MLP(' + arr[3] + ') explicitly predicts a nothing label.')
						else:										#  The MLP will NOT explicitly predict the nothing class.
							self.query_to_probability['explicit-nothing'] = False
							if self.verbose:
								print('    MLP(' + arr[3] + ') does NOT predict a nothing label.')
																	#  Create an MLP object for the pipeline.
						self.query_to_probability['pipeline'] = MLP(arr[3], executable=arr[4])
						if self.verbose:
							print('    MLP(' + arr[3] + ') to compute probabilities from matching costs.')

					elif self.query_to_probability['mode'] == 'cnn':#  A CNN will perform query-buffer ---> probabilities.
						if arr[2] == '*':							#  The CNN will explicitly predict the nothing class.
							self.query_to_probability['explicit-nothing'] = True
							if self.verbose:
								print('    CNN(' + arr[3] + ') explicitly predicts a nothing label.')
						else:										#  The CNN will NOT explicitly predict the nothing class.
							self.query_to_probability['explicit-nothing'] = False
							if self.verbose:
								print('    CNN(' + arr[3] + ') does NOT predict a nothing label.')
																	#  Create a CNN object for the pipeline.
						self.query_to_probability['pipeline'] = CNN(arr[3], executable=arr[4])
						if self.verbose:
							print('    CNN(' + arr[3] + ') to compute probabilities from matching costs.')

				elif arr[0] == 'matching-cost-to-probability':		#  We will compute probabilities from matching costs.
																	#  Save mode.
					self.matchingcost_to_probability['mode'] = arr[1]
																	#  An MLP will perform matching-costs ---> probabilities.
					if self.matchingcost_to_probability['mode'] == 'mlp':

						if arr[2] == '*':							#  The MLP will explicitly predict the nothing class.
							self.matchingcost_to_probability['explicit-nothing'] = True
							if self.verbose:
								print('    MLP(' + arr[3] + ') explicitly predicts a nothing label.')
						else:										#  The MLP will NOT explicitly predict the nothing class.
							self.matchingcost_to_probability['explicit-nothing'] = False
							if self.verbose:
								print('    MLP(' + arr[3] + ') does NOT predict a nothing label.')
																	#  Create an MLP object for the pipeline.
						self.matchingcost_to_probability['pipeline'] = MLP(arr[3], executable=arr[4])
						if self.verbose:
							print('    MLP(' + arr[3] + ') to compute probabilities from matching costs.')

				elif arr[0] == 'confidence-to-probability':			#  We will compute probabilities from confidence scores.
																	#  Save mode.
					self.confidence_to_probability['mode'] = arr[1]
																	#  An isotonic lookup table will convert confidences ---> probabilitues.
					if self.confidence_to_probability['mode'] == 'isotonic':
						self.load_isotonic_map(arr[2])				#  Load the table.
																	#  An MLP will perform confidences ---> probabilities.
					elif self.confidence_to_probability['mode'] == 'mlp':

						if arr[2] == '*':							#  The MLP will explicitly predict the nothing class.
							self.confidence_to_probability['explicit-nothing'] = True
							if self.verbose:
								print('    MLP(' + arr[3] + ') explicitly predicts a nothing label.')
						else:										#  The MLP will NOT explicitly predict the nothing class.
							self.confidence_to_probability['explicit-nothing'] = False
							if self.verbose:
								print('    MLP(' + arr[3] + ') does NOT predict a nothing label.')
																	#  Create an MLP object for the pipeline.
						self.confidence_to_probability['pipeline'] = MLP(arr[3], executable=arr[4])
						if self.verbose:
							print('    MLP(' + arr[3] + ') to compute probabilities from confidence scores.')

				elif arr[0] == 'prediction-source':					#  Set source of system prediction. {'nearest-neighbor', 'knn', 'prob'}
					self.prediction_source = arr[1]
					if self.prediction_source == 'knn':
						self.knn = max(1, int(arr[2]))

		fh.close()

		return

	#################################################################
	#  Classification prep.                                         #
	#################################################################

	#  Return a dictionary of the form: key: action-label ==> val: inf
	def prepare_label_matching_costs_table(self):
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

	#  Return a list of k Nones and a list of k +infs.
	def prepare_knn_lists(self, k):
		return [None for i in range(0, k)], [float('inf') for i in range(0, k)], [None for i in range(0, k)]

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
			metadata[label]['db-index'] = None
			metadata[label]['template-indices'] = None
			metadata[label]['query-indices'] = None

		if include_nothing_label:									#  Allowed here for completeness, but the DB is never expected
			metadata['*'] = {}										#  to include "nothing" snippets.
			metadata['*']['db-index'] = None
			metadata['*']['template-indices'] = None
			metadata['*']['query-indices'] = None

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
		#############################################################
		#  The Classifier can pass the query buffer directly to a   #
		#  Multi-Layer-Perceptron to compute label probabilities.   #
		#############################################################
		if self.query_to_probability['mode'] == 'mlp':				#  Expect self.query_to_probability['pipeline'] to be an MLP object.
																	#  These first three initializations are for completeness: using an MLP to
																	#  compute probabilities from the query by-passes computation of matching costs.
																	#  In this case, confidences will be equal to probabilities.
																	#  This saves some trouble for the derived classes AtemporalClassifier and TemporalClassifier;
																	#  They can both expect the same thing, more or less.
																	#  Initialize all costs for all predictable labels as +inf.
			matching_costs = self.prepare_label_matching_costs_table()
			confidences = self.prepare_probabilities_table( self.query_to_probability['explicit-nothing'] )
			metadata = self.prepare_metadata_table()				#  Initialize all metadata.
																	#  Initialize all to zero. (Explicitly predicting the nothing-label?)
			probabilities = self.prepare_probabilities_table( self.query_to_probability['explicit-nothing'] )

			if type(self).__name__ == 'AtemporalClassifier' or type(self).__name__ == 'TemporalClassifier':
				labels = self.labels('train')
			else:
				labels = sorted(np.unique(self.y_train))

			if self.query_to_probability['explicit-nothing']:
				labels_and_nothing = labels + ['*']					#  Append the nothing-label.

			assert isinstance(self.query_to_probability['pipeline'], MLP), \
			  'In order to estimate label probabilities directly from the query, Classifier.query_to_probability[\'pipeline\'] must be an MLP object.'

			mlp_input = []											#  Concatenated descriptors in...
			for descriptor in query:
				mlp_input += list(descriptor)
			y_hat = self.query_to_probability['pipeline'].run( mlp_input )

			nearest_neighbor_label = None
			greatest_probability = float('-inf')

			if self.query_to_probability['explicit-nothing']:
				for i in range(0, len(labels_and_nothing)):			#  ...{N+1} probabilities out.
					probabilities[ labels_and_nothing[i] ] = y_hat[i]
					confidences[ labels_and_nothing[i] ] = y_hat[i]	#  In this case, let confidences be probabilities.

					if y_hat[i] > greatest_probability:				#  nearest_neighbor_label = argmax(probabilities)
						greatest_probability = y_hat[i]
						nearest_neighbor_label = labels_and_nothing[i]

					if i < len(labels_and_nothing) - 1:				#  Matching costs never expected to contain a value for the nothing-label.
						matching_costs[ labels_and_nothing[i] ] = y_hat[i]
			else:
				for i in range(0, len(labels)):						#  ...{N} probabilities out.
					probabilities[ labels[i] ] = y_hat[i]
					confidences[ labels[i] ] = y_hat[i]				#  In this case, let matching costs and confidences be probabilities.
					matching_costs[ labels[i] ] = y_hat[i]

					if y_hat[i] > greatest_probability:				#  nearest_neighbor_label = argmax(probabilities)
						greatest_probability = y_hat[i]
						nearest_neighbor_label = labels[i]

		#############################################################
		#  The Classifier can pass the query buffer directrly to a  #
		#  Convolutional Neural Network to compute probabilities.   #
		#############################################################
		elif self.query_to_probability['mode'] == 'cnn':			#  Expect self.query_to_probability['pipeline'] to be a CNN object.
																	#  These first three initializations are for completeness: using an MLP to
																	#  compute probabilities from the query by-passes computation of matching costs.
																	#  In this case, confidences will be equal to probabilities.
																	#  This saves some trouble for the derived classes AtemporalClassifier and TemporalClassifier;
																	#  They can both expect the same thing, more or less.
																	#  Initialize all costs for all predictable labels as +inf.
			matching_costs = self.prepare_label_matching_costs_table()
			confidences = self.prepare_probabilities_table( self.query_to_probability['explicit-nothing'] )
			metadata = self.prepare_metadata_table()				#  Initialize all metadata.
																	#  Initialize all to zero. (Explicitly predicting the nothing-label?)
			probabilities = self.prepare_probabilities_table( self.query_to_probability['explicit-nothing'] )

			if type(self).__name__ == 'AtemporalClassifier' or type(self).__name__ == 'TemporalClassifier':
				labels = self.labels('train')
			else:
				labels = sorted(np.unique(self.y_train))

			if self.query_to_probability['explicit-nothing']:
				labels_and_nothing = labels + ['*']					#  Append the nothing-label.

			assert isinstance(self.query_to_probability['pipeline'], CNN), \
			  'In order to estimate label probabilities directly from the query, Classifier.query_to_probability[\'pipeline\'] must be a CNN object.'

			cnn_input = []											#  Concatenated descriptors in...
			for descriptor in query:
				cnn_input += list(descriptor)
			y_hat = self.query_to_probability['pipeline'].run( cnn_input )

			nearest_neighbor_label = None
			greatest_probability = float('-inf')

			if self.query_to_probability['explicit-nothing']:
				for i in range(0, len(labels_and_nothing)):			#  ...{N+1} probabilities out.
					probabilities[ labels_and_nothing[i] ] = y_hat[i]
					confidences[ labels_and_nothing[i] ] = y_hat[i]	#  In this case, let confidences be probabilities.

					if y_hat[i] > greatest_probability:				#  nearest_neighbor_label = argmax(probabilities)
						greatest_probability = y_hat[i]
						nearest_neighbor_label = labels_and_nothing[i]

					if i < len(labels_and_nothing) - 1:				#  Matching costs never expected to contain a value for the nothing-label.
						matching_costs[ labels_and_nothing[i] ] = y_hat[i]
			else:
				for i in range(0, len(labels)):						#  ...{N} probabilities out.
					probabilities[ labels[i] ] = y_hat[i]
					confidences[ labels[i] ] = y_hat[i]				#  In this case, let matching costs and confidences be probabilities.
					matching_costs[ labels[i] ] = y_hat[i]

					if y_hat[i] > greatest_probability:				#  nearest_neighbor_label = argmax(probabilities)
						greatest_probability = y_hat[i]
						nearest_neighbor_label = labels[i]

		#############################################################
		#  The Classifier can skip DTW and simply compute Euclidean #
		#  distances between unwaroed query and unwarped exemplars. #
		#############################################################
		elif self.query_to_probability['mode'] == 'euclidean':		#  No DTW.
			nearest_neighbor_label, matching_costs, metadata = self.Euclidean_match(query)

																	#  Sort both labels and costs ascending by cost.
			sorted_labels_costs = sorted([x for x in matching_costs.items()], key=lambda x: x[1])
			match_labels = [x[0] for x in sorted_labels_costs]
			match_costs = [x[1] for x in sorted_labels_costs]
																	#  Compute confidences according to self.confidence_function.
																	#  Get a dictionary of key:label ==> val:confidence for all recognizable labels.
			confidences = self.match_confidences(match_costs, match_labels)
																	#  Weigh confidence by weight of winning sample.
			if nearest_neighbor_label is not None:
				confidences[ nearest_neighbor_label ] *= self.w_train[ metadata[nearest_neighbor_label]['db-index'] ]

			#########################################################
			#  The Classifier can pass the matching costs to a      #
			#  Multi-Layer Perceptron to estimate probabilities.    #
			#                                                       #
			#  In this case, classify() returns:                    #
			#    nearest_neighbor_label                             #
			#    matching_costs in R^N                              #
			#    confidences    in R^N                              #
			#    probabilities  in R^{N+1}                          #
			#    metadata                                           #
			#########################################################
			if self.matchingcost_to_probability['mode'] == 'mlp':	#  Expect self.matchingcost_to_probability['pipeline'] is an MLP object.
																	#  Initialize all to zero. (Explicitly predicting the nothing-label?)
				probabilities = self.prepare_probabilities_table( self.matchingcost_to_probability['explicit-nothing'] )

				if type(self).__name__ == 'AtemporalClassifier' or type(self).__name__ == 'TemporalClassifier':
					labels = self.labels('train')
				else:
					labels = sorted(np.unique(self.y_train))

				if self.matchingcost_to_probability['explicit-nothing']:
					labels_and_nothing = labels + ['*']				#  Append the nothing-label.

				assert isinstance(self.matchingcost_to_probability['pipeline'], MLP), \
				  'In order to estimate label probabilities from matching costs, Classifier.matchingcost_to_probability[\'pipeline\'] must be an MLP object.'
			  														#  Costs in...
				y_hat = self.matchingcost_to_probability['pipeline'].run( [matching_costs[label] for label in labels] )

				if self.matchingcost_to_probability['explicit-nothing']:
					for i in range(0, len(labels_and_nothing)):		#  ...{N+1} probabilities out.
						probabilities[ labels_and_nothing[i] ] = y_hat[i]
				else:
					for i in range(0, len(labels)):					#  ...{N} probabilities out.
						probabilities[ labels[i] ] = y_hat[i]

			else:													#  Probabilities to be computed some other way.
				#####################################################
				#  The Classifier can use the matching costs to     #
				#  compute confidences according to                 #
				#  self.confidence_function and then convert these  #
				#  to probabilities.                                #
				#                                                   #
				#  In this case, classify() returns:                #
				#    nearest_neighbor_label                         #
				#    matching_costs in R^N                          #
				#    confidences    in R^N                          #
				#    probabilities  in R^N                          #
				#    metadata                                       #
				#####################################################
																	#  Expect self.confidence_to_probability['pipeline'] is an isotonic lookup table.
				if self.confidence_to_probability['mode'] == 'isotonic':
																	#  Initialize all to zero. DO NOT explicitly predict the nothing-label.
					probabilities = self.prepare_probabilities_table(False)

					for label, confidence in confidences.items():
						brackets = sorted(self.confidence_to_probability['pipeline'].keys())
						i = 0
						while i < len(brackets) and not (confidence > brackets[i][0] and confidence <= brackets[i][1]):
							i += 1

						probabilities[label] = self.confidence_to_probability['pipeline'][ brackets[i] ]

					prob_norm = sum( probabilities.values() )		#  Normalize lookup-table probabilities.
					for k in probabilities.keys():
						if prob_norm > 0.0:
							probabilities[k] /= prob_norm
						else:
							probabilities[k] = 0.0

				#####################################################
				#  The Classifier can pass confidence scores to a   #
				#  Multi-Layer Perceptron (MLP) to estimate         #
				#  probabilities.                                   #
				#                                                   #
				#  In this case, classify() returns:                #
				#    nearest_neighbor_label                         #
				#    matching_costs in R^N                          #
				#    confidences    in R^N                          #
				#    probabilities  in R^{N+1}                      #
				#    metadata                                       #
				#####################################################
																	#  Expect self.confidence_to_probability['pipeline'] is an MLP object.
				elif self.confidence_to_probability['mode'] == 'mlp':
																	#  Initialize all to zero. (Explicitly predicting the nothing-label?)
					probabilities = self.prepare_probabilities_table( self.confidence_to_probability['explicit-nothing'] )

					if type(self).__name__ == 'AtemporalClassifier' or type(self).__name__ == 'TemporalClassifier':
						labels = self.labels('train')
					else:
						labels = sorted(np.unique(self.y_train))

					if self.confidence_to_probability['explicit-nothing']:
						labels_and_nothing = labels + ['*']			#  Append the nothing-label.

					assert isinstance(self.confidence_to_probability['pipeline'], MLP), \
					  'In order to estimate label probabilities from confidence scores, Classifier.confidence_to_probability[\'pipeline\'] must be an MLP object.'
				  													#  Confidences in...
					y_hat = self.confidence_to_probability['pipeline'].run( [confidences[label] for label in labels] )

					if self.confidence_to_probability['explicit-nothing']:
						for i in range(0, len(labels_and_nothing)):	#  ...{N+1} probabilities out.
							probabilities[ labels_and_nothing[i] ] = y_hat[i]
					else:
						for i in range(0, len(labels)):				#  ...{N} probabilities out.
							probabilities[ labels[i] ] = y_hat[i]

				#####################################################
				#  The Classifier can simply normalize confidence   #
				#  scores and consider them probabilities, though   #
				#  this is discouraged.                             #
				#                                                   #
				#  In this case, classify() returns:                #
				#    nearest_neighbor_label                         #
				#    matching_costs in R^N                          #
				#    confidences    in R^N                          #
				#    probabilities  in R^N                          #
				#    metadata                                       #
				#####################################################
				else:												#  No isotonic map and no MLP?
																	#  Initialize all to zero. DO NOT explicitly predict the nothing-label.
					probabilities = self.prepare_probabilities_table(False)

					for label, confidence in confidences.items():	#  Then probability = (normalized) confidence, which is sloppy, but... meh.
						probabilities[label] = confidence

					prob_norm = sum( probabilities.values() )		#  Normalize probabilities.
					for k in probabilities.keys():
						if prob_norm > 0.0:
							probabilities[k] /= prob_norm
						else:
							probabilities[k] = 0.0

		#############################################################
		#  Otherwise the basis for the Classifier's decision will   #
		#  partially depend on matching costs computed using DTW.   #
		#  Matching costs can be used themselves or passed to a     #
		#  Multi-Layer-Perceptron for interpretation.               #
		#############################################################
		else:														#  Run DTW matching to find a nearest-neighbor label, costs, and confidences.

			#########################################################
			#  Use k-Nearest Neighbor voting to predict label.      #
			#########################################################
			if self.prediction_source == 'knn':
				nearest_neighbor_label, knn_labels, knn_costs, knn_indices, matching_costs, metadata = self.DTW_kNN(self.knn, query)
																	#  Both 'knn_labels' and 'knn_costs' are already sorted ascending by cost.

																	#  Compute confidences according to self.confidence_function.
																	#  Get a dictionary of key:label ==> val:confidence for all recognizable labels.
				confidences, support_indices = self.knn_confidences(knn_costs, knn_labels, knn_indices)
																	#  Weigh confidence by average weight of vote-winning samples.
				confidences[ nearest_neighbor_label ] *= np.mean( [self.w_train[x] for x in support_indices[nearest_neighbor_label] if x is not None] )

			#########################################################
			#  Use Nearest-Neighbor matching to predict label.      #
			#########################################################
			else:
				nearest_neighbor_label, matching_costs, metadata = self.DTW_match(query)
																	#  Sort both labels and costs ascending by cost.
				sorted_labels_costs = sorted([x for x in matching_costs.items()], key=lambda x: x[1])
				match_labels = [x[0] for x in sorted_labels_costs]
				match_costs = [x[1] for x in sorted_labels_costs]
																	#  Compute confidences according to self.confidence_function.
																	#  Get a dictionary of key:label ==> val:confidence for all recognizable labels.
				confidences = self.match_confidences(match_costs, match_labels)
																	#  Weigh confidence by weight of winning sample.
				if nearest_neighbor_label is not None:
					confidences[ nearest_neighbor_label ] *= self.w_train[ metadata[nearest_neighbor_label]['db-index'] ]

			#########################################################
			#  The Classifier can pass the matching costs to a      #
			#  Multi-Layer Perceptron to estimate probabilities.    #
			#                                                       #
			#  In this case, classify() returns:                    #
			#    nearest_neighbor_label                             #
			#    matching_costs in R^N                              #
			#    confidences    in R^N                              #
			#    probabilities  in R^{N+1}                          #
			#    metadata                                           #
			#########################################################
			if self.matchingcost_to_probability['mode'] == 'mlp':	#  Expect self.matchingcost_to_probability['pipeline'] is an MLP object.
																	#  Initialize all to zero. (Explicitly predicting the nothing-label?)
				probabilities = self.prepare_probabilities_table( self.matchingcost_to_probability['explicit-nothing'] )

				if type(self).__name__ == 'AtemporalClassifier' or type(self).__name__ == 'TemporalClassifier':
					labels = self.labels('train')
				else:
					labels = sorted(np.unique(self.y_train))

				if self.matchingcost_to_probability['explicit-nothing']:
					labels_and_nothing = labels + ['*']				#  Append the nothing-label.

				assert isinstance(self.matchingcost_to_probability['pipeline'], MLP), \
				  'In order to estimate label probabilities from matching costs, Classifier.matchingcost_to_probability[\'pipeline\'] must be an MLP object.'
			  														#  Costs in...
				y_hat = self.matchingcost_to_probability['pipeline'].run( [matching_costs[label] for label in labels] )

				if self.matchingcost_to_probability['explicit-nothing']:
					for i in range(0, len(labels_and_nothing)):		#  ...{N+1} probabilities out.
						probabilities[ labels_and_nothing[i] ] = y_hat[i]
				else:
					for i in range(0, len(labels)):					#  ...{N} probabilities out.
						probabilities[ labels[i] ] = y_hat[i]

			else:													#  Probabilities to be computed some other way.
				#####################################################
				#  The Classifier can use the matching costs to     #
				#  compute confidences according to                 #
				#  self.confidence_function and then convert these  #
				#  to probabilities.                                #
				#                                                   #
				#  In this case, classify() returns:                #
				#    nearest_neighbor_label                         #
				#    matching_costs in R^N                          #
				#    confidences    in R^N                          #
				#    probabilities  in R^N                          #
				#    metadata                                       #
				#####################################################
																	#  Expect self.confidence_to_probability['pipeline'] is an isotonic lookup table.
				if self.confidence_to_probability['mode'] == 'isotonic':
																	#  Initialize all to zero. DO NOT explicitly predict the nothing-label.
					probabilities = self.prepare_probabilities_table(False)

					for label, confidence in confidences.items():
						brackets = sorted(self.confidence_to_probability['pipeline'].keys())
						i = 0
						while i < len(brackets) and not (confidence > brackets[i][0] and confidence <= brackets[i][1]):
							i += 1

						probabilities[label] = self.confidence_to_probability['pipeline'][ brackets[i] ]

					prob_norm = sum( probabilities.values() )		#  Normalize lookup-table probabilities.
					for k in probabilities.keys():
						if prob_norm > 0.0:
							probabilities[k] /= prob_norm
						else:
							probabilities[k] = 0.0

				#####################################################
				#  The Classifier can pass confidence scores to a   #
				#  Multi-Layer Perceptron (MLP) to estimate         #
				#  probabilities.                                   #
				#                                                   #
				#  In this case, classify() returns:                #
				#    nearest_neighbor_label                         #
				#    matching_costs in R^N                          #
				#    confidences    in R^N                          #
				#    probabilities  in R^{N+1}                      #
				#    metadata                                       #
				#####################################################
																	#  Expect self.confidence_to_probability['pipeline'] is an MLP object.
				elif self.confidence_to_probability['mode'] == 'mlp':
																	#  Initialize all to zero. (Explicitly predicting the nothing-label?)
					probabilities = self.prepare_probabilities_table( self.confidence_to_probability['explicit-nothing'] )

					if type(self).__name__ == 'AtemporalClassifier' or type(self).__name__ == 'TemporalClassifier':
						labels = self.labels('train')
					else:
						labels = sorted(np.unique(self.y_train))

					if self.confidence_to_probability['explicit-nothing']:
						labels_and_nothing = labels + ['*']			#  Append the nothing-label.

					assert isinstance(self.confidence_to_probability['pipeline'], MLP), \
					  'In order to estimate label probabilities from confidence scores, Classifier.confidence_to_probability[\'pipeline\'] must be an MLP object.'
				  													#  Confidences in...
					y_hat = self.confidence_to_probability['pipeline'].run( [confidences[label] for label in labels] )

					if self.confidence_to_probability['explicit-nothing']:
						for i in range(0, len(labels_and_nothing)):	#  ...{N+1} probabilities out.
							probabilities[ labels_and_nothing[i] ] = y_hat[i]
					else:
						for i in range(0, len(labels)):				#  ...{N} probabilities out.
							probabilities[ labels[i] ] = y_hat[i]

				#####################################################
				#  The Classifier can simply normalize confidence   #
				#  scores and consider them probabilities, though   #
				#  this is discouraged.                             #
				#                                                   #
				#  In this case, classify() returns:                #
				#    nearest_neighbor_label                         #
				#    matching_costs in R^N                          #
				#    confidences    in R^N                          #
				#    probabilities  in R^N                          #
				#    metadata                                       #
				#####################################################
				else:												#  No isotonic map and no MLP?
																	#  Initialize all to zero. DO NOT explicitly predict the nothing-label.
					probabilities = self.prepare_probabilities_table(False)

					for label, confidence in confidences.items():	#  Then probability = (normalized) confidence, which is sloppy, but... meh.
						probabilities[label] = confidence

					prob_norm = sum( probabilities.values() )		#  Normalize probabilities.
					for k in probabilities.keys():
						if prob_norm > 0.0:
							probabilities[k] /= prob_norm
						else:
							probabilities[k] = 0.0

		return nearest_neighbor_label, matching_costs, confidences, probabilities, metadata

	#  Performs DTW matching on the given query snippet.
	#  If cutoff conditions apply (such as "Only consider Grab(Helmet) if vector[helmet] > 0.0"), they apply here.
	#  Returns
	#    'nearest_neighbor_label', a string
	#    'label_matching_costs',   a dictionary:  key: action-label ==> val: matching-cost
	#    'metadata',               a dictionary:  key: action-label ==> val: {key:'db-index'         ==> val: index of best matching DB snippet
	#                                                                         key:'template-indices' ==> val: frames in T snippet
	#                                                                         key:'query-indices'    ==> val: frames in Q snippet
	#                                                                        }
	def DTW_match(self, query):
																	#  Initialize all costs for all predictable labels as +inf.
		label_matching_costs = self.prepare_label_matching_costs_table()
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
				dist, _, query_indices, template_indices = DTW.DTW(query, template, self.diagonal_cost, self.L_type)

				if least_cost > dist:								#  A preferable match over all!
					least_cost = dist								#  Save the cost.
					nearest_neighbor_label = template_label[:]		#  Save the (tentative) prediction.

				if label_matching_costs[template_label] > dist:		#  A preferable match within class!
					label_matching_costs[template_label] = dist		#  Save the preferable cost.

					metadata[template_label]['db-index'] = db_index	#  Save information about the best match found so far.
					metadata[template_label]['template-indices'] = template_indices
					metadata[template_label]['query-indices'] = query_indices

			db_index += 1

		return nearest_neighbor_label, label_matching_costs, metadata

	#  Performs DTW matching on the given query snippet but makes its prediction according to a vote among k nearest neighbors.
	#  This method does not enforce per-class exclusivity as DTW_match() does: the idea here is that the correct match *should* have
	#  several low-cost exemplars.
	#  If cutoff conditions apply (such as "Only consider Grab(Helmet) if vector[helmet] > 0.0"), they apply here.
	#  Returns
	#    'nearest_neighbor_label', a string
	#    'knn_labels',             a length-k list of strings
	#    'knn_costs',              a length-k list of floats
	#    'knn_indices',            a length-k list of integer indices into the DB
	#    'label_matching_costs',   a dictionary:  key: action-label ==> val: matching-cost
	#    'metadata',               a dictionary:  key: action-label ==> val: {key:'db-index'         ==> val: index of the most recently counted DB snippet
	#                                                                                                         for the winning category
	#                                                                         key:'template-indices' ==> val: frames in T snippet
	#                                                                         key:'query-indices'    ==> val: frames in Q snippet
	#                                                                        }
	def DTW_kNN(self, k, query):
																	#  Initialize: [None, None, ... ], [+inf, +inf, ...], and [None, None, ... ].
		knn_labels, knn_costs, knn_indices = self.prepare_knn_lists(k)
																	#  Initialize all costs for all predictable labels as +inf.
		label_matching_costs = self.prepare_label_matching_costs_table()
		metadata = self.prepare_metadata_table()					#  Initialize all metadata.

		nearest_neighbor_label = None								#  Initialize best label to None.
		max_votes = 0

		db_index = 0												#  Index into self.X_train let us know which sample best matches the query.
		for template in self.X_train:								#  For every training-set sample, 'template'...
			template_label = self.y_train[db_index]					#  Save the true label for this template sequence.

			conditions_passed = False
			if self.conditions is not None:
				conditions_passed = self.test_cutoff_conditions(template_label, query)

			if self.conditions is None or conditions_passed:		#  Either we have no conditions, or our conditions give us reason to run DTW.
																	#  What is the distance between this query and this template?
				dist, _, query_indices, template_indices = DTW.DTW(query, template, self.diagonal_cost, self.L_type)

				i = 0												#  Iterate until we exceed k or can make an improvement.
				while i < k and knn_costs[i] <= dist:
					i += 1
				if i < k:
					knn_labels  = knn_labels[ :i] + [ template_label[:] ] + knn_labels[ i + 1:]
					knn_costs   = knn_costs[  :i] + [ dist ]              + knn_costs[  i + 1:]
					knn_indices = knn_indices[:i] + [ db_index ]          + knn_indices[i + 1:]

				if label_matching_costs[template_label] > dist:		#  A preferable match within class!
					label_matching_costs[template_label] = dist		#  Save the preferable cost.

					metadata[template_label]['db-index'] = db_index	#  Save information about the best match found so far.
					metadata[template_label]['template-indices'] = template_indices
					metadata[template_label]['query-indices'] = query_indices

			db_index += 1

		for label in label_matching_costs.keys():					#  Try every label.
			votes = knn_labels.count(label)
			if votes > max_votes:									#  Prediction is the winner by vote.
				nearest_neighbor_label = label
				max_votes = votes

		return nearest_neighbor_label, knn_labels, knn_costs, knn_indices, label_matching_costs, metadata

	#  This method DOES NOT PERFORM DYNAMIC TIME-WARPING!
	#  Therefore, keep in mind that if template and query do not have the same number of frames, they will be considered infinitely far away.
	#  If cutoff conditions apply (such as "Only consider Grab(Helmet) if vector[helmet] > 0.0"), they apply here.
	#  Returns
	#    'nearest_neighbor_label', a string
	#    'label_matching_costs',   a dictionary:  key: action-label ==> val: matching-cost
	#    'metadata',               a dictionary:  key: action-label ==> val: {key:'db-index'         ==> val: index of best matching DB snippet
	#                                                                         key:'template-indices' ==> val: frames in T snippet
	#                                                                         key:'query-indices'    ==> val: frames in Q snippet
	#                                                                        }
	def Euclidean_match(self, query):
																	#  Initialize all costs for all predictable labels as +inf.
		label_matching_costs = self.prepare_label_matching_costs_table()
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
				if len(query) == len(template):						#  If we're not time-warping, then snippet lengths must match.
					dist = 0.0										#  What is the distance between this query and this template?
					for frame_ctr in range(0, len(query)):
						dist += np.linalg.norm(np.array(query[frame_ctr]) - np.array(template[frame_ctr]))
				else:
					dist = float('inf')

				if least_cost > dist:								#  A preferable match over all!
					least_cost = dist								#  Save the cost.
					nearest_neighbor_label = template_label[:]		#  Save the (tentative) prediction.

				if label_matching_costs[template_label] > dist:		#  A preferable match within class!
					label_matching_costs[template_label] = dist		#  Save the preferable cost.

					metadata[template_label]['db-index'] = db_index	#  Save information about the best match found so far.
					metadata[template_label]['template-indices'] = [x for x in range(0, len(template))]
					metadata[template_label]['query-indices'] = [x for x in range(0, len(query))]

			db_index += 1

		return nearest_neighbor_label, label_matching_costs, metadata

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

	#################################################################
	#  Confidence computation.                                      #
	#################################################################

	#  Given two lists, matching costs sorted ascending and labels for those costs sorted accordingly, compute a confidence score for each label.
	#  Returns:
	#    confidence_table = {key: label ==> val: confidence}
	#      for all labels in training set, regardless of which labels are passed or omitted in the given 'sorted_labels'.
	def match_confidences(self, sorted_costs, sorted_labels):
		confidence_table = {}										#  Initialize all confidences to zero.
		if type(self).__name__ == 'AtemporalClassifier' or type(self).__name__ == 'TemporalClassifier':
			for label in self.labels('train'):
				confidence_table[label] = 0.0
		else:
			for label in np.unique(self.y_train):
				confidence_table[label] = 0.0

		if self.confidence_function.startswith('sum'):
			sorted_confidences = self.conf_sumn(sorted_costs, int(self.confidence_function[3:]))
		elif self.confidence_function == 'n-min-obsv':
			sorted_confidences = self.conf_nminobsrv(sorted_costs)
		elif self.confidence_function == 'min-obsv':				#  DON'T ACTUALLY USE THIS CONFIDENCE FUNCTION!
			sorted_confidences = self.conf_minobsrv(sorted_costs)
		elif self.confidence_function == 'max-marg':
			sorted_confidences = self.conf_maxmarg(sorted_costs)
		elif self.confidence_function == '2over1':
			sorted_confidences = self.conf_2over1(sorted_costs)

		for i in range(0, len(sorted_labels)):
			confidence_table[ sorted_labels[i] ] = sorted_confidences[i]

		return confidence_table

	#  Given two lists, matching costs sorted ascending and labels for those costs sorted accordingly, compute a confidence score for each label.
	#  When using kNN voting, the confidence of a label is the sum of the confidences of its representative samples in the length-k list, minus
	#  the sum of the confidences of all other labels in the length-k list.
	#  This means that it is possible to have negative confidences.
	#  Returns:
	#    confidence_table = {key: label ==> val: confidence}
	#      for all labels in training set, regardless of which labels are passed or omitted in the given 'sorted_labels'.
	#    support_indices = {key: label ==> val: list of indices into DB of samples supporting this label}
	#      for all labels in training set, regardless of which labels are passed or omitted in the given 'sorted_labels'.
	def knn_confidences(self, knn_costs, knn_labels, knn_indices):
		confidence_table = {}
		support_indices = {}

		if self.confidence_function.startswith('sum'):
			sorted_confidences = self.conf_sumn(knn_costs, int(self.confidence_function[3:]))
		elif self.confidence_function == 'n-min-obsv':
			sorted_confidences = self.conf_nminobsrv(knn_costs)
		elif self.confidence_function == 'min-obsv':				#  DON'T ACTUALLY USE THIS CONFIDENCE FUNCTION!
			sorted_confidences = self.conf_minobsrv(knn_costs)
		elif self.confidence_function == 'max-marg':
			sorted_confidences = self.conf_maxmarg(knn_costs)
		elif self.confidence_function == '2over1':
			sorted_confidences = self.conf_2over1(knn_costs)

		if type(self).__name__ == 'AtemporalClassifier' or type(self).__name__ == 'TemporalClassifier':
			for label in self.labels('train'):
				support = sum( [sorted_confidences[i] for i in range(0, len(knn_labels)) if knn_labels[i] == label] )
				opposition = sum( [sorted_confidences[i] for i in range(0, len(knn_labels)) if knn_labels[i] != label] )
				confidence_table[label] = support - opposition
				support_indices[label] = [knn_indices[i] for i in range(0, len(knn_labels)) if knn_labels[i] == label]
		else:
			for label in np.unique(self.y_train):
				support = sum( [sorted_confidences[i] for i in range(0, len(knn_labels)) if knn_labels[i] == label] )
				opposition = sum( [sorted_confidences[i] for i in range(0, len(knn_labels)) if knn_labels[i] != label] )
				confidence_table[label] = support - opposition
				support_indices[label] = [knn_indices[i] for i in range(0, len(knn_labels)) if knn_labels[i] == label]

		return confidence_table, support_indices

	#  Given a list of costs only (sorted ascending), return a list of confidence scores per item in the same order.
	#  (Sum of n minimal distances) / my distance
	def conf_sumn(self, costs, n):
		s = sum(costs[:n])
		scores = []
		for cost in costs:
			score = s / (cost + self.epsilon)
			if np.isnan(score):
				score = 0.0
			scores.append(score)
		return scores

	#  Given a list of costs only (sorted ascending), return a list of confidence scores per item in the same order.
	#  Minimum distance observed / my distance
	#  DO NOT ACTUALLY USE THIS CONFIDENCE FUNCTION! It's an illustrative fail-case ONLY.
	def conf_minobsrv(self, costs):
		scores = []
		for cost in costs:
			scores.append( costs[0] / (cost + self.epsilon) )
			if np.isnan(scores[-1]):
				scores[-1] = 0.0
		return scores

	#  Given a list of costs only (sorted ascending), return a list of confidence scores per item in the same order.
	#  Normalized minimum distance observed / my distance
	def conf_nminobsrv(self, costs):
		scores = []
		for cost in costs:
			scores.append( costs[0] / (cost + self.epsilon) )
		s = sum(scores)
		for i in range(0, len(scores)):
			scores[i] /= s
			if np.isnan(scores[i]):
				scores[i] = 0.0
		return scores

	#  Given a list of costs only (sorted ascending), return a list of confidence scores per item in the same order.
	#  Second-best distance minus best distance. Worst match gets zero.
	def conf_maxmarg(self, costs):
		scores = []
		for i in range(0, len(costs) - 1):
			scores.append(costs[i + 1] - costs[i])
			if np.isnan(scores[-1]):
				scores[-1] = 0.0
		scores.append(0.0)
		return scores

	#  Given a list of costs only (sorted ascending), return a list of confidence scores per item in the same order.
	#  Second-best distance over best distance. Worst match gets zero.
	def conf_2over1(self, costs):
		scores = []
		for i in range(0, len(costs) - 1):
			scores.append(costs[i + 1] / (costs[i] + self.epsilon))
			if np.isnan(scores[-1]):
				scores[-1] = 0.0
		scores.append(0.0)
		return scores

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

	#################################################################
	#  Label manipulation.                                          #
	#################################################################

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

	def load_proxies_from_file(self, filename):
		fh = open(filename, 'r')
		for line in fh.readlines():
			if len(line) > 1 and line[0] != '#':
				arr = line.strip().split('\t')
				if arr[0] in self.y_train and arr[1] in self.y_train:
					self.proxy_label(arr[0], arr[1])
		fh.close()
		return

	#  When a label is a 'proxy', it is present in the DB (y_train), but it endorses another label if the proxy label is picked.
	#  However, when selected as the system's prediction, a hidden label becomes a no-vote.
	def proxy_label(self, proxy_label, target_label):
		self.proxy_labels[proxy_label] = target_label
		return

	#  Remove the given 'label' from the table of proxy labels.
	def unproxy_label(self, label):
		if label in self.proxy_labels:
			del self.proxy_labels[label]
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
																	### Matching Costs
		classification_stats['_costs'] = []							#  key:_costs ==> val:[ (timestamp-start, timestamp-end, enactment-source,
																	#                        cost_0, cost_1, cost_2, ..., cost_N, ground-truth-label),
																	#                       (timestamp-start, timestamp-end, enactment-source,
																	#                        cost_0, cost_1, cost_2, ..., cost_N, ground-truth-label),
																	#                       ... ]
																	### Predictions vs. Ground-Truth
		classification_stats['_tests'] = []							#  key:_tests ==> val:[ ( prediction, ground-truth,
																	#                         confidence-of-prediction, probability-of-pred,
																	#                         enactment-source, timestamp, DB-index, fair ),
																	#                       ( prediction, ground-truth,
																	#                         confidence-of-prediction, probability-of-pred,
																	#                         enactment-source, timestamp, DB-index, fair ),
																	#                       ... ]
		classification_stats['_matches'] = []						### Matching information
																	#  key:_matches ==> val: [ ( prediction, nearest-neighbor-label, ground-truth,
																	#                            DB-index,
																	#                            query-enactment, query-start-time-incl., query-end-time-excl.
																	#                            Q-indices, T-indices, fair ),
																	#                          ( prediction, ground-truth,
																	#                            DB-index,
																	#                            query-enactment, query-start-time-incl., query-end-time-excl.
																	#                            Q-indices, T-indices, fair ),
																	#                          ... ]
																	### Confidence Scores for each label
		classification_stats['_test-conf'] = []						#  key:_test-conf ==> val:[ (c_0, c_1, c_2, ..., c_N) for test 0,
																	#                           (c_0, c_1, c_2, ..., c_N) for test 1,
																	#                           (c_0, c_1, c_2, ..., c_N) for test 2,
																	#                           ... ]
																	### Probabilities for each label
		classification_stats['_test-prob'] = []						#  key:_test-prob ==> val:[ (p_0, p_1, p_2, ..., p_N) for test 0,
																	#                           (p_0, p_1, p_2, ..., p_N) for test 1,
																	#                           (p_0, p_1, p_2, ..., p_N) for test 2,
																	#                           ... ]
																	### Smoothed Probabilities
		classification_stats['_test-smooth-prob'] = []				#  key:_test-smooth-prob ==> val:[ (sp_0, sp_1, sp_2, ..., sp_N) for test 0,
																	#                                  (sp_0, sp_1, sp_2, ..., sp_N) for test 1,
																	#                                  (sp_0, sp_1, sp_2, ..., sp_N) for test 2,
																	#                                  ... ]
																	### Confidence Scores for winning labels
		classification_stats['_conf'] = []							#  key:_conf  ==> val:[ (confidence-for-label, label, ground-truth,
																	#                        source-enactment, first-snippet-timestamp, final-snippet-timestamp),
																	#                       (confidence-for-label, label, ground-truth,
																	#                        source-enactment, first-snippet-timestamp, final-snippet-timestamp),
																	#                       ... ]
																	### Probabilities for winning labels
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
	def confusion_matrix(self, predictions_truths, sets='both', drop_hidden=True, drop_proxies=True):
		labels = self.labels(sets)

		if drop_proxies:
			for proxy_label in self.proxy_labels.keys():
				labels.remove(proxy_label)
		if drop_hidden:
			for hidden_label in self.hidden_labels.keys():
				labels.remove(hidden_label)

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
	def write_confusion_matrix(self, predictions_truths, file_timestamp=None, sets='both', drop_hidden=True, drop_proxies=True):
		if file_timestamp is None:
			file_timestamp = self.time_stamp()						#  Build a distinct substring so I don't accidentally overwrite results.

		labels = self.labels(sets)

		if drop_proxies:
			for proxy_label in self.proxy_labels.keys():
				labels.remove(proxy_label)
		if drop_hidden:
			for hidden_label in self.hidden_labels.keys():
				labels.remove(hidden_label)

		labels.append('*')											#  Add the nothing-label, a posteriori.
		num_classes = len(labels)

		M = self.confusion_matrix(predictions_truths, sets)

		fh = open('confusion-matrix_' + file_timestamp + '.txt', 'w')
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

		'''
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
		for k, v in sorted( [x for x in stats.items() if x[0] not in ['_tests', '_costs', '_test-conf', '_test-prob', '_test-smooth-prob', '_conf', '_prob', '*']] ):
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
		for k, v in sorted( [x for x in stats.items() if x[0] not in ['_tests', '_costs', '_test-conf', '_test-prob', '_test-smooth-prob', '_conf', '_prob', '*']] ):
			fh.write(k + '\t' + str(v['tp']) + '\t' + str(v['fp']) + '\t' + str(v['fn']) + '\t' + str(v['support']) + '\t')
			fh.write(str(len( [x for x in stats['_tests'] if x[1] == k and x[-1]] )) + '\n')
		fh.write('\n')

		fh.write('Test Set Survey:\n')
		fh.write('================\n')
		for label in train_labels:
			fh.write(label + '\t' + str(len( [x for x in stats['_tests'] if x[1] == label and x[-1]] )) + '\n')
		fh.write('\n')
		fh.close()
		'''

		self.write_test_itemization(stats, file_timestamp)			#  Log test itemization.
		self.write_per_test_confidences(stats, file_timestamp)		#  Log per-test confidences.
		self.write_per_test_probabilities(stats, file_timestamp)	#  Log per-test probabilities.
		self.write_evaluation_scores(stats, file_timestamp)			#  Log scores.
		self.write_detailed_matches(stats, file_timestamp)			#  Log match details.
		self.write_timing(file_timestamp)							#  Log timing.

		return

	#  Write the details of each match in the order in which they were performed.
	#  'stats' is dictionary with key:_matches ==> val: [ ( prediction, ground-truth,
	#                                                       DB-index,
	#                                                       query-enactment, query-start-time-incl., query-end-time-excl.
	#                                                       Q-indices, T-indices, fair ),
	#                                                     ( prediction, ground-truth,
	#                                                       DB-index,
	#                                                       query-enactment, query-start-time-incl., query-end-time-excl.
	#                                                       Q-indices, T-indices, fair ),
	#                                                     ... ]
	#  Each line of this file reads: Pred. <tab>  G.T. <tab> DB-Index <tab> Query-Enactment <tab> Query-Start-Timecode(incl.) <tab> Query-End-Timecode(excl.) <tab> Query-Indices <tab> Template-Indices <tab> fair/unfair
	def write_detailed_matches(self, stats, file_timestamp=None):
		if file_timestamp is None:
			file_timestamp = self.time_stamp()						#  Build a distinct substring so I don't accidentally overwrite results.

		train_labels = self.labels('train')							#  List of strings.

		fh = open('match-details_' + file_timestamp + '.txt', 'w')
		fh.write('#  Classifier match details completed at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		if len(sys.argv) > 1:										#  Was this called from a script? Save the command-line call.
			fh.write('#  COMMAND:\n')
			fh.write('#    ' + ' '.join(sys.argv) + '\n')
		fh.write('#  RECOGNIZABLE ACTIONS:\n')
		fh.write('#    ' + '\t'.join(train_labels) + '\n')
		fh.write('#  Note that the prediction does not necessarily equal the nearest database neighbor!\n')
		fh.write('#  FORMAT per LINE:\n')
		fh.write('#    Pred.    Nearest-Neighbor-Label    G.T.    DB-Index    Query-Enactment    Query-Start-Timecode(incl.)    Query-End-Timecode(excl.)    Query-Indices    Template-Indices    fair/unfair\n')
		for match in stats['_matches']:
			if match[0] is None:									#  Write Prediction.
				fh.write('*\t')
			else:
				fh.write(match[0] + '\t')

			if match[1] is None:									#  Write Nearest-Neighbor-Label.
				fh.write('*\t')
			else:
				fh.write(match[1] + '\t')

			if match[2] is None:									#  Write Ground-Truth.
				fh.write('*\t')
			else:
				fh.write(match[2] + '\t')
			fh.write(str(match[3]) + '\t')							#  Write Database Index.
																	#  Write Query Enactment and start (inclusive) and end (exclusive) time stamps.
			fh.write(match[4] + '\t' + str(match[5]) + '\t' + str(match[6]) + '\t')
			fh.write(' '.join([str(x) for x in match[7]]) + '\t')	#  Write space-separated string of Query Frame Indices
			fh.write(' '.join([str(x) for x in match[8]]) + '\t')	#  Write space-separated string of Template Frame Indices
			if match[9]:
				fh.write('fair\n')
			else:
				fh.write('unfair\n')
		fh.close()

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

	#  Compute and record scores used to evaluate classifier performance.
	def write_evaluation_scores(self, stats, file_timestamp=None):
		if file_timestamp is None:
			file_timestamp = self.time_stamp()						#  Build a distinct substring so I don't accidentally overwrite results.

		prediction_full_seq = []
		ground_truth_full_seq = []

		snippet_ctr = 0
		fair_ctr = 0

		for i in range(0, len(stats['_tests'])):
			if stats['_tests'][i][-1]:								#  Only include fair tests!
				if stats['_tests'][i][0] is not None:
					prediction_full_seq.append( stats['_tests'][i][0] )
				else:
					prediction_full_seq.append( '*' )

				if stats['_tests'][i][1] is not None:
					ground_truth_full_seq.append( stats['_tests'][i][1] )
				else:
					ground_truth_full_seq.append( '*' )

				fair_ctr += 1
			snippet_ctr += 1

		overlap_score = self.modified_overlap_score(prediction_full_seq, ground_truth_full_seq)
		seg_edit_score = self.segmental_edit_score(prediction_full_seq, ground_truth_full_seq)

		train_labels = self.labels('train')							#  List of strings.

		fh = open('evaluation_' + file_timestamp + '.txt', 'w')
		fh.write('#  Classifier evaluation scores, completed at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		if len(sys.argv) > 1:										#  Was this called from a script? Save the command-line call.
			fh.write('#  COMMAND:\n')
			fh.write('#    ' + ' '.join(sys.argv) + '\n')
		fh.write('#  RECOGNIZABLE ACTIONS:\n')
		fh.write('#    ' + '\t'.join(train_labels) + '\n')
		fh.write('Total test snippets, including nothings and unfairs: ' + str(snippet_ctr) + '\n')
		fh.write('Total fair test snippets, including nothings:        ' + str(fair_ctr) + '\n')
		fh.write('\n')
		fh.write('Modified Overlap Score:\t' + str(overlap_score) + '\n')
		fh.write('Segmental Edit Score:  \t' + str(seg_edit_score) + '\n')
		fh.write('\n')
		fh.write('pred. full:\t' + '\t'.join(prediction_full_seq) + '\n')
		fh.write('G.T. full: \t' + '\t'.join(ground_truth_full_seq) + '\n')
		fh.write('\n')
		fh.write('pred. seg:\t' + '\t'.join(self.prediction_segment(prediction_full_seq)) + '\n')
		fh.write('G.T. seg: \t' + '\t'.join(self.ground_truth_segment(ground_truth_full_seq)) + '\n')
		fh.close()

		return

	#  Compute the modified overlap score.
	#  Notice that this metric does not at all consider whether the maximal overlap is correct!
	def modified_overlap_score(self, pred_full_seq, gt_full_seq):
																	#  [ [start, end), [start, end), ..., [start, end)], [label, label, ..., label]
		prediction_segment_borders, prediction_segment_labels = self.segment_borders(pred_full_seq)
																	#  [ [start, end), [start, end), ..., [start, end)], [label, label, ..., label]
		ground_truth_segment_borders, ground_truth_segment_labels = self.segment_borders(gt_full_seq)

		score = 0.0

		for i in range(0, len(ground_truth_segment_labels)):		#  For every ground-truth segment...
			start_index = ground_truth_segment_borders[i][0]		#  (Inclusive)
			end_index   = ground_truth_segment_borders[i][1]		#  (Exclusive)

			max_intersection_over_union = 0.0
			for prediction in prediction_segment_borders:
				prediction_start_index = prediction[0]				#  (Inclusive)
				prediction_end_index   = prediction[1]				#  (Exclusive)
																	#  Check all prediction segments that overlap the ground-truth segment at all.
				if prediction_start_index < end_index and prediction_end_index > start_index:
					intersection = min(end_index, prediction_end_index) - max(start_index, prediction_start_index)
					union        = max(end_index, prediction_end_index) - min(start_index, prediction_start_index)
					max_intersection_over_union = max(intersection / union, max_intersection_over_union)
			score += max_intersection_over_union

		return score / float(len(ground_truth_segment_labels)) * 100.0

	#  Compute the segmental edit score.
	def segmental_edit_score(self, pred_full_seq, gt_full_seq):
		gt_seq = self.ground_truth_segment(gt_full_seq)				#  Compute the collapsed sequences.
		pred_seq = self.prediction_segment(pred_full_seq)
																	#  Compute edit distance.
		edit_distance, del_ctr, ins_ctr, sub_ctr = self.levenshtein_distance(gt_seq, pred_seq)
		if self.verbose:
			print('Levenshtein deletions:     ' + str(del_ctr))
			print('Levenshtein insertions:    ' + str(ins_ctr))
			print('Levenshtein substitutions: ' + str(sub_ctr))
																	#  Return a score that has first been normalized
																	#  by the longer of the two collapsed sequences.
		return (1.0 - float(edit_distance) / float(max(len(pred_seq), len(gt_seq)))) * 100.0

	#  Return a list of "collapsed" prediction labels from the given list of all predictions.
	#  This creates a sort of map of label regions.
	#  If, frame-by-frame, the predicted labels are {[AA], [BBBBB], [CCC]},
	#  then the returned list should be ['A', 'B', 'C'].
	def prediction_segment(self, predictions):
		segment_map = []
		current_label = None
		for prediction in predictions:
			if prediction != current_label:
				segment_map.append(prediction)
				current_label = prediction
		return segment_map

	#  Return a list of "collapsed" ground-truth labels, a sort of map of label regions.
	#  If, frame-by-frame, the ground-truth labels are {[AA], [BBBBB], [CCC]},
	#  then the returned list should be ['A', 'B', 'C'].
	#  This method relabels according to the relabeling lookup-table.
	def ground_truth_segment(self, gt_seq=None):
		segment_map = []

		if gt_seq is None:											#  Not given a buffer/window sequence? No problem: read it from file(s) or y_test.
			if type(self).__name__ == 'AtemporalClassifier':		#  The Atemporal class has a y_test attribute.
				current_label = None

				for label in self.y_test:
					if label in self.relabelings:
						label = self.relabelings[label]

					if label != current_label:
						segment_map.append(label)
						current_label = label

			elif type(self).__name__ == 'TemporalClassifier':		#  The Temporal class must proceed through time.
				current_label = None

				for enactment_input in self.enactment_inputs:
					fh = open(enactment_input + '.enactment', 'r')	#  Read in the input-enactment.
					lines = fh.readlines()
					fh.close()
					for line in lines:
						if line[0] != '#':
							arr = line.strip().split('\t')
							ground_truth_label = arr[2]				#  Save the true label (these include the nothing-labels.)

							if ground_truth_label in self.relabelings:
								ground_truth_label = self.relabelings[ground_truth_label]

							if ground_truth_label != current_label:
								segment_map.append(ground_truth_label)
								current_label = ground_truth_label
		else:														#  Use the given sequence of buffer/window labels.
			current_label = None

			for label in gt_seq:
				if label != current_label:
					segment_map.append(label)
					current_label = label

		return segment_map

	#  Return two lists:
	#    indices = [ (incl-start-index, excl-end-index), (incl-start-index, excl-end-index), ..., (incl-start-index, excl-end-index) ]
	#              for each continuous run under the same label.
	#    labels =  [ label,                              label,                            , ..., label                              ]
	#              for each continuous run of that label.
	def segment_borders(self, seq):
		indices = []
		labels = []

		current_label = None
		current_start = 0
		index = 0
		for label in seq:
			if label != current_label:
				if current_label is not None:
					indices.append( (current_start, index) )
					labels.append( current_label )

				current_label = label
				current_start = index
			index += 1

		indices.append( (current_start, index) )					#  Close the final segment.
		labels.append( current_label )

		return indices, labels

	#  Derive from the given uncollapsed sequence and return a dictionary of the form:
	#    key:label ==> val: length of longest continuous run for that label.
	def segment_max_lengths(self, seq):
		max_lengths = {}
		for label in np.unique(seq):								#  Initialize all labels' max lengths to zero.
			max_lengths[label] = 0

		current_label = None
		current_ctr = 0
		for label in seq:											#  Find the longest run of the target label.
			if label != current_label:
				if current_label is not None:						#  Is this a superlative count for the label?
					max_lengths[current_label] = max(max_lengths[current_label], current_ctr)

				current_label = label								#  Change current label.
				current_ctr = 0										#  Reset counter.
			else:
				current_ctr += 1
																	#  Close the final segment.
		max_lengths[current_label] = max(max_lengths[current_label], current_ctr)

		return max_lengths

	#  Compute the Levenshtein distance between the two given sequences.
	#  Let one be the sequence of (collapsed) ground-truth labels, and the other be the (collapsed) sequence of predictions.
	def levenshtein_distance(self, seq1, seq2):
		D = np.zeros((len(seq1) + 1, len(seq2) + 1))				#  Build zero-matrix.
		for i in range(0, len(seq1) + 1):							#  Initialize leftmost column.
			D[i][0] = i
		for i in range(0, len(seq2) + 1):							#  Initialize top row.
			D[0][i] = i

		for s1 in range(1, len(seq1) + 1):
			for s2 in range(1, len(seq2) + 1):
				if seq1[s1 - 1] == seq2[s2 - 1]:					#  Matched predictions.
					D[s1][s2] = D[s1 - 1][s2 - 1]
				else:
					a = D[s1][s2 - 1]								#  Cumulative cost left.
					b = D[s1 - 1][s2]								#  Cumulative cost above.
					c = D[s1 - 1][s2 - 1]							#  Cumulative cost above-left.
					if a <= b and a <= c:
						D[s1][s2] = a + 1
					elif b <= a and b <= c:
						D[s1][s2] = b + 1
					else:
						D[s1][s2] = c + 1

		del_ctr = 0													#  Now trace the cheapest past and count the types of edits.
		ins_ctr = 0
		sub_ctr = 0
		s1 = len(seq1)
		s2 = len(seq2)
		while s1 > 1 or s2 > 1:
			if seq1[s1 - 1] != seq2[s2 - 1]:						#  Tokens do not match.
				l  = D[s1][s2 - 1]
				u  = D[s1 - 1][s2]
				ul = D[s1 - 1][s2 - 1]
				least = min(l, u, ul)

				if least == l:										#  Insertion.
					ins_ctr += 1
					s2 -= 1
				elif least == u:									#  Deletion.
					del_ctr += 1
					s1 -= 1
				else:												#  Substitution.
					sub_ctr += 1
					s1 -= 1
					s2 -= 1
			else:													#  Tokens match, no edit necessary. Proceed diagonally.
				s1 -= 1
				s2 -= 1

		return D[len(seq1)][len(seq2)], del_ctr, ins_ctr, sub_ctr

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
