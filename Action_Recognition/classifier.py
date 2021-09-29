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

'''

'''
class MetricLearner():
	def __init__(self, **kwargs):
		return

	#
	def func(self):
		return

'''
Similar to the Enactment class in enactment.py but completely separated from the raw materials and file structure.
This class handles and manipulates the *.enactment files. This is a strictly post-processing class.
'''
class ProcessedEnactment():
	def __init__(self, name, **kwargs):
		assert isinstance(name, str), 'Argument \'name\' in ProcessedEnactment must be a string.'
		self.enactment_name = name

		self.width = None											#  Read from file.
		self.height = None
		self.fps = None
		self.gaussian_parameters = None
		self.recognizable_objects = None
		self.encoding_structure = None
		self.vector_length = None
		self.object_detection_source = None

		if 'verbose' in kwargs:
			assert isinstance(kwargs['verbose'], bool), \
			       'Argument \'verbose\' passed to Classifier must be a Boolean.'
			self.verbose = kwargs['verbose']
		else:
			self.verbose = False									#  Default to False.

		#############################################################
		#  The main attributes of this class.                       #
		#############################################################
		self.frames = {}											#  key:time stamp ==> val:{key:file               ==> val:file path;
																	#                          key:ground-truth-label ==> val:label (incl. "*");
																	#                          key:vector             ==> val:vector}
		self.actions = []											#  List of tuples: (label, start time, start frame, end time, end frame).
																	#  Note that we avoid using the Action class defined in "enactment.py"
																	#  because these are not supposed to be mutable after processing.
		self.load_from_file()

	def load_from_file(self):
		if self.verbose:
			print('>>> Loading "' + self.enactment_name + '" from file.')

		reading_dimensions = False
		reading_fps = False
		reading_gaussian = False
		reading_object_detection_source = False
		reading_recognizable_objects = False
		reading_encoding_structure = False

		self.frames = {}											#  (Re)set.
		self.actions = []

		fh = open(self.enactment_name + '.enactment', 'r')
		lines = fh.readlines()
		fh.close()

		num_lines = len(lines)
		prev_ctr = 0
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.

		ctr = 0
		prev_action = None
		current_action = None
		for line in lines:
			if line[0] == '#':										#  Header line
				if 'WIDTH & HEIGHT' in line:						#  Read this enactment's dimensions.
					reading_dimensions = True
				elif reading_dimensions:
					arr = line[1:].strip().split('\t')
					self.width = int(arr[0])
					self.height = int(arr[1])
					reading_dimensions = False

				if 'FPS' in line:									#  Read this enactment's frames-per-second.
					reading_fps = True
				elif reading_fps:
					arr = line[1:].strip().split('\t')
					self.fps = int(arr[0])
					reading_fps = False

				if 'GAUSSIAN' in line:								#  Read this enactment's Gaussian parameters.
					reading_gaussian = True
				elif reading_gaussian:
					arr = line[1:].strip().split('\t')
					self.gaussian_parameters = tuple([float(x) for x in arr])
					reading_gaussian = False

				if 'OBJECT DETECTION SOURCE' in line:				#  Read this enactment's object-detection source.
					reading_object_detection_source = True
				elif reading_object_detection_source:
					arr = line[1:].strip().split('\t')
					self.object_detection_source = arr[0]
					reading_object_detection_source = False

				if 'RECOGNIZABLE OBJECTS' in line:					#  Read this enactment's recognizable objects.
					reading_recognizable_objects = True
				elif reading_recognizable_objects:
					arr = line[1:].strip().split('\t')
					self.recognizable_objects = tuple(arr)
					self.vector_length = 12 + len(self.recognizable_objects)
					reading_recognizable_objects = False

				if 'ENCODING STRUCTURE' in line:					#  Read this enactment's encoding structure (contents of each non-comment line).
					reading_encoding_structure = True
				elif reading_encoding_structure:
					arr = line[1:].strip().split('\t')
					self.encoding_structure = tuple(arr)
					reading_encoding_structure = False
			else:													#  Vector line
				arr = line.strip().split('\t')
				time_stamp = float(arr[0])

				self.frames[time_stamp] = {}
				self.frames[time_stamp]['file']               = arr[1]
				self.frames[time_stamp]['ground-truth-label'] = arr[2]
				self.frames[time_stamp]['vector']             = tuple([float(x) for x in arr[3:]])

				if self.frames[time_stamp]['ground-truth-label'] != current_action:
					prev_action = current_action
					current_action = self.frames[time_stamp]['ground-truth-label']
																	#  Cap the outgoing action.
					if prev_action is not None and prev_action != '*':
						self.actions[-1] = (self.actions[-1][0], self.actions[-1][1], self.actions[-1][2], time_stamp, self.frames[time_stamp]['file'])
																	#  Create the incoming action.
					if current_action is not None and current_action != '*':
						self.actions.append( (current_action, time_stamp, self.frames[time_stamp]['file'], None, None) )

			if self.verbose:
				if int(round(float(ctr) / float(num_lines) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(ctr) / float(num_lines) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(ctr) / float(num_lines) * 100.0))) + '%]')
					sys.stdout.flush()
			ctr += 1

		if self.verbose:
			print('')

		return

	#################################################################
	#  Retrieval: load data from files and the file system          #
	#################################################################

	def get_frames(self):
		return sorted([x for x in self.frames.items()], key=lambda x: x[0])

	#################################################################
	#  Recap: list the current state of the enactment               #
	#################################################################

	#  Number of action instances--NOT the number of unique action labels.
	#  Note that "actions" are not the same as snippets.
	#  Snippets commit to a fixed length, say 10 frames. An action lasts as long as the label it was given.
	#  From a single action, if it is long enough, you can derive several snippets.
	def num_actions(self):
		return len(self.actions)

	#  Return a list of unique action labels.
	#  This list is derived from the current state of the self.actions attribute--NOT from the JSON source.
	def labels(self):
		return list(np.unique([x[0] for x in self.actions]))

	def num_frames(self):
		return len(self.frames)

	#  Print out a formatted list of this enactment's actions.
	#  Actions are label intervals: from whenever one begins to whenever it ends.
	def itemize_actions(self):
		maxlen_label = 0
		maxlen_timestamp = 0
		maxlen_filepath = 0
																	#  Action = (label, start time, start frame, end time, end frame).
		maxlen_index = len(str(len(self.actions) - 1))
		for action in self.actions:									#  Discover extrema
			maxlen_label = max(maxlen_label, len(action[0]))
			maxlen_timestamp = max(maxlen_timestamp, len(str(action[1])), len(str(action[3])))
			maxlen_filepath = max(maxlen_filepath, len(action[2].split('/')[-1]), len(action[4].split('/')[-1]))
		i = 0
		for action in self.actions:									#  Print all nice and tidy like.
			print('[' + str(i) + ' '*(maxlen_index - len(str(i))) + ']: ' + \
			      action[0] + ' '*(maxlen_label - len(action[0])) + ': incl. ' + \
			      str(action[1]) + ' '*(maxlen_timestamp - len(str(action[1]))) + ' ' + \
			      action[2].split('/')[-1] + ' '*(maxlen_filepath - len(action[2].split('/')[-1])) + ' --> excl. ' + \
			      str(action[3]) + ' '*(maxlen_timestamp - len(str(action[3]))) + ' ' + \
			      action[4].split('/')[-1] + ' '*(maxlen_filepath - len(action[4].split('/')[-1])) )
			i += 1
		return

	#  Print out a formatted list of this enactment's frames.
	#  This print out respects no borders: just start from the beginning and print out a summary of every frame until the end.
	def itemize_frames(self, precision=3):
		maxlen_timestamp = 0
		maxlen_filename = 0
		maxlen_label = 0

		maxlen_index = len(str(len(self.frames) - 1))
		for time_stamp, frame in sorted(self.frames.items()):		#  Discover extrema
			maxlen_timestamp = max(maxlen_timestamp, len(str(time_stamp)))
			maxlen_filename = max(maxlen_filename, len(frame['file'].split('/')[-1]))
			maxlen_label = max(maxlen_label, len(frame['ground-truth-label']))

		formatstr = '{:.' + str(precision) + 'f}'

		i = 0
		for time_stamp, frame in sorted(self.frames.items()):
			print('[' + str(i) + ' '*(maxlen_index - len(str(i))) + ']: ' + \
			      frame['ground-truth-label'] + ' '*(maxlen_label - len(frame['ground-truth-label'])) + ': ' + \
			      str(time_stamp) + ' '*(maxlen_timestamp - len(str(time_stamp))) + ' ' + \
			      frame['file'].split('/')[-1] + ' '*(maxlen_filename - len(frame['file'].split('/')[-1])) + ' = ' + \
			      '[' + formatstr.format(frame['vector'][0]) + ', ' + \
			            formatstr.format(frame['vector'][1]) + ', ' + \
			            formatstr.format(frame['vector'][2]) + ', ' + \
			            formatstr.format(frame['vector'][3]) + ', ' + \
			            '... ' + \
			            formatstr.format(frame['vector'][-4]) + ', ' + \
			            formatstr.format(frame['vector'][-3]) + ', ' + \
			            formatstr.format(frame['vector'][-2]) + ', ' + \
			            formatstr.format(frame['vector'][-1]) + ']')
			i += 1

		return

	#  Return a list of action tuples derived from an existing ProcessedEnactment action tuple (or from all existing action tuples.)
	#  The action tuples in this list reflect an ATEMPORAL examination of the Enactment because we already know where true boundaries are.
	def snippets_from_action(self, window_length, stride, index=None):
		if index is None:
			indices = [i for i in range(0, len(self.actions))]
		else:
			indices = [index]

		video_frames = [y[1]['file'] for y in sorted([x for x in self.frames.items()], key=lambda x: x[0])]
		time_stamps = sorted([x[0] for x in self.frames.items()])

		snippet_actions = []										#  To be returned: a list of action tuples.
																	#  Action = (label, start time, start frame, end time, end frame).
		for index in indices:
																	#  Get a list of all frame indices for this action.
																	#  (The +1 at the end ensures that we take the last snippet.)
			frame_indices = range(video_frames.index(self.actions[index][2]), video_frames.index(self.actions[index][4]) + 1)
			for i in range(0, len(frame_indices) - window_length, stride):
				snippet_actions.append( (self.actions[index][0],                           \
				                         time_stamps[ frame_indices[i] ],                  \
				                         video_frames[ frame_indices[i] ],                 \
				                         time_stamps[ frame_indices[i + window_length] ],  \
				                         video_frames[ frame_indices[i + window_length] ]) )

		return snippet_actions

	#  March through time by 'stride', and when the ground-truth label of every frame within 'window_length' is the same, add it to a list and return that list.
	#  The action tuples in this list reflect a temporal examination of the Enactment because we do NOT know where true boundaries are.
	def snippets_from_frames(self, window_length, stride):
		video_frames = [y[1]['file'] for y in sorted([x for x in self.frames.items()], key=lambda x: x[0])]
		time_stamps = sorted([x[0] for x in self.frames.items()])
		num_frames = len(time_stamps)

		snippet_actions = []										#  To be returned: a list of Action objects.

		for i in range(0, num_frames - window_length, stride):		#  March through time by 'stride'. Halt 'window_length' short of the end of time.
			buffer_labels = [self.frames[time_stamps[i + x]]['ground-truth-label'] for x in range(0, window_length)]
																	#  Labels in the buffer are uniform and not nothing.
			if buffer_labels[0] != '*' and buffer_labels.count(buffer_labels[0]) == window_length:
				snippet_actions.append( (buffer_labels[0],                \
				                         time_stamps[i],                  \
				                         video_frames[i],                 \
				                         time_stamps[i + window_length],  \
				                         video_frames[i + window_length]) )

		return snippet_actions

'''
The Classifier object really serves as a repository for attributes and functions used by both its derived classes:
  TemporalClassifier (simulating a real-time system)
  AtemporalClassifier (which has prior knowledge about where sequences begin and end.)
'''
class Classifier():
	def __init__(self, **kwargs):
		self.confidence_function_names = ['sum2', 'sum3', 'sum4', 'sum5', 'n-min-obsv', 'min-obsv', 'max-marg', '2over1']
		self.hand_schema_names = ['left-right', 'strong-hand']

		if 'conf_func' in kwargs:									#  Were we given a confidence function?
			assert isinstance(kwargs['conf_func'], str) and kwargs['conf_func'] in self.confidence_function_names, \
			       'Argument \'conf_func\' passed to Classifier must be a string in {' + ', '.join(self.confidence_function_names) + '}.'
			self.confidence_function = kwargs['conf_func']
		else:
			self.confidence_function = 'sum2'						#  Default to 'sum2'

		if 'verbose' in kwargs:
			assert isinstance(kwargs['verbose'], bool), \
			       'Argument \'verbose\' passed to Classifier must be a boolean.'
			self.verbose = kwargs['verbose']
		else:
			self.verbose = False									#  Default to False.

		if 'threshold' in kwargs:									#  Were we given a threshold?
			assert isinstance(kwargs['threshold'], float), 'Argument \'threshold\' passed to Classifier must be a float.'
			self.threshold = kwargs['threshold']
		else:
			self.threshold = 0.0
																	#  Were we given an isotonic mapping file?
		if 'isotonic_file' in kwargs and kwargs['isotonic_file'] is not None:
			assert isinstance(kwargs['isotonic_file'], str), 'Argument \'isotonic_file\' passed to Classifier must be a string.'
			self.load_isotonic_map(kwargs['isotonic_file'])
		else:
			self.isotonic_map = None
																	#  Were we given a cut-off conditions file?
		if 'conditions_file' in kwargs and kwargs['conditions_file'] is not None:
			assert isinstance(kwargs['conditions_file'], str), 'Argument \'conditions_file\' passed to Classifier must be a string.'
			self.load_conditions(kwargs['conditions_file'])
		else:
			self.conditions = None

		if 'hands_coeff' in kwargs:									#  Were we given a hands-subvector coefficient?
			assert isinstance(kwargs['hands_coeff'], float), 'Argument \'hands_coeff\' passed to Classifier must be a float.'
			self.hands_coeff = kwargs['hands_coeff']
		else:
			self.hands_coeff = 1.0

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
			self.hand_schema = 'left-right'

		if 'presence_threshold' in kwargs:							#  Were we given an object presence threshold?
			assert isinstance(kwargs['presence_threshold'], float) and kwargs['presence_threshold'] > 0.0 and kwargs['presence_threshold'] <= 1.0, \
			       'Argument \'presence_threshold\' passed to Classifier must be a float in (0.0, 1.0].'
			self.object_presence_threshold = kwargs['presence_threshold']
		else:
			self.object_presence_threshold = 0.8

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
		self.R = rpy2.robjects.r									#  Shortcut to the R backend.
		self.DTW = importr('dtw')									#  Shortcut to the R DTW library.

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

	#  Unify this object's outputs with a unique time stamp.
	def time_stamp(self):
		now = datetime.datetime.now()								#  Build a distinct substring so I don't accidentally overwrite results.
		file_timestamp = now.strftime("%d") + now.strftime("%m") + now.strftime("%Y")[-2:] + 'T' + now.strftime("%H:%M:%S").replace(':', '')
		return file_timestamp

	#################################################################
	#  Loading.                                                     #
	#################################################################

	#  Load a database from file. This loads directly into X_train and y_train. No "allocations."
	#  Rearrange according to schems and apply subvector coefficients before saving internally to the "training set."
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
					self.robject_colors = {}						#  (Re)set
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
					db_entry_enactment_source = action_arr[1]
					db_entry_start_time       = float(action_arr[2])
					db_entry_start_frame      = action_arr[3]
					db_entry_end_time         = float(action_arr[4])
					db_entry_end_frame        = action_arr[5]
					self.y_train.append( label )
					self.X_train.append( [] )
																	#  Be able to lookup the frames of a matched database sample.
					self.train_sample_lookup[sample_ctr] = (db_entry_enactment_source, db_entry_start_time, db_entry_start_frame, \
					                                                                   db_entry_end_time,   db_entry_end_frame)
					sample_ctr += 1
		return

	def load_isotonic_map(self, isotonic_file):
		if self.verbose:
			print('>>> Loading isotonic map from "' + isotonic_file + '".')

		self.isotonic_map = {}										#  key:(lower-bound, upper-bound) ==> val:probability

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
				self.isotonic_map[ (lb, ub) ] = p
		fh.close()

		if self.verbose:
			for k, v in sorted(self.isotonic_map.items()):
				print('    [' + "{:.6f}".format(k[0]) + ', ' + "{:.6f}".format(k[1]) + '] ==> ' + "{:.6f}".format(v))

		return

	def load_conditions(self, conditions_file):
		if self.verbose:
			print('>>> Loading cut-off conditions from "' + conditions_file + '".')

		self.conditions = {}

		fh = open(conditions_file, 'r')
		for line in fh.readlines():
			if line[0] != '#':
				arr = line.strip().split('\t')
				action = arr[0]
				condition = arr[1]
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
			for key in sorted(self.conditions.keys()):
				header_str = '    In order to consider "' + key + '": '
				print(header_str + ' AND '.join(self.conditions[key]['and']))
				if len(self.conditions[key]['or']) > 0:
					print(' '*(len(header_str)) + ' OR '.join(self.conditions[key]['or']))
		return

	#################################################################
	#  Shared classification engine.                                #
	#################################################################

	#  This is a core classification routine, usable by Atemporal and Temporal subclasses alike.
	#  Given a single query sequence, return:
	#    - Matching costs over all classes
	#    - Confidences over all classes
	#    - Probability distribution over all classes
	#    - Metadata (nearest neighbor indices, alignment sequences) over all classes
	#    - Timing
	def classify(self, query_seq):
		query = np.array(query_seq)									#  Convert to numpy.
		rq, cq = query.shape										#  Save number of rows and number of columns.
		query_R = self.R.matrix(query, nrow=rq, ncol=cq)			#  Convert to R matrix.

		least_cost = float('inf')
		nearest_neighbor_label = None
																	#  In defense of hashing versus building a flat list:
																	#  a database is likely to have many more samples than there are labels.
																	#  As a hash table, updating takes O(1) * |X_train| rather than
																	#  O(n) * |X_train| to find matching_costs[ matching_costs.index(template_label) ].
		matching_costs = {}											#  Matching cost for the nearest neighbor per class.
		for label in self.labels('train'):							#  Initialize everything to infinitely far away.
			matching_costs[label] = float('inf')

		metadata = {}												#  Track information about the best match per class.
		for label in self.labels('both'):							#  Include labels the classifier may not know; these will simply be empty.
			metadata[label] = {}

		timing = {}
		timing['dtw-classification'] = []							#  This is a coarser grain: time each classification process.
		timing['test-cutoff-conditions'] = []						#  Prepare to collect times for calling test_cutoff_conditions().
		timing['dtw-R-call'] = []									#  Prepare to collect times for running R's DTW.
		timing['compute-confidence'] = []							#  Prepare to collect times for computing confidence scores.
		timing['isotonic-lookup'] = []								#  Prepare to collect times for bucket-search.

		t0_start = time.process_time()								#  Start timer.
		db_index = 0												#  Index into self.X_train let us know which sample best matches the query.
		for template_seq in self.X_train:							#  For every training-set sample, 'template'...
			template_label = self.y_train[db_index]					#  Save the true label for this template sequence.

			conditions_passed = False
			if self.conditions is not None:
				t1_start = time.process_time()						#  Start timer.
				conditions_passed = self.cutoff_conditions(template_label, query_seq)
				t1_stop = time.process_time()						#  Stop timer.
				timing['test-cutoff-conditions'].append(t1_stop - t1_start)

			if self.conditions is None or conditions_passed:		#  Either we have no conditions, or our conditions give us reason to run DTW.

				template = np.array(template_seq)					#  Convert to numpy.
				rt, ct = template.shape								#  Save number of rows and number of columns.
																	#  Convert to R matrix.
				template_R = self.R.matrix(template, nrow=rt, ncol=ct)

				t1_start = time.process_time()						#  Start timer.
																	#  What is the cost of aligning this template with this query?
				alignment = self.R.dtw(template_R, query_R, open_begin=self.open_begin, open_end=self.open_end)
				t1_stop = time.process_time()						#  Stop timer.
				timing['dtw-R-call'].append(t1_stop - t1_start)

				dist = alignment.rx('normalizedDistance')[0][0]		#  (Normalized) cost of matching this query to this template
																	#  Save sequences of aligned frames (we might render them side by side)
				template_indices = [int(x) for x in list(alignment.rx('index1s')[0])]
				query_indices = [int(x) for x in list(alignment.rx('index2s')[0])]

				if least_cost > dist:								#  A preferable match over all!
					least_cost = dist								#  Save the cost.
					nearest_neighbor_label = template_label[:]		#  Save the (tentative) prediction.

				if matching_costs[template_label] > dist:			#  A preferable match within class!
					matching_costs[template_label] = dist			#  Save the preferable cost.

					metadata[template_label]['db-index'] = db_index	#  Save information about the best match found so far.
					metadata[template_label]['template-indices'] = template_indices
					metadata[template_label]['query-indices'] = query_indices

			db_index += 1

		t1_start = time.process_time()								#  Start timer.
																	#  Get a dictionary of key:label ==> val:confidence.
		confidences = self.compute_confidences(sorted([x for x in matching_costs.items()], key=lambda x: x[1]))
		t1_stop = time.process_time()								#  Stop timer.
		timing['compute-confidence'].append(t1_stop - t1_start)

		probabilities = {}											#  If we apply isotonic mapping, then this is a different measure than confidence.
		for label in self.labels('train'):
			probabilities[label] = 0.0

		if self.isotonic_map is not None:							#  We have an isotonic mapping to apply.
			t1_start = time.process_time()							#  Start timer.
			for label, confidence in confidences.items():
				brackets = sorted(self.isotonic_map.keys())
				i = 0
				while i < len(brackets) and not (confidence >= brackets[i][0] and confidence < brackets[i][1]):
					i += 1

				probabilities[label] = self.isotonic_map[ brackets[i] ]
			t1_stop = time.process_time()							#  Stop timer.
			timing['isotonic-lookup'].append(t1_stop - t1_start)
		else:														#  No mapping; probability = confidence, which is sloppy, but... meh.
			for label, confidence in confidences.items():
				probabilities[label] = confidence
		t0_stop = time.process_time()								#  Stop timer.
		timing['dtw-classification'].append(t0_stop - t0_start)

		return matching_costs, confidences, probabilities, metadata, timing

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
																	#  12 is the offset past the hand encodings, into the props sub-vector.
																	#  If anything required in the AND list has a zero signal, then this frame fails.
						if vector[ self.recognizable_objects.index(self.conditions[candidate_label]['and'][i]) + 12 ] == 0.0:
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
																	#  12 is the offset past the hand encodings, into the props sub-vector.
																	#  If anything required in the AND list has a zero signal, then this frame fails.
						if vector[ self.recognizable_objects.index(self.conditions[candidate_label]['or'][i]) + 12 ] > 0.0:
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

			return passed_and or passed_or

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

		elif self.confidence_function == 'sum3':					#  (Sum of three minimal distances) / my distance
			s = sum([x[1] for x in labels_costs[:3]])
			for i in range(0, len(labels_costs)):
				conf[ labels_costs[i][0] ] = s / (labels_costs[i][1] + self.epsilon)

		elif self.confidence_function == 'sum4':					#  (Sum of four minimal distances) / my distance
			s = sum([x[1] for x in labels_costs[:4]])
			for i in range(0, len(labels_costs)):
				conf[ labels_costs[i][0] ] = s / (labels_costs[i][1] + self.epsilon)

		elif self.confidence_function == 'sum5':					#  (Sum of five minimal distances) / my distance
			s = sum([x[1] for x in labels_costs[:5]])
			for i in range(0, len(labels_costs)):
				conf[ labels_costs[i][0] ] = s / (labels_costs[i][1] + self.epsilon)

		elif self.confidence_function == 'n-min-obsv':				#  Normalized minimum distance observed / my distance
			min_d = labels_costs[0][1]
			for i in range(0, len(labels_costs)):
				conf[ labels_costs[i][0] ] = min_d / (labels_costs[i][1] + self.epsilon)
			s = sum(conf.values())
			for i in range(0, len(labels_costs)):
				conf[ labels_costs[i][0] ] /= s

		elif self.confidence_function == 'min-obsv':				#  Minimum distance observed / my distance
			min_d = labels_costs[0][1]								#  DO NOT ACTUALLY USE THIS FUNCTION! it's an illustrative fail-case ONLY.
			for i in range(0, len(labels_costs)):
				conf[ labels_costs[i][0] ] = min_d / (labels_costs[i][1] + self.epsilon)

		elif self.confidence_function == 'max-marg':				#  Second-best distance minus best distance. Worst match gets zero.
			for i in range(0, len(labels_costs)):
				if i < len(labels_costs) - 1:
					conf[ labels_costs[i][0] ] = labels_costs[i + 1][1] - labels_costs[i][1]
				else:
					conf[ labels_costs[i][0] ] = 0.0

		elif self.confidence_function == '2over1':					#  Second-best distance over best distance. Worst match gets zero.
			for i in range(0, len(labels_costs)):
				if i < len(labels_costs) - 1:
					conf[ labels_costs[i][0] ] = labels_costs[i + 1][1] / (labels_costs[i][1] + self.epsilon)
				else:
					conf[ labels_costs[i][0] ] = 0.0

		return conf

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
																	#  Skip one-hot encoded LH_0.
																	#  Skip one-hot encoded LH_1.
																	#  Skip one-hot encoded LH_2.
		vec[6] *= self.hands_coeff									#  Weigh RH_x.
		vec[7] *= self.hands_coeff									#  Weigh RH_y.
		vec[8] *= self.hands_coeff									#  Weigh RH_z.
																	#  Skip one-hot encoded RH_0.
																	#  Skip one-hot encoded RH_1.
																	#  Skip one-hot encoded RH_2.
		for i in range(12, len(vector)):							#  Weigh props_i.
			vec[i] *= self.props_coeff

		return tuple(vec)

	#  We would want to do this when rendering seismographs: prevent them from being HUGE.
	def undo_vector_coefficients(self, vector):
		vec = [x for x in vector]									#  Convert to a list so we can manipulate it.

		vec[0] /= self.hands_coeff									#  Weigh LH_x.
		vec[1] /= self.hands_coeff									#  Weigh LH_y.
		vec[2] /= self.hands_coeff									#  Weigh LH_z.
																	#  Skip one-hot encoded LH_0.
																	#  Skip one-hot encoded LH_1.
																	#  Skip one-hot encoded LH_2.
		vec[6] /= self.hands_coeff									#  Weigh RH_x.
		vec[7] /= self.hands_coeff									#  Weigh RH_y.
		vec[8] /= self.hands_coeff									#  Weigh RH_z.
																	#  Skip one-hot encoded RH_0.
																	#  Skip one-hot encoded RH_1.
																	#  Skip one-hot encoded RH_2.
		for i in range(12, len(vector)):							#  Weigh props_i.
			vec[i] /= self.props_coeff

		return tuple(vec)

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

	#  Given 'predictions_truths' is a list of tuples: (predicted label, true label).
	def confusion_matrix(self, predictions_truths):
		labels = self.labels('train')								#  Only count the labels the classifier is equipped to recognize!
		num_classes = len(labels)

		M = np.zeros((num_classes, num_classes), dtype='uint16')

		for pred_gt in predictions_truths:
			prediction = pred_gt[0]
			ground_truth_label = pred_gt[1]
			if prediction is not None:
				i = labels.index(prediction)
				if ground_truth_label in labels:
					j = labels.index(ground_truth_label)
					M[i, j] += 1

		return M

	#  Write the confusion matrix to file.
	def write_confusion_matrix(self, predictions_truths, file_timestamp=None):
		if file_timestamp is None:
			file_timestamp = self.time_stamp()						#  Build a distinct substring so I don't accidentally overwrite results.

		num_classes = self.num_labels()
		labels = self.labels('both')

		M = self.confusion_matrix(predictions_truths)

		fh = open('confusion-matrix-' + file_timestamp + '.txt', 'w')
		fh.write('#  Classifier confusion matrix made at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		fh.write('\t' + '\t'.join(labels) + '\n')					#  Write the column headers
		for i in range(0, len(labels)):
			fh.write(labels[i] + '\t' + '\t'.join([str(x) for x in M[i]]) + '\n')
		fh.close()

		return

	#  Writes two files: "confidences-winners-<time stamp>.txt" and "confidences-all-<time stamp>.txt".
	#  The former writes only the confidence scores for predictions.
	#  The latter writes the confidence scores for all labels, even those that were not selected by nearest-neighbor.
	def write_confidences(self, predictions_truths, confidences, file_timestamp=None):
		if file_timestamp is None:
			file_timestamp = self.time_stamp()						#  Build a distinct substring so I don't accidentally overwrite results.

																	#  Zip these up so we can sort them DESCENDING by confidence.
		pred_gt_conf = sorted(list(zip(predictions_truths, confidences)), key=lambda x: x[1], reverse=True)
		pred_gt = [x[0] for x in pred_gt_conf]						#  Now separate them again.
		conf = [x[1] for x in pred_gt_conf]

		fh = open('confidences-winners-' + file_timestamp + '.txt', 'w')
		fh.write('#  Classifier predictions made at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		fh.write('#  Winning labels only.\n')
		fh.write('#  Confidence function is "' + self.confidence_function + '"\n')
		fh.write('#  Confidence    Predicted-Label    Ground-Truth-Label\n')
		for i in range(0, len(pred_gt)):
			c = conf[i]
			prediction = pred_gt[i][0]
			ground_truth_label = pred_gt[i][1]

			if prediction is not None:
				fh.write(str(c) + '\t' + prediction + '\t' + ground_truth_label + '\n')
			else:
				fh.write(str(c) + '\t' + 'NO-DECISION' + '\t' + ground_truth_label + '\n')
		fh.close()

		fh = open('confidences-all-' + file_timestamp + '.txt', 'w')
		fh.write('#  Classifier predictions made at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		fh.write('#  All labels.\n')
		fh.write('#  Confidence function is "' + self.confidence_function + '"\n')
		fh.write('#  Confidence    Predicted-Label    Ground-Truth-Label\n')
		for i in range(0, len(pred_gt)):
			c = conf[i]
			prediction = pred_gt[i][0]
			ground_truth_label = pred_gt[i][1]

			if prediction is not None:
				fh.write(str(c) + '\t' + prediction + '\t' + ground_truth_label + '\n')
			else:
				fh.write(str(c) + '\t' + 'NO-DECISION' + '\t' + ground_truth_label + '\n')
		fh.close()

		return

	#  Avoid repeating time-consuming experiments. Save results to file.
	#  'stats' is dictionary with key:'_tests' ==> val:[(prediction, ground-truth), (prediction, ground-truth), ... ]
	#                             key:'_conf'  ==> val:[ confidence,                 confidence,                ... ]
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
			fh.write('#  ' + ' '.join(sys.argv) + '\n')

		conf_correct = []											#  Accumulate confidences when the classifier is correct.
		conf_incorrect = []											#  Accumulate confidences when the classifier is incorrect.

		decision_ctr = 0											#  Count times the classifier made a decision.
		no_decision_ctr = 0											#  Count times the classifier abstained from making a decision.

		for i in range(0, len(stats['_tests'])):					#  'i' also indexes into 'stats[_conf]'.
			prediction = stats['_tests'][i][0]
			true_label = stats['_tests'][i][1]
			confidence = stats['_conf'][i]

			if prediction is not None:
				decision_ctr += 1
				if prediction == true_label:
					conf_correct.append(confidence)
				else:
					conf_incorrect.append(confidence)
			else:
				no_decision_ctr += 1
				if type(self).__name__ == 'AtemporalClassifier':	#  We happen to know that for atemporal classification
					conf_incorrect.append(confidence)				#  there is never a correct abstention.

		avg_conf_correct = np.mean(conf_correct)					#  Compute average confidence when correct.
		avg_conf_incorrect = np.mean(conf_incorrect)				#  Compute average confidence when incorrect.

		stddev_conf_correct = np.std(conf_correct)					#  Compute standard deviation of confidence when correct.
		stddev_conf_incorrect = np.std(conf_incorrect)				#  Compute standard deviation of confidence when incorrect.

		fh.write('Classification:\n')
		fh.write('===============\n')
		fh.write('\tAccuracy\tPrecision\tRecall\tF1-score\tSupport\n')
		meanAcc = []
		for k, v in stats.items():
			if k != '_tests' and k != '_conf':
				if v['support'] > 0:								#  ONLY ATTEMPT TO CLASSIFY IF THIS IS A "FAIR" QUESTION
					if v['tp'] + v['fp'] + v['fn'] == 0:
						acc    = 0.0
					else:
						acc    = float(v['tp']) / float(v['tp'] + v['fp'] + v['fn'])
					meanAcc.append(acc)

					if v['tp'] + v['fp'] == 0:
						prec   = 0.0
					else:
						prec   = float(v['tp']) / float(v['tp'] + v['fp'])

					if v['tp'] + v['fn'] == 0:
						recall = 0.0
					else:
						recall = float(v['tp']) / float(v['tp'] + v['fn'])

					if prec + recall == 0:
						f1     = 0.0
					else:
						f1     = float(2 * prec * recall) / float(prec + recall)

					support = v['support']
					fh.write(k + '\t' + str(acc) + '\t' + str(prec) + '\t' + str(recall) + '\t' + str(f1) + '\t' + str(support) + '\n')
		fh.write('\n')
		fh.write('Mean Avg. Accuracy:\n')
		fh.write('===================\n')
		if len(meanAcc) > 0:
			fh.write('\t' + str(np.mean(meanAcc)) + '\n')
		else:
			fh.write('N/A\n')

		fh.write('Total decisions made = ' + str(decision_ctr) + '\n')
		fh.write('Total non-decisions made = ' + str(no_decision_ctr) + '\n')
		fh.write('\n')
		fh.write('Avg. Confidence when correct = ' + str(avg_conf_correct) + '\n')
		fh.write('Avg. Confidence when incorrect = ' + str(avg_conf_incorrect) + '\n')
		fh.write('\n')
		fh.write('Std.Dev. Confidence when correct = ' + str(stddev_conf_correct) + '\n')
		fh.write('Std.Dev. Confidence when incorrect = ' + str(stddev_conf_incorrect) + '\n')

		fh.close()

		return

	#  Look for:
	#    - total													Total time taken.
	#    - load-enactment											Times taken to load enactments.
	#    - dtw-classification										This is a coarser grain: time each classification process.
	#    - test-cutoff-conditions									Times taken to test cutoff conditions.
	#    - dtw-R-call												Times taken to run R's DTW.
	#    - compute-confidence										Times taken to compute confidence scores.
	#    - isotonic-lookup											Times taken to look up probabilities from computed confidence scores.
	#    - make-tentative-prediction								Times taken to compute the least cost match.
	#    - sort-confidences											Times taken to put confidence scores in label-order.
	#    - sort-probabilities										Times taken to put probabilities in label-order.
	#    - push-temporal-buffer										Times taken to update temporal-buffer.
	#    - temporal-smoothing										Times taken to perform temporal smoothing.
	#    - make-temporally-smooth-decision							Times taken to make a final decision, given temporally-smoothed probabilities.
	#    - render-side-by-side										Times taken to produce a video showing the query and best-matched template.
	#    - render-annotated-source									Times taken to produce that annotated source image.
	#    - render-rolling-buffer									Times taken to produce the rolling buffer seismograph.
	#    - render-confidence										Times taken to produce the confidences seismograph.
	#    - render-probabilities										Times taken to produce the probabilities seismograph.
	#    - render-smoothed-probabilities							Times taken to produce the smoothed-probabilities seismograph.
	def write_timing(self, file_timestamp=None):
		if file_timestamp is None:
			file_timestamp = self.time_stamp()						#  Build a distinct substring so I don't accidentally overwrite results.

		fh = open('timing-' + file_timestamp + '.txt', 'w')
		fh.write('#  Times for classifier tasks, completed at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		if len(sys.argv) > 1:										#  Was this called from a script? Save the command-line call.
			fh.write('#  ' + ' '.join(sys.argv) + '\n')
																	#  'total' MUST be in the table.
		fh.write('TOTAL TIME\t' + str(self.timing['total']) + '\n\n')
																	#  Report enactment-loading times.
		if 'load-enactment' in self.timing and len(self.timing['load-enactment']) > 0:
			fh.write('Avg. enactment-loading time\t' + str(np.mean(self.timing['load-enactment'])) + '\t' + str(np.mean(self.timing['load-enactment']) / self.timing['total'] * 100.0) + '%\n')
			fh.write('Std.dev enactment-loading time\t' + str(np.std(self.timing['load-enactment'])) + '\n\n')
		else:
			fh.write('Avg. enactment-loading time\tN/A\n')
			fh.write('Std.dev enactment-loading time\tN/A\n\n')
																	#  Report DTW-classification times.
		if 'dtw-classification' in self.timing and len(self.timing['dtw-classification']) > 0:
			fh.write('Avg. DTW time (per query pair)\t' + str(np.mean(self.timing['dtw-classification'])) + '\t' + str(np.mean(self.timing['dtw-classification']) / self.timing['total'] * 100.0) + '%\n')
			fh.write('Std.dev DTW time (per query pair)\t' + str(np.std(self.timing['dtw-classification'])) + '\n\n')
		else:
			fh.write('Avg. DTW time (per query pair)\tN/A\n')
			fh.write('Std.dev DTW time (per query pair)\tN/A\n\n')
																	#  Report cutoff-condition testing times.
		if 'test-cutoff-conditions' in self.timing and len(self.timing['test-cutoff-conditions']) > 0:
			fh.write('Avg. cutoff-condition testing time\t' + str(np.mean(self.timing['test-cutoff-conditions'])) + '\t' + str(np.mean(self.timing['test-cutoff-conditions']) / self.timing['total'] * 100.0) + '%\n')
			fh.write('Std.dev cutoff-condition testing time\t' + str(np.std(self.timing['test-cutoff-conditions'])) + '\n\n')
		else:
			fh.write('Avg. cutoff-condition testing time\tN/A\n')
			fh.write('Std.dev cutoff-condition testing time\tN/A\n\n')
																	#  Report DTW times.
		if 'dtw-R-call' in self.timing and len(self.timing['dtw-R-call']) > 0:
			fh.write('Avg. DTW time (per template-query pair)\t' + str(np.mean(self.timing['dtw-R-call'])) + '\t' + str(np.mean(self.timing['dtw-R-call']) / self.timing['total'] * 100.0) + '%\n')
			fh.write('Std.dev DTW time (per template-query pair)\t' + str(np.std(self.timing['dtw-R-call'])) + '\n\n')
		else:
			fh.write('Avg. DTW time (per template-query pair)\tN/A\n')
			fh.write('Std.dev DTW time (per template-query pair)\tN/A\n\n')
																	#  Report cutoff-condition testing times.
		if 'compute-confidence' in self.timing and len(self.timing['compute-confidence']) > 0:
			fh.write('Avg. confidence computation time\t' + str(np.mean(self.timing['compute-confidence'])) + '\t' + str(np.mean(self.timing['compute-confidence']) / self.timing['total'] * 100.0) + '%\n')
			fh.write('Std.dev confidence computation time\t' + str(np.std(self.timing['compute-confidence'])) + '\n\n')
		else:
			fh.write('Avg. confidence computation time\tN/A\n')
			fh.write('Std.dev confidence computation time\tN/A\n\n')
																	#  Report isotonic lookup times.
		if 'isotonic-lookup' in self.timing and len(self.timing['isotonic-lookup']) > 0:
			fh.write('Avg. probability lookup time\t' + str(np.mean(self.timing['isotonic-lookup'])) + '\t' + str(np.mean(self.timing['isotonic-lookup']) / self.timing['total'] * 100.0) + '%\n')
			fh.write('Std.dev probability lookup time\t' + str(np.std(self.timing['isotonic-lookup'])) + '\n\n')
		else:
			fh.write('Avg. probability lookup time\tN/A\n')
			fh.write('Std.dev probability lookup time\tN/A\n\n')
																	#  Report least-distance-finding times.
		if 'make-tentative-prediction' in self.timing and len(self.timing['make-tentative-prediction']) > 0:
			fh.write('Avg. tentative decision-making time\t' + str(np.mean(self.timing['make-tentative-prediction'])) + '\t' + str(np.mean(self.timing['make-tentative-prediction']) / self.timing['total'] * 100.0) + '%\n')
			fh.write('Std.dev tentative decision-making time\t' + str(np.std(self.timing['make-tentative-prediction'])) + '\n\n')
		else:
			fh.write('Avg. tentative decision-making time\tN/A\n')
			fh.write('Std.dev tentative decision-making time\tN/A\n\n')
																	#  Report confidence score-sorting times.
		if 'sort-confidences' in self.timing and len(self.timing['sort-confidences']) > 0:
			fh.write('Avg. confidence sorting time\t' + str(np.mean(self.timing['sort-confidences'])) + '\t' + str(np.mean(self.timing['sort-confidences']) / self.timing['total'] * 100.0) + '%\n')
			fh.write('Std.dev confidence sorting time\t' + str(np.std(self.timing['sort-confidences'])) + '\n\n')
		else:
			fh.write('Avg. confidence sorting time\tN/A\n')
			fh.write('Std.dev confidence sorting time\tN/A\n\n')
																	#  Report probability-sorting times.
		if 'sort-probabilities' in self.timing and len(self.timing['sort-probabilities']) > 0:
			fh.write('Avg. probability sorting time\t' + str(np.mean(self.timing['sort-probabilities'])) + '\t' + str(np.mean(self.timing['sort-probabilities']) / self.timing['total'] * 100.0) + '%\n')
			fh.write('Std.dev probability sorting time\t' + str(np.std(self.timing['sort-probabilities'])) + '\n\n')
		else:
			fh.write('Avg. probability sorting time\tN/A\n')
			fh.write('Std.dev probability sorting time\tN/A\n\n')
																	#  Report temporal-buffer update times.
		if 'push-temporal-buffer' in self.timing and len(self.timing['push-temporal-buffer']) > 0:
			fh.write('Avg. temporal-buffer update time\t' + str(np.mean(self.timing['push-temporal-buffer'])) + '\t' + str(np.mean(self.timing['push-temporal-buffer']) / self.timing['total'] * 100.0) + '%\n')
			fh.write('Std.dev temporal-buffer update time\t' + str(np.std(self.timing['push-temporal-buffer'])) + '\n\n')
		else:
			fh.write('Avg. temporal-buffer update time\tN/A\n')
			fh.write('Std.dev temporal-buffer update time\tN/A\n\n')
																	#  Report temporal-smoothing times.
		if 'temporal-smoothing' in self.timing and len(self.timing['temporal-smoothing']) > 0:
			fh.write('Avg. temporal-smoothing time\t' + str(np.mean(self.timing['temporal-smoothing'])) + '\t' + str(np.mean(self.timing['temporal-smoothing']) / self.timing['total'] * 100.0) + '%\n')
			fh.write('Std.dev temporal-smoothing time\t' + str(np.std(self.timing['temporal-smoothing'])) + '\n\n')
		else:
			fh.write('Avg. temporal-smoothing time\tN/A\n')
			fh.write('Std.dev temporal-smoothing time\tN/A\n\n')
																	#  Report final decision-making times.
		if 'make-temporally-smooth-decision' in self.timing and len(self.timing['make-temporally-smooth-decision']) > 0:
			fh.write('Avg. temporally-smoothed classification time\t' + str(np.mean(self.timing['make-temporally-smooth-decision'])) + '\t' + str(np.mean(self.timing['make-temporally-smooth-decision']) / self.timing['total'] * 100.0) + '%\n')
			fh.write('Std.dev temporally-smoothed classification time\t' + str(np.std(self.timing['make-temporally-smooth-decision'])) + '\n\n')
		else:
			fh.write('Avg. temporally-smoothed classification time\tN/A\n')
			fh.write('Std.dev temporally-smoothed classification time\tN/A\n\n')
																	#  Report side-by-side rendering times.
		if 'render-side-by-side' in self.timing and len(self.timing['render-side-by-side']) > 0:
			fh.write('Avg. side-by-side video rendering time\t' + str(np.mean(self.timing['render-side-by-side'])) + '\t' + str(np.mean(self.timing['render-side-by-side']) / self.timing['total'] * 100.0) + '%\n')
			fh.write('Std.dev side-by-side video rendering time\t' + str(np.std(self.timing['render-side-by-side'])) + '\n\n')
		else:
			fh.write('Avg. side-by-side video rendering time\tN/A\n')
			fh.write('Std.dev side-by-side video rendering time\tN/A\n\n')
																	#  Report annotation times.
		if 'render-annotated-source' in self.timing and len(self.timing['render-annotated-source']) > 0:
			fh.write('Avg. video annotation time\t' + str(np.mean(self.timing['render-annotated-source'])) + '\t' + str(np.mean(self.timing['render-annotated-source']) / self.timing['total'] * 100.0) + '%\n')
			fh.write('Std.dev video annotation time\t' + str(np.std(self.timing['render-annotated-source'])) + '\n\n')
		else:
			fh.write('Avg. video annotation time\tN/A\n')
			fh.write('Std.dev video annotation time\tN/A\n\n')
																	#  Report rolling-buffer seismograph rendering times.
		if 'render-rolling-buffer' in self.timing and len(self.timing['render-rolling-buffer']) > 0:
			fh.write('Avg. rolling buffer rendering time\t' + str(np.mean(self.timing['render-rolling-buffer'])) + '\t' + str(np.mean(self.timing['render-rolling-buffer']) / self.timing['total'] * 100.0) + '%\n')
			fh.write('Std.dev rolling buffer rendering time\t' + str(np.std(self.timing['render-rolling-buffer'])) + '\n\n')
		else:
			fh.write('Avg. rolling buffer rendering time\tN/A\n')
			fh.write('Std.dev rolling buffer rendering time\tN/A\n\n')
																	#  Report confidence seismograph rendering times.
		if 'render-confidence' in self.timing and len(self.timing['render-confidence']) > 0:
			fh.write('Avg. confidence rendering time\t' + str(np.mean(self.timing['render-confidence'])) + '\t' + str(np.mean(self.timing['render-confidence']) / self.timing['total'] * 100.0) + '%\n')
			fh.write('Std.dev confidence rendering time\t' + str(np.std(self.timing['render-confidence'])) + '\n\n')
		else:
			fh.write('Avg. confidence rendering time\tN/A\n')
			fh.write('Std.dev confidence rendering time\tN/A\n\n')
																	#  Report probabilities seismograph rendering times.
		if 'render-probabilities' in self.timing and len(self.timing['render-probabilities']) > 0:
			fh.write('Avg. probabilities rendering time\t' + str(np.mean(self.timing['render-probabilities'])) + '\t' + str(np.mean(self.timing['render-probabilities']) / self.timing['total'] * 100.0) + '%\n')
			fh.write('Std.dev probabilities rendering time\t' + str(np.std(self.timing['render-probabilities'])) + '\n\n')
		else:
			fh.write('Avg. probabilities rendering time\tN/A\n')
			fh.write('Std.dev probabilities rendering time\tN/A\n\n')
																	#  Report smoothed-probabilities seismograph rendering times.
		if 'render-smoothed-probabilities' in self.timing and len(self.timing['render-smoothed-probabilities']) > 0:
			fh.write('Avg. smoothed-probabilities rendering time\t' + str(np.mean(self.timing['render-smoothed-probabilities'])) + '\t' + str(np.mean(self.timing['render-smoothed-probabilities']) / self.timing['total'] * 100.0) + '%\n')
			fh.write('Std.dev smoothed-probabilities rendering time\t' + str(np.std(self.timing['render-smoothed-probabilities'])) + '\n\n')
		else:
			fh.write('Avg. smoothed-probabilities rendering time\tN/A\n')
			fh.write('Std.dev smoothed-probabilities rendering time\tN/A\n\n')

		fh.close()
		return

'''
Why "atemporal"? Because sequence boundaries are given; they need not be discovered frame by frame.
Give this classifier enactment files, and it will divvy them up, trying to be fair, turn them into a dataset,
and perform "atemporal" classification.

In the interpreter:
  atemporal = AtemporalClassifier(window_size=10, stride=2, train=['BackBreaker1', 'Enactment1', 'Enactment2', 'Enactment3', 'Enactment4', 'Enactment5', 'Enactment6', 'Enactment7', 'Enactment9', 'Enactment10', 'MainFeederBox1', 'Regulator1', 'Regulator2'], test=['Enactment11', 'Enactment12'], verbose=True)
  atemporal = AtemporalClassifier(window_size=10, stride=2, divide=['BackBreaker1', 'Enactment1', 'Enactment2', 'Enactment3', 'Enactment4', 'Enactment5', 'Enactment6', 'Enactment7', 'Enactment9', 'Enactment10', 'MainFeederBox1', 'Regulator1', 'Regulator2', 'Enactment11', 'Enactment12'], verbose=True)

Alternatively, you can give this class only a test set, not training set, and load a database file like the TemporalClassifier uses.
In the interpreter:
  atemporal = AtemporalClassifier(window_size=10, stride=2, test=['Enactment11', 'Enactment12'], verbose=True)
  atemporal.relabel_from_file('relabels_07sep21.txt')
  atemporal.commit()
  atemporal.load_db('10f.db')
'''
class AtemporalClassifier(Classifier):
	def __init__(self, **kwargs):
		super(AtemporalClassifier, self).__init__(**kwargs)

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
		else:
			train_portion = 0.8
		test_portion = 1.0 - train_portion

		if 'test_portion' in kwargs:								#  Were we given a portion of divided enactments to allocate to the test set?
			assert isinstance(kwargs['test_portion'], float) and kwargs['test_portion'] > 0.0 and kwargs['test_portion'] < 1.0, \
			       'Argument \'test_portion\' passed to AtemporalClassifier must be a float in (0.0, 1.0).'
			test_portion = kwargs['test_portion']
		else:
			test_portion = 0.2
		train_portion = 1.0 - test_portion

		if 'minimum_length' in kwargs:								#  Were we given a minimum sequence length?
			assert isinstance(kwargs['minimum_length'], int) and kwargs['minimum_length'] > 0, \
			       'Argument \'minimum_length\' passed to AtemporalClassifier must be an int > 0.'
			self.minimum_length = kwargs['minimum_length']
		else:
			self.minimum_length = 2

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
		#############################################################
		#  Load ProcessedEnactments from the given enactment names. #
		#############################################################
		if self.verbose and len(train_list) > 0:
			print('>>> Loading atemporal training set.')
		for enactment in train_list:
			pe = ProcessedEnactment(enactment, verbose=self.verbose)
			for i in range(0, pe.num_actions()):
																	#  Mark this action in this enactment for the training set.
				self.allocation[ (enactment, i, pe.actions[i][0]) ] = 'train'
		if self.verbose and len(train_list) > 0:
			print('')

		if self.verbose and len(test_list) > 0:
			print('>>> Loading atemporal test set.')
		for enactment in test_list:
			pe = ProcessedEnactment(enactment, verbose=self.verbose)
			for i in range(0, pe.num_actions()):
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

		classification_stats = {}
		classification_stats['_tests'] = []							#  key:_tests ==> val:[(prediction, ground-truth), (prediction, ground-truth), ... ]
		classification_stats['_conf'] = []							#  key:_conf  ==> val:[confidence, confidence, ... ]

		for label in self.labels('both'):
			classification_stats[label] = {}						#  key:label ==> val:{key:tp      ==> val:true positive count
			classification_stats[label]['tp']      = 0				#                     key:fp      ==> val:false positive count
			classification_stats[label]['fp']      = 0				#                     key:fn      ==> val:false negative count
			classification_stats[label]['fn']      = 0				#                     key:support ==> val:instance in training set}
			classification_stats[label]['support'] = len([x for x in self.y_train if x == label])

		num_labels = len(self.y_test)
		prev_ctr = 0
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.
		success_ctr = 0
		mismatch_ctr = 0

		self.timing = {}											#  (Re)set.
		self.timing['total'] = 0
		self.timing['dtw-classification'] = []						#  This is a coarser grain: time each classification process.
		self.timing['test-cutoff-conditions'] = []					#  Prepare to collect times for calling test_cutoff_conditions().
		self.timing['dtw-R-call'] = []								#  Prepare to collect times for running R's DTW.
		self.timing['compute-confidence'] = []						#  Prepare to collect times for computing confidence scores.
		self.timing['isotonic-lookup'] = []							#  Prepare to collect times for bucket-search.
		self.timing['make-tentative-prediction'] = []				#  Prepare to collect least-distance-finding times.
		self.timing['make-decision'] = []							#  Prepare to collect final decision-making runtimes.
		if self.render:
			self.timing['render-side-by-side'] = []					#  Prepare to collect rendering times.

		t0_start = time.process_time()								#  Start timer.
		for i in range(0, len(self.y_test)):
			query = self.X_test[i]									#  Bookmark the query and ground-truth label
			ground_truth_label = self.y_test[i]
																	#  Call the parent class's core matching engine.
			matching_costs, confidences, probabilities, metadata, timing = super(AtemporalClassifier, self).classify(query)
			self.timing['dtw-classification'] += timing['dtw-classification']
			self.timing['test-cutoff-conditions'] += timing['test-cutoff-conditions']
			self.timing['dtw-R-call'] += timing['dtw-R-call']
			self.timing['compute-confidence'] += timing['compute-confidence']
			self.timing['isotonic-lookup'] += timing['isotonic-lookup']

			t1_start = time.process_time()							#  Start timer.
			least_cost = float('inf')
			prediction = None
			for k, v in matching_costs.items():						#  Find the best match.
				if v < least_cost:
					least_cost = v
					prediction = k
			t1_stop = time.process_time()							#  Stop timer.
			self.timing['make-tentative-prediction'].append(t1_stop - t1_start)

																	#  Before consulting the threshold and possibly witholding judgment,
																	#  save the confidence score for what *would* have been picked.
																	#  We use these values downstream in the pipeline for isotonic regression.
			classification_stats['_conf'].append( confidences[prediction] )

			t1_start = time.process_time()							#  Start timer.
			if probabilities[prediction] < self.threshold:			#  Is it above the threshold?
				prediction = None
			t1_stop = time.process_time()							#  Stop timer.
			self.timing['make-decision'].append(t1_stop - t1_start)

			if prediction == ground_truth_label:					#  This is the atemporal tester: ground_truth_label will never be None.
				classification_stats[ground_truth_label]['tp'] += 1
			elif prediction is not None:
				classification_stats[prediction]['fp']         += 1
				classification_stats[ground_truth_label]['fn'] += 1
																	#  Add this prediction-truth tuple to our list.
			classification_stats['_tests'].append( (prediction, ground_truth_label) )

			if self.render:											#  Put the query and the template side by side.
				if prediction is not None:							#  If there's no prediction, there is nothing to render.
					t1_start = time.process_time()					#  Start timer.
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
			self.X_train = []										#  To become a list of lists of vectors
			self.y_train = []										#  To become a list of lables (strings)
			self.train_sample_lookup = {}

		if sets == 'test' or sets == 'both':
			self.X_test = []										#  To become a list of lists of vectors
			self.y_test = []										#  To become a list of lables (strings)
			self.test_sample_lookup = {}

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
		for enactment_action, set_name in self.allocation.items():
			if set_name in rev_allocation:
				rev_allocation[set_name].append(enactment_action)

		if 'train' in rev_allocation and len(rev_allocation['train']) > 0:
			if self.verbose:
				print('>>> Collecting snippets of length ' + str(self.window_size) + ', stride ' + str(self.stride) + ' from training set enactments.')
			num = len(rev_allocation['train'])
			prev_ctr = 0
			max_ctr = os.get_terminal_size().columns - 7			#  Leave enough space for the brackets, space, and percentage.
			ctr = 0													#  Count through allocations.
			sample_ctr = 0											#  Count through snippets.

			for enactment_action in rev_allocation['train']:
				enactment_name = enactment_action[0]
				action_index = enactment_action[1]
				action_label = enactment_action[2]

				pe = ProcessedEnactment(enactment_name, verbose=False)
				enactment_frames = pe.get_frames()					#  Retrieve a list of tuples: (time stamp, frame dictionary).
																	#  Dictionary has keys 'file', 'ground-truth-label', and 'vector'.
																	#  I don't trust the fidelity of float-->str-->float conversions.
																	#  Separate frame file paths and use these to index into 'enactment_frames'.
				video_frames = [x[1]['file'] for x in enactment_frames]
				snippets = pe.snippets_from_action(self.window_size, self.stride, action_index)
				for snippet in snippets:							#  Each 'snippet' = (label, start time, start frame, end time, end frame).
					seq = []
					for i in range(0, self.window_size):			#  Build the snippet sequence.
						vec = enactment_frames[video_frames.index(snippet[2]) + i][1]['vector'][:]
						if self.hand_schema == 'strong-hand':		#  Re-arrange for "strong-hand-first" encoding?
							vec = strong_hand_encode(vec)
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
				enactment_frames = pe.get_frames()					#  Retrieve a list of tuples: (time stamp, frame dictionary).
																	#  Dictionary has keys 'file', 'ground-truth-label', and 'vector'.
																	#  I don't trust the fidelity of float-->str-->float conversions.
																	#  Separate frame file paths and use these to index into 'enactment_frames'.
				video_frames = [x[1]['file'] for x in enactment_frames]
				snippets = pe.snippets_from_action(self.window_size, self.stride, action_index)
				for snippet in snippets:							#  Each 'snippet' = (label, start time, start frame, end time, end frame).
					seq = []
					for i in range(0, self.window_size):			#  Build the snippet sequence.
						vec = enactment_frames[video_frames.index(snippet[2]) + i][1]['vector'][:]
						if self.hand_schema == 'strong-hand':		#  Re-arrange for "strong-hand-first" encoding?
							vec = self.strong_hand_encode(vec)
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
																	#  These come ONE-indexed from R.
				q_index = metadata[prediction]['query-indices'][j] - 1
			else:
				q_index = None
			if j < t_alignment_length:
																	#  These come ONE-indexed from R.
				t_index = metadata[prediction]['template-indices'][j] - 1
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

			if self.object_detection_source == 'GT':
				fh = open(q_enactment + '_props.txt', 'r')
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

				fh = open(t_enactment + '_props.txt', 'r')
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
			#else:

			q_img = cv2.addWeighted(q_img, 1.0, q_mask_canvas, 0.7, 0)
			q_img = cv2.cvtColor(q_img, cv2.COLOR_RGBA2RGB)			#  Flatten alpha

			t_img = cv2.addWeighted(t_img, 1.0, t_mask_canvas, 0.7, 0)
			t_img = cv2.cvtColor(t_img, cv2.COLOR_RGBA2RGB)			#  Flatten alpha

			if self.object_detection_source == 'GT':
				fh = open(q_enactment + '_props.txt', 'r')
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

				fh = open(t_enactment + '_props.txt', 'r')
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
			#else:

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
				cv2.putText(canvas, 'From ' + q_enactment + ', ' + q_frames[q_index].split('/')[-1], \
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
				cv2.putText(canvas, 'From ' + t_enactment + ', ' + t_frames[t_index].split('/')[-1], \
				                    (self.side_by_side_source_super['x'] + half_width, self.side_by_side_source_super['y']), cv2.FONT_HERSHEY_SIMPLEX, \
				                    self.side_by_side_source_super['fontsize'], (255, 255, 255), 2)
			vid.write(canvas)
		vid.release()
		return

'''
Give this classifier a database file and enactment files.
The database becomes the training set, self.X_train, and the enactments will be marched through in simulated real time.
Buffers-full of vectors from the enactments are given to the classification engine.
This constitutes "temporal" classification because sequence boundaries are not known a priori.

In the interpreter:
temporal = TemporalClassifier(rolling_buffer_length=10, rolling_buffer_stride=2, db_file='10f.db', inputs=['Enactment11', 'Enactment12'], verbose=True)
'''
class TemporalClassifier(Classifier):
	def __init__(self, **kwargs):
		super(TemporalClassifier, self).__init__(**kwargs)

		if 'rolling_buffer_length' in kwargs:						#  Were we given a rolling buffer length?
			assert isinstance(kwargs['rolling_buffer_length'], int) and kwargs['rolling_buffer_length'] > 0, \
			       'Argument \'rolling_buffer_length\' passed to Classifier must be an integer > 0.'
			self.rolling_buffer_length = kwargs['rolling_buffer_length']
		else:
			self.rolling_buffer_length = 10							#  Default to 10.

		if 'rolling_buffer_stride' in kwargs:						#  Were we given a rolling buffer stride?
			assert isinstance(kwargs['rolling_buffer_stride'], int) and kwargs['rolling_buffer_stride'] > 0, \
			       'Argument \'rolling_buffer_stride\' passed to Classifier must be an integer > 0.'
			self.rolling_buffer_stride = kwargs['rolling_buffer_stride']
		else:
			self.rolling_buffer_stride = 2							#  Default to 2.

		if 'temporal_buffer_length' in kwargs:						#  Were we given a temporal buffer length?
			assert isinstance(kwargs['temporal_buffer_length'], int) and kwargs['temporal_buffer_length'] > 0, \
			       'Argument \'temporal_buffer_length\' passed to Classifier must be an integer > 0.'
			self.temporal_buffer_length = kwargs['temporal_buffer_length']
		else:
			self.temporal_buffer_length = 3							#  Default to 3.

		if 'temporal_buffer_stride' in kwargs:						#  Were we given a temporal buffer stride?
			assert isinstance(kwargs['temporal_buffer_stride'], int) and kwargs['temporal_buffer_stride'] > 0, \
			       'Argument \'temporal_buffer_stride\' passed to Classifier must be an integer > 0.'
			self.temporal_buffer_stride = kwargs['temporal_buffer_stride']
		else:
			self.temporal_buffer_stride = 1							#  Default to 1.

		if 'db_file' in kwargs:										#  Were we given a database file?
			assert isinstance(kwargs['db_file'], str), 'Argument \'db_file\' passed to TemporalClassifier must be a string.'
			self.database_file = kwargs['db_file']
		else:
			self.database_file = None

		if 'inputs' in kwargs:										#  Were we given a list of enactments to use in simulated real time?
			assert isinstance(kwargs['inputs'], list), 'Argument \'inputs\' passed to TemporalClassifier must be a list of enactment name strings.'
			self.enactment_inputs = kwargs['inputs']
		else:
			self.enactment_inputs = []

		if 'relabel' in kwargs:										#  Were we given a relableing file?
			assert isinstance(kwargs['relabel'], str), 'Argument \'relabel\' passed to TemporalClassifier must be a string: the filepath for a relabeling file.'
			self.relabelings = {}
			fh = open(kwargs['relabel'], 'r')
			for line in fh.readlines():
				if line[0] != '#':
					arr = line.strip().split('\t')
					self.relabelings[arr[0]] = arr[1]
			fh.close()
		else:
			self.relabelings = {}									#  key: old label ==> val: new label

		if 'render_modes' in kwargs:								#  Were we given render modes?
			assert isinstance(kwargs['render_modes'], list), \
			       'Argument \'render_modes\' passed to TemporalClassifier must be a list of zero or more strings in {rolling-buffer, confidence, probabilities, smoothed}.'
			self.render_modes = kwargs['render_modes']
			self.render = True
		else:
			self.render_modes = []

		#############################################################
		#  Main attributes for this class.                          #
		#############################################################
		self.X_train = []											#  Contains lists of tuples (sequences of vectors).
		self.y_train = []											#  Contains strings (labels).
																	#  Becomes 'query'.
		self.rolling_buffer = [None for i in range(0, self.rolling_buffer_length)]
		self.rolling_buffer_filling = True							#  Buffer has yet to reach capacity
																	#  Holds probability distributions.
		self.temporal_buffer = [None for i in range(0, self.temporal_buffer_length)]
		self.temporal_buffer_filling = True							#  Buffer has yet to reach capacity
																	#  Holds buffer-fulls of ground-truth labels.
																	#  Used to determine whether a evaluation is "fair."
		self.buffer_labels = [None for i in range(0, self.rolling_buffer_length)]

		self.train_sample_lookup = {}								#  key: index into X_train ==> val: (source enactment, start time, start frame,
		self.vector_length = None									#                                                      end time,   end frame)
		self.width = None
		self.height = None

		#############################################################
		#  Attributes used for rendering and display.               #
		#############################################################
		self.seismograph_length = 10								#  Cache length.
		self.max_seismograph_linewidth = 15

		self.seismograph_enactment_name_super = {}					#  Where to locate and how to type the source of an annotated frame.
		self.seismograph_enactment_name_super['x'] = 10
		self.seismograph_enactment_name_super['y'] = 50
		self.seismograph_enactment_name_super['fontsize'] = 1.0

		self.seismograph_ground_truth_label_super = {}				#  Where to locate and how to type the label of an annotated frame.
		self.seismograph_ground_truth_label_super['x'] = 10
		self.seismograph_ground_truth_label_super['y'] = 90
		self.seismograph_ground_truth_label_super['fontsize'] = 1.0

		self.seismograph_prediction_super = {}						#  Where to locate and how to type the predicted label in an annotated frame.
		self.seismograph_prediction_super['x'] = 10
		self.seismograph_prediction_super['y'] = 130
		self.seismograph_prediction_super['fontsize'] = 1.0

		if self.database_file is not None:
			self.load_db(self.database_file)

	#################################################################
	#  Labels.                                                      #
	#################################################################

	#  Return a list of unique action labels as evidences by the snippet labels in 'y_train' and/or the test enactments.
	#  It is possible that the training set and test enactments contain different action labels.
	def labels(self, sets='both'):
		if sets == 'train':
			return list(np.unique(self.y_train))
		if sets == 'test':
			labels = {}												#  Used for its set-like properties.
			for enactment_inputs in self.enactment_inputs:			#  Include all (potentially unknown) labels from the input.
				pe = ProcessedEnactment(enactment_inputs, verbose=False)
				for x in pe.labels():
					if x in self.relabelings:						#  Allow that we may have asked to relabel some or all of these.
						labels[ self.relabelings[x] ] = True
					else:
						labels[x] = True
			return sorted([x for x in labels.keys()])				#  Return a sorted list of unique labels.

		labels = {}													#  Used for its set-like properties.
		for enactment_inputs in self.enactment_inputs:				#  Include all (potentially unknown) labels from the input.
			pe = ProcessedEnactment(enactment_inputs, verbose=False)
			for x in pe.labels():
				if x in self.relabelings:							#  Allow that we may have asked to relabel some or all of these.
					labels[ self.relabelings[x] ] = True
				else:
					labels[x] = True
		for label in list(np.unique(self.y_train)):					#  Also include the training set labels.
			labels[label] = True
		return sorted([x for x in labels.keys()])					#  Return a sorted list of unique labels.

	#  Return a list of unique action labels as evidences by the snippet labels in 'y_train' and/or the test enactments.
	#  It is possible that the training set and test enactments contain different action labels.
	def num_labels(self, sets='both'):
		labels = self.labels(sets)
		return len(labels)

	#################################################################
	#  Classify.                                                    #
	#################################################################

	#  March through each vector/frame in input enactments as if they were received in real time.
	#  Note that this is not our best approximation of a deployed system, since it avoids all the work
	#  of image processing/detection by network, centroid location, etc. As such, this method is best for
	#  testing the accuracy of the matching mechanism.
	#  And, yes, this is "simulated real time," but if we know we won't save the results of a test, then why perform that test?
	#  This is what the 'skip_unfair' parameter is for: check the accumulated ground-truth labels first BEFORE running DTW.
	#  Save time; run more tests.
	def classify(self, skip_unfair=True):
		assert isinstance(skip_unfair, bool), 'Argument \'skip_unfair\' passed to TemporalClassifier.classify() must be a Boolean.'

		classification_stats = {}
		classification_stats['_tests'] = []							#  key:_tests ==> val:[(prediction, ground-truth), (prediction, ground-truth), ... ]
		classification_stats['_conf'] = []							#  key:_conf  ==> val:[confidence, confidence, ... ]

		for label in self.labels('both'):							#  May include "unfair" labels, but will not include the "*" nothing-label.
			classification_stats[label] = {}						#  key:label ==> val:{key:tp      ==> val:true positive count
			classification_stats[label]['tp']      = 0				#                     key:fp      ==> val:false positive count
			classification_stats[label]['fp']      = 0				#                     key:fn      ==> val:false negative count
			classification_stats[label]['fn']      = 0				#                     key:support ==> val:instance in training set}
			classification_stats[label]['support'] = len([x for x in self.y_train if x == label])

		self.timing['total'] = 0									#  Prepare to capture time taken by the entire classification run.
		self.timing['load-enactment'] = []							#  Prepare to capture enactment loading times.
		self.timing['dtw-classification'] = []						#  This is a coarser grain: time each classification process.
		self.timing['test-cutoff-conditions'] = []					#  Prepare to collect times for calling test_cutoff_conditions().
		self.timing['dtw-R-call'] = []								#  Prepare to collect times for running R's DTW.
		self.timing['compute-confidence'] = []						#  Prepare to collect times for computing confidence scores.
		self.timing['isotonic-lookup'] = []							#  Prepare to collect times for bucket-search.
		self.timing['make-tentative-prediction'] = []				#  Prepare to collect least-distance-finding times.
		self.timing['sort-confidences'] = []						#  Prepare to collect confidence-score sorting times.
		self.timing['sort-probabilities'] = []						#  Prepare to collect probability sorting times.
		self.timing['push-temporal-buffer'] = []					#  Prepare to collect temporal-buffer update times.
		self.timing['temporal-smoothing'] = []						#  Prepare to collect temporal-smoothing runtimes.
		self.timing['make-temporally-smooth-decision'] = []			#  Prepare to collect final decision-making runtimes.
		if self.render:
			self.timing['render-annotated-source'] = []				#  Prepare to collect rendering times.
			self.timing['render-rolling-buffer'] = []
			self.timing['render-confidence'] = []
			self.timing['render-probabilities'] = []
			self.timing['render-smoothed-probabilities'] = []

		t0_start = time.process_time()								#  Start timer.
		for enactment_input in self.enactment_inputs:				#  Treat each input enactment as a separate slice of time.
																	#  (Re)initialize the rolling buffer.
			self.rolling_buffer = [None for i in range(0, self.rolling_buffer_length)]
			self.rolling_buffer_filling = True						#  Buffer has yet to reach capacity
																	#  (Re)initialize the temporal buffer.
			self.temporal_buffer = [None for i in range(0, self.temporal_buffer_length)]
			self.temporal_buffer_filling = True						#  Buffer has yet to reach capacity
																	#  (Re)initialize the ground-truth labels buffer.
			self.buffer_labels = [None for i in range(0, self.rolling_buffer_length)]

			if self.render:											#  If we're rendering, we may need to cache some things
				if 'confidence' in self.render_modes:				#  we do not need when simply making predictions.
					confidence_store = []
				if 'probabilities' in self.render_modes:
					probability_store = []
				if 'smoothed' in self.render_modes:
					smoothed_probability_store = []

			vector_buffer = []										#  Takes the place of X_test or an input stream.
			ground_truth_buffer = []								#  Can include "*" or "nothing" labels.
			time_stamp_buffer = []
			frame_path_buffer = []
			t1_start = time.process_time()							#  Start timer.
			fh = open(enactment_input + '.enactment', 'r')			#  Read in the input-enactment.
			lines = fh.readlines()
			fh.close()
			for line in lines:
				if line[0] != '#':
					arr = line.strip().split('\t')
					timestamp = float(arr[0])						#  Save the time stamp.
					frame_filename = arr[1]							#  Save the frame file path.
					ground_truth_label = arr[2]						#  Save the true label (these include the nothing-labels.)
					vector = [float(x) for x in arr[3:]]

					if self.hand_schema == 'strong-hand':			#  Apply hand-schema (if applicable.)
						vector = self.strong_hand_encode(vector)
																	#  Apply coefficients (if applicable.)
					vector_buffer.append( self.apply_vector_coefficients(vector) )

					if ground_truth_label in self.relabelings:
						ground_truth_buffer.append( self.relabelings[ground_truth_label] )
					else:
						ground_truth_buffer.append( ground_truth_label )

					time_stamp_buffer.append( timestamp )
					frame_path_buffer.append( frame_filename )
			t1_stop = time.process_time()							#  Stop timer.
			self.timing['load-enactment'].append(t1_stop - t1_start)

			num = len(vector_buffer)
			prev_ctr = 0
			max_ctr = os.get_terminal_size().columns - 7			#  Leave enough space for the brackets, space, and percentage.
			if self.verbose:
				if skip_unfair:
					print('>>> Classifying from "' + enactment_input + '" in simulated real time (skipping unfair tests.)')
				else:
					print('>>> Classifying from "' + enactment_input + '" in simulated real time.')

			tentative_prediction = None								#  Initially nothing.
			prediction = None

			if self.render:											#  Rendering? Create the video file now and add to it through the loop.
				pe = ProcessedEnactment(enactment_input, verbose=False)
				self.width = pe.width
				self.height = pe.height
				if 'rolling-buffer' in self.render_modes:
					vid_rolling_buffer = cv2.VideoWriter(enactment_input + '_seismograph.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
					                                     pe.fps, (pe.width * 2, pe.height) )
				if 'confidence' in self.render_modes:
					vid_confidence = cv2.VideoWriter(enactment_input + '_confidence_seismograph.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
					                                 pe.fps, (pe.width * 2, pe.height) )
				if 'probabilities' in self.render_modes:
					vid_probabilities = cv2.VideoWriter(enactment_input + '_probabilities_seismograph.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
					                                    pe.fps, (pe.width * 2, pe.height) )
				if 'smoothed' in self.render_modes:
					vid_smoothed_probabilities = cv2.VideoWriter(enactment_input + '_smoothed-probabilities_seismograph.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
					                                             pe.fps, (pe.width * 2, pe.height) )

			#########################################################
			#  Pseudo-boot-up is over. March through the buffer.    #
			#########################################################
			for frame_ctr in range(0, len(vector_buffer)):			#  March through vector_buffer.
				self.push_rolling( vector_buffer[frame_ctr] )		#  Push vector to rolling buffer.
																	#  Push G.T. label to rolling G.T. buffer.
				self.push_ground_truth_buffer( ground_truth_buffer[frame_ctr] )
																	#  Are the contents of the ground-truth buffer "fair"?
				fair = self.full() and self.uniform() and self.fair()

				#####################################################
				#  If the rolling buffer is full, then classify.    #
				#  This will always produce a tentative prediction  #
				#  based on nearest-neighbor. Actual predictions are#
				#  subject to threshold and smoothed probabilities. #
				#####################################################
				#if len(self.rolling_buffer) == self.rolling_buffer_length:
				if self.is_rolling_buffer_full():

					tentative_prediction = None						#  (Re)set.
					prediction = None

					#if True:
					if fair or not skip_unfair:
																	#  Call the parent class's core matching engine.
						matching_costs, confidences, probabilities, metadata, timing = super(TemporalClassifier, self).classify(self.rolling_buffer)
						self.timing['dtw-classification'] += timing['dtw-classification']
						self.timing['test-cutoff-conditions'] += timing['test-cutoff-conditions']
						self.timing['dtw-R-call'] += timing['dtw-R-call']
						self.timing['compute-confidence'] += timing['compute-confidence']
						self.timing['isotonic-lookup'] += timing['isotonic-lookup']

					#################################################
					#  matching_costs: key: label ==> val: cost     #
					#  confidences:    key: label ==> val: score    #
					#  probabilities:  key: label ==> val: prob.    #
					#  metadata:       key: label ==>               #
					#                    val: {query-indices,       #
					#                          template-indices,    #
					#                          db-index}            #
					#################################################
					#if True:
					if fair or not skip_unfair:
						t1_start = time.process_time()				#  Start timer.
						least_cost = float('inf')					#  Tentative prediction always determined by least matching cost.
						for k, v in matching_costs.items():
							if v < least_cost:
								least_cost = v
								tentative_prediction = k
								tentative_confidence = confidences[k]
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['make-tentative-prediction'].append(t1_stop - t1_start)

						t1_start = time.process_time()				#  Start timer.
						sorted_confidences = []
						for label in self.labels('train'):			#  Maintain the order of label scores.
							sorted_confidences.append( confidences[label] )
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['sort-confidences'].append(t1_stop - t1_start)

						t1_start = time.process_time()				#  Start timer.
						sorted_probabilities = []
						for label in self.labels('train'):			#  Maintain the order of label scores.
							sorted_probabilities.append( probabilities[label] )
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['sort-probabilities'].append(t1_stop - t1_start)

						t1_start = time.process_time()				#  Start timer.
						self.push_temporal( sorted_probabilities )	#  Add this probability distribution to the temporal buffer.
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['push-temporal-buffer'].append(t1_stop - t1_start)

						if self.render and 'confidence' in self.render_modes:
							confidence_store = self.push_buffer(sorted_confidences, confidence_store)

						if self.render and 'probabilities' in self.render_modes:
							probability_store = self.push_buffer(sorted_probabilities, probability_store)

				#####################################################
				#  Smooth the contents of the temporal buffer and   #
				#  make a prediction (or abstain from predicting).  #
				#####################################################
				#if len(self.temporal_buffer) == self.temporal_buffer_length:
				#if True:
				if fair or not skip_unfair:
					t1_start = time.process_time()					#  Start timer.
					smoothed_probabilities = list(np.mean(np.array([x for x in self.temporal_buffer if x is not None]), axis=0))
					t1_stop = time.process_time()					#  Stop timer.
					self.timing['temporal-smoothing'].append(t1_stop - t1_start)

					if self.render and 'smooth' in self.render_modes:
						smoothed_probability_store = self.push_buffer(smoothed_probabilities, smoothed_probability_store)

					t1_start = time.process_time()					#  Start timer.
					if smoothed_probabilities[ self.labels('train').index(tentative_prediction) ] > self.threshold:
						prediction = tentative_prediction
					else:
						prediction = None
					t1_stop = time.process_time()					#  Stop timer.
					self.timing['make-temporally-smooth-decision'].append(t1_stop - t1_start)
																	#  Only measure performance when conditions are fair.
				if self.full() and self.uniform() and self.fair():
					ground_truth_label = self.buffer_labels[0]
					if prediction == ground_truth_label:
						classification_stats[ground_truth_label]['tp'] += 1
					elif prediction is not None:
						classification_stats[prediction]['fp']  += 1
						classification_stats[ground_truth_label]['fn'] += 1

					classification_stats['_tests'].append( (prediction, ground_truth_label) )
					classification_stats['_conf'].append( tentative_confidence )

				#####################################################
				#  Rendering?                                       #
				#####################################################
				if self.render:
																	#  Whatever you are rendering, you will need the reference source frame.
					t1_start = time.process_time()					#  Start timer.
					if prediction is not None:
						annotated_prediction = prediction
					else:
						annotated_prediction = ''
					annotated_source_frame = self.render_annotated_source_frame(enactment_input, frame_path_buffer[frame_ctr],     \
					                                                            time_stamp=time_stamp_buffer[frame_ctr],           \
					                                                            ground_truth_label=ground_truth_buffer[frame_ctr], \
					                                                            prediction=annotated_prediction)
					t1_stop = time.process_time()					#  Stop timer.
					self.timing['render-annotated-source'].append(t1_stop - t1_start)

					if 'rolling-buffer' in self.render_modes:
						t1_start = time.process_time()				#  Stop timer.
						graph = self.render_rolling_buffer_seismograph()
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['render-rolling-buffer'].append(t1_stop - t1_start)
						concat_frame = np.zeros((self.height, self.width * 2, 3), dtype='uint8')
						concat_frame[:, :self.width, :] = annotated_source_frame[:, :, :]
						concat_frame[:, self.width:, :] = graph[:, :, :]
						vid_rolling_buffer.write(concat_frame)

					if 'confidence' in self.render_modes:
						t1_start = time.process_time()				#  Stop timer.
						graph = self.render_confidence_seismograph(confidence_store)
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['render-confidence'].append(t1_stop - t1_start)
						concat_frame = np.zeros((self.height, self.width * 2, 3), dtype='uint8')
						concat_frame[:, :self.width, :] = annotated_source_frame[:, :, :]
						concat_frame[:, self.width:, :] = graph[:, :, :]
						vid_confidence.write(concat_frame)

					if 'probabilities' in self.render_modes:
						t1_start = time.process_time()				#  Stop timer.
						graph = self.render_probabilities_seismograph(probability_store)
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['render-probabilities'].append(t1_stop - t1_start)
						concat_frame = np.zeros((self.height, self.width * 2, 3), dtype='uint8')
						concat_frame[:, :self.width, :] = annotated_source_frame[:, :, :]
						concat_frame[:, self.width:, :] = graph[:, :, :]
						vid_probabilities.write(concat_frame)

					if 'smoothed' in self.render_modes:
						t1_start = time.process_time()				#  Stop timer.
						graph = self.render_smoothed_probabilities_seismograph(smoothed_probability_store)
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['render-smoothed-probabilities'].append(t1_stop - t1_start)
						concat_frame = np.zeros((self.height, self.width * 2, 3), dtype='uint8')
						concat_frame[:, :self.width, :] = annotated_source_frame[:, :, :]
						concat_frame[:, self.width:, :] = graph[:, :, :]
						vid_smoothed_probabilities.write(concat_frame)

				if self.verbose:									#  Progress bar.
					if int(round(float(frame_ctr) / float(num - 1) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
						prev_ctr = int(round(float(frame_ctr) / float(num - 1) * float(max_ctr)))
						sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(frame_ctr) / float(num - 1) * 100.0))) + '%]')
						sys.stdout.flush()

			if self.render:											#  Rendering? Close and save the newly created video.
				if 'rolling-buffer' in self.render_modes:
					vid_rolling_buffer.release()
				if 'confidence' in self.render_modes:
					vid_confidence.release()
				if 'probabilities' in self.render_modes:
					vid_probabilities.release()
				if 'smoothed' in self.render_modes:
					vid_smoothed_probabilities.release()
			if self.verbose:
				print('')
		t0_stop = time.process_time()
		self.timing['total'] = t0_stop - t0_start
		return classification_stats

	#################################################################
	#  Buffer update.                                               #
	#################################################################

	#  Add the given vector to the rolling buffer, kicking out old vectors if necessary.
	def push_rolling(self, vector):
		if self.rolling_buffer_filling:
			i = 0
			while i < self.rolling_buffer_length and self.rolling_buffer[i] is not None:
				i += 1
			self.rolling_buffer[i] = vector[:]
			if self.is_rolling_buffer_full():
				self.rolling_buffer_filling = False
		else:
			if None not in self.rolling_buffer:
				self.rolling_buffer = self.rolling_buffer[self.rolling_buffer_stride:] + [None for i in range(0, self.rolling_buffer_stride)]
			self.rolling_buffer[ self.rolling_buffer.index(None) ] = vector[:]
		return

	#  Add the given ground-truth label to the ground-truth buffer, kicking out old labels if necessary.
	def push_ground_truth_buffer(self, label):
		if self.rolling_buffer_filling:
			i = 0
			while i < self.rolling_buffer_length and self.buffer_labels[i] is not None:
				i += 1
			self.buffer_labels[i] = label[:]
		else:
			if None not in self.buffer_labels:
				self.buffer_labels = self.buffer_labels[self.rolling_buffer_stride:] + [None for i in range(0, self.rolling_buffer_stride)]
			self.buffer_labels[ self.buffer_labels.index(None) ] = label[:]
		return

	#  Add the given probability distribution to the temporal buffer, kicking out old distributions if necessary.
	def push_temporal(self, distribution):
		if self.temporal_buffer_filling:
			i = 0
			while i < self.temporal_buffer_length and self.temporal_buffer[i] is not None:
				i += 1
			self.temporal_buffer[i] = distribution[:]
			if self.is_temporal_buffer_full():
				self.temporal_buffer_filling = False
		else:
			if None not in self.temporal_buffer:
				self.temporal_buffer = self.temporal_buffer[self.temporal_buffer_stride:] + [None for i in range(0, self.temporal_buffer_stride)]
			self.temporal_buffer[ self.temporal_buffer.index(None) ] = distribution[:]
		return

	def is_rolling_buffer_full(self):
		return None not in self.rolling_buffer

	def is_temporal_buffer_full(self):
		return None not in self.temporal_buffer

	#  A more generic function used by the caches for rendering.
	#  (This assumes the rendering caches use a stride of 1.)
	#  (Also, unlike the above push operations, this one has to return an updated buffer.)
	def push_buffer(self, vector, buffer):
		if len(buffer) == self.seismograph_length:					#  If the buffer is at its maximum length, then shift 'stride' elements out.
			buffer = buffer[1:]
		buffer.append( vector )										#  Append the latest.
		return buffer

	#################################################################
	#  Buffer testing.                                              #
	#################################################################

	#  Is self.buffer_labels currently full?
	def full(self):
		return None not in self.buffer_labels

	#  Does self.buffer_labels currently the same label?
	def uniform(self):
		return self.buffer_labels.count(self.buffer_labels[0]) == self.rolling_buffer_length

	#  Is self.buffer_labels free of any labels that are not in the database?
	def fair(self):
		valid_labels = self.labels('train')
		return all([x in valid_labels for x in self.buffer_labels])

	#################################################################
	#  Profiling.                                                   #
	#################################################################

	#  How many snippets will we encounter as we march through time with the current rolling buffer size and stride?
	def itemize(self, skip_unfair=True):
		maxindexlen = 0
		maxlabellen = 0
		total_snippets = 0											#  Across all enactments.
		labels = self.labels('train')

		for enactment_input in self.enactment_inputs:
			pe = ProcessedEnactment(enactment_input, verbose=False)
			snippets = pe.snippets_from_frames(self.rolling_buffer_length, self.rolling_buffer_stride)
			for snippet in snippets:
				label = snippet[0]
				if label in self.relabelings:						#  Look up.
					label = self.relabelings[label]
																	#  Just because something has been relabeled does not mean that the DB knows it.
				if (skip_unfair and label in labels) or not skip_unfair:
					total_snippets += 1
					if len(label) > maxlabellen:
						maxlabellen = len(label)
		maxindexlen = len(str(total_snippets))

		for enactment_input in self.enactment_inputs:
			pe = ProcessedEnactment(enactment_input, verbose=False)
			print('>>> "' + enactment_input + '" in simulated real-time:')
			ctr = 0
			for snippet in pe.snippets_from_frames(self.rolling_buffer_length, self.rolling_buffer_stride):
				label = snippet[0]
				if label in self.relabelings:
					label = self.relabelings[label]

				if (skip_unfair and label in labels) or not skip_unfair:
					print_str = '    [' + str(ctr) + ']:' + ' '*(maxindexlen - len(str(ctr))) + \
					                 label + ' '*(maxlabellen - len(label)) + '\t' + \
					                 str(snippet[1]) + '\t-->\t' + str(snippet[3])
					print(print_str)
					ctr += 1;
			print('')

		if skip_unfair:
			print('>>> Total (fair) snippets in test set: ' + str(total_snippets))
		else:
			print('>>> Total snippets in test set: ' + str(total_snippets))
		return

	def itemize_snippets(self):
		maxlabellen = 0
		maxcountlen = 0
		training_counts = {}
		testing_counts = {}

		X_test = []													#  These are perishable, local variables.
		y_test = []													#  We hold snippets and labels here *as if* they did not exist in time.

		for label in list(np.unique(self.y_train)):
			if len(label) > maxlabellen:
				maxlabellen = len(label)
			training_counts[label] = len([x for x in self.y_train if x == label])
			if len(str(training_counts[label])) > maxcountlen:
				maxcountlen = len(str(training_counts[label]))

		for enactment in self.enactment_inputs:
			pe = ProcessedEnactment(enactment, verbose=False)
			snippets = pe.snippets_from_frames(self.rolling_buffer_length, self.rolling_buffer_stride)
			for snippet in snippets:
				label = snippet[0]
				if label in self.relabelings:
					label = self.relabelings[label]
				X_test.append( (snippet[1], snippet[2], snippet[3], snippet[4]) )
				y_test.append( label )

				if label not in testing_counts:
					testing_counts[label] = 0
				testing_counts[label] += 1
				if len(label) > maxlabellen:
					maxlabellen = len(label)
				if len(str(testing_counts[label])) > maxcountlen:
					maxcountlen = len(str(testing_counts[label]))

		num_fair_labels = len([x for x in testing_counts.keys() if x in training_counts])
		num_fair_snippets = len([x for x in y_test if x in training_counts])

		if len(self.X_train) > 0:
			print('    Training set: ' + str(len(self.labels('train'))) + ' unique labels, ' + str(len(self.X_train)) + ' total ' + str(self.rolling_buffer_length) + '-frame snippets: ' + \
			           "{:.2f}".format(float(len(self.X_train)) / float(len(self.X_train) + len(X_test)) * 100.0) + ' %.')

			for label in self.labels('train'):
				print('        ' + label + ': ' + ' '*(maxlabellen - len(label)) + \
				                   str(training_counts[label]) + ' '*(maxcountlen - len(str(training_counts[label]))) + \
				   ' snippets, ' + "{:.3f}".format(float(training_counts[label]) / float(len(self.y_train)) * 100.0) + ' % of train.')

		if len(X_test) > 0:
			print('    Test set:     ' + str(len(self.labels('test'))) + ' unique labels, ' + str(len(X_test))  + ' total ' + str(self.rolling_buffer_length) + '-frame snippets: ' + \
			           "{:.2f}".format(float(len(X_test)) / float(len(self.X_train) + len(X_test)) * 100.0) + ' %.')
			print('                  ' + str(num_fair_labels) + ' unique fair labels, ' + str(num_fair_snippets) + ' total fair ' + str(self.rolling_buffer_length) + '-frame snippets: ' + \
			           "{:.2f}".format(float(num_fair_snippets) / float(len(self.X_train) + len(X_test)) * 100.0) + ' %.')

			for label in sorted(list(np.unique(y_test))):
				if label not in list(np.unique(self.y_train)):			#  Let users know that a label in the test set is not in the training set.
					print('     !! ' + label + ': ' + ' '*(maxlabellen - len(label)) + \
					                   str(testing_counts[label]) + ' '*(maxcountlen - len(str(testing_counts[label]))) + \
					   ' snippets, ' + "{:.3f}".format(float(testing_counts[label]) / float(len(y_test)) * 100.0) + ' % of test.')
				else:
					print('        ' + label + ': ' + ' '*(maxlabellen - len(label)) + \
					                   str(testing_counts[label]) + ' '*(maxcountlen - len(str(testing_counts[label]))) + \
					   ' snippets, ' + "{:.3f}".format(float(testing_counts[label]) / float(len(y_test)) * 100.0) + ' % of test.')
		return

	#  Print out every input-enactment's timestamps and labels, just to check what is currently on hand.
	def preview(self):
		for enactment_input in self.enactment_inputs:
			fh = open(enactment_input + '.enactment', 'r')			#  Read in the input-enactment.
			lines = fh.readlines()
			fh.close()

			for line in lines:
				if line[0] != '#':
					arr = line.strip().split('\t')
					timestamp = float(arr[0])						#  Save the time stamp.
					ground_truth_label = arr[2]						#  Save the true label (these include the nothing-labels.)
					if ground_truth_label in self.relabelings:
						ground_truth_label = self.relabelings[ground_truth_label]

					print(enactment_input + ':\t' + str(timestamp) + '\t' + ground_truth_label)
		return

	#################################################################
	#  Rendering.                                                   #
	#################################################################

	#  Take the given frame name, superimpose object masks, centroids, other data.
	def render_annotated_source_frame(self, enactment, file_name, **kwargs):
		vid_frame = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
		masks = self.get_masks_for_frame(enactment, file_name)
		maskcanvas = np.zeros((self.height, self.width, 3), dtype='uint8')
		for mask in masks:											#  Each is (mask file name, recog-object, bounding-box).
			if mask[1] not in ['LeftHand', 'RightHand']:
				mask_img = cv2.imread(mask[0], cv2.IMREAD_UNCHANGED)
				mask_img[mask_img > 1] = 1							#  All values greater than 1 become 1.
																	#  Extrude to three channels.
				mask_img = mask_img[:, :, None] * np.ones(3, dtype='uint8')[None, None, :]
																	#  Convert this to a graphical overlay:
				mask_img[:, :, 0] *= self.robject_colors[ mask[1] ][2]
				mask_img[:, :, 1] *= self.robject_colors[ mask[1] ][1]
				mask_img[:, :, 2] *= self.robject_colors[ mask[1] ][0]

				maskcanvas += mask_img								#  Add mask to mask accumulator.
				maskcanvas[maskcanvas > 255] = 255					#  Clip accumulator to 255.
																	#  Add mask accumulator to source frame.
		vid_frame = cv2.addWeighted(vid_frame, 1.0, maskcanvas, 0.7, 0)
																	#  Flatten alpha.
		vid_frame = cv2.cvtColor(vid_frame, cv2.COLOR_RGBA2RGB)

		for mask in masks:											#  Second pass: draw bounding boxes and 2D centroids.
			if mask[1] not in ['LeftHand', 'RightHand']:
				bbox = mask[2]
				cv2.rectangle(vid_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (self.robject_colors[ mask[1] ][2], \
				                                                                  self.robject_colors[ mask[1] ][1], \
				                                                                  self.robject_colors[ mask[1] ][0]), 1)
				cv2.circle(vid_frame, (int(round(np.mean([bbox[0], bbox[2]]))), int(round(np.mean([bbox[1], bbox[3]])))), \
				           5, (self.robject_colors[ mask[1] ][2], \
				               self.robject_colors[ mask[1] ][1], \
				               self.robject_colors[ mask[1] ][0]), 3)

		if 'time_stamp' in kwargs:
			assert isinstance(kwargs['time_stamp'], float), \
			       'Argument \'time_stamp\' passed to TemporalClassifier.render_annotated_source_frame() must be a float.'
			time_stamp_str = str(kwargs['time_stamp'])
		else:
			time_stamp_str = ''

		cv2.putText(vid_frame, enactment + ', ' + time_stamp_str, (self.seismograph_enactment_name_super['x'], self.seismograph_enactment_name_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.seismograph_enactment_name_super['fontsize'], (255, 255, 255, 255), 3)

		if 'ground_truth_label' in kwargs:
			assert isinstance(kwargs['ground_truth_label'], str), \
			       'Argument \'ground_truth_label\' passed to TemporalClassifier.render_annotated_source_frame() must be a string.'
																	#  Bright green for truth!
			cv2.putText(vid_frame, kwargs['ground_truth_label'], (self.seismograph_ground_truth_label_super['x'], self.seismograph_ground_truth_label_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.seismograph_ground_truth_label_super['fontsize'], (0, 255, 0, 255), 3)

		if 'prediction' in kwargs:
			assert isinstance(kwargs['prediction'], str), \
			       'Argument \'prediction\' passed to TemporalClassifier.render_annotated_source_frame() must be a string.'
			cv2.putText(vid_frame, kwargs['prediction'], (self.seismograph_prediction_super['x'], self.seismograph_prediction_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.seismograph_prediction_super['fontsize'], (255, 255, 255, 255), 3)

		return vid_frame

	#  Create a seismograph-like plot of the rolling buffer's components.
	def render_rolling_buffer_seismograph(self):
																	#  All-white RGB
		graph = np.ones((self.height, self.width, 3), dtype='uint8') * 255

		buffer_len = self.rolling_buffer_length
		vector_len = self.vector_length
		img_center = (int(round(float(self.width) * 0.5)), int(round(float(self.height) * 0.5)))

																	#  Build a (buffer_len) X (vector_len) matrix:
																	#  [LHx, LHy, LHz, LH0, LH1, LH2, RHx, ..., num_objects] in frame 0
																	#  [LHx, LHy, LHz, LH0, LH1, LH2, RHx, ..., num_objects] in frame 1
		B = []														#  [LHx, LHy, LHz, LH0, LH1, LH2, RHx, ..., num_objects] in frame 2
		if self.rolling_buffer_filling:								#  Buffer has yet to reach capacity
			for i in range(len([x for x in self.rolling_buffer if x is not None]), self.rolling_buffer_length):
				B.append( [0.0 for x in range(0, vector_len)] )
			for i in range(0, self.rolling_buffer.index(None)):
				B.append( [x for x in self.undo_vector_coefficients(self.rolling_buffer[i])] )
		else:														#  Buffer reached capacity at some point but may contain Nones left from stride
			for i in range(0, len(self.rolling_buffer)):
				if self.rolling_buffer[i] is not None:
					B.append( [x for x in self.undo_vector_coefficients(self.rolling_buffer[i])] )
				else:
					B.append( [0.0 for x in range(0, vector_len)] )
		B = np.array(B)

		x_intervals = int(np.floor(float(self.width) / float(vector_len)))
		y_intervals = int(np.floor(float(self.height) / float(buffer_len)))
		nudge_right = 10

		for y in range(0, buffer_len):
			color_ctr = 2

			for x in range(0, vector_len):
				point_a = (x * x_intervals + nudge_right,  y      * y_intervals)
				point_b = (x * x_intervals + nudge_right, (y + 1) * y_intervals)

				if x >= 0 and x < 6:								#  Always green for left hand/strong hand
					color = (0, 255, 0)
				elif x >= 6 and x < 12:								#  Always red for right hand/weak hand
					color = (0, 0, 255)
				else:
					if color_ctr == 0:
						color = (0, 0, 255)
					elif color_ctr == 1:
						color = (0, 255, 0)
					else:
						color = (255, 0, 0)

				graph = cv2.line(graph, point_a, point_b, color, max(1, int(round(float(self.max_seismograph_linewidth) * B[y, x]))) )

				if x >= 12:
					tmp = np.zeros((self.height, self.width, 3), dtype='uint8')
					pt = (x * x_intervals, self.height - 10)
					cv2.putText(tmp, self.recognizable_objects[x - 12], img_center, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
					tmp = self.rotate_text(tmp, 90.0, (x * x_intervals, self.height - 10))
					indices = np.where(tmp > 0)
					for i in range(0, len(indices[0])):
						graph[indices[0][i], indices[1][i]] = tmp[indices[0][i], indices[1][i]]

					color_ctr += 1
					if color_ctr == 3:
						color_ctr = 0
		return graph

	#  Create a seismograph-like plot of confidence scores.
	#  Receives 'confidence_accumulator', a buffer of 'self.seismograph_length' confidence vectors for all recognizable actions.
	def render_confidence_seismograph(self, confidence_accumulator):
																	#  All-white RGB
		graph = np.ones((self.height, self.width, 3), dtype='uint8') * 255

		labels = self.labels('train')
		buffer_len = self.seismograph_length
		vector_len = len(labels)
		img_center = (int(round(float(self.width) * 0.5)), int(round(float(self.height) * 0.5)))
																	#  Build a (buffer_len) X (length of labels) matrix:
																	#  [class0, class1, class2, ..., classn] in frame 0
																	#  [class0, class1, class2, ..., classn] in frame 1
		B = []														#  [class0, class1, class2, ..., classn] in frame 2
		if len(confidence_accumulator) < buffer_len:
			for i in range(0, buffer_len - len(confidence_accumulator)):
				B.append( [0.0 for x in range(0, vector_len)] )
		for i in range(0, len(confidence_accumulator)):
			B.append( [x for x in confidence_accumulator[i]] )
		B = np.array(B)

		x_intervals = int(np.floor(float(self.width) / float(vector_len)))
		y_intervals = int(np.floor(float(self.height) / float(buffer_len)))
		nudge_right = 30
		label_margin = 5

		for y in range(0, buffer_len):
			color_ctr = 0

			for x in range(0, vector_len):
				point_a = (x * x_intervals + nudge_right,  y      * y_intervals)
				point_b = (x * x_intervals + nudge_right, (y + 1) * y_intervals)

				if color_ctr == 0:
					color = (0, 0, 255)
				elif color_ctr == 1:
					color = (0, 255, 0)
				else:
					color = (255, 0, 0)

				graph = cv2.line(graph, point_a, point_b, color, max(1, int(round(float(self.max_seismograph_linewidth) * B[y, x]))) )

				tmp = np.zeros((self.height, self.width, 3), dtype='uint8')
				pt = (x * x_intervals, self.height - 10)
				cv2.putText(tmp, labels[x], img_center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
				tmp = self.rotate_text(tmp, 90.0, (x * x_intervals + nudge_right - label_margin, self.height - 10))
				indices = np.where(tmp > 0)
				for i in range(0, len(indices[0])):
					graph[indices[0][i], indices[1][i]] = tmp[indices[0][i], indices[1][i]]

				color_ctr += 1
				if color_ctr == 3:
					color_ctr = 0

		return graph

	#  Create a seismograph-like plot of probabilities.
	#  Receives 'probability_accumulator', a buffer of 'self.seismograph_length' probability distributions for all recognizable actions.
	def render_probabilities_seismograph(self, probability_accumulator):
																	#  All-white RGB
		graph = np.ones((self.height, self.width, 3), dtype='uint8') * 255

		labels = self.labels('train')
		buffer_len = self.seismograph_length
		vector_len = len(labels)
		img_center = (int(round(float(self.width) * 0.5)), int(round(float(self.height) * 0.5)))
																	#  Build a (buffer_len) X (length of labels) matrix:
																	#  [class0, class1, class2, ..., classn] in frame 0
																	#  [class0, class1, class2, ..., classn] in frame 1
		B = []														#  [class0, class1, class2, ..., classn] in frame 2
		if len(probability_accumulator) < buffer_len:
			for i in range(0, buffer_len - len(probability_accumulator)):
				B.append( [0.0 for x in range(0, vector_len)] )
		for i in range(0, len(probability_accumulator)):
			B.append( [x for x in probability_accumulator[i]] )
		B = np.array(B)

		x_intervals = int(np.floor(float(self.width) / float(vector_len)))
		y_intervals = int(np.floor(float(self.height) / float(buffer_len)))
		nudge_right = 30
		label_margin = 5

		for y in range(0, buffer_len):
			color_ctr = 0

			for x in range(0, vector_len):
				point_a = (x * x_intervals + nudge_right,  y      * y_intervals)
				point_b = (x * x_intervals + nudge_right, (y + 1) * y_intervals)

				if color_ctr == 0:
					color = (0, 0, 255)
				elif color_ctr == 1:
					color = (0, 255, 0)
				else:
					color = (255, 0, 0)

				graph = cv2.line(graph, point_a, point_b, color, max(1, int(round(float(self.max_seismograph_linewidth) * B[y, x]))) )

				tmp = np.zeros((self.height, self.width, 3), dtype='uint8')
				pt = (x * x_intervals, self.height - 10)
				cv2.putText(tmp, labels[x], img_center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
				tmp = self.rotate_text(tmp, 90.0, (x * x_intervals + nudge_right - label_margin, self.height - 10))
				indices = np.where(tmp > 0)
				for i in range(0, len(indices[0])):
					graph[indices[0][i], indices[1][i]] = tmp[indices[0][i], indices[1][i]]

				color_ctr += 1
				if color_ctr == 3:
					color_ctr = 0

		return graph

	#  Create a seismograph-like plot of smoothed probabilities.
	#  Receives 'smoothed_probability_accumulator', a buffer of 'self.seismograph_length' probability distributions for all recognizable actions.
	def render_smoothed_probabilities_seismograph(self, smoothed_probability_accumulator):
																	#  All-white RGB
		graph = np.ones((self.height, self.width, 3), dtype='uint8') * 255

		labels = self.labels('train')
		buffer_len = self.seismograph_length
		vector_len = len(labels)
		img_center = (int(round(float(self.width) * 0.5)), int(round(float(self.height) * 0.5)))
																	#  Build a (buffer_len) X (length of labels) matrix:
																	#  [class0, class1, class2, ..., classn] in frame 0
																	#  [class0, class1, class2, ..., classn] in frame 1
		B = []														#  [class0, class1, class2, ..., classn] in frame 2
		if len(smoothed_probability_accumulator) < buffer_len:
			for i in range(0, buffer_len - len(smoothed_probability_accumulator)):
				B.append( [0.0 for x in range(0, vector_len)] )
		for i in range(0, len(smoothed_probability_accumulator)):
			B.append( [x for x in smoothed_probability_accumulator[i]] )
		B = np.array(B)

		x_intervals = int(np.floor(float(self.width) / float(vector_len)))
		y_intervals = int(np.floor(float(self.height) / float(buffer_len)))
		nudge_right = 30
		label_margin = 5

		for y in range(0, buffer_len):
			color_ctr = 0

			for x in range(0, vector_len):
				point_a = (x * x_intervals + nudge_right,  y      * y_intervals)
				point_b = (x * x_intervals + nudge_right, (y + 1) * y_intervals)

				if color_ctr == 0:
					color = (0, 0, 255)
				elif color_ctr == 1:
					color = (0, 255, 0)
				else:
					color = (255, 0, 0)

				graph = cv2.line(graph, point_a, point_b, color, max(1, int(round(float(self.max_seismograph_linewidth) * B[y, x]))) )

				tmp = np.zeros((self.height, self.width, 3), dtype='uint8')
				pt = (x * x_intervals, self.height - 10)
				cv2.putText(tmp, labels[x], img_center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
				tmp = self.rotate_text(tmp, 90.0, (x * x_intervals + nudge_right - label_margin, self.height - 10))
				indices = np.where(tmp > 0)
				for i in range(0, len(indices[0])):
					graph[indices[0][i], indices[1][i]] = tmp[indices[0][i], indices[1][i]]

				color_ctr += 1
				if color_ctr == 3:
					color_ctr = 0

		return graph

	def rotate_text(self, src, angle, true_point):
		center = (int(round(float(self.width) * 0.5)), int(round(float(self.height) * 0.5)))
		img = src[:, :, :]

		rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
		img = cv2.warpAffine(img, rotMat, (self.width, self.height))

		transMat = np.array([[1.0, 0.0, true_point[0] - center[0]], \
		                     [0.0, 1.0, true_point[1] - center[1]]])
		img = cv2.warpAffine(img, transMat, (self.width, self.height))

		return img

	#  Return a list of tuples: [ (mask file name, recog-object, bounding-box), (mask file name, recog-object, bounding-box), ... ].
	#  One tuple for each recognizable object in the given enactment, at the given frame.
	def get_masks_for_frame(self, enactment, frame_path):
		masks = []
		fh = open(enactment + '_props.txt', 'r')
		for line in fh.readlines():
			if line[0] != '#':
				arr = line.strip().split('\t')
				file_path = arr[1]
				r_object = arr[3]
				bbox_str = arr[6]
				mask_path = arr[7]
				if file_path == frame_path:
					bbox_arr = bbox_str.split(';')
					bbox = tuple([int(x) for x in bbox_arr[0].split(',')] + [int(x) for x in bbox_arr[1].split(',')])
					masks.append( (mask_path, r_object, bbox) )
		fh.close()
		return masks
