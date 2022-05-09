import numpy as np
import os
import sys

'''
Similar to the Enactment class but completely separated from the raw materials and file structure.
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
																	#  Note that we avoid using the enactment.Action class because these are
																	#  not supposed to be mutable after processing.
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

				if 'GAUSSIAN' in line:								#  Read this enactment's Gaussian3D parameters.
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

	#  Return the length in frames of the index-th action.
	def action_len(self, index):
		video_frames = [y[1]['file'] for y in sorted([x for x in self.frames.items()], key=lambda x: x[0])]
		return video_frames.index(self.actions[index][4]) - video_frames.index(self.actions[index][2])

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
																	#  Action = (label, start-time, start-frame, end-time, end-frame).
		for index in indices:
																	#  Get a list of all frame indices for this action.
																	#  (The +1 at the end ensures that we take the last snippet.)
			frame_indices = range(video_frames.index(self.actions[index][2]), video_frames.index(self.actions[index][4]) + 1)

			if window_length > len(frame_indices):					#  CAUTION!!!  This allows you to take a snippet that extends past the labeled frames!!
																	#  You'll include some frames of the adjacent label!!!
				snippet_actions.append( (self.actions[index][0],                           \
				                         time_stamps[  frame_indices[0]                 ], \
				                         video_frames[ frame_indices[0]                 ], \
				                         time_stamps[  frame_indices[0] + window_length ], \
				                         video_frames[ frame_indices[0] + window_length ]) )
			else:
				for i in range(0, len(frame_indices) - window_length, stride):
					snippet_actions.append( (self.actions[index][0],                           \
					                         time_stamps[  frame_indices[i]                 ], \
					                         video_frames[ frame_indices[i]                 ], \
					                         time_stamps[  frame_indices[i + window_length] ], \
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
