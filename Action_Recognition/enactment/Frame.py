import cv2
import matplotlib.pyplot as plt
import numpy as np


'''
Frames contain information about the files (normal, depth, colormap), visible objects, and poses at a given time-stamp.
'''
class Frame():
	def __init__(self, **kwargs):
		self.time_stamp = None										#  Float
		self.file_name = None										#  Name of the file alone; no path.
		self.path = None											#  This should leave off the last directory. e.g. Enactment1/Users/vr1/POV/
																	#  We want to use this attribute to refer to either the normal-view, color-map, or depth-map.
		self.ground_truth_label = None								#  None = no label != "*" = no action
		self.width = None											#  Image width
		self.height = None											#  Image height

		self.head_pose = None										#  x, y, z, RotationMatrix
		self.left_hand_pose = None									#  x, y, z, state
		self.right_hand_pose = None									#  x, y, z, state
		self.left_hand_global_pose = None							#  x, y, z, state
		self.right_hand_global_pose = None							#  x, y, z, state

		self.detections = []										#  List of RecognizableObjects

		if 'time_stamp' in kwargs:
			assert isinstance(kwargs['time_stamp'], float), 'Argument \'time_stamp\' passed to Frame must be a float.'
			self.time_stamp = kwargs['time_stamp']

		if 'file_name' in kwargs:
			assert isinstance(kwargs['file_name'], str), 'Argument \'file_name\' passed to Frame must be a string.'
			self.file_name = kwargs['file_name']

		if 'path' in kwargs:
			assert isinstance(kwargs['path'], str), 'Argument \'path\' passed to Frame must be a string.'
			self.path = kwargs['path']

		if 'ground_truth_label' in kwargs:
			assert isinstance(kwargs['ground_truth_label'], str), 'Argument \'ground_truth_label\' passed to Frame must be a string.'
			self.ground_truth_label = kwargs['ground_truth_label']

		if 'width' in kwargs:
			assert isinstance(kwargs['width'], int) and kwargs['width'] > 0, 'Argument \'width\' passed to Frame must be an integer > 0.'
			self.width = kwargs['width']

		if 'height' in kwargs:
			assert isinstance(kwargs['height'], int) and kwargs['height'] > 0, 'Argument \'height\' passed to Frame must be an integer > 0.'
			self.height = kwargs['height']

	def fullpath(self, mode='video'):
		assert isinstance(mode, str) and (mode == 'video' or mode == 'color' or mode == 'depth'), 'The argument \'mode\' in Frame.fullpath() must be a string in {video, color, depth}.'
		if mode == 'color':
			return self.path + '/ColorMapCameraFrames/' + self.file_name
		if mode == 'depth':
			return self.path + '/DepthMapCameraFrames/' + self.file_name
		return self.path + '/NormalViewCameraFrames/' + self.file_name

	#  Here 'pose' is (x, y, z, RotMat)
	def set_head_pose(self, pose):
		self.head_pose = pose
		return

	#  Here 'pose' is (x, y, z, state) or None
	def set_left_hand_pose(self, pose):
		self.left_hand_pose = pose
		return

	#  Here 'pose' is (x, y, z, state) or None
	def set_right_hand_pose(self, pose):
		self.right_hand_pose = pose
		return

	#  Here 'pose' is (x, y, z, state) or None
	def set_left_hand_global_pose(self, pose):
		self.left_hand_global_pose = pose
		return

	#  Here 'pose' is (x, y, z, state) or None
	def set_right_hand_global_pose(self, pose):
		self.right_hand_global_pose = pose
		return

	#  Disable the targeted detections. If no detections are targeted, target all.
	def disable_detections(self, indices=None):
		if indices is None:
			indices = [x for x in range(0, len(self.detections))]
		for i in indices:
			self.detections[i].disable()
		return

	#  Enable the targeted detections. If no detections are targeted, target all.
	def enable_detections(self, indices=None):
		if indices is None:
			indices = [x for x in range(0, len(self.detections))]
		for i in indices:
			self.detections[i].enable()
		return

	def print(self, **kwargs):
		if 'index' in kwargs:										#  Were we given an index?
			assert isinstance(kwargs['index'], int) and kwargs['index'] >= 0, 'Argument \'index\' passed to Frame.print() must be an integer >= 0.'
			index = kwargs['index']
		else:
			index = None

		if 'pad_index' in kwargs:									#  Were we given a maximum number of spaces for the index?
			assert isinstance(kwargs['pad_index'], int) and kwargs['pad_index'] >= 0, 'Argument \'pad_index\' passed to Frame.print() must be an integer >= 0.'
			pad_index = kwargs['pad_index']
		else:
			pad_index = 0

		if 'pad_time' in kwargs:									#  Were we given a maximum number of spaces for the time stamp?
			assert isinstance(kwargs['pad_time'], int) and kwargs['pad_time'] >= 0, 'Argument \'pad_time\' passed to Frame.print() must be an integer >= 0.'
			pad_time = kwargs['pad_time']
		else:
			pad_time = 0

		if 'pad_file' in kwargs:									#  Were we given a maximum number of spaces for the file name?
			assert isinstance(kwargs['pad_file'], int) and kwargs['pad_file'] >= 0, 'Argument \'pad_file\' passed to Frame.print() must be an integer >= 0.'
			pad_file = kwargs['pad_file']
		else:
			pad_file = 0

		if 'pad_label' in kwargs:									#  Were we given a maximum number of spaces for the label?
			assert isinstance(kwargs['pad_label'], int) and kwargs['pad_label'] >= 0, 'Argument \'pad_label\' passed to Frame.print() must be an integer >= 0.'
			pad_label = kwargs['pad_label']
		else:
			pad_label = 0

		if 'pad_dim' in kwargs:										#  Were we given a maximum number of spaces for each image dimension?
			assert isinstance(kwargs['pad_dim'], int) and kwargs['pad_dim'] >= 0, 'Argument \'pad_dim\' passed to Frame.print() must be an integer >= 0.'
			pad_dim = kwargs['pad_dim']
		else:
			pad_dim = 0

		if 'precision' in kwargs:										#  Were we given a maximum number of spaces for each image dimension?
			assert isinstance(kwargs['precision'], int) and kwargs['precision'] > 0, 'Argument \'precision\' passed to Frame.print() must be an integer > 0.'
			precision = kwargs['precision']
		else:
			precision = 1

		format_string = '{:.' + str(precision) + 'f}'

		if index is not None:
			outstr = '[' + str(index) + ']:' + ' '*(max(1, pad_index - len(str(index)) + 1))
		else:
			outstr = ''
		outstr += str(self.time_stamp) + ' '*(max(1, pad_time - len(str(self.time_stamp)) + 1))
		outstr += self.file_name + ' '*(max(1, pad_file - len(self.file_name) + 1))
		outstr += '(' + str(self.width) + ' '*(max(1, pad_dim - len(str(self.width)) + 1)) + 'x '
		outstr += str(self.height) + ' '*(max(1, pad_dim - len(str(self.width)) + 1)) + '): '
		outstr += self.ground_truth_label + ' '*(max(1, pad_label - len(self.ground_truth_label) + 1)) + '['

		if self.head_pose is not None:
			outstr += format_string.format(self.head_pose[0]) + ' '
			outstr += format_string.format(self.head_pose[1]) + ' '
			outstr += format_string.format(self.head_pose[2]) + '; '
		else:
			outstr += '*' + ' '
			outstr += '*' + ' '
			outstr += '*' + '; '

		if self.left_hand_pose is not None:
			outstr += format_string.format(self.left_hand_pose[0]) + ' '
			outstr += format_string.format(self.left_hand_pose[1]) + ' '
			outstr += format_string.format(self.left_hand_pose[2]) + ' '
			outstr += str(self.left_hand_pose[3]) + '; '
		else:
			outstr += '*' + ' '
			outstr += '*' + ' '
			outstr += '*' + ' '
			outstr += '*' + '; '

		if self.right_hand_pose is not None:
			outstr += format_string.format(self.right_hand_pose[0]) + ' '
			outstr += format_string.format(self.right_hand_pose[1]) + ' '
			outstr += format_string.format(self.right_hand_pose[2]) + ' '
			outstr += str(self.right_hand_pose[3])
		else:
			outstr += '*' + ' '
			outstr += '*' + ' '
			outstr += '*' + ' '
			outstr += '*'

		outstr += ']'
		print(outstr)
		return

	#  Show the frame in a pop-up window.
	def show(self, mode='video'):
		assert isinstance(mode, str) and (mode == 'video' or mode == 'color' or mode == 'depth'), 'The argument \'mode\' in Frame.show() must be a string in {video, color, depth}.'
		img = cv2.imread(self.fullpath(mode), cv2.IMREAD_UNCHANGED)
		if len(img.shape) > 2 and img.shape[2] > 3:					#  Don't even ask for channel 3 if this is grayscale
			img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)				#  Drop alpha, if it was there
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)					#  Reverse colors for MatPlotLib
		imgplot = plt.imshow(img)
		plt.show()
		return
