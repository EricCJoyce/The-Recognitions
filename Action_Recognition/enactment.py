import cv2
import datetime
from itertools import product										#  For color-testing when we allow for compression artifacts
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
import time

'''
The "cone of attention" that boosts or lowers object signals in the props subvector according to their proximities to gaze and hands.
'''
class Gaussian():
	def __init__(self, **kwargs):
		self.mu = (0.0, 0.0, 0.0)									#  Centered on the head
		self.sigma_gaze = (2.0, 1.5, 3.0)							#  In METERS
																	#  No mus for hands: their centers are hand-centroids
		self.sigma_hand = (0.5, 0.5, 0.5)							#  In METERS

		if 'mu' in kwargs:
			assert isinstance(kwargs['mu'], tuple) and len(kwargs['mu']) == 3 and \
			       isinstance(kwargs['mu'][0], float) and isinstance(kwargs['mu'][1], float) and isinstance(kwargs['mu'][2], float), \
			       'Argument \'mu\' passed to Gaussian must be a 3-tuple of floats.'
			self.mu = kwargs['mu']

		if 'sigma_gaze' in kwargs:
			assert isinstance(kwargs['sigma_gaze'], tuple) and len(kwargs['sigma_gaze']) == 3 and \
			       isinstance(kwargs['sigma_gaze'][0], float) and isinstance(kwargs['sigma_gaze'][1], float) and isinstance(kwargs['sigma_gaze'][2], float), \
			       'Argument \'sigma_gaze\' passed to Gaussian must be a 3-tuple of floats.'
			self.sigma_gaze = kwargs['sigma_gaze']

		if 'sigma_hand' in kwargs:
			assert isinstance(kwargs['sigma_hand'], tuple) and len(kwargs['sigma_hand']) == 3 and \
			       isinstance(kwargs['sigma_hand'][0], float) and isinstance(kwargs['sigma_hand'][1], float) and isinstance(kwargs['sigma_hand'][2], float), \
			       'Argument \'sigma_hand\' passed to Gaussian must be a 3-tuple of floats.'
			self.sigma_hand = kwargs['sigma_hand']

	#  'object_centroid' is a Numpy array.
	#  'LH_centroid' is a Numpy array or None if the left hand is not visible.
	#  'RH_centroid' is a Numpy array or None if the right hand is not visible.
	def weigh(self, object_centroid, LH_centroid, RH_centroid):
		x = np.array([object_centroid[0] - self.mu[0], \
		              object_centroid[1] - self.mu[1], \
		              object_centroid[2] - self.mu[2]])
		C = np.array([[self.sigma_gaze[0] * self.sigma_gaze[0], 0.0,                                     0.0                                    ], \
		              [0.0,                                     self.sigma_gaze[1] * self.sigma_gaze[1], 0.0                                    ], \
		              [0.0,                                     0.0,                                     self.sigma_gaze[2] * self.sigma_gaze[2]]])
		C_inv = np.linalg.inv(C)
		f_head = np.exp(-0.5 * np.dot(np.dot(x.T, C_inv), x))

		if LH_centroid is not None:
			x = np.array([object_centroid[0] - LH_centroid[0], \
			              object_centroid[1] - LH_centroid[1], \
			              object_centroid[2] - LH_centroid[2]])
			C = np.array([[self.sigma_hand[0] * self.sigma_hand[0], 0.0,                                     0.0                                    ], \
			              [0.0,                                     self.sigma_hand[1] * self.sigma_hand[1], 0.0                                    ], \
			              [0.0,                                     0.0,                                     self.sigma_hand[2] * self.sigma_hand[2]]])
			C_inv = np.linalg.inv(C)
			f_Lhand = np.exp(-0.5 * np.dot(np.dot(x.T, C_inv), x))
		else:
			f_Lhand = 0.0

		if RH_centroid is not None:
			x = np.array([object_centroid[0] - RH_centroid[0], \
			              object_centroid[1] - RH_centroid[1], \
			              object_centroid[2] - RH_centroid[2]])
			C = np.array([[self.sigma_hand[0] * self.sigma_hand[0], 0.0,                                     0.0                                    ], \
			              [0.0,                                     self.sigma_hand[1] * self.sigma_hand[1], 0.0                                    ], \
			              [0.0,                                     0.0,                                     self.sigma_hand[2] * self.sigma_hand[2]]])
			C_inv = np.linalg.inv(C)
			f_Rhand = np.exp(-0.5 * np.dot(np.dot(x.T, C_inv), x))
		else:
			f_Rhand = 0.0

		return max(f_head, f_Lhand, f_Rhand)

	#  Return a list of this object's parameters.
	def parameter_list(self):
		return list(self.mu) + list(self.sigma_gaze) + list(self.sigma_hand)

'''
Recognizable Object, whether determined by ground-truth or by deep network detection.
'''
class RObject():
	def __init__(self, **kwargs):
		self.parent_frame = None									#  Full file path of the frame in which this detection occurs.

		self.instance_name = None									#  What is this instance called? e.g. Target_ControlPanel
		self.object_name = None										#  What is this object called? e.g. ControlPanel

		self.centroid = None										#  What is the 3D centroid of this object?
		self.detection_source = None								#  What was the source of this detection? e.g. ground-truth, mask-rcnn-0028

		self.mask_path = None										#  File path to the mask for this object detection, this frame
		self.bounding_box = None									#  The bounding box for this object detection,
																	#  (upper-left x, upper-left y, lower-right x, lower-right y).
		self.confidence = None										#  Confidence score for this detection, this frame. In [0.0, 1.0].

		self.colormaps = []											#  Only applies if ground-truth was used to recognize this object.
		self.colorsrc = None										#  A single RObject may contain several colors if it is a composite object.

		self.enabled = True											#  Let's allow ourselves the ability to toggle detections.
																	#  This will be helpful in deciding which are too small to bother about.
		self.epsilon = (0, 0, 0)

		if 'parent_frame' in kwargs:
			assert isinstance(kwargs['parent_frame'], str), 'Argument \'parent_frame\' passed to RObject must be a string.'
			self.parent_frame = kwargs['parent_frame']

		if 'instance_name' in kwargs:
			assert isinstance(kwargs['instance_name'], str), 'Argument \'instance_name\' passed to RObject must be a string.'
			self.instance_name = kwargs['instance_name']

		if 'object_name' in kwargs:
			assert isinstance(kwargs['object_name'], str), 'Argument \'object_name\' passed to RObject must be a string.'
			self.object_name = kwargs['object_name']

		if 'centroid' in kwargs:
			assert isinstance(kwargs['centroid'], tuple), 'Argument \'centroid\' passed to RObject must be a tuple.'
			self.centroid = kwargs['centroid']

		if 'detection_source' in kwargs:
			assert isinstance(kwargs['detection_source'], str), 'Argument \'detection_source\' passed to RObject must be a string.'
			self.detection_source = kwargs['detection_source']

		if 'mask_path' in kwargs:
			assert isinstance(kwargs['mask_path'], str), 'Argument \'mask_path\' passed to RObject must be a string.'
			self.mask_path = kwargs['mask_path']

		if 'bounding_box' in kwargs:
			assert isinstance(kwargs['bounding_box'], tuple) and \
			       len(kwargs['bounding_box']) == 4, 'Argument \'bounding_box\' passed to RObject must be a 4-tuple.'
			self.bounding_box = kwargs['bounding_box']

		if 'confidence' in kwargs:
			assert isinstance(kwargs['confidence'], float) and \
			       kwargs['confidence'] >= 0.0 and kwargs['confidence'] <= 1.0, 'Argument \'confidence\' passed to RObject must be a float in [0.0, 1.0].'
			self.confidence = kwargs['confidence']

		if 'colors' in kwargs:
			assert isinstance(kwargs['colors'], list) and \
			       [len(x) for x in kwargs['colors']].count(3) == len(kwargs['colors']), 'Argument \'colors\' passed to RObject must be a list of 3-tuples.'
			self.colormaps = kwargs['colors']

		if 'colorsrc' in kwargs:
			assert isinstance(kwargs['colorsrc'], str), 'Argument \'colorsrc\' passed to RObject must be a string.'
			self.colorsrc = kwargs['colorsrc']

	#  Return the 2D center (x, y) of this object, according either to its average (time-costly) or its bounding-box.
	def center(self, method='bbox'):
		assert isinstance(method, str) and (method == 'bbox' or method == 'avg'), 'Argument \'method\' passed to RObject.center() must be a string in {avg, bbox}.'

		if method == 'avg':
			if self.mask_path is not None:
				img = cv2.imread(self.mask_path, cv2.IMREAD_UNCHANGED)
				indices = np.where(img > 0)
				return (int(round(np.mean(indices[1]))), int(round(np.mean(indices[0]))))
			return None

		if self.bounding_box is not None:
			return ( int(round(float(self.bounding_box[0] + self.bounding_box[2]) * 0.5)), \
			         int(round(float(self.bounding_box[1] + self.bounding_box[3]) * 0.5)) )

		return None

	#  Override from outside this class.
	#  (Because, though we could look up a Recognizable Object's corresponding depth map, this class has no sense of the camera.)
	def set_centroid(self, c):
		self.centroid = c[:]
		return

	#  Render this Recognizable Object's mask to a new file with the given path 'mask_path'.
	#  Also update the internal attributes, self.mask_path and self.bounding_box.
	def render_mask(self, mask_path):
		img = cv2.imread(self.colorsrc, cv2.IMREAD_UNCHANGED)		#  Open the color map file for the given frame.
		if img.shape[2] > 3:
			img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

		mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')

		for color in self.colormaps:								#  For every color in this object...
			color = color[::-1]
			epsilonHi = tuple(map(lambda i, j: i + j, color, self.epsilon))
			epsilonLo = tuple(map(lambda i, j: i - j, color, self.epsilon))
			indices = np.all(img >= epsilonLo, axis=-1) & np.all(img <= epsilonHi, axis=-1)
			if True in indices:										#  We were able to match this color.
				mask += indices * np.uint8(1)						#  Accumulate into 'mask'

		np.clip(mask, 0, 1)											#  Clamp
		mask *= 255													#  Now convert all values > 0 to 255
		indices = np.where(mask > 0)
																	#  Now compute bounding box, once
		self.bounding_box = (min(indices[1]), min(indices[0]), max(indices[1]), max(indices[0]))

		self.mask_path = mask_path
		cv2.imwrite(self.mask_path, mask)

		return

	#  Recognizable Object's mask must exist first. This effect will be performed, and it will be saved back to file.
	def erode_mask(self, erosion):
		mask = cv2.imread(self.mask_path, cv2.IMREAD_UNCHANGED)
		if mask is not None:
			kernel = np.ones((erosion, erosion), np.uint8)			#  Create kernel.
			mask = cv2.erode(mask, kernel, iterations=1)			#  Perform erosion.

			indices = np.where(mask > 0)							#  Re-compute bounding box.
			self.bounding_box = (min(indices[1]), min(indices[0]), max(indices[1]), max(indices[0]))

			cv2.imwrite(self.mask_path, mask)						#  Save mask.
		return

	#  Recognizable Object's mask must exist first. This effect will be performed, and it will be saved back to file.
	def dilate_mask(self, erosion):
		mask = cv2.imread(self.mask_path, cv2.IMREAD_UNCHANGED)
		if mask is not None:
			kernel = np.ones((erosion, erosion), np.uint8)			#  Create kernel.
			mask = cv2.dilate(mask, kernel, iterations=1)			#  Perform dilation.

			indices = np.where(mask > 0)							#  Re-compute bounding box.
			self.bounding_box = (min(indices[1]), min(indices[0]), max(indices[1]), max(indices[0]))

			cv2.imwrite(self.mask_path, mask)						#  Save mask.
		return

	#  Return the area in pixels of the bounding box (if it exists)
	def get_bbox_area(self):
		if self.bounding_box is not None:
			return (self.bounding_box[2] - self.bounding_box[0]) * (self.bounding_box[3] - self.bounding_box[1])
		return 0

	#  Return the area in pixels of the mask (if it exists)
	def get_mask_area(self):
		if self.mask_path is not None:
			img = cv2.imread(self.mask_path, cv2.IMREAD_UNCHANGED)
			indices = np.where(img > 0)
			return len(indices[0])
		return 0

	def disable(self):
		self.enabled = False
		return

	def enable(self):
		self.enabled = True
		return

	def print(self):
		if self.parent_frame is not None:
			print('Recognizable-Object in ' + self.parent_frame + ':')
		else:
			print('Recognizable-Object in *:')
		if self.instance_name is not None:
			print('  Instance:   ' + self.instance_name)
		else:
			print('  Instance:   *')
		if self.object_name is not None:
			print('  Class:      ' + self.object_name)
		else:
			print('  Class:      *')
		if self.centroid is not None:
			print('  Centroid:   (' + str(self.centroid[0]) + ', ' + str(self.centroid[1]) + ', ' + str(self.centroid[2]) + ')')
		else:
			print('  Centroid:   *')
		if self.detection_source is not None:
			print('  Detection:  ' + self.detection_source)
		else:
			print('  Detection:  *')
		if self.mask_path is not None:
			print('  Mask:       ' + self.mask_path)
		else:
			print('  Mask:       *')
		if self.bounding_box is not None:
			print('  B.Box:      (' + str(self.bounding_box[0]) + ', ' + str(self.bounding_box[1]) + '), (' + str(self.bounding_box[2]) + ', ' + str(self.bounding_box[3]) + ')')
		else:
			print('  B.Box:      *')
		if self.confidence is not None:
			print('  Confidence: ' + str(self.confidence))
		else:
			print('  Confidence: *')
		if len(self.colormaps) > 0:
			colorstr = ''
			for i in range(0, len(self.colormaps)):
				color = self.colormaps[i]
				colorstr += '(' + str(color[0]) + ', ' + str(color[1]) + ', ' + str(color[2]) + ')'
				if i < len(self.colormaps) - 1:
					colorstr += ', '
			print('  Colors:      ' + colorstr)
		else:
			print('  Colors:     *')
		if self.colorsrc is not None:
			print('  Color Src:  ' + self.colorsrc)
		else:
			print('  Color Src:  *')

		return

	#  If self.mask_path exists, show the masked object in a pop-up window.
	def show(self, mode='video'):
		assert isinstance(mode, str) and (mode == 'video' or mode == 'color' or mode == 'depth'), 'The argument \'mode\' in RObject.show() must be a string in {video, color, depth}.'
		if self.mask_path is not None:
			mask = cv2.imread(self.mask_path, cv2.IMREAD_UNCHANGED)
			mask = np.clip(mask, 0, 1)								#  Knock {0, 255} down to {0, 1}

			arr = self.parent_frame.split('/')
			path = '/'.join(arr[:-2])
			print(path)
			if mode == 'color':
				img = cv2.imread(path + '/ColorMapCameraFrames/' + arr[-1], cv2.IMREAD_UNCHANGED)
			elif mode == 'depth':
				img = cv2.imread(path + '/DepthMapCameraFrames/' + arr[-1], cv2.IMREAD_UNCHANGED)
			else:
				img = cv2.imread(path + '/NormalViewCameraFrames/' + arr[-1], cv2.IMREAD_UNCHANGED)
			if len(img.shape) > 2 and img.shape[2] > 3:				#  Don't even ask for channel 3 if this is grayscale
				img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)			#  Drop alpha

			if len(img.shape) > 2:									#  RGB
				img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)			#  Reverse colors for MatPlotLib
				img[:, :, 0] *= mask								#  Mask out each channel by broadcasting the mask
				img[:, :, 1] *= mask
				img[:, :, 2] *= mask
			else:													#  Grayscale
				img *= mask											#  Mask out by broadcasting the mask

			imgplot = plt.imshow(img)
			plt.show()

		return

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

		self.detections = []										#  List of RObjects

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

'''
This is a "loose" sort of data type in that Actions are not really binding until their host Enactment calls apply_actions_to_frames().
Actions contain a label and delineate where things begin and end.
Beginnings are considered inclusive; endings are considered exclusive, mimicking Pythonic slice notation.
'''
class Action():
	def __init__(self, start_time, end_time, **kwargs):
		self.start_time = start_time
		self.end_time = end_time

		self.start_frame = None
		self.end_frame = None
		self.label = None

		if 'start_frame' in kwargs:									#  Were we given a start_frame?
			assert isinstance(kwargs['start_frame'], str), 'Argument \'start_frame\' passed to Action must be a string.'
			self.start_frame = kwargs['start_frame']
		if 'end_frame' in kwargs:									#  Were we given a end_frame?
			assert isinstance(kwargs['end_frame'], str), 'Argument \'end_frame\' passed to Action must be a string.'
			self.end_frame = kwargs['end_frame']
		if 'label' in kwargs:										#  Were we given a label?
			assert isinstance(kwargs['label'], str), 'Argument \'label\' passed to Action must be a string.'
			self.label = kwargs['label']

	def print(self, **kwargs):
		if 'index' in kwargs:										#  Were we given an index?
			assert isinstance(kwargs['index'], int) and kwargs['index'] >= 0, 'Argument \'index\' passed to Action.print() must be an integer >= 0.'
			index = kwargs['index']
		else:
			index = None

		if 'pad_index' in kwargs:									#  Were we given a maximum number of spaces for the index?
			assert isinstance(kwargs['pad_index'], int) and kwargs['pad_index'] >= 0, 'Argument \'pad_index\' passed to Action.print() must be an integer >= 0.'
			pad_index = kwargs['pad_index']
		else:
			pad_index = 0

		if 'pad_label' in kwargs:									#  Were we given a maximum number of spaces for the label?
			assert isinstance(kwargs['pad_label'], int) and kwargs['pad_label'] >= 0, 'Argument \'pad_label\' passed to Action.print() must be an integer >= 0.'
			pad_label = kwargs['pad_label']
		else:
			pad_label = 0

		if 'pad_time' in kwargs:									#  Were we given a maximum number of spaces for time stamps?
			assert isinstance(kwargs['pad_time'], int) and kwargs['pad_time'] >= 0, 'Argument \'pad_time\' passed to Action.print() must be an integer >= 0.'
			pad_time = kwargs['pad_time']
		else:
			pad_time = 0

		if 'pad_path' in kwargs:									#  Were we given a maximum number of spaces for file paths?
			assert isinstance(kwargs['pad_path'], int) and kwargs['pad_path'] >= 0, 'Argument \'pad_path\' passed to Action.print() must be an integer >= 0.'
			pad_path = kwargs['pad_path']
		else:
			pad_path = 0

		if index is not None:
			outstr = '[' + str(index) + ']:' + ' '*(max(1, pad_index - len(str(index)) + 1))
		else:
			outstr = ''
		outstr += self.label + ':' + ' '*(max(1, pad_label - len(self.label) + 1))
		outstr += str(self.start_time) + ' '*(max(1, pad_time - len(str(self.start_time)) + 1))
		if self.start_frame is not None:
			outstr += 'incl.(' + self.start_frame.split('/')[-1] + ')' + ' '*(max(1, pad_path - len(self.start_frame.split('/')[-1]) + 1))
		outstr += '-- '
		outstr += str(self.end_time) + ' '*(max(1, pad_time - len(str(self.end_time)) + 1))
		if self.end_frame is not None:
			outstr += 'excl.(' + self.end_frame.split('/')[-1] + ')'
		print(outstr)
		return

'''
Interface for the recorded VR session.
'''
class Enactment():
	def __init__(self, name, **kwargs):
		self.enactment_name = name									#  Also the name of the directory.
		self.width = None
		self.height = None
		self.focal_length = None									#  Can be set explicitly or computed from metadata.
		self.verbose = False										#  False by default.
		self.epsilon = 0											#  Amount, per color channel, by which we allow colors to deviate.

		#############################################################
		#  The main attributes of this class.                       #
		#############################################################

		self.frames = {}											#  key:time_stamp ==> val:Frame object
		self.actions = []											#  List of Action objects arranged in chronological order.
		self.rules = None											#  Must be loaded from file (or not).
		self.object_detection_source = None							#  Determined by how objects are parsed.
		self.recognizable_objects = []								#  As determined by a rules file; not necessarily what can be seen in this Enactment.

		if 'fps' in kwargs:											#  Were we given a frames-per-second rate?
			assert isinstance(kwargs['fps'], int), 'Argument \'fps\' passed to Enactment must be an integer.'
			self.fps = kwargs['fps']
		else:														#  Fetch fps from metadata
			metadata = self.load_metadata()
			self.fps = int(1.0 / metadata['keyframeSamplingInterval'])

		if 'verbose' in kwargs:
			assert isinstance(kwargs['verbose'], bool), 'Argument \'verbose\' passed to Enactment must be a Boolean.'
			self.verbose = kwargs['verbose']

		if 'user' in kwargs:										#  Were we given a user? (Becomes part of the file path)
			assert isinstance(kwargs['user'], str) and '/' not in kwargs['user'], 'Argument \'user\' passed to Enactment must be a string without directory delimiters.'
			self.user = kwargs['user']
		else:														#  Fetch maximum depth from metadata
			self.user = 'vr1'

		if 'enactment_file' in kwargs:								#  Were we given an enactment file? (Helpful for getting width, height,
																	#  object_detection_source, and recognizable_objects.)
			assert isinstance(kwargs['enactment_file'], str), 'Argument \'enactment_file\' passed to Enactment must be a string.'
			fh = open(kwargs['enactment_file'], 'r')
			reading_dimensions = False
			reading_framerate = False
			reading_object_detection_source = False
			reading_recognizable_objects = False
			for line in fh.readlines():
				if line[0] == '#':
					if 'WIDTH & HEIGHT' in line:					#  Next line contains width and height.
						reading_dimensions = True
					elif reading_dimensions:
						arr = line[1:].strip().split()
						self.width = int(arr[0])
						self.height = int(arr[1])
						reading_dimensions = False

					if 'FPS' in line:								#  Next line contains frames-per-second.
						reading_framerate = True
					elif reading_framerate:
						arr = line[1:].strip()
						self.fps = int(arr)
						reading_framerate = False

					if 'OBJECT DETECTION SOURCE' in line:			#  Next line contains object-detection source.
						reading_object_detection_source = True
					elif reading_object_detection_source:
						self.object_detection_source = line[1:].strip()
						reading_object_detection_source = False

					if 'RECOGNIZABLE OBJECTS' in line:				#  Next line contains recognizable objects.
						reading_recognizable_objects = True
					elif reading_recognizable_objects:
						self.recognizable_objects = line[1:].strip().split('\t')
						reading_recognizable_objects = False
			fh.close()

		if 'wh' in kwargs:											#  Were we given a tuple (width, height)?
			assert isinstance(kwargs['wh'], tuple) and \
			       len(kwargs['wh']) == 2 and \
			       isinstance(kwargs['wh'][0], int) and kwargs['wh'][0] > 0 and \
			       isinstance(kwargs['wh'][1], int) and kwargs['wh'][1] > 0, 'Argument \'wh\' passed to Enactment must be a tuple of integers > 0.'
			self.width = kwargs['wh'][0]
			self.height = kwargs['wh'][1]
		elif self.width is None or self.height is None:				#  If not, check every file for bogies.
			discovered_w = {}
			discovered_h = {}

			video_frames = self.load_frame_sequence(True)
			num_frames = len(video_frames)
			prev_ctr = 0
			max_ctr = os.get_terminal_size().columns - 7			#  Leave enough space for the brackets, space, and percentage.
			if self.verbose:
				print('>>> Scanning enactment frames for consistent image dimensions.')

			for i in range(0, num_frames):
				img = cv2.imread(video_frames[i], cv2.IMREAD_UNCHANGED)
				if img.shape[1] not in discovered_w:
					discovered_w[ img.shape[1] ] = 0
				if img.shape[0] not in discovered_h:
					discovered_h[ img.shape[0] ] = 0
				if self.verbose:
					if int(round(float(i) / float(num_frames) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
						prev_ctr = int(round(float(i) / float(num_frames) * float(max_ctr)))
						sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(i) / float(num_frames) * 100.0))) + '%]')
						sys.stdout.flush()
				discovered_w[ img.shape[1] ] += 1
				discovered_h[ img.shape[0] ] += 1
			if self.verbose:
				print('')

			if len(discovered_w) != 1 or len(discovered_h) != 1:
				assert False, 'Frames discovered for Enactment "' + self.name + '" do not have consistent shape.'
			else:
				self.width = [x for x in discovered_w.keys()][0]
				self.height = [x for x in discovered_h.keys()][0]

		if 'min_depth' in kwargs:									#  Were we given a minimum depth?
			assert isinstance(kwargs['min_depth'], float), 'Argument \'min_depth\' passed to Enactment must be a float.'
			self.min_depth = kwargs['min_depth']
		else:														#  Fetch minimum depth from metadata
			metadata = self.load_metadata()
			self.min_depth = metadata['depthImageRange']['x']

		if 'max_depth' in kwargs:									#  Were we given a maximum depth?
			assert isinstance(kwargs['max_depth'], float), 'Argument \'max_depth\' passed to Enactment must be a float.'
			self.max_depth = kwargs['max_depth']
		else:														#  Fetch maximum depth from metadata
			metadata = self.load_metadata()
			self.max_depth = metadata['depthImageRange']['y']

		if 'focal_length' in kwargs:								#  Were we given a focal length?
			assert isinstance(kwargs['focal_length'], float), 'Argument \'focal_length\' passed to Enactment must be a float.'
			self.focal_length = kwargs['focal_length']

		#############################################################
		#  Attributes that facilitate display and visualization.    #
		#############################################################
		self.robject_colors = {}									#  key:recognizable-object(string) ==> val:(r, g, b)

		self.gt_label_super = {}									#  Where to locate and how to type the ground-truth label super-imposition
		self.gt_label_super['x'] = 10
		self.gt_label_super['y'] = 50
		self.gt_label_super['fontsize'] = 1.0
		self.filename_super = {}									#  Where to locate and how to type the video-frame file name super-imposition
		self.filename_super['x'] = 10
		self.filename_super['y'] = 90
		self.filename_super['fontsize'] = 1.0
		self.LH_super = {}
		self.LH_super['x'] = 10
		self.LH_super['y'] = self.height - 90
		self.LH_super['fontsize'] = 1.0
		self.RH_super = {}
		self.RH_super['x'] = 10
		self.RH_super['y'] = self.height - 50
		self.RH_super['fontsize'] = 1.0

		labels = self.load_json_action_labels()						#  Constructor loads ground-truth action labels.
		head_data = self.load_head_poses()							#  Used here to load time stamps.

		camdata = self.load_camera_intrinsics()						#  Establish the focal length
		fov = float(camdata['fov'])									#  We don't use camdata['focalLength'] anymore because IT'S IN MILLIMETERS
		if self.focal_length is None:								#  This value is IN PIXELS!
																	#  ALSO: NOTE THAT THIS IS THE *VERTICAL* F.o.V. !!
			self.focal_length = self.height * 0.5 / np.tan(fov * np.pi / 180.0)

		video_frames = self.load_frame_sequence(True)
		time_stamps = [x['timestamp'] for x in head_data['list']]

		assert len(video_frames) == len(time_stamps), 'Number of time stamps and number of frames do not agree in enactment "' + self.enactment_name + '".'

		prev_ctr = 0
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.

		#############################################################
		#  Load poses into the object.                              #
		#############################################################

		self.load_sensor_poses()									#  By default, use sensor poses--though they be noisy.
																	#  Once masks are computed or loaded, we can compute IK poses from LeftHand & RightHand.
		#############################################################
		#  Load ground-truth action labels into the object.         #
		#############################################################

		fixes = []													#  Prepare to detect and correct label conflicts and overlaps.

		if os.path.exists(self.enactment_name + '/labels.txt'):		#  If an explicit frame-labeling file exists, use that.
			if self.verbose:
				print('>>> Found ' + self.enactment_name + '/labels.txt')
			fh = open(self.enactment_name + '/labels.txt', 'r')
			current_label = None
			start_time = None
			end_time = None
			for line in fh.readlines():
				if line[0] != '#':
					arr = line.strip().split('\t')					#  timestamp    frame-path    label
					time_stamp = float(arr[0])
					label = arr[2]

					if label == '*' and current_label is None:
						current_label = label

					if label != current_label:						#  A change!
						if current_label != '*':					#  Something other than nothing has ended.
							end_time = time_stamp
							self.actions.append( Action(start_time, end_time, label=current_label) )
						start_time = time_stamp
						current_label = label
			fh.close()

			if current_label != '*':								#  An action ran all the way to the end of the enactment; close it!
				end_time = time_stamp
				self.actions.append( Action(start_time, end_time, label=current_label) )
		else:														#  Otherwise, use the JSON.
			if self.verbose:
				print('>>> Reading labels from JSON')
			for i in range(0, len(labels['list'])):
				action = labels['list'][i]
				action_start_time = action['startTime']
				action_end_time = action['endTime']
				label = action['stepDescription']

				self.actions.append( Action(action_start_time, action_end_time, label=label) )
																	#  Sort temporally
		self.actions = sorted(self.actions, key=lambda x: x.start_time)

		for i in range(0, len(self.actions) - 1):					#  Examine all neighbors for conflicts
																	#  LABEL OVERLAP!
			if self.actions[i].end_time > self.actions[i + 1].start_time:
				if self.verbose:
					fixes.append(i)
																	#  ADJUST TIMES!
				mid = self.actions[i + 1].start_time - (self.actions[i].end_time - self.actions[i + 1].start_time) * 0.5
				self.actions[i].end_time = mid						#  Short the end of the out-going action
				self.actions[i + 1].start_time = mid				#  Postpone the beginning of the in-coming action

																	#  Potential conflicts have been resolved.
		for i in range(0, len(self.actions)):						#  Now sync time stamps with frames.
																	#  Find the frame that minimizes (squared) difference between
																	#  its time_stamp and the action's start_time
			self.actions[i].start_frame = video_frames[ np.argmin([(x - self.actions[i].start_time) ** 2 for x in time_stamps]) ]
																	#  Find the frame that minimizes (squared) difference between
																	#  its time_stamp and the action's end_time
			self.actions[i].end_frame = video_frames[ np.argmin([(x - self.actions[i].end_time) ** 2 for x in time_stamps]) ]

		if self.verbose:
			for i in range(0, len(fixes)):
				print('    * Fixed label conflict between "' + self.actions[ fixes[i] ].label + '" and "' + self.actions[ fixes[i] + 1 ].label + '"')
			print('')

		self.apply_actions_to_frames()

	#################################################################
	#  Output.                                                      #
	#################################################################

	#  Formatted representation of an enactment.
	def write_text_enactment_file(self, gaussian):
		if self.verbose:
			print('>>> Writing enactment to file.')

		num_frames = len(self.frames)
		prev_ctr = 0
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.

		fh = open(self.enactment_name + '.enactment', 'w')
		fh.write('#  Enactment vectors derived from FactualVR enactment materials.\n')
		fh.write('#  This file created ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '.\n')
		fh.write('#  WIDTH & HEIGHT:\n')
		fh.write('#    ' + str(self.width) + '\t' + str(self.height) + '\n')
		fh.write('#  FPS:\n')
		fh.write('#    ' + str(self.fps) + '\n')
		fh.write('#  GAUSSIAN: (mu_x, mu_y, mu_z, sigma_gaze_x, sigma_gaze_y, sigma_gaze_z, sigma_hand_x, sigma_hand_y, sigma_hand_z)\n')
		fh.write('#    ' + '\t'.join([str(x) for x in gaussian.parameter_list()]) + '\n')
		fh.write('#  OBJECT DETECTION SOURCE:\n')
		fh.write('#    ' + self.object_detection_source + '\n')
		fh.write('#  RECOGNIZABLE OBJECTS:\n')
		fh.write('#    ' + '\t'.join(self.recognizable_objects) + '\n')
		fh.write('#  ENCODING STRUCTURE:\n')
		structure_string = '\t'.join(['timestamp', 'filename', 'label', \
		                              'LHx', 'LHy', 'LHz', 'LH0', 'LH1', 'LH2', \
		                              'RHx', 'RHy', 'RHz', 'RH0', 'RH1', 'RH2'] + self.recognizable_objects)
		fh.write('#    ' + structure_string + '\n')

		ctr = 0
		for time_stamp, frame in sorted(self.frames.items()):
			fh.write(str(time_stamp) + '\t')
			fh.write(frame.fullpath() + '\t')
			fh.write(frame.ground_truth_label + '\t')
			if frame.left_hand_pose is not None:					#  Write the LEFT-HAND subvector
				fh.write(str(frame.left_hand_pose[0]) + '\t')
				fh.write(str(frame.left_hand_pose[1]) + '\t')
				fh.write(str(frame.left_hand_pose[2]) + '\t')
				if frame.left_hand_pose[3] == 0:
					fh.write('1.0\t0.0\t0.0\t')
				elif frame.left_hand_pose[3] == 1:
					fh.write('0.0\t1.0\t0.0\t')
				elif frame.left_hand_pose[3] == 2:
					fh.write('0.0\t0.0\t1.0\t')
				else:												#  This case should never happen, but be prepared!
					fh.write('0.0\t0.0\t0.0\t')
			else:
				fh.write('0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t')
			if frame.right_hand_pose is not None:					#  Write the RIGHT-HAND subvector
				fh.write(str(frame.right_hand_pose[0]) + '\t')
				fh.write(str(frame.right_hand_pose[1]) + '\t')
				fh.write(str(frame.right_hand_pose[2]) + '\t')
				if frame.right_hand_pose[3] == 0:
					fh.write('1.0\t0.0\t0.0\t')
				elif frame.right_hand_pose[3] == 1:
					fh.write('0.0\t1.0\t0.0\t')
				elif frame.right_hand_pose[3] == 2:
					fh.write('0.0\t0.0\t1.0\t')
				else:												#  This case should never happen, but be prepared!
					fh.write('0.0\t0.0\t0.0\t')
			else:
				fh.write('0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t')

			props_subvector = [0.0 for i in range(0, len(self.recognizable_objects))]
			for detection in frame.detections:
				if detection.enabled:
					if frame.left_hand_pose is not None:
						lh = np.array(frame.left_hand_pose[:3])
					else:
						lh = None
					if frame.right_hand_pose is not None:
						rh = np.array(frame.right_hand_pose[:3])
					else:
						rh = None
					g = gaussian.weigh(np.array(detection.centroid), lh, rh)
					props_subvector[ self.recognizable_objects.index(detection.object_name) ] = max(g, props_subvector[ self.recognizable_objects.index(detection.object_name) ])

			fh.write('\t'.join([str(x) for x in props_subvector]) + '\n')

			if self.verbose:
				if int(round(float(ctr) / float(num_frames) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(ctr) / float(num_frames) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(ctr) / float(num_frames) * 100.0))) + '%]')
					sys.stdout.flush()
			ctr += 1

		fh.close()

		if self.verbose:
			print('')

		return

	#  Write a hand-pose file.
	def write_hand_pose_file(self, file_name=None):
		if self.verbose:
			print('>>> Writing hand poses to file.')

		num_frames = len(self.frames)
		prev_ctr = 0
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.

		if file_name is None:
			fh = open(self.enactment_name + '.handposes', 'w')
		else:
			fh = open(file_name + '.handposes', 'w')

		fh.write('#  Hand poses from FactualVR enactment materials.\n')
		fh.write('#  This file created ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '.\n')
		fh.write('#  FPS:\n')
		fh.write('#    ' + str(self.fps) + '\n')
		fh.write('#  ENCODING STRUCTURE:\n')
		structure_string = '\t'.join(['timestamp', 'filename', 'label', \
		                              'LHx', 'LHy', 'LHz', 'LH0', 'LH1', 'LH2', \
		                              'RHx', 'RHy', 'RHz', 'RH0', 'RH1', 'RH2'])
		fh.write('#    ' + structure_string + '\n')

		ctr = 0
		for time_stamp, frame in sorted(self.frames.items()):
			fh.write(str(time_stamp) + '\t')
			fh.write(frame.fullpath() + '\t')
			fh.write(frame.ground_truth_label + '\t')
			if frame.left_hand_pose is not None:					#  Write the LEFT-HAND subvector
				fh.write(str(frame.left_hand_pose[0]) + '\t')
				fh.write(str(frame.left_hand_pose[1]) + '\t')
				fh.write(str(frame.left_hand_pose[2]) + '\t')
				if frame.left_hand_pose[3] == 0:
					fh.write('1.0\t0.0\t0.0\t')
				elif frame.left_hand_pose[3] == 1:
					fh.write('0.0\t1.0\t0.0\t')
				elif frame.left_hand_pose[3] == 2:
					fh.write('0.0\t0.0\t1.0\t')
				else:												#  This case should never happen, but be prepared!
					fh.write('0.0\t0.0\t0.0\t')
			else:
				fh.write('0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t')
			if frame.right_hand_pose is not None:					#  Write the RIGHT-HAND subvector
				fh.write(str(frame.right_hand_pose[0]) + '\t')
				fh.write(str(frame.right_hand_pose[1]) + '\t')
				fh.write(str(frame.right_hand_pose[2]) + '\t')
				if frame.right_hand_pose[3] == 0:
					fh.write('1.0\t0.0\t0.0\t')
				elif frame.right_hand_pose[3] == 1:
					fh.write('0.0\t1.0\t0.0\t')
				elif frame.right_hand_pose[3] == 2:
					fh.write('0.0\t0.0\t1.0\t')
				else:												#  This case should never happen, but be prepared!
					fh.write('0.0\t0.0\t0.0\t')
			else:
				fh.write('0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t')
			fh.write('\n')

			if self.verbose:
				if int(round(float(ctr) / float(num_frames - 1) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(ctr) / float(num_frames - 1) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(ctr) / float(num_frames - 1) * 100.0))) + '%]')
					sys.stdout.flush()
			ctr += 1

		fh.close()

		if self.verbose:
			print('')

		return

	#  Write all action labels for all frames to file.
	def write_all_frame_labels(self, file_name=None):
		if self.verbose:
			print('>>> Writing frame action-labels to file.')

		num_frames = len(self.frames)
		prev_ctr = 0
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.

		if file_name is None:
			fh = open(self.enactment_name + '.labels', 'w')
		else:
			fh = open(file_name + '.labels', 'w')

		fh.write('#  Action labels from FactualVR enactment.\n')
		fh.write('#  This file created ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '.\n')
		fh.write('#  ENCODING STRUCTURE:\n')
		structure_string = '\t'.join(['timestamp', 'filename', 'label'])
		fh.write('#    ' + structure_string + '\n')

		ctr = 0
		for time_stamp, frame in sorted(self.frames.items()):
			fh.write(str(time_stamp) + '\t')
			fh.write(frame.fullpath() + '\t')
			fh.write(frame.ground_truth_label + '\n')

			if self.verbose:
				if int(round(float(ctr) / float(num_frames - 1) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(ctr) / float(num_frames - 1) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(ctr) / float(num_frames - 1) * 100.0))) + '%]')
					sys.stdout.flush()
			ctr += 1

		fh.close()

		if self.verbose:
			print('')

		return

	#################################################################
	#  Editing: make changes to actions, poses, detections.         #
	#################################################################

	#  Relabel all action instances with 'old_label' to 'new_label'
	def relabel(self, old_label, new_label):
		for action in self.actions:
			if action.label == old_label:
				action.label = new_label
		self.apply_actions_to_frames()								#  Refresh all Frame labels
		return

	#  The given file may have some comment lines, starting with "#",
	#  but apart from that, each line should have the format:
	#  old-label <tab> new-label
	def relabel_from_file(self, relabel_file):
		fh = open(relabel_file, 'r')
		for line in fh.readlines():
			if line[0] != '#':
				arr = line.strip().split('\t')
				old_label = arr[0]
				new_label = arr[1]

				for action in self.actions:
					if action.label == old_label:
						action.label = new_label
		fh.close()

		self.apply_actions_to_frames()								#  Refresh all Frame labels
		return

	#  Return a list of all RObjects matching the given condition.
	def list_detections(self, condition):
		if self.verbose:
			print('>>> Searching for detections according to condition.')

		ret = []

		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.
		num_frames = self.num_frames()
		prev_ctr = 0
		i = 0

		for time_stamp, frame in sorted(self.frames.items()):
			for detection in frame.detections:
				if condition(detection):
					ret.append( detection )

			if self.verbose:
				if int(round(float(i) / float(num_frames) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(i) / float(num_frames) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(i) / float(num_frames) * 100.0))) + '%]')
					sys.stdout.flush()
			i += 1

		return ret

	#  Return a list of tuples: (timestamp, index) for all RObjects matching the given condition.
	def locate_detections(self, condition):
		if self.verbose:
			print('>>> Searching for detections according to condition.')

		ret = []

		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.
		num_frames = self.num_frames()
		prev_ctr = 0
		i = 0

		for time_stamp, frame in sorted(self.frames.items()):
			detection_index = 0
			for detection in frame.detections:
				if condition(detection):
					ret.append( (time_stamp, detection_index) )
				detection_index += 1

			if self.verbose:
				if int(round(float(i) / float(num_frames) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(i) / float(num_frames) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(i) / float(num_frames) * 100.0))) + '%]')
					sys.stdout.flush()
			i += 1

		return ret

	#  This method receives a lambda-function as its argument.
	#  For example, to disable all detections with bounding boxes less than 400 pixels:
	#    disable_detections( (lambda detection: detection.get_bbox_area() < 400) )
	#  To disable all detections:
	#    disable_detections( (lambda detection: True) )
	def disable_detections(self, condition):
		if self.verbose:
			print('>>> Disabling detections according to condition.')

		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.
		num_frames = self.num_frames()
		prev_ctr = 0
		i = 0

		for time_stamp, frame in sorted(self.frames.items()):
			for detection in frame.detections:
				if condition(detection):
					detection.disable()

			if self.verbose:
				if int(round(float(i) / float(num_frames) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(i) / float(num_frames) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(i) / float(num_frames) * 100.0))) + '%]')
					sys.stdout.flush()
			i += 1

		return

	#  This method receives a lambda-function as its argument.
	#  To enable all detections:
	#    enable_detections( (lambda detection: True) )
	def enable_detections(self, condition):
		if self.verbose:
			print('>>> Enabling detections according to condition.')

		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.
		num_frames = self.num_frames()
		prev_ctr = 0
		i = 0

		for time_stamp, frame in sorted(self.frames.items()):
			for detection in frame.detections:
				if condition(detection):
					detection.enable()

			if self.verbose:
				if int(round(float(i) / float(num_frames) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(i) / float(num_frames) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(i) / float(num_frames) * 100.0))) + '%]')
					sys.stdout.flush()
			i += 1

		return

	#  Delete an action. Relinquish its frames to the nothing-label "*".
	def drop_action(self, index):
		assert isinstance(index, int) and index >= 0 and index < len(self.actions), \
		       'Argument \'index\' in Enactment.drop_action() must be an integer >= 0 and < Enactment.num_actions().'
		self.actions.pop(index)
		self.apply_actions_to_frames()								#  Refresh all Frame labels
		return

	#  Compare this with the below Enactment.load_sensor_poses().
	def compute_IK_poses(self):
		for time_stamp, frame in sorted(self.frames.items()):
			found_left_hand = False
			found_right_hand = False
			for detection in frame.detections:
				if detection.object_name == 'LeftHand':
					pose = tuple(list(detection.centroid) + [frame.left_hand_pose[3]])
					frame.set_left_hand_pose(pose)
					frame.set_left_hand_global_pose(pose)
					found_left_hand = True
				elif detection.object_name == 'RightHand':
					pose = tuple(list(detection.centroid) + [frame.right_hand_pose[3]])
					frame.set_right_hand_pose(pose)
					frame.set_right_hand_global_pose(pose)
					found_right_hand = True
			if not found_left_hand:
				frame.set_left_hand_pose(None)
				frame.set_left_hand_global_pose(None)
			if not found_right_hand:
				frame.set_right_hand_pose(None)
				frame.set_right_hand_global_pose(None)
		return

	#  Without creating conflicts and without erasing Actions altogether, erode the boundaries of Actions.
	#  If 'index' is provided, then target only that Action. If no 'index' is provided, then erode all Actions.
	#  Note: enactment.erode_action(n, i) is equivalent to calling
	#        enactment.erode_action_head(n, i) and enactment.erode_action_tail(n, i, True).
	def erode_action(self, frames, index=None):
		if index is None:
			indices = [i for i in range(0, len(self.actions))]
		else:
			indices = [index]
		for index in indices:
			self.erode_action_head(frames, index)
			self.erode_action_tail(frames, index)
		self.apply_actions_to_frames()								#  Refresh all Frame labels
		return

	#  The head is the onset of an action.
	def erode_action_head(self, frames, index=None, refresh=False):
		if index is None:
			indices = [i for i in range(0, len(self.actions))]
		else:
			indices = [index]

		sorted_frames = sorted(self.frames.items())
		sorted_frame_names = [x[1].file_name for x in sorted_frames]

		for index in indices:										#  Index into self.actions
																	#  Find the indices of the start and end frames in sorted_frames
			start_index = sorted_frame_names.index(self.actions[index].start_frame.split('/')[-1])
			end_index = sorted_frame_names.index(self.actions[index].end_frame.split('/')[-1])

			if start_index + frames < end_index - 1:				#  We can make this head-erosion and not kill the action.
				self.actions[index].start_time = sorted_frames[start_index + frames][1].time_stamp
				self.actions[index].start_frame = sorted_frames[start_index + frames][1].path + '/NormalViewCameraFrames/' + sorted_frames[start_index + frames][1].file_name
			else:													#  Making this head-erosion would kill the action. Leave a frame.
				self.actions[index].start_time = sorted_frames[end_index - 2][1].time_stamp
				self.actions[index].start_frame = sorted_frames[end_index - 2][1].path + '/NormalViewCameraFrames/' + sorted_frames[end_index - 2][1].file_name

		if refresh:
			self.apply_actions_to_frames()							#  Refresh all Frame labels
		return

	#  The tail is the fading of an action.
	def erode_action_tail(self, frames, index=None, refresh=False):
		if index is None:
			indices = [i for i in range(0, len(self.actions))]
		else:
			indices = [index]

		sorted_frames = sorted(self.frames.items())
		sorted_frame_names = [x[1].file_name for x in sorted_frames]

		for index in indices:										#  Index into self.actions
																	#  Find the indices of the start and end frames in sorted_frames
			start_index = sorted_frame_names.index(self.actions[index].start_frame.split('/')[-1])
			end_index = sorted_frame_names.index(self.actions[index].end_frame.split('/')[-1])

			if end_index - frames > start_index:					#  We can make this tail-erosion and not kill the action.
				self.actions[index].end_time = sorted_frames[end_index - frames][1].time_stamp
				self.actions[index].end_frame = sorted_frames[end_index - frames][1].path + '/NormalViewCameraFrames/' + sorted_frames[end_index - frames][1].file_name
			else:													#  Making this tail-erosion would kill the action. Leave a frame.
				self.actions[index].end_time = sorted_frames[start_index + 1][1].time_stamp
				self.actions[index].end_frame = sorted_frames[start_index + 1][1].path + '/NormalViewCameraFrames/' + sorted_frames[start_index + 1][1].file_name

		if refresh:
			self.apply_actions_to_frames()							#  Refresh all Frame labels
		return

	#  Without creating conflicts, dilate the boundaries of Actions.
	#  If 'index' is provided, then target only that Action. If no 'index' is provided, then dilate all Actions.
	#  Note: enactment.dilate_action(n, i) is equivalent to calling
	#        enactment.dilate_action_head(n, i) and enactment.dilate_action_tail(n, i, True).
	def dilate_action(self, frames, index=None):
		if index is None:
			indices = [i for i in range(0, len(self.actions))]
		else:
			indices = [index]
		for index in indices:
			self.dilate_action_head(frames, index)
			self.dilate_action_tail(frames, index)
		self.apply_actions_to_frames()								#  Refresh all Frame labels
		return

	#  The head is the onset of an action.
	def dilate_action_head(self, frames, index=None, refresh=False):
		if index is None:
			indices = [i for i in range(0, len(self.actions))]
		else:
			indices = [index]

		sorted_frames = sorted(self.frames.items())
		sorted_frame_names = [x[1].file_name for x in sorted_frames]

		for index in indices:										#  Index into self.actions
																	#  Find the indices of the start and end frames in sorted_frames
			start_index = sorted_frame_names.index(self.actions[index].start_frame.split('/')[-1])
			end_index = sorted_frame_names.index(self.actions[index].end_frame.split('/')[-1])

			proposed_start = max(0, start_index - frames)			#  Don't allow extension before the dawn of time
			for i in range(0, len(self.actions)):					#  Temper extended start time against collisions
				if i != index and \
				   sorted_frame_names.index(self.actions[i].start_frame.split('/')[-1]) < proposed_start and \
				   sorted_frame_names.index(self.actions[i].end_frame.split('/')[-1]) > proposed_start:
					proposed_start = sorted_frame_names.index(self.actions[i].end_frame.split('/')[-1])

			self.actions[index].start_time = sorted_frames[proposed_start][1].time_stamp
			self.actions[index].start_frame = sorted_frames[proposed_start][1].path + '/NormalViewCameraFrames/' + sorted_frames[proposed_start][1].file_name

		if refresh:
			self.apply_actions_to_frames()							#  Refresh all Frame labels
		return

	#  The tail is the fading of an action.
	def dilate_action_tail(self, frames, index=None, refresh=False):
		if index is None:
			indices = [i for i in range(0, len(self.actions))]
		else:
			indices = [index]

		sorted_frames = sorted(self.frames.items())
		sorted_frame_names = [x[1].file_name for x in sorted_frames]

		for index in indices:										#  Index into self.actions
																	#  Find the indices of the start and end frames in sorted_frames
			start_index = sorted_frame_names.index(self.actions[index].start_frame.split('/')[-1])
			end_index = sorted_frame_names.index(self.actions[index].end_frame.split('/')[-1])

																	#  Don't allow extension past the end of time
			proposed_end = min(len(sorted_frames) - 1, end_index + frames)
			for i in range(0, len(self.actions)):					#  Temper extended end time against collisions
				if i != index and \
				   sorted_frame_names.index(self.actions[i].start_frame.split('/')[-1]) < proposed_end and \
				   sorted_frame_names.index(self.actions[i].end_frame.split('/')[-1]) > proposed_end:
					proposed_end = sorted_frame_names.index(self.actions[i].start_frame.split('/')[-1]) - 1

			self.actions[index].end_time = sorted_frames[proposed_end][1].time_stamp
			self.actions[index].end_frame = sorted_frames[proposed_end][1].path + '/NormalViewCameraFrames/' + sorted_frames[proposed_end][1].file_name

		if refresh:
			self.apply_actions_to_frames()							#  Refresh all Frame labels
		return

	#################################################################
	#  Update: apply changes made to actions to enactment frames.   #
	#################################################################

	#  Whatever the current state of the 'actions' attribute, use those in-s and out-s to label all Enactment Frames.
	def apply_actions_to_frames(self):
		if self.verbose:
			print('>>> Applying labels to frames.')

		head_data = self.load_head_poses()
		video_frames = self.load_frame_sequence(True)
		time_stamps = [x['timestamp'] for x in head_data['list']]

		num_frames = len(video_frames)
		prev_ctr = 0
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.

		for i in range(0, num_frames):
			time_stamp = time_stamps[i]
			applied = False

			for j in range(0, len(self.actions)):					#  Snap to frame indices. Beginning is INCLUSIVE; ending is EXCLUSIVE.
				if i >= video_frames.index(self.actions[j].start_frame) and \
				   i < video_frames.index(self.actions[j].end_frame):
					self.frames[time_stamp].ground_truth_label = self.actions[j].label
					applied = True
			if not applied:
				self.frames[time_stamp].ground_truth_label = '*'	#  Signify that this frame was examined and simply has no label.

			if self.verbose:
				if int(round(float(i) / float(num_frames) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(i) / float(num_frames) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(i) / float(num_frames) * 100.0))) + '%]')
					sys.stdout.flush()
		if self.verbose:
			print('')

		return

	#################################################################
	#  Object-Logic: apply rules and conditions to group color-map- #
	#  coded parts into recognizable objects with state.            #
	#################################################################

	#  Use the color map provided with the enactment and the rules dictionary we may (or may not) have
	#  to determine 'detections' for every Frame of this Enactment.
	def logical_parse(self, rules_file=None):
		if rules_file is not None:
			self.load_rules(rules_file)

		for time_stamp, frame in self.frames.items():				#  Dump any existing object parses.
			frame.detections = []
		self.recognizable_objects = []								#  (Re)set the list
		self.robject_colors = {}									#  (Re)set color lookup

		colormap = self.load_colormap()
		self.object_detection_source = 'GT'							#  Using rules and color maps? Then you're using Ground Truth.

		part_lookup_colorAndLink = {}								#  e.g. key:Target1_AuxiliaryFeederBox_Auxiliary.Jaw1
																	#         ==>  val:{rgb:(255, 205, 0),
																	#                   link:OpenClose0/Users/Subprops/Target1_AuxiliaryFeederBox_Auxiliary.Jaw1.fvr}
		color_lookup_part = {}										#  e.g. key:(255, 205, 0) ==> Target1_AuxiliaryFeederBox_Auxiliary.Jaw1

		for colorkeyed_item in colormap['list']:					#  Iterate over all color-keyed items
			part_lookup_colorAndLink[ colorkeyed_item['label'] ] = {}
			part_lookup_colorAndLink[ colorkeyed_item['label'] ]['rgb'] = (colorkeyed_item['color']['r'], colorkeyed_item['color']['g'], colorkeyed_item['color']['b'])
																	#  Ignore what we ignore
			if self.rules is not None and colorkeyed_item['label'] not in self.rules['IGNORE']:
				color_lookup_part[ (colorkeyed_item['color']['r'], colorkeyed_item['color']['g'], colorkeyed_item['color']['b']) ] = colorkeyed_item['label']
			if len(colorkeyed_item['metadataPath']) > 0:
				part_lookup_colorAndLink[ colorkeyed_item['label'] ]['link'] = self.enactment_name + '/' + colorkeyed_item['metadataPath']
			else:
				part_lookup_colorAndLink[ colorkeyed_item['label'] ]['link'] = None

		base_classes = {}											#  Lookup table of classes, instances, and each instance's parts-strings:
		if self.rules is not None:
			for k, v in self.rules['DEF'].items():					#  key:class-name ==> val:{ [instance-name] ==> [ part-name, part-name, ... ]
				if v not in base_classes:							#                           [instance-name] ==> [ part-name, part-name, ... ]
					base_classes[v] = {}							#                              ...
				base_classes[v][k] = []								#                         }

		#  Iterate over all PARTS:
		#    Does the part have a '.' in it? Then look up the instance name among 'base_classes': Add part to that instance's (all instances) list.
		#    Does the part NOT have a '.' in it? Then look up the part name (whole) among 'base_classes': Add part (same as instance) to instance's list.
		for k, v in part_lookup_colorAndLink.items():				#  Target1_AuxiliaryFeederBox_Auxiliary.Jaw1 ==> {'rgb', 'link'}
			whole_name = k
			if '.' in k:
				instance_name = k.split('.')[0]
			else:
				instance_name = k

			for class_k, class_v in base_classes.items():			#  ControlPanel ==> {'Spare_ControlPanel_Main': [],
																	#                    'Target1_ControlPanel_Main': [],
																	#                    'Target2_ControlPanel_Main': []}
				for instance_k, instance_v in class_v.items():
					if (instance_k == instance_name or whole_name == instance_k) and k not in instance_v:
						instance_v.append(k)

		#  Iterate over all COLLECT rules:
		#    fill in their parts
		if self.rules is not None:
			for k, v in self.rules['COLL'].items():					#  key:(part-name, part-name, ...) ==> val:(label, label, ..., instance-name)
				instance_name = v[-1]
				for class_k, class_v in base_classes.items():		#  key:class-name ==> val:{key:instance-name ==> val:[part-names, ... ]}
					if instance_name in class_v:
						for part_name in k:
							class_v[instance_name].append(part_name)

		#  Expanded classes cover conditional class assignment
		expanded_classes = {}
		for k, v in base_classes.items():
			expanded_classes[k] = {}
			for kk, vv in v.items():
				expanded_classes[k][kk] = vv[:]
		if self.rules is not None:
			for k, v in self.rules['COND'].items():
				#  COND_rules:
				#  e.g. MainFeederBox ==> ('Door', 'hingeStatus', int(0)) ==> MainFeederBox_Closed
				#                         ('Door', 'hingeStatus', int(1)) ==> MainFeederBox_Open
				#                         ('Door', 'hingeStatus', int(2)) ==> MainFeederBox_Unknown

				#  classes:
				#  e.g. SafetyPlank ==> 'Spare_TransferFeederBox_Transfer.SafetyPlank'   ==> []
				#                       'Target1_TransferFeederBox_Transfer.SafetyPlank' ==> []
				#                       'Target2_TransferFeederBox_Transfer.SafetyPlank' ==> []
				save_dict = {}
				for kk, vv in expanded_classes[k].items():
					save_dict[kk] = vv

				del expanded_classes[k]

				for cond_class_name in v.values():
					expanded_classes[cond_class_name] = save_dict

		#  Unaffected by conditionals
		part_lookup_instances = {}									#  e.g. key:Spare_AuxiliaryFeederBox_Auxiliary.Connector1
																	#       val:[Spare_AuxiliaryFeederBox_Auxiliary, Disconnect0]
		for class_name, inst_parts_dict in base_classes.items():	#  AuxiliaryFeederBox ==> {Spare_AuxiliaryFeederBox_Auxiliary ==> [],
			for k, v in inst_parts_dict.items():					#                          Target1_AuxiliaryFeederBox_Auxiliary ==> [],
				for vv in v:										#                          Target2_AuxiliaryFeederBox_Auxiliary ==> []}
					if vv not in part_lookup_instances:
						part_lookup_instances[vv] = []
					part_lookup_instances[vv].append(k)

		#  Iterate over all PARTS:
		#    Consult all classes ==> instances and find the unique instance in which this part appears
		instance_lookup_classes = {}
		for k, v in expanded_classes.items():						#  class-name ==> instance ==> [parts]
			for kk, vv in v.items():								#                 instance ==> [parts] ...
				if kk not in instance_lookup_classes:				#  instance-name ==> [parts]
					instance_lookup_classes[kk] = []
				instance_lookup_classes[kk].append(k)				#  instance-name ==> [all classes to which this instance CAN belong]

		instance_lookup_pseudoclass = {}
		for k, v in base_classes.items():							#  class-name ==> instance ==> [parts]
			for kk, vv in v.items():								#                 instance ==> [parts] ...
				instance_lookup_pseudoclass[kk] = k					#  instance-name ==> [parts]
																	#  instance-name ==> pseudo-class to which this instance belongs
																	#  (Pseudo-classes may yet be subject to conditionals)

		#############################################################
		#  Write the parse log file.                                #
		#############################################################

		now = datetime.datetime.now()								#  Build a distinct substring so I don't accidentally overwrite results.
		file_timestamp = now.strftime("%d") + now.strftime("%m") + now.strftime("%Y")[-2:] + 'T' + now.strftime("%H:%M:%S").replace(':', '')

		fh = open('logical_parse-' + file_timestamp + '.log', 'w')
		if rules_file is not None:
			fh.write('#  Log file of Enactment.logical_parse("' + rules_file + '"), run at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		else:
			fh.write('#  Log file of Enactment.logical_parse(), run at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		log_str = 'PARTS:\n'										#  Print all parts: r g b    part-name
		ctr = 1														#  (Note: this will include things ignored and "permitted")
		for k, v in sorted(part_lookup_colorAndLink.items()):
			log_str += '\t' + str(ctr) + ':\t' + ' '.join([str(x) for x in v['rgb']]) + '\t' + k + '\n'
			ctr += 1

		log_str += 'INSTANCES:\n'									#  Print all instances: instance-name    class-name OR class-name class-name class-name...
		ctr = 1														#                       followed by all colors that make up this instance
																	#  instance_lookup_classes{} = instance-name ==> [class, class, ... ]
		for k, v in sorted(instance_lookup_classes.items()):
			log_str += '\t' + str(ctr) + ':\t' + k + '\t' + ' '.join(v) + '\n'

			colorstr = '\t\t'										#  All colors for all parts of this instance
			for k2, v2 in base_classes.items():						#  class-name ==> {instance ==> [parts]; instance ==> [parts]; ... }
				for k3, v3 in v2.items():							#  instance-name ==> [parts]
					if k3 == k:
						for coloritem in v3:
							colorstr += '  (' + str(part_lookup_colorAndLink[coloritem]['rgb'][0]) + ' ' + \
							                    str(part_lookup_colorAndLink[coloritem]['rgb'][1]) + ' ' + \
							                    str(part_lookup_colorAndLink[coloritem]['rgb'][2]) + ')'
			log_str += colorstr + '\n'
			ctr += 1

		log_str += 'TYPES:\n'										#  Print all classes: class-anme    instance instance instance...
		ctr = 1
		for k, v in sorted(expanded_classes.items()):
			self.recognizable_objects.append(k)						#  Build up list of recognizable-objects: NOT NECESSARILY WHAT IS SEEN IN THE ENACTMENT!
																	#  Initialize with random colors
			self.robject_colors[k] = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
			log_str += '\t' + str(ctr) + ':\t' + k + '\t' + ' '.join(v.keys()) + '\n'
			ctr += 1

		fh.write(log_str)
		fh.close()

		if self.verbose:
			print(log_str)

		#############################################################
		#  Ok. We have built all the tools we'll need.              #
		#  Proceed to the task of parsing and grouping.             #
		#  Write detections to self.frames.detections.              #
		#############################################################
		if self.verbose:
			print('>>> Grouping parts and objects for all frames.')

		num_frames = len(self.frames)
		prev_ctr = 0
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.

		ctr = 0
		for time_stamp, frame in sorted(self.frames.items()):
			img = cv2.imread(frame.fullpath('color'), cv2.IMREAD_UNCHANGED)
			if img.shape[2] > 3:
				img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

			#########################################################
			#  Go through every unique color in the frame.          #
			#  If that color is in our lookup table, then which     #
			#  rules apply to it? Is it part of a composite object  #
			#  or an object in its own right? Apply rules-logic and #
			#  save color and frame information to RObjects per     #
			#  Frame. This is time-consuming, but it allows us a    #
			#  lot of logical freedom.                              #
			#########################################################
																	#  Do not search every color on record.
																	#  First get only those unique colors present in this frame.
			present_colors = [tuple(x) for x in np.unique(img.reshape(-1, img.shape[2]), axis=0)]
			for present_color in present_colors:
																	#  Compression artifacts persist... build in some potential forgiveness.
				color_variations = [tuple(y) for y in np.unique([tuple(x) for x in product((-self.epsilon, 0, self.epsilon), repeat=3)], axis=0)]
				for color_variation in color_variations:
																	#  Reverse 'present_color' for mapping because it is BGR.
																	#  Color is RGB
					color = tuple(map(lambda i, j: i + j, present_color[::-1], color_variation))

					if color in color_lookup_part:					#  Is the RGB color on record?

						part_name = color_lookup_part[color]		#  Identify the part name keyed by this color.

						rev_color = color[::-1]						#  Bloody OpenCV, man... why do you do everything backwards?
																	#  (We will save colors internally as R, G, B... like sane people do.)
						indices = np.all(img == rev_color, axis=-1)

						if True in indices:							#  We were able to match this color.
							if part_name in self.rules['PERMIT']:	#  This is a special case: do not bother looking for components
								i = 0								#  (But this code goes here anyway, just in case)
								while i < len(frame.detections) and frame.detections[i].instance_name != part_name:
									i += 1
								if i < len(frame.detections):		#  If this instance exists in this Frame, add a color to it.
									frame.detections[i].colormaps.append( (color[0], color[1], color[2]) )
								else:								#  If this instance does not exist in this Frame, create it.
									frame.detections.append( RObject(parent_frame=frame.fullpath(), \
									                                 instance_name=part_name, object_name=part_name, \
									                                 colors=[(color[0], color[1], color[2])], colorsrc=frame.fullpath('color'), \
									                                 detection_source='GT', confidence=1.0) )
							else:
																	#  Retrieve a list of all instances this part comprises
								instances = part_lookup_instances[part_name]

								#  A PART MAY BELONG TO SEVERAL INSTANCES, so FOR EACH INSTANCE...
								#  And each instance may be subject to condition, affecting the class-label assigned to this part.

								for instance_name in instances:
																	#  Only ONE label in the pseudo-class lookup--though these may be subject to condition
									pseudo_class = instance_lookup_pseudoclass[instance_name]
																	#  CLASS ASSIGNMENT IS SUBJECT TO CONDITION!
									if pseudo_class in self.rules['COND']:
										#  Find the first test to pass...
										assigned_class = None		#  Find the condition that determines 'pseudo-class-name's assigned class

										#  rules['COND']:	key:pseudo-class-name ==> val:{key:(part, attribute, value) ==> val:class-name
										#                                                  key:(part, attribute, value) ==> val:class-name
										#                                                  key:(part, attribute, value) ==> val:class-name
										#                                                         ... }

										#  rules['COND'][pseudo_class] = key:(part, attribute, value) ==> val:class-name
										#                                key:(part, attribute, value) ==> val:class-name
										#                                key:(part, attribute, value) ==> val:class-name
										#                          e.g.  key:(*, hingeStatus, 0) ==> val:SafetyPlank_Closed
										#                          e.g.  key:(Door, hingeStatus, 0) ==> val:MainFeederBox_Closed
										#                          e.g.  key:(Jaw, hingeStatus, 0) ==> val:Disconnect_Closed
																	#  Check all conditional rules
										for cond_rule, resultant_class in self.rules['COND'][pseudo_class].items():
											determining_component = cond_rule[0]
											determining_attribute = cond_rule[1]
											determining_value     = cond_rule[2]
																	#  None stands in for '*', which is a reference to "self"
											if determining_component is None:
												determining_component = part_name
											else:					#  The determining component is given--but may still be artificial
																	#  Check in COLL rules: a moniker will have been assigned
												#  Is the pseudo-class a collection (COLL)?
												defined_among_collections = False
												for coll_parts, coll_labels in self.rules['COLL'].items():
													if coll_labels[-1] == instance_name:
														defined_among_collections = True
														i = 0
														while i < len(coll_parts) and coll_labels[i] != determining_component:
															i += 1
														determining_component = coll_parts[i]
														break

												if not defined_among_collections:
													determining_component = instance_name + '.' + determining_component

																	#  Open the JSON
											jsonfh = open(part_lookup_colorAndLink[determining_component]['link'], 'r')
																	#  Convert string to JSON
											line = jsonfh.readlines()[0]
											jsonfh.close()			#  Find the string inside the JSON
											component_history = json.loads(line)
																	#  Make that a JSON, too.... what the hell?
											unpacked_string = json.loads(component_history['frames'])
											i = 0
											j = 0
											while j < len(unpacked_string['list']):
												if unpacked_string['list'][j]['timestamp'] == time_stamp:
													val = unpacked_string['list'][j][determining_attribute]
													if val == determining_value:
														assigned_class = resultant_class
													break
												j += 1
											i += 1

										if assigned_class is not None:
											i = 0					#  Does this instance already exist in this Frame, under this class?
											while i < len(frame.detections) and not(frame.detections[i].instance_name == instance_name and \
											                                        frame.detections[i].object_name == assigned_class):
												i += 1
																	#  If this instance exists in this Frame, add a color to it.
											if i < len(frame.detections):
												frame.detections[i].colormaps.append( (color[0], color[1], color[2]) )
											else:					#  If this instance does not exist in this Frame, create it.
												frame.detections.append( RObject(parent_frame=frame.fullpath(), \
												                                 instance_name=instance_name, object_name=assigned_class, \
												                                 colors=[(color[0], color[1], color[2])], colorsrc=frame.fullpath('color'), \
												                                 detection_source='GT', confidence=1.0) )
									else:							#  Class assignment is NOT subject to condition
										i = 0						#  Does this instance already exist in this Frame, under this class?
										while i < len(frame.detections) and not(frame.detections[i].instance_name == instance_name and \
								        		                                frame.detections[i].object_name == pseudo_class):
											i += 1
																	#  If this instance exists in this Frame, add a color to it.
										if i < len(frame.detections):
											frame.detections[i].colormaps.append( (color[0], color[1], color[2]) )
										else:						#  If this instance does not exist in this Frame, create it.
											frame.detections.append( RObject(parent_frame=frame.fullpath(), \
											                                 instance_name=instance_name, object_name=pseudo_class, \
											                                 colors=[(color[0], color[1], color[2])], colorsrc=frame.fullpath('color'), \
											                                 detection_source='GT', confidence=1.0) )

			if self.verbose:
				if int(round(float(ctr) / float(num_frames) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(ctr) / float(num_frames) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(ctr) / float(num_frames) * 100.0))) + '%]')
					sys.stdout.flush()
			ctr += 1

		return

	#  Write all parsed RObjects to mask images and write a mask lookup-table, "<self.enactment_name>_props.txt".
	def render_parsed(self):
		if self.verbose:
			print('>>> Rendering Recognizable-Object masks.')

		directory_name = self.enactment_name + '/' + self.object_detection_source
		if os.path.isdir(directory_name):							#  If 'directory_name' exists, delete it
			shutil.rmtree(directory_name)
		os.mkdir(directory_name)									#  Create directory_name

		K = self.K()												#  Retrieve camera matrix
		K_inv = np.linalg.inv(K)									#  Build inverse K-matrix
																	#  Build the flip matrix
		flip = np.array([[-1.0,  0.0, 0.0], \
		                 [ 0.0, -1.0, 0.0], \
		                 [ 0.0,  0.0, 1.0]], dtype='float64')

		mask_ctr = 0
		frame_ctr = 0
		fh = open(self.enactment_name + '_props.txt', 'w')			#  Create "<self.enactment_name>_props.txt"
		fh.write('#  Enactment object-mask lookup-table, created ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		fh.write('#  WIDTH & HEIGHT:\n')
		fh.write('#    ' + str(self.width) + '\t' + str(self.height) + '\n')
		fh.write('#  FPS:\n')
		fh.write('#    ' + str(self.fps) + '\n')
		if self.object_detection_source is not None:
			fh.write('#  OBJECT DETECTION SOURCE:\n')
			fh.write('#    ' + self.object_detection_source + '\n')
		fh.write('#  RECOGNIZABLE OBJECTS:\n')
																	#  self.recognizable_objects will have been filled if we just performed logical_parse()
		fh.write('#    ' + '\t'.join(self.recognizable_objects) + '\n')
		fh.write('#  FORMAT:\n')
		fh.write('#    timestamp    image-filename    instance    class    detection-source    confidence    bounding-box    mask-filename    3D-centroid-Avg    3D-centroid-BBox\n')

		num_frames = self.num_frames()
		prev_ctr = 0
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.

		for time_stamp, frame in sorted(self.frames.items()):
																	#  Cache the Frame's depth map once, here.
			depthmap = cv2.imread(frame.fullpath('depth'), cv2.IMREAD_UNCHANGED)

			for detection in frame.detections:
				detection.render_mask(directory_name + '/mask_' + str(mask_ctr) + '.png')

				fh.write(str(time_stamp) + '\t')					#  Write time_stamp
				fh.write(frame.fullpath() + '\t')					#  Write video file full path
				fh.write(detection.instance_name + '\t')			#  Write instance name
				fh.write(detection.object_name + '\t')				#  Write object name
				fh.write(detection.detection_source + '\t')			#  Write detection source
				fh.write(str(detection.confidence) + '\t')			#  Write detection confidence
				fh.write(str(detection.bounding_box[0]) + ',' + str(detection.bounding_box[1]) + ';' + \
				         str(detection.bounding_box[2]) + ',' + str(detection.bounding_box[3]) + '\t')
				fh.write(detection.mask_path + '\t')				#  Write mask path

				center = detection.center('avg')
																	#  In meters
				d = self.min_depth + (float(depthmap[min(center[1], self.height - 1), min(center[0], self.width - 1)]) / 255.0) * (self.max_depth - self.min_depth)
				centroid = np.dot(K_inv, np.array([center[0], center[1], 1.0]))
				centroid *= d										#  Scale by known depth (meters from head)
				pt = np.dot(flip, centroid)							#  Flip point
				detection.set_centroid( (pt[0], pt[1], pt[2]) )
																	#  Write the 3D centroid (determined by pixels' average)
				fh.write(str(detection.centroid[0]) + ',' + str(detection.centroid[1]) + ',' + str(detection.centroid[2]) + '\t')

				center = detection.center('bbox')
																	#  In meters
				d = self.min_depth + (float(depthmap[min(center[1], self.height - 1), min(center[0], self.width - 1)]) / 255.0) * (self.max_depth - self.min_depth)
				centroid = np.dot(K_inv, np.array([center[0], center[1], 1.0]))
				centroid *= d										#  Scale by known depth (meters from head)
				pt = np.dot(flip, centroid)							#  Flip point
				detection.set_centroid( (pt[0], pt[1], pt[2]) )
																	#  Write the 3D centroid (determined by bounding box)
				fh.write(str(detection.centroid[0]) + ',' + str(detection.centroid[1]) + ',' + str(detection.centroid[2]) + '\n')

				mask_ctr += 1

			if self.verbose:
				if int(round(float(frame_ctr) / float(num_frames) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(frame_ctr) / float(num_frames) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(frame_ctr) / float(num_frames) * 100.0))) + '%]')
					sys.stdout.flush()

			frame_ctr += 1

		fh.close()
		return

	#  Write all detected RObjects (masks already exist) to a mask lookup-table, "<self.enactment_name>_<detector_name>_detections.txt"
	def render_detected(self, detector_name=None):
		K = self.K()												#  Retrieve camera matrix
		K_inv = np.linalg.inv(K)									#  Build inverse K-matrix
																	#  Build the flip matrix
		flip = np.array([[-1.0,  0.0, 0.0], \
		                 [ 0.0, -1.0, 0.0], \
		                 [ 0.0,  0.0, 1.0]], dtype='float64')

		mask_ctr = 0
		frame_ctr = 0
		if detector_name is None:
			fh = open(self.enactment_name + '_detections.txt', 'w')	#  Create "<self.enactment_name>_detections.txt"
		else:
			fh = open(self.enactment_name + '_' + detector_name + '_detections.txt', 'w')

		fh.write('#  Enactment object-mask lookup-table, created ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		fh.write('#  WIDTH & HEIGHT:\n')
		fh.write('#    ' + str(self.width) + '\t' + str(self.height) + '\n')
		fh.write('#  FPS:\n')
		fh.write('#    ' + str(self.fps) + '\n')
		if self.object_detection_source is not None:
			fh.write('#  OBJECT DETECTION SOURCE:\n')
			fh.write('#    ' + self.object_detection_source + '\n')
		fh.write('#  RECOGNIZABLE OBJECTS:\n')
																	#  self.recognizable_objects will have been filled if we just performed logical_parse()
		fh.write('#    ' + '\t'.join(self.recognizable_objects) + '\n')
		fh.write('#  FORMAT:\n')
		fh.write('#    timestamp    image-filename    instance    class    detection-source    confidence    bounding-box    mask-filename    3D-centroid-Avg    3D-centroid-BBox\n')

		num_frames = self.num_frames()
		prev_ctr = 0
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.

		for time_stamp, frame in sorted(self.frames.items()):
																	#  Cache the Frame's depth map once, here.
			depthmap = cv2.imread(frame.fullpath('depth'), cv2.IMREAD_UNCHANGED)

			for detection in frame.detections:
				fh.write(str(time_stamp) + '\t')					#  Write time_stamp
				fh.write(frame.fullpath() + '\t')					#  Write video file full path
				fh.write('*' + '\t')								#  Write instance name (detections do not pick up "instances")
				fh.write(detection.object_name + '\t')				#  Write object name
				fh.write(detection.detection_source + '\t')			#  Write detection source
				fh.write(str(detection.confidence) + '\t')			#  Write detection confidence
				fh.write(str(detection.bounding_box[0]) + ',' + str(detection.bounding_box[1]) + ';' + \
				         str(detection.bounding_box[2]) + ',' + str(detection.bounding_box[3]) + '\t')
				fh.write(detection.mask_path + '\t')				#  Write mask path

				center = detection.center('avg')
																	#  In meters
				d = self.min_depth + (float(depthmap[min(center[1], self.height - 1), min(center[0], self.width - 1)]) / 255.0) * (self.max_depth - self.min_depth)
				centroid = np.dot(K_inv, np.array([center[0], center[1], 1.0]))
				centroid *= d										#  Scale by known depth (meters from head)
				pt = np.dot(flip, centroid)							#  Flip point
				detection.set_centroid( (pt[0], pt[1], pt[2]) )
																	#  Write the 3D centroid (determined by pixels' average)
				fh.write(str(detection.centroid[0]) + ',' + str(detection.centroid[1]) + ',' + str(detection.centroid[2]) + '\t')

				center = detection.center('bbox')
																	#  In meters
				d = self.min_depth + (float(depthmap[min(center[1], self.height - 1), min(center[0], self.width - 1)]) / 255.0) * (self.max_depth - self.min_depth)
				centroid = np.dot(K_inv, np.array([center[0], center[1], 1.0]))
				centroid *= d										#  Scale by known depth (meters from head)
				pt = np.dot(flip, centroid)							#  Flip point
				detection.set_centroid( (pt[0], pt[1], pt[2]) )
																	#  Write the 3D centroid (determined by bounding box)
				fh.write(str(detection.centroid[0]) + ',' + str(detection.centroid[1]) + ',' + str(detection.centroid[2]) + '\n')

				mask_ctr += 1

			if self.verbose:
				if int(round(float(frame_ctr) / float(num_frames) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(frame_ctr) / float(num_frames) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(frame_ctr) / float(num_frames) * 100.0))) + '%]')
					sys.stdout.flush()

			frame_ctr += 1

		fh.close()
		return

	#  Load RObject parse data from file.
	def load_parsed_objects(self, props_file=None, centroid_src='bbox'):
		assert isinstance(centroid_src, str) and (centroid_src == 'bbox' or centroid_src == 'avg'), \
		       'Argument \'centroid_src\' passed to Enactment.load_parsed_objects() must be a string in {avg, bbox}.'
		if props_file is None:
			props_file = self.enactment_name + '_props.txt'

		if self.verbose:
			print('>>> Loading parsed Recognizable-Objects from file "' + props_file + '".')
			print('    Object 3D centroids come from "' + centroid_src + '"')

		timed_frames = sorted(self.frames.items())
		time_stamps = [x[0] for x in timed_frames]
		frames = [x[1].file_name for x in timed_frames]				#  Full-path names

		for time_stamp, frame in timed_frames:
			frame.detections = []									#  Clear out any previous detections.
		self.recognizable_objects = []								#  (Re)set
		self.robject_colors = {}									#  (Re)set

		prev_ctr = 0
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.

		fh = open(props_file, 'r')
		lines = fh.readlines()
		fh.close()
		num_lines = len(lines)
		line_ctr = 0
		reading_robjects = False
		reading_object_detection_source = False
		for line in lines:
			if line[0] == '#':										#  Comment/header
				if 'RECOGNIZABLE OBJECTS' in line:					#  Flag meaning that the following line contains all recognizable objects.
					reading_robjects = True
				elif reading_robjects:
					arr = line[1:].strip().split('\t')
					self.recognizable_objects = arr[:]				#  Copy the list of objects.
					for robject in self.recognizable_objects:		#  Initialize with random colors, just so this attribute isn't vacant.
						self.robject_colors[robject] = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
					reading_robjects = False

				if 'OBJECT DETECTION SOURCE' in line:				#  Flag meaning that the following line contains the object detection source.
					reading_object_detection_source = True
				elif reading_object_detection_source:
					arr = line[1:].strip().split('\t')
					self.object_detection_source = arr[0]			#  Copy the object detection source.
					reading_object_detection_source = False
			else:
				arr = line.strip().split('\t')
				time_stamp = float(arr[0])							#  Retrieve time stamp
				file_path = arr[1]									#  Retrieve file path
				file_name = arr[1].split('/')[-1]
				instance_name = arr[2]								#  Instance name
				object_name = arr[3]								#  The class of object that instance is
				detection_source = arr[4]
				confidence = float(arr[5])
				bbox = arr[6].split(';')
				bbox = bbox[0].split(',') + bbox[1].split(',')
				bbox = tuple([int(x) for x in bbox])
				mask_path = arr[7]									#  Path to this instance's mask
				centroid_3d_avg = tuple([float(x) for x in arr[8].split(',')])
				centroid_3d_bbox = tuple([float(x) for x in arr[9].split(',')])

				if centroid_src == 'bbox':							#  Why use the full-path file name and not the floats?
																	#  Because I don't trust float-to-string-to-float conversions.
					timed_frames[ frames.index(file_name) ][1].detections.append( RObject(parent_frame=file_path, \
					                                                                      instance_name=instance_name, \
					                                                                      object_name=object_name, \
					                                                                      detection_source=detection_source, \
					                                                                      mask_path=mask_path, \
					                                                                      bounding_box=bbox, \
					                                                                      confidence=confidence, \
					                                                                      centroid=centroid_3d_bbox) )
				else:
					timed_frames[ frames.index(file_name) ][1].detections.append( RObject(parent_frame=file_path, \
					                                                                      instance_name=instance_name, \
					                                                                      object_name=object_name, \
					                                                                      detection_source=detection_source, \
					                                                                      mask_path=mask_path, \
					                                                                      bounding_box=bbox, \
					                                                                      confidence=confidence, \
					                                                                      centroid=centroid_3d_avg) )
			if self.verbose:
				if int(round(float(line_ctr) / float(num_lines) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(line_ctr) / float(num_lines) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(line_ctr) / float(num_lines) * 100.0))) + '%]')
					sys.stdout.flush()
			line_ctr += 1

		return

	#  Load rules from the given file and determine the classes/objects/props to be parsed.
	#    self.rules['IGNORE']:  key:string ==> True
	#    self.rules['PERMIT']:  key:string ==> True
	#    self.rules['DEF']:     key:instance-name ==> val:class-name
	#    self.rules['COLL']:    key:(part-name, part-name, ...) ==> val:(label, label, ..., class-name)
	#    self.rules['COND']:    key:pseudo-class-name ==> val:{key:(part, attribute, value) ==> val:class-name
	#                                                          key:(part, attribute, value) ==> val:class-name
	#                                                          key:(part, attribute, value) ==> val:class-name
	#                                                                               ... }
	def load_rules(self, rules_file):
		IGNORE_rules = {}											#  Things we do NOT wish to consider
		PERMIT_rules = {}											#  Things we wish to consider but which are not classes

		#  For all colored items, if it contains a '.' then initially assume that it is a PART of the INSTANCE to the LEFT
		#  These assumptions may or may not be overridden.
		#  An instance IS NEVER a class-name: they all have something prefixed like Target1_ DeezNuts_, etc.

		DEF_rules = {}												#  key:instance-name ==> val:class-name

		COLL_rules = {}												#  key:(part-name, part-name, ...) ==> val:(label, label, ..., instance-name)

		COND_rules = {}												#  key:pseudo-class-name ==> val:{key:(part, attribute, value) ==> val:class-name
																	#                                 key:(part, attribute, value) ==> val:class-name
																	#                                 key:(part, attribute, value) ==> val:class-name
																	#                                                ... }
		fh = open(rules_file, 'r')									#  Load object-parsing rules
		lines = fh.readlines()
		fh.close()
		for line in lines:
			if line[0] != '#' and len(line.strip()) > 0:			#  Skip comments and empties
				arr = line.strip().split('\t')

				if arr[0] == 'IGNORE':								#  Ignore the Environment
					IGNORE_rules[ arr[1] ] = True

				elif arr[0] == 'PERMIT':							#  Track e.g. the hands but do not consider them "props" per se.
					PERMIT_rules[ arr[1] ] = True

				elif arr[0] == 'DEFINE':							#  e.g. Global_SubstationTag_Yellow2 is a YellowTag
					DEF_rules[ arr[1] ] = arr[2]					#  DEF_rules['Global_SubstationTag_Yellow2'] ==> 'YellowTag'

				elif arr[0] == 'COND':								#  e.g. Spare_AuxiliaryFeederBox_Auxiliary is an AuxiliaryFeederBox_Closed
																	#       if component Door has hingeStatus == int(0)
					if arr[1] not in COND_rules:
						COND_rules[ arr[1] ] = {}					#  COND_rules['AuxiliaryFeederBox'][ ( 'Door', 'hingeStatus', int(0) ) ] ==>
																	#                                                 'AuxiliaryFeederBox_Closed'
					if arr[3] == '*':
						defining_component = None
					else:
						defining_component = arr[3]

					defining_attribute = arr[4]

					if arr[5] == 'int':
						defining_value = int(arr[6])
					elif arr[5] == 'float':
						defining_value = float(arr[6])
					elif arr[5] == 'str':
						defining_value = arr[6]

					COND_rules[ arr[1] ][ (defining_component, defining_attribute, defining_value) ] = arr[2]

				elif arr[0] == 'COLLECT':							#  e.g. Target1_TransferFeederBox_Transfer.Connector1 is known internally as Contact
																	#       Target1_TransferFeederBox_Transfer.Jaw1	is known internally as  Jaw
																	#       in the instance Disconnect19
					instance_name = arr[-1]
					rule = arr[1:-1]								#  COLL_rules[ ( Target2_TransferFeederBox_Transfer.Connector1,
					keys = []										#                Target2_TransferFeederBox_Transfer.Jaw1         ) ]
					labels = []										#   ==> (Contact, Jaw, Disconnect31)
					for i in range(0, len(rule), 3):
						keys.append(rule[i])
						labels.append(rule[i + 2])
					keys = tuple(keys)
					labels = tuple(labels + [instance_name])

					COLL_rules[ keys ] = labels

		self.rules = {}
		self.rules['IGNORE'] = IGNORE_rules
		self.rules['PERMIT'] = PERMIT_rules
		self.rules['DEF'] = DEF_rules
		self.rules['COLL'] = COLL_rules
		self.rules['COND'] = COND_rules
		return

	#  Load an RGB 3-tuple for each string in self.recognizable_objects.
	def load_color_lookup(self, color_file):
		self.random_colors()										#  What if the colors file is missing something? Initialize with randoms.

		fh = open(color_file, 'r')									#  Now attempt to open the given file.
		for line in fh.readlines():
			if line[0] != '#':
				arr = line.strip().split('\t')
				self.robject_colors[ arr[0] ] = (int(arr[1]), int(arr[2]), int(arr[3]))
		fh.close()

		return

	#  Invent an RGB 3-tuple for each string in self.recognizable_objects.
	def random_colors(self):
		self.robject_colors = {}									#  (Re)set
		for robject in self.recognizable_objects:
			self.robject_colors[robject] = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
		return

	#################################################################
	#  Recap: list the current state of the enactment               #
	#################################################################

	#  Number of action instances--NOT the number of unique action labels.
	def num_actions(self):
		return len(self.actions)

	#  Return a list of unique action labels.
	#  This list is derived from the current state of the self.actions attribute--NOT from the JSON source.
	def labels(self):
		return list(np.unique([x.label for x in self.actions]))

	def num_frames(self):
		return len(self.frames)

	#  Return a list of indices into sorted self.frames covering all Frames that make up self.action[index].
	def action_frameset(self, index):
		assert isinstance(index, int) and index >= 0 and index < len(self.actions), \
		       'Argument \'index\' in Enactment.action_frameset() must be an integer >= 0 and < Enactment.num_actions().'
		f = []

		sorted_frames = sorted(self.frames.items())
		sorted_frame_names = [x[1].file_name for x in sorted_frames]

		for i in range(0, len(sorted_frames)):
			if i >= sorted_frame_names.index(self.actions[index].start_frame.split('/')[-1]) and \
			   i <  sorted_frame_names.index(self.actions[index].end_frame.split('/')[-1]):
				f.append(i)

		return f

	#  Return a list of tuples(time stamp, Frame) covering a given (as opposed to an indexed) Action.
	def action_to_frames(self, action, full_tuple=True):
		f = []

		sorted_frames = sorted(self.frames.items())
		sorted_frame_names = [x[1].file_name for x in sorted_frames]

		for i in range(0, len(sorted_frames)):
			if i >= sorted_frame_names.index(action.start_frame.split('/')[-1]) and \
			   i <  sorted_frame_names.index(action.end_frame.split('/')[-1]):
				if full_tuple:										#  Return the pair (time stamp, Frame object).
					f.append(sorted_frames[i])
				else:												#  Return the Frame object only.
					f.append(sorted_frames[i][1])

		return f

	#  Print out a formatted list of this enactment's actions
	def itemize_actions(self):
		maxlen_label = 0
		maxlen_timestamp = 0
		maxlen_filepath = 0

		maxlen_index = len(str(len(self.actions) - 1))
		for action in self.actions:									#  Discover extrema
			maxlen_label = max(maxlen_label, len(action.label))
			maxlen_timestamp = max(maxlen_timestamp, len(str(action.start_time)), len(str(action.end_time)))
			maxlen_filepath = max(maxlen_filepath, len(action.start_frame.split('/')[-1]), len(action.end_frame.split('/')[-1]))
		i = 0
		for action in self.actions:									#  Print all nice and tidy like
			action.print(index=i, pad_index=maxlen_index, pad_label=maxlen_label, pad_time=maxlen_timestamp, pad_path=maxlen_filepath)
			i += 1
		return

	#  Print out a formatted list of this enactment's frames
	def itemize_frames(self, precision=3):
		maxlen_timestamp = 0
		maxlen_filename = 0
		maxlen_label = 0
		maxlen_dimension = 0

		maxlen_index = len(str(len(self.frames) - 1))
		for time_stamp, frame in sorted(self.frames.items()):		#  Discover extrema
			maxlen_timestamp = max(maxlen_timestamp, len(str(time_stamp)))
			maxlen_filename = max(maxlen_filename, len(frame.file_name))
			maxlen_label = max(maxlen_label, len(frame.ground_truth_label))
			maxlen_dimension = max(maxlen_dimension, len(str(frame.width)), len(str(frame.height)))

		i = 0
		for time_stamp, frame in sorted(self.frames.items()):
			frame.print(index=i, pad_index=maxlen_index, \
			            pad_time=maxlen_timestamp, \
			            pad_file=maxlen_filename, \
			            pad_label=maxlen_label, \
			            pad_dim=maxlen_dimension, \
			            precision=precision)
			i += 1

		return

	#  Return a list of Actions derived from an existing Enactment Action (or from all existing Actions.)
	#  The Actions in this list reflect an atemporal examination of the Enactment because we already know where true Action boundaries are.
	def snippets_from_action(self, window_length, stride, index=None):
		if index is None:
			indices = [i for i in range(0, len(self.actions))]
		else:
			indices = [index]

		head_data = self.load_head_poses()
		video_frames = self.load_frame_sequence(True)
		time_stamps = [x['timestamp'] for x in head_data['list']]

		snippet_actions = []										#  To be returned: a list of Action objects.

		for index in indices:
																	#  Get a list of all frame indices for this action.
																	#  (The +1 at the end ensures that we take the last snippet.)
			frame_indices = range(video_frames.index(self.actions[index].start_frame), video_frames.index(self.actions[index].end_frame) + 1)
			for i in range(0, len(frame_indices) - window_length, stride):
				snippet_actions.append( Action(time_stamps[ frame_indices[i] ], time_stamps[ frame_indices[i + window_length] ], \
				                               start_frame=video_frames[ frame_indices[i] ], end_frame=video_frames[ frame_indices[i + window_length] ], \
				                               label=self.actions[index].label) )

		return snippet_actions

	#  March through time by 'stride', and when the ground-truth label of every frame within 'window_length' is the same, add it to a list and return that list.
	#  The Actions in this list reflect a temporal examination of the Enactment because we do NOT know where true Action boundaries are.
	def snippets_from_frames(self, window_length, stride):
		head_data = self.load_head_poses()
		video_frames = self.load_frame_sequence(True)
		time_stamps = [x['timestamp'] for x in head_data['list']]
		num_frames = len(time_stamps)

		snippet_actions = []										#  To be returned: a list of Action objects.

		for i in range(0, num_frames - window_length, stride):		#  March through time by 'stride'. Halt 'window_length' short of the end of time.
			buffer_labels = [self.frames[time_stamps[i + x]].ground_truth_label for x in range(0, window_length)]
																	#  Labels in the buffer are uniform and not nothing.
			if buffer_labels[0] is not None and buffer_labels[0] != '*' and buffer_labels.count(buffer_labels[0]) == window_length:
				snippet_actions.append( Action(time_stamps[i], time_stamps[i + window_length], \
				                               start_frame=video_frames[i], end_frame=video_frames[i + window_length], \
				                               label=buffer_labels[0]) )

		return snippet_actions

	#  Return a dictionary of all depths for all pixels of all frames in this enactment.
	#  key:depth(in meters) ==> val:count.
	def compute_depth_histogram(self):
		histogram = {}												#  Prepare the histogram--IN METERS
		for i in range(0, 256):
			d = (float(i) / 255.0) * (self.max_depth - self.min_depth) + self.min_depth
			histogram[d] = 0

		if self.verbose:
			print('>>> Computing depth histogram.')

		depth_frames = self.load_depth_sequence(True)
		num_frames = len(depth_frames)
		prev_ctr = 0
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.

		ctr = 0
		for depth_frame in depth_frames:
			depth_map = cv2.imread(depth_frame, cv2.IMREAD_UNCHANGED)

			histg = [int(x) for x in cv2.calcHist([depth_map], [0], None, [256], [0, 256])]
			for i in range(0, 256):
				d = (float(i) / 255.0) * (self.max_depth - self.min_depth) + self.min_depth
				histogram[d] += histg[i]

			if self.verbose:
				if int(round(float(ctr) / float(num_frames) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(ctr) / float(num_frames) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(ctr) / float(num_frames) * 100.0))) + '%]')
					sys.stdout.flush()
			ctr += 1

		if self.verbose:
			print('')

		return histogram

	#################################################################
	#  Rendering: see that what is computed is what you expect.     #
	#################################################################

	#  If a specific index is not given, then each action will be rendered to video.
	#  Include in the rendering as many details as have been prepared. If we computed masks, show the masks. If centroids, show centroids....
	def render_annotated_action(self, index=None):
		if index is None:
			indices = [i for i in range(0, len(self.actions))]
		else:
			indices = [index]

		K = self.K()												#  Build the camera matrix
		sorted_frames = sorted(self.frames.items())
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.

		for index in indices:
			if self.verbose:
				print('>>> Rendering "' + self.enactment_name + '_annotated_' + str(index) + '.avi' + '"')

			action_frames = self.action_frameset(index)				#  Retrieve a list of indices into self.frames for self.action[index]
			num_frames = len(action_frames)
			prev_ctr = 0

			vid = cv2.VideoWriter( self.enactment_name + '_annotated_' + str(index) + '.avi', \
			                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
			                       self.fps, \
			                      (self.width, self.height) )
			i = 0
			for f in action_frames:
																	#  Load the video frame
				img = cv2.imread(sorted_frames[f][1].fullpath(), cv2.IMREAD_UNCHANGED)
				if img.shape[2] == 4:								#  Do these guys have alpha channels? I forget.
					img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

				maskcanvas = np.zeros((self.height, self.width, 3), dtype='uint8')
				for detection in sorted_frames[f][1].detections:	#  Render overlays
					if detection.object_name is not None and detection.enabled and detection.object_name in self.robject_colors:
						object_name = detection.object_name
						mask = cv2.imread(detection.mask_path, cv2.IMREAD_UNCHANGED)
						mask[mask > 1] = 1							#  All values greater than 1 become 1
																	#  Extrude to three channels
						mask = mask[:, :, None] * np.ones(3, dtype='uint8')[None, None, :]
																	#  Convert this to a graphical overlay:
						mask[:, :, 0] *= self.robject_colors[ object_name ][2]
						mask[:, :, 1] *= self.robject_colors[ object_name ][1]
						mask[:, :, 2] *= self.robject_colors[ object_name ][0]

						maskcanvas += mask							#  Add mask to mask accumulator
						maskcanvas[maskcanvas > 255] = 255			#  Clip accumulator to 255

				img = cv2.addWeighted(img, 1.0, maskcanvas, 0.7, 0)	#  Add mask accumulator to source frame
				img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)			#  Flatten alpha

				for detection in sorted_frames[f][1].detections:	#  Render bounding boxes and (bbox) centroids
					if detection.object_name is not None and detection.enabled and detection.object_name in self.robject_colors:
						object_name = detection.object_name
						center_bbox = detection.center('bbox')
						bbox = detection.bounding_box
						cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (self.robject_colors[ object_name ][2], \
						                                                            self.robject_colors[ object_name ][1], \
						                                                            self.robject_colors[ object_name ][0]), 1)
						cv2.circle(img, (center_bbox[0], center_bbox[1]), 5, (self.robject_colors[ object_name ][2], \
						                                                      self.robject_colors[ object_name ][1], \
						                                                      self.robject_colors[ object_name ][0]), 3)

				if sorted_frames[f][1].left_hand_pose is not None:	#  Does the left hand project into the camera?
					x = np.array(sorted_frames[f][1].left_hand_pose[:3]).reshape(3, 1)
					p = np.dot(K, x)
					if p[2] != 0.0:
						p /= p[2]
						x = int(round(p[0][0]))						#  Round and discretize to pixels
						y = int(round(p[1][0]))
						if x >= 0 and x < self.width and y >= 0 and y < self.height:
							cv2.line(img, (self.width - x - 5, self.height - y - 5), \
							              (self.width - x + 5, self.height - y + 5), (0, 255, 0, 255), 3)
							cv2.line(img, (self.width - x - 5, self.height - y + 5), \
							              (self.width - x + 5, self.height - y - 5), (0, 255, 0, 255), 3)
																	#  Write the hand subvector
							cv2.putText(img, "{:.2f}".format(sorted_frames[f][1].left_hand_pose[0]) + ', ' + \
							                 "{:.2f}".format(sorted_frames[f][1].left_hand_pose[1]) + ', ' + \
							                 "{:.2f}".format(sorted_frames[f][1].left_hand_pose[2]) + ': ' + str(sorted_frames[f][1].left_hand_pose[3]), \
							            (self.LH_super['x'], self.LH_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.LH_super['fontsize'], (0, 255, 0, 255), 3)

				if sorted_frames[f][1].right_hand_pose is not None:	#  Does the right hand project into the camera?
					x = np.array(sorted_frames[f][1].right_hand_pose[:3]).reshape(3, 1)
					p = np.dot(K, x)
					if p[2] != 0.0:
						p /= p[2]
						x = int(round(p[0][0]))						#  Round and discretize to pixels
						y = int(round(p[1][0]))
						if x >= 0 and x < self.width and y >= 0 and y < self.height:
							cv2.line(img, (self.width - x - 5, self.height - y - 5), \
							              (self.width - x + 5, self.height - y + 5), (0, 0, 255, 255), 3)
							cv2.line(img, (self.width - x - 5, self.height - y + 5), \
							              (self.width - x + 5, self.height - y - 5), (0, 0, 255, 255), 3)
																	#  Write the hand subvector
							cv2.putText(img, "{:.2f}".format(sorted_frames[f][1].right_hand_pose[0]) + ', ' + \
							                 "{:.2f}".format(sorted_frames[f][1].right_hand_pose[1]) + ', ' + \
							                 "{:.2f}".format(sorted_frames[f][1].right_hand_pose[2]) + ': ' + str(sorted_frames[f][1].right_hand_pose[3]), \
							            (self.RH_super['x'], self.RH_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.RH_super['fontsize'], (0, 0, 255, 255), 3)

																	#  Write the true action
				cv2.putText(img, self.actions[index].label, (self.gt_label_super['x'], self.gt_label_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.gt_label_super['fontsize'], (0, 255, 0, 255), 3)
																	#  Write the frame file name
				cv2.putText(img, sorted_frames[f][1].file_name, (self.filename_super['x'], self.filename_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.filename_super['fontsize'], (0, 255, 0, 255), 3)

				vid.write(img)

				if self.verbose:
					if int(round(float(i) / float(num_frames - 1) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
						prev_ctr = int(round(float(i) / float(num_frames - 1) * float(max_ctr)))
						sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(i) / float(num_frames - 1) * 100.0))) + '%]')
						sys.stdout.flush()
				i += 1

			vid.release()
			if self.verbose:
				print('')

		return

	#  Render a video that overlays recognizable-object masks, superimposes centroids, ground-truth labels, a frame counter, and hand data.
	#  The works, baby.
	def render_annotated_video(self):
		K = self.K()												#  Build the camera matrix
		sorted_frames = sorted(self.frames.items())
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.

		if self.verbose:
			print('>>> Rendering "' + self.enactment_name + '_annotated.avi' + '"')

		num_frames = len(sorted_frames)
		prev_ctr = 0

		vid = cv2.VideoWriter( self.enactment_name + '_annotated.avi', \
		                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
		                       self.fps, \
		                      (self.width, self.height) )
		i = 0
		for time_stamp, frame in sorted_frames:
																	#  Load the video frame
			img = cv2.imread(frame.fullpath(), cv2.IMREAD_UNCHANGED)
			if img.shape[2] == 4:									#  Do these guys have alpha channels? I forget.
				img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

			maskcanvas = np.zeros((self.height, self.width, 3), dtype='uint8')
			for detection in frame.detections:						#  Render overlays
				if detection.object_name is not None and detection.enabled and detection.object_name in self.robject_colors:
					object_name = detection.object_name
					mask = cv2.imread(detection.mask_path, cv2.IMREAD_UNCHANGED)
					mask[mask > 1] = 1								#  All values greater than 1 become 1
																	#  Extrude to three channels
					mask = mask[:, :, None] * np.ones(3, dtype='uint8')[None, None, :]
																	#  Convert this to a graphical overlay:
					mask[:, :, 0] *= self.robject_colors[ object_name ][2]
					mask[:, :, 1] *= self.robject_colors[ object_name ][1]
					mask[:, :, 2] *= self.robject_colors[ object_name ][0]

					maskcanvas += mask								#  Add mask to mask accumulator
					maskcanvas[maskcanvas > 255] = 255				#  Clip accumulator to 255

			img = cv2.addWeighted(img, 1.0, maskcanvas, 0.7, 0)		#  Add mask accumulator to source frame
			img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)				#  Flatten alpha

			for detection in frame.detections:						#  Render bounding boxes and (bbox) centroids
				if detection.object_name is not None and detection.enabled and detection.object_name in self.robject_colors:
					object_name = detection.object_name
					center_bbox = detection.center('bbox')
					bbox = detection.bounding_box
					cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (self.robject_colors[ object_name ][2], \
					                                                            self.robject_colors[ object_name ][1], \
					                                                            self.robject_colors[ object_name ][0]), 1)
					cv2.circle(img, (center_bbox[0], center_bbox[1]), 5, (self.robject_colors[ object_name ][2], \
					                                                      self.robject_colors[ object_name ][1], \
					                                                      self.robject_colors[ object_name ][0]), 3)

			if frame.left_hand_pose is not None:					#  Does the left hand project into the camera?
				x = np.array(frame.left_hand_pose[:3]).reshape(3, 1)
				p = np.dot(K, x)
				if p[2] != 0.0:
					p /= p[2]
					x = int(round(p[0][0]))							#  Round and discretize to pixels
					y = int(round(p[1][0]))
					if x >= 0 and x < self.width and y >= 0 and y < self.height:
						cv2.line(img, (self.width - x - 5, self.height - y - 5), \
						              (self.width - x + 5, self.height - y + 5), (0, 255, 0, 255), 3)
						cv2.line(img, (self.width - x - 5, self.height - y + 5), \
						              (self.width - x + 5, self.height - y - 5), (0, 255, 0, 255), 3)
																	#  Write the hand subvector
						cv2.putText(img, "{:.2f}".format(frame.left_hand_pose[0]) + ', ' + \
						                 "{:.2f}".format(frame.left_hand_pose[1]) + ', ' + \
						                 "{:.2f}".format(frame.left_hand_pose[2]) + ': ' + str(frame.left_hand_pose[3]), \
						            (self.LH_super['x'], self.LH_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.LH_super['fontsize'], (0, 255, 0, 255), 3)

			if frame.right_hand_pose is not None:					#  Does the right hand project into the camera?
				x = np.array(frame.right_hand_pose[:3]).reshape(3, 1)
				p = np.dot(K, x)
				if p[2] != 0.0:
					p /= p[2]
					x = int(round(p[0][0]))							#  Round and discretize to pixels
					y = int(round(p[1][0]))
					if x >= 0 and x < self.width and y >= 0 and y < self.height:
						cv2.line(img, (self.width - x - 5, self.height - y - 5), \
						              (self.width - x + 5, self.height - y + 5), (0, 0, 255, 255), 3)
						cv2.line(img, (self.width - x - 5, self.height - y + 5), \
						              (self.width - x + 5, self.height - y - 5), (0, 0, 255, 255), 3)
																	#  Write the hand subvector
						cv2.putText(img, "{:.2f}".format(frame.right_hand_pose[0]) + ', ' + \
						                 "{:.2f}".format(frame.right_hand_pose[1]) + ', ' + \
						                 "{:.2f}".format(frame.right_hand_pose[2]) + ': ' + str(frame.right_hand_pose[3]), \
						            (self.RH_super['x'], self.RH_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.RH_super['fontsize'], (0, 0, 255, 255), 3)

																	#  Write the true action
			if frame.ground_truth_label == '*':						#  White asterisk for nothing
				cv2.putText(img, frame.ground_truth_label, (self.gt_label_super['x'], self.gt_label_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.gt_label_super['fontsize'], (255, 255, 255, 255), 3)
			else:													#  Bright green for truth!
				cv2.putText(img, frame.ground_truth_label, (self.gt_label_super['x'], self.gt_label_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.gt_label_super['fontsize'], (0, 255, 0, 255), 3)
																	#  Write the frame file name
			cv2.putText(img, frame.file_name, (self.filename_super['x'], self.filename_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.filename_super['fontsize'], (0, 255, 0, 255), 3)

			vid.write(img)

			if self.verbose:
				if int(round(float(i) / float(num_frames) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(i) / float(num_frames) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(i) / float(num_frames) * 100.0))) + '%]')
					sys.stdout.flush()
			i += 1

		vid.release()
		if self.verbose:
			print('')

		return

	#  This is time-consuming but informative.
	#  If a specific index is not given, then each action will be rendered to video.
	#  Include in the rendering as many details as have been prepared. If we computed masks, show the masks. If centroids, show centroids....
	def render_gaussian_weighted_action(self, gaussian, index=None):
		if index is None:
			indices = [i for i in range(0, len(self.actions))]
		else:
			indices = [index]

		K = self.K()												#  Build the camera matrix
		K_inv = np.linalg.inv(K)									#  Build inverse K-matrix
																	#  Build the flip matrix
		flip = np.array([[-1.0,  0.0, 0.0], \
		                 [ 0.0, -1.0, 0.0], \
		                 [ 0.0,  0.0, 1.0]], dtype='float64')

		sorted_frames = sorted(self.frames.items())
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.

		for index in indices:
			if self.verbose:
				print('>>> Rendering "' + self.enactment_name + '_Gaussian-weighted_' + str(index) + '.avi' + '"')

			action_frames = self.action_frameset(index)				#  Retrieve a list of indices into self.frames for self.action[index]
			num_frames = len(action_frames)
			prev_ctr = 0

			vid = cv2.VideoWriter( self.enactment_name + '_Gaussian-weighted_' + str(index) + '.avi', \
			                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
			                       self.fps, \
			                      (self.width, self.height) )
			i = 0
			for f in action_frames:
				frame = np.zeros((self.height, self.width, 3), dtype='uint8')
																	#  Load the depth map.
				depthmap = cv2.imread(sorted_frames[f][1].fullpath('depth'), cv2.IMREAD_UNCHANGED)
				if sorted_frames[f][1].left_hand_pose is not None:
					lh = sorted_frames[f][1].left_hand_pose[:3]		#  Leave off hand state.
				else:
					lh = None
				if sorted_frames[f][1].right_hand_pose is not None:
					rh = sorted_frames[f][1].right_hand_pose[:3]	#  Leave off hand state.
				else:
					rh = None
																	#  Array of mask images: load everything, ONCE.
				detection_masks = [cv2.imread(detection.mask_path, cv2.IMREAD_UNCHANGED) for detection in sorted_frames[f][1].detections if detection.enabled]

				for y in range(0, self.height):						#  So sloooooooow... iterate over every pixel and project.
					for x in range(0, self.width):
						d = self.min_depth + (float(depthmap[y, x]) / 255.0) * (self.max_depth - self.min_depth)
						centroid = np.dot(K_inv, np.array([x, y, 1.0]))
						centroid *= d								#  Scale by known depth (meters from head).
						pt = np.dot(flip, centroid)					#  Flip point.
						g = gaussian.weigh(pt, lh, rh)				#  Weigh the pixel.

						detection_ctr = 0
						found = False
						for detection in [det for det in sorted_frames[f][1].detections if det.enabled]:
							if detection.enabled and detection.object_name in self.robject_colors and detection_masks[detection_ctr][y, x] > 0:
								frame[y, x, 0] = int(round(self.robject_colors[detection.object_name][2] * g))
								frame[y, x, 1] = int(round(self.robject_colors[detection.object_name][1] * g))
								frame[y, x, 2] = int(round(self.robject_colors[detection.object_name][0] * g))
								found = True
								break
							detection_ctr += 1

						if not found:								#  No object: color gray.
							frame[y, x, 0] = int(round(255.0 * g))
							frame[y, x, 1] = int(round(255.0 * g))
							frame[y, x, 2] = int(round(255.0 * g))

				for detection in sorted_frames[f][1].detections:	#  Render bounding boxes and (bbox) centroids
					if detection.object_name is not None and detection.enabled and detection.object_name in self.robject_colors:
						object_name = detection.object_name
						center_bbox = detection.center('bbox')
						bbox = detection.bounding_box
						cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (self.robject_colors[ object_name ][2], \
						                                                              self.robject_colors[ object_name ][1], \
						                                                              self.robject_colors[ object_name ][0]), 1)
						cv2.circle(frame, (center_bbox[0], center_bbox[1]), 5, (self.robject_colors[ object_name ][2], \
						                                                        self.robject_colors[ object_name ][1], \
						                                                        self.robject_colors[ object_name ][0]), 3)

				if sorted_frames[f][1].left_hand_pose is not None:	#  Does the left hand project into the camera?
					x = np.array(sorted_frames[f][1].left_hand_pose[:3]).reshape(3, 1)
					p = np.dot(K, x)
					if p[2] != 0.0:
						p /= p[2]
						x = int(round(p[0][0]))						#  Round and discretize to pixels
						y = int(round(p[1][0]))
						if x >= 0 and x < self.width and y >= 0 and y < self.height:
							cv2.line(frame, (self.width - x - 5, self.height - y - 5), \
							                (self.width - x + 5, self.height - y + 5), (0, 255, 0, 255), 3)
							cv2.line(frame, (self.width - x - 5, self.height - y + 5), \
							                (self.width - x + 5, self.height - y - 5), (0, 255, 0, 255), 3)
																	#  Write the hand subvector
							cv2.putText(frame, "{:.2f}".format(sorted_frames[f][1].left_hand_pose[0]) + ', ' + \
							                   "{:.2f}".format(sorted_frames[f][1].left_hand_pose[1]) + ', ' + \
							                   "{:.2f}".format(sorted_frames[f][1].left_hand_pose[2]) + ': ' + str(sorted_frames[f][1].left_hand_pose[3]), \
							            (self.LH_super['x'], self.LH_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.LH_super['fontsize'], (0, 255, 0, 255), 3)

				if sorted_frames[f][1].right_hand_pose is not None:	#  Does the right hand project into the camera?
					x = np.array(sorted_frames[f][1].right_hand_pose[:3]).reshape(3, 1)
					p = np.dot(K, x)
					if p[2] != 0.0:
						p /= p[2]
						x = int(round(p[0][0]))						#  Round and discretize to pixels
						y = int(round(p[1][0]))
						if x >= 0 and x < self.width and y >= 0 and y < self.height:
							cv2.line(frame, (self.width - x - 5, self.height - y - 5), \
							                (self.width - x + 5, self.height - y + 5), (0, 0, 255, 255), 3)
							cv2.line(frame, (self.width - x - 5, self.height - y + 5), \
							                (self.width - x + 5, self.height - y - 5), (0, 0, 255, 255), 3)
																	#  Write the hand subvector
							cv2.putText(frame, "{:.2f}".format(sorted_frames[f][1].right_hand_pose[0]) + ', ' + \
							                   "{:.2f}".format(sorted_frames[f][1].right_hand_pose[1]) + ', ' + \
							                   "{:.2f}".format(sorted_frames[f][1].right_hand_pose[2]) + ': ' + str(sorted_frames[f][1].right_hand_pose[3]), \
							            (self.RH_super['x'], self.RH_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.RH_super['fontsize'], (0, 0, 255, 255), 3)

																	#  Write the true action
				if sorted_frames[f][1].ground_truth_label == '*':	#  White asterisk for nothing
					cv2.putText(frame, sorted_frames[f][1].ground_truth_label, (self.gt_label_super['x'], self.gt_label_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.gt_label_super['fontsize'], (255, 255, 255, 255), 3)
				else:												#  Bright green for truth!
					cv2.putText(frame, sorted_frames[f][1].ground_truth_label, (self.gt_label_super['x'], self.gt_label_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.gt_label_super['fontsize'], (0, 255, 0, 255), 3)
																	#  Write the frame file name
				cv2.putText(frame, sorted_frames[f][1].file_name, (self.filename_super['x'], self.filename_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.filename_super['fontsize'], (0, 255, 0, 255), 3)

				vid.write(frame)

				if self.verbose:
					if int(round(float(i) / float(num_frames - 1) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
						prev_ctr = int(round(float(i) / float(num_frames - 1) * float(max_ctr)))
						sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(i) / float(num_frames - 1) * 100.0))) + '%]')
						sys.stdout.flush()
				i += 1

			vid.release()
			if self.verbose:
				print('')

		return

	#  This is time-consuming but informative.
	#  Include in the rendering as many details as have been prepared. If we computed masks, show the masks. If centroids, show centroids....
	def render_gaussian_weighted_video(self, gaussian):
		K = self.K()												#  Build the camera matrix
		K_inv = np.linalg.inv(K)									#  Build inverse K-matrix
																	#  Build the flip matrix
		flip = np.array([[-1.0,  0.0, 0.0], \
		                 [ 0.0, -1.0, 0.0], \
		                 [ 0.0,  0.0, 1.0]], dtype='float64')
		sorted_frames = sorted(self.frames.items())
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.

		if self.verbose:
			print('>>> Rendering "' + self.enactment_name + '_Gaussian-weighted.avi' + '"')

		num_frames = len(sorted_frames)
		prev_ctr = 0

		vid = cv2.VideoWriter( self.enactment_name + '_Gaussian-weighted.avi', \
		                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
		                       self.fps, \
		                      (self.width, self.height) )
		i = 0
		for time_stamp, frame in sorted_frames:
			img = np.zeros((self.height, self.width, 3), dtype='uint8')
																	#  Load the depth map.
			depthmap = cv2.imread(frame.fullpath('depth'), cv2.IMREAD_UNCHANGED)
			if frame.left_hand_pose is not None:
				lh = frame.left_hand_pose[:3]						#  Leave off hand state.
			else:
				lh = None
			if frame.right_hand_pose is not None:
				rh = frame.right_hand_pose[:3]						#  Leave off hand state.
			else:
				rh = None
																	#  Array of mask images: load everything, ONCE.
			detection_masks = [cv2.imread(detection.mask_path, cv2.IMREAD_UNCHANGED) for detection in frame.detections if detection.enabled]

			for y in range(0, self.height):							#  So sloooooooow... iterate over every pixel and project.
				for x in range(0, self.width):
					d = self.min_depth + (float(depthmap[y, x]) / 255.0) * (self.max_depth - self.min_depth)
					centroid = np.dot(K_inv, np.array([x, y, 1.0]))
					centroid *= d									#  Scale by known depth (meters from head).
					pt = np.dot(flip, centroid)						#  Flip point.
					g = gaussian.weigh(pt, lh, rh)					#  Weigh the pixel.

					detection_ctr = 0
					found = False
					for detection in [det for det in frame.detections if det.enabled]:
						if detection.enabled and detection.object_name in self.robject_colors and detection_masks[detection_ctr][y, x] > 0:
							img[y, x, 0] = int(round(self.robject_colors[detection.object_name][2] * g))
							img[y, x, 1] = int(round(self.robject_colors[detection.object_name][1] * g))
							img[y, x, 2] = int(round(self.robject_colors[detection.object_name][0] * g))
							found = True
							break
						detection_ctr += 1

					if not found:									#  No object: color gray.
						img[y, x, 0] = int(round(255.0 * g))
						img[y, x, 1] = int(round(255.0 * g))
						img[y, x, 2] = int(round(255.0 * g))

			for detection in frame.detections:						#  Render bounding boxes and (bbox) centroids
				if detection.object_name is not None and detection.enabled and detection.object_name in self.robject_colors:
					object_name = detection.object_name
					center_bbox = detection.center('bbox')
					bbox = detection.bounding_box
					cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (self.robject_colors[ object_name ][2], \
					                                                            self.robject_colors[ object_name ][1], \
					                                                            self.robject_colors[ object_name ][0]), 1)
					cv2.circle(img, (center_bbox[0], center_bbox[1]), 5, (self.robject_colors[ object_name ][2], \
					                                                      self.robject_colors[ object_name ][1], \
					                                                      self.robject_colors[ object_name ][0]), 3)

			if frame.left_hand_pose is not None:					#  Does the left hand project into the camera?
				x = np.array(frame.left_hand_pose[:3]).reshape(3, 1)
				p = np.dot(K, x)
				if p[2] != 0.0:
					p /= p[2]
					x = int(round(p[0][0]))							#  Round and discretize to pixels
					y = int(round(p[1][0]))
					if x >= 0 and x < self.width and y >= 0 and y < self.height:
						cv2.line(img, (self.width - x - 5, self.height - y - 5), \
						              (self.width - x + 5, self.height - y + 5), (0, 255, 0, 255), 3)
						cv2.line(img, (self.width - x - 5, self.height - y + 5), \
						              (self.width - x + 5, self.height - y - 5), (0, 255, 0, 255), 3)
																	#  Write the hand subvector
						cv2.putText(img, "{:.2f}".format(frame.left_hand_pose[0]) + ', ' + \
						                 "{:.2f}".format(frame.left_hand_pose[1]) + ', ' + \
						                 "{:.2f}".format(frame.left_hand_pose[2]) + ': ' + str(frame.left_hand_pose[3]), \
						            (self.LH_super['x'], self.LH_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.LH_super['fontsize'], (0, 255, 0, 255), 3)

			if frame.right_hand_pose is not None:					#  Does the right hand project into the camera?
				x = np.array(frame.right_hand_pose[:3]).reshape(3, 1)
				p = np.dot(K, x)
				if p[2] != 0.0:
					p /= p[2]
					x = int(round(p[0][0]))							#  Round and discretize to pixels
					y = int(round(p[1][0]))
					if x >= 0 and x < self.width and y >= 0 and y < self.height:
						cv2.line(img, (self.width - x - 5, self.height - y - 5), \
						              (self.width - x + 5, self.height - y + 5), (0, 0, 255, 255), 3)
						cv2.line(img, (self.width - x - 5, self.height - y + 5), \
						              (self.width - x + 5, self.height - y - 5), (0, 0, 255, 255), 3)
																	#  Write the hand subvector
						cv2.putText(img, "{:.2f}".format(frame.right_hand_pose[0]) + ', ' + \
						                 "{:.2f}".format(frame.right_hand_pose[1]) + ', ' + \
						                 "{:.2f}".format(frame.right_hand_pose[2]) + ': ' + str(frame.right_hand_pose[3]), \
						            (self.RH_super['x'], self.RH_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.RH_super['fontsize'], (0, 0, 255, 255), 3)

																	#  Write the true action
			if frame.ground_truth_label == '*':						#  White asterisk for nothing
				cv2.putText(img, frame.ground_truth_label, (self.gt_label_super['x'], self.gt_label_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.gt_label_super['fontsize'], (255, 255, 255, 255), 3)
			else:													#  Bright green for truth!
				cv2.putText(img, frame.ground_truth_label, (self.gt_label_super['x'], self.gt_label_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.gt_label_super['fontsize'], (0, 255, 0, 255), 3)
																	#  Write the frame file name
			cv2.putText(img, frame.file_name, (self.filename_super['x'], self.filename_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.filename_super['fontsize'], (0, 255, 0, 255), 3)

			vid.write(img)

			if self.verbose:
				if int(round(float(i) / float(num_frames) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(i) / float(num_frames) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(i) / float(num_frames) * 100.0))) + '%]')
					sys.stdout.flush()
			i += 1

		vid.release()
		if self.verbose:
			print('')

		return

	#  Render a video that overlays color-segmentation and depth so that we can see if anything has been misfiled.
	def render_composite_video(self):
		vid = cv2.VideoWriter( self.enactment_name + '_composite.avi', \
		                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
		                       self.fps, \
		                      (self.width, self.height) )
		i = 0
		frames = self.load_frame_sequence()
		num_frames = len(frames)
		prev_ctr = 0
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.

		for framename in frames:
			normal_exists = os.path.exists(self.enactment_name + '/Users/' + self.user + '/POV/NormalViewCameraFrames/' + framename)
			segmap_exists = os.path.exists(self.enactment_name + '/Users/' + self.user + '/POV/ColorMapCameraFrames/' + framename)
			dmap_exists   = os.path.exists(self.enactment_name + '/Users/' + self.user + '/POV/DepthMapCameraFrames/' + framename)

			assert normal_exists and segmap_exists and dmap_exists, 'A camera, color-map, or depth-map frame is out of sync with the others.'

			frame = cv2.imread(self.enactment_name + '/Users/' + self.user + '/POV/NormalViewCameraFrames/' + framename, cv2.IMREAD_UNCHANGED)
			if frame.shape[2] == 4:									#  Do these guys have alpha channels? I forget.
				frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

			seg_oly = cv2.imread(self.enactment_name + '/Users/' + self.user + '/POV/ColorMapCameraFrames/' + framename, cv2.IMREAD_UNCHANGED)
			if seg_oly.shape[2] == 4:								#  Some of these guys have an alpha channel; some of these guys don't
				seg_oly = cv2.cvtColor(seg_oly, cv2.COLOR_BGRA2BGR)

			d_oly = cv2.imread(self.enactment_name + '/Users/' + self.user + '/POV/DepthMapCameraFrames/' + framename, cv2.IMREAD_UNCHANGED)
			if len(d_oly.shape) != 2:
				d_oly = d_oly[:, :, 0]								#  Isolate single channel
			d_oly = cv2.cvtColor(d_oly, cv2.COLOR_GRAY2BGR)			#  Restore color and alpha channels so we can map it to the rest

			d_oly = cv2.addWeighted(d_oly, 1.0, seg_oly, 0.7, 0)
			frame = cv2.addWeighted(frame, 1.0, d_oly, 0.7, 0)
			frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)			#  Flatten alpha

			vid.write(frame)

			if self.verbose:
				if int(round(float(i) / float(num_frames) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(i) / float(num_frames) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(i) / float(num_frames) * 100.0))) + '%]')
					sys.stdout.flush()

			i += 1

		vid.release()
		return

	#  Make *.ply files showing the head, right hand, and left hand.
	def render_skeleton_poses(self):
		total_V = len(self.frames) * 3
		total_E = len(self.frames) - 1 + 2 * len(self.frames)

		#############################################################  Write skeleton in the head frame
		fh = open(self.enactment_name + '.skeleton.headframe.ply', 'w')
		fh.write('ply' + '\n')
		fh.write('format ascii 1.0' + '\n')
		fh.write('comment Head = white' + '\n')
		fh.write('comment Left hand = green' + '\n')
		fh.write('comment Right hand = red' + '\n')
		fh.write('element vertex ' + str(total_V) + '\n')
		fh.write('property float x' + '\n')
		fh.write('property float y' + '\n')
		fh.write('property float z' + '\n')
		fh.write('property uchar red' + '\n')
		fh.write('property uchar green' + '\n')
		fh.write('property uchar blue' + '\n')
		fh.write('element edge ' + str(total_E) + '\n')
		fh.write('property int vertex1' + '\n')
		fh.write('property int vertex2' + '\n')
		fh.write('end_header' + '\n')

		lines = []
		ctr = 0
		for time_stamp, frame in sorted(self.frames.items()):
			fh.write(str(frame.head_pose[0]) + ' ' + str(frame.head_pose[1]) + ' ' + str(frame.head_pose[2]) + ' 255 255 255' + '\n')
			fh.write(str(frame.head_pose[3][:, 0].dot(frame.left_hand_pose[:3]) + frame.head_pose[0]) + ' ' + \
			         str(frame.head_pose[3][:, 1].dot(frame.left_hand_pose[:3]) + frame.head_pose[1]) + ' ' + \
			         str(frame.head_pose[3][:, 2].dot(frame.left_hand_pose[:3]) + frame.head_pose[2]) + \
			         ' 0 255 0' + '\n')
			fh.write(str(frame.head_pose[3][:, 0].dot(frame.right_hand_pose[:3]) + frame.head_pose[0]) + ' ' + \
			         str(frame.head_pose[3][:, 1].dot(frame.right_hand_pose[:3]) + frame.head_pose[1]) + ' ' + \
			         str(frame.head_pose[3][:, 2].dot(frame.right_hand_pose[:3]) + frame.head_pose[2]) + \
			         ' 255 0 0' + '\n')
			lines.append( (ctr, ctr + 1) )
			lines.append( (ctr, ctr + 2) )
			if ctr < total_V - 3:
				lines.append( (ctr, ctr + 3) )
			ctr += 3

		for line in lines:
			fh.write(str(line[0]) + ' ' + str(line[1]) + '\n')
		fh.close()

		#############################################################  Write skeleton in the global frame
		fh = open(self.enactment_name + '.skeleton.global.ply', 'w')
		fh.write('ply' + '\n')
		fh.write('format ascii 1.0' + '\n')
		fh.write('comment Head = white' + '\n')
		fh.write('comment Left hand = green' + '\n')
		fh.write('comment Right hand = red' + '\n')
		fh.write('element vertex ' + str(total_V) + '\n')
		fh.write('property float x' + '\n')
		fh.write('property float y' + '\n')
		fh.write('property float z' + '\n')
		fh.write('property uchar red' + '\n')
		fh.write('property uchar green' + '\n')
		fh.write('property uchar blue' + '\n')
		fh.write('element edge ' + str(total_E) + '\n')
		fh.write('property int vertex1' + '\n')
		fh.write('property int vertex2' + '\n')
		fh.write('end_header' + '\n')

		lines = []
		ctr = 0
		for time_stamp, frame in sorted(self.frames.items()):
			fh.write(str(frame.head_pose[0]) + ' ' + str(frame.head_pose[1]) + ' ' + str(frame.head_pose[2]) + ' 255 255 255' + '\n')
			fh.write(str(frame.left_hand_global_pose[0]) + ' ' + \
			         str(frame.left_hand_global_pose[1]) + ' ' + \
			         str(frame.left_hand_global_pose[2]) + ' 0 255 0' + '\n')
			fh.write(str(frame.right_hand_global_pose[0]) + ' ' + \
			         str(frame.right_hand_global_pose[1]) + ' ' + \
			         str(frame.right_hand_global_pose[2]) + ' 255 0 0' + '\n')
			lines.append( (ctr, ctr + 1) )
			lines.append( (ctr, ctr + 2) )
			if ctr < total_V - 3:
				lines.append( (ctr, ctr + 3) )
			ctr += 3

		for line in lines:
			fh.write(str(line[0]) + ' ' + str(line[1]) + '\n')
		fh.close()

		return

	#  Make color-coded "centipedes".
	def render_action_poses(self, stem_verbs=True):
		all_actions = self.labels()

		colors = np.array([ [int(x * 255)] for x in np.linspace(0.0, 1.0, len(all_actions)) ], np.uint8)
		colors = cv2.applyColorMap(colors, cv2.COLORMAP_JET)
		action_colors = {}
		i = 0
		for act in all_actions:
			action_colors[act] = [int(x) for x in colors[i][0]]
			action_colors[act].reverse()
			action_colors[act] = tuple(action_colors[act])
			i += 1

		#############################################################  Write the color-coded action centipede in the head frame
		frames_p = {}												#  Build a reverse dictionary:
		for frame in self.frames.values():							#  key:label ==> val:temporal list of dict/frames
			if frame.ground_truth_label != '*':
				frames_p[ frame.ground_truth_label ] = []

		total_V = 0													#  frames_p[label] = [ {[timestamp]  = timestamp
		total_E = 0													#                       [head]       = (x, y, z)
																	#                       [lefthand]   = (x, y, z)
																	#                       [righthand]} = (x, y, z)
																	#                       [file]       = string,
																	#                      {[timestamp]  = timestamp
																	#                       [head]       = (x, y, z)
		for label in frames_p.keys():								#                       [lefthand]   = (x, y, z)
																	#                       [righthand]} = (x, y, z)
			for timestamp, frame in self.frames.items():			#                       [file]       = string,
																	#
				if frame.ground_truth_label == label:				#                           ... ,
																	#                      {[timestamp]  = timestamp
					frames_p[label].append( {} )					#                       [head]       = (x, y, z)
																	#                       [lefthand]   = (x, y, z)
																	#                       [righthand]} = (x, y, z)
																	#                       [file]       = string     ]
					frames_p[label][-1]['timestamp'] = timestamp
					frames_p[label][-1]['head'] = frame.head_pose
					frames_p[label][-1]['lefthand'] = frame.left_hand_pose
					frames_p[label][-1]['righthand'] = frame.right_hand_pose
					frames_p[label][-1]['globalLH'] = frame.left_hand_global_pose
					frames_p[label][-1]['globalRH'] = frame.right_hand_global_pose
					frames_p[label][-1]['file'] = frame.file_name
					total_V += 3
			if len(frames_p[label]) > 0:
				total_E += len(frames_p[label]) * 2 + len(frames_p[label]) - 1

		for timestamp, frame in frames_p.items():					#  Sort temporally
			frames_p[timestamp] = sorted(frame, key=lambda x: x['timestamp'])

		fh = open(self.enactment_name + '.actions.headframe.ply', 'w')
		fh.write('ply' + '\n')
		fh.write('format ascii 1.0' + '\n')
		for k, v in action_colors.items():
			fh.write('comment ' + k + ' = (' + ' '.join([str(x) for x in v]) + ')\n')
		fh.write('element vertex ' + str(total_V) + '\n')
		fh.write('property float x' + '\n')
		fh.write('property float y' + '\n')
		fh.write('property float z' + '\n')
		fh.write('property uchar red' + '\n')
		fh.write('property uchar green' + '\n')
		fh.write('property uchar blue' + '\n')
		fh.write('element edge ' + str(total_E) + '\n')
		fh.write('property int vertex1' + '\n')
		fh.write('property int vertex2' + '\n')
		fh.write('end_header' + '\n')

		lines = []
		ctroffset = 0
		for act, v in frames_p.items():								#  For every LABEL...
			ctr = 0
			for pose in v:
				fh.write(str(pose['head'][0]) + ' ' + str(pose['head'][1]) + ' ' + str(pose['head'][2]) + ' ' + ' '.join([str(x) for x in action_colors[act]]) + '\n')

				fh.write(str(pose['head'][3][:, 0].dot(pose['lefthand'][:3]) + pose['head'][0]) + ' ' + \
				         str(pose['head'][3][:, 1].dot(pose['lefthand'][:3]) + pose['head'][1]) + ' ' + \
				         str(pose['head'][3][:, 2].dot(pose['lefthand'][:3]) + pose['head'][2]) + ' ' + \
				         ' '.join([str(x) for x in action_colors[act]]) + ' ' + '\n')
				fh.write(str(pose['head'][3][:, 0].dot(pose['righthand'][:3]) + pose['head'][0]) + ' ' + \
				         str(pose['head'][3][:, 1].dot(pose['righthand'][:3]) + pose['head'][1]) + ' ' + \
				         str(pose['head'][3][:, 2].dot(pose['righthand'][:3]) + pose['head'][2]) + ' ' + \
				         ' '.join([str(x) for x in action_colors[act]]) + ' ' + '\n')
				lines.append( (ctroffset, ctroffset + 1) )
				lines.append( (ctroffset, ctroffset + 2) )
				if ctr > 0:
					lines.append( (ctroffset - 3, ctroffset) )
				ctroffset += 3
				ctr += 1

		for line in lines:
			fh.write(str(line[0]) + ' ' + str(line[1]) + '\n')
		fh.close()

		#############################################################  Write the color-coded action centipede in the global frame
		fh = open(self.enactment_name + '.actions.global.ply', 'w')
		fh.write('ply' + '\n')
		fh.write('format ascii 1.0' + '\n')
		for k, v in action_colors.items():
			fh.write('comment ' + k + ' = (' + ' '.join([str(x) for x in v]) + ')\n')
		fh.write('element vertex ' + str(total_V) + '\n')
		fh.write('property float x' + '\n')
		fh.write('property float y' + '\n')
		fh.write('property float z' + '\n')
		fh.write('property uchar red' + '\n')
		fh.write('property uchar green' + '\n')
		fh.write('property uchar blue' + '\n')
		fh.write('element edge ' + str(total_E) + '\n')
		fh.write('property int vertex1' + '\n')
		fh.write('property int vertex2' + '\n')
		fh.write('end_header' + '\n')

		lines = []
		ctroffset = 0
		for act, v in frames_p.items():								#  For every LABEL...
			ctr = 0
			for pose in v:
				fh.write(str(pose['head'][0]) + ' ' + str(pose['head'][1]) + ' ' + str(pose['head'][2]) + ' ' + ' '.join([str(x) for x in action_colors[act]]) + '\n')

				fh.write(str(pose['globalLH'][0]) + ' ' + \
				         str(pose['globalLH'][1]) + ' ' + \
				         str(pose['globalLH'][2]) + ' ' + \
				         ' '.join([str(x) for x in action_colors[act]]) + ' ' + '\n')
				fh.write(str(pose['globalRH'][0]) + ' ' + \
				         str(pose['globalRH'][1]) + ' ' + \
				         str(pose['globalRH'][2]) + ' ' + \
				         ' '.join([str(x) for x in action_colors[act]]) + ' ' + '\n')
				lines.append( (ctroffset, ctroffset + 1) )
				lines.append( (ctroffset, ctroffset + 2) )
				if ctr > 0:
					lines.append( (ctroffset - 3, ctroffset) )
				ctroffset += 3
				ctr += 1

		for line in lines:
			fh.write(str(line[0]) + ' ' + str(line[1]) + '\n')
		fh.close()

		return

	#  Render a single frame as a point cloud and save it to "<self.enactment_name>_<time_stamp>_<mode>.ply".
	def render_point_cloud(self, frame_key, **kwargs):
		if 'mode' in kwargs:
			assert isinstance(kwargs['mode'], str) and \
			       (kwargs['mode'] == 'video' or kwargs['mode'] == 'color' or kwargs['mode'] == 'depth'), \
			       'Argument \'mode\' passed to Enactment.render_point_cloud() must be a string in {video, color, depth}.'
			mode = kwargs['mode']
		else:
			mode = 'video'

		if 'gaussian' in kwargs:
			assert isinstance(kwargs['gaussian'], Gaussian), \
			       'Argument \'gaussian\' passed to Enactment.render_point_cloud() must be a Gaussian object.'
			gaussian = kwargs['gaussian']
		else:
			gaussian = None

		if frame_key in self.frames:
			K = self.K()											#  Build the camera matrix
			K_inv = np.linalg.inv(K)								#  Build inverse K-matrix
																	#  Build the flip matrix
			flip = np.array([[-1.0,  0.0, 0.0], \
			                 [ 0.0, -1.0, 0.0], \
			                 [ 0.0,  0.0, 1.0]], dtype='float64')
																	#  We will always need the depth map
			depthmap = cv2.imread(self.frames[frame_key].fullpath('depth'), cv2.IMREAD_UNCHANGED)
			colormap = cv2.imread(self.frames[frame_key].fullpath('color'), cv2.IMREAD_UNCHANGED)
			videoframe = cv2.imread(self.frames[frame_key].fullpath('video'), cv2.IMREAD_UNCHANGED)
			if videoframe.shape[2] > 3:
				videoframe = cv2.cvtColors(videoframe, cv2.COLOR_RGBA2RGB)

			if self.frames[frame_key].left_hand_pose is not None:
				lh = self.frames[frame_key].left_hand_pose[:3]		#  Leave off hand state.
			else:
				lh = None
			if self.frames[frame_key].right_hand_pose is not None:
				rh = self.frames[frame_key].right_hand_pose[:3]		#  Leave off hand state.
			else:
				rh = None

			ply_file_name = self.enactment_name + '_' + str(frame_key) + '_' + mode + '.ply'
			fh = open(ply_file_name, 'w')
			fh.write('ply' + '\n')
			fh.write('format ascii 1.0' + '\n')
			fh.write('comment Point cloud of frame ' + str(frame_key) + ', enactment "' + self.enactment_name + '".' + '\n')
			if mode == 'video':
				fh.write('comment Rendered from the video frame.' + '\n')
			elif mode == 'color':
				fh.write('comment Rendered from the color map.' + '\n')
			elif mode == 'depth':
				fh.write('comment Rendered from the depth map.' + '\n')
			if gaussian is not None:
				fh.write('comment Gaussian-weighting applied to points\' colors.' + '\n')
			fh.write('element vertex ' + str(self.width * self.height) + '\n')
			fh.write('property float x' + '\n')
			fh.write('property float y' + '\n')
			fh.write('property float z' + '\n')
			fh.write('property uchar red' + '\n')
			fh.write('property uchar green' + '\n')
			fh.write('property uchar blue' + '\n')
			fh.write('end_header' + '\n')

			if self.verbose:
				print('>>> Rendering frame ' + str(frame_key) + ' to point cloud "' + ply_file_name + '"')
																	#  Array of mask images: load everything, ONCE.
			detection_masks = [cv2.imread(detection.mask_path, cv2.IMREAD_UNCHANGED) for detection in self.frames[frame_key].detections if detection.enabled]

			max_ctr = os.get_terminal_size().columns - 7			#  Leave enough space for the brackets, space, and percentage.
			prev_ctr = 0
			i = 0

			for y in range(0, self.height):							#  So sloooooooow... iterate over every pixel and project.
				for x in range(0, self.width):
					d = self.min_depth + (float(depthmap[y, x]) / 255.0) * (self.max_depth - self.min_depth)
					centroid = np.dot(K_inv, np.array([x, y, 1.0]))
					centroid *= d									#  Scale by known depth (meters from head).
					pt = np.dot(flip, centroid)						#  Flip point.
					if gaussian is not None:
						g = gaussian.weigh(pt, lh, rh)				#  Weigh the pixel.

					detection_ctr = 0
					found = False
					for detection in [det for det in self.frames[frame_key].detections if det.enabled]:
						if detection.object_name in self.robject_colors and detection_masks[detection_ctr][y, x] > 0:
							if gaussian is not None:
								fh.write(str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]) + ' ' + \
								         str(int(round(self.robject_colors[detection.object_name][0] * g))) + ' ' + \
								         str(int(round(self.robject_colors[detection.object_name][1] * g))) + ' ' + \
								         str(int(round(self.robject_colors[detection.object_name][2] * g))) + '\n')
							else:
								fh.write(str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]) + ' ' + \
								         str(self.robject_colors[detection.object_name][0]) + ' ' + \
								         str(self.robject_colors[detection.object_name][1]) + ' ' + \
								         str(self.robject_colors[detection.object_name][2]) + '\n')
							found = True
							break
						detection_ctr += 1

					if not found:									#  No object: color according to 'mode'.
						if mode == 'video':
							if gaussian is not None:
								fh.write(str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]) + ' ' + \
								         str(int(round(videoframe[y, x, 2] * g))) + ' ' + \
								         str(int(round(videoframe[y, x, 1] * g))) + ' ' + \
								         str(int(round(videoframe[y, x, 0] * g))) + '\n')
							else:
								fh.write(str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]) + ' ' + \
								         str(videoframe[y, x, 2]) + ' ' + \
								         str(videoframe[y, x, 1]) + ' ' + \
								         str(videoframe[y, x, 0]) + '\n')
						elif mode == 'color':
							if gaussian is not None:
								fh.write(str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]) + ' ' + \
								         str(int(round(colormap[y, x, 2] * g))) + ' ' + \
								         str(int(round(colormap[y, x, 1] * g))) + ' ' + \
								         str(int(round(colormap[y, x, 0] * g))) + '\n')
							else:
								fh.write(str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]) + ' ' + \
								         str(colormap[y, x, 2]) + ' ' + \
								         str(colormap[y, x, 1]) + ' ' + \
								         str(colormap[y, x, 0]) + '\n')
						else:
							if gaussian is not None:
								fh.write(str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]) + ' ' + \
								         str(int(round(255.0 * g))) + ' ' + \
								         str(int(round(255.0 * g))) + ' ' + \
								         str(int(round(255.0 * g))) + '\n')
							else:
								fh.write(str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]) + ' ' + \
								         str(depthmap[y, x]) + ' ' + \
								         str(depthmap[y, x]) + ' ' + \
								         str(depthmap[y, x]) + '\n')

					if self.verbose:
						if int(round(float(i) / float(self.width * self.height) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
							prev_ctr = int(round(float(i) / float(self.width * self.height) * float(max_ctr)))
							sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(i) / float(self.width * self.height) * 100.0))) + '%]')
							sys.stdout.flush()
					i += 1

			fh.close()

		elif self.verbose:
			print('    Key ' + str(frame_key) + ' not found in frames table.')

		return

	#################################################################
	#  Retrieval: load data from files and the file system          #
	#################################################################

	#  Default hand poses come from JSON files in the enactment file structure.
	#  These are collected from hand-held paddles relaying their positions relative to "lighthouses,"
	#  and sometimes the data are rather noisy.
	#  Compare this with the above Enactment.compute_IK_poses().
	def load_sensor_poses(self):
		head_data = self.load_head_poses()							#  Used here to load time stamps.
		left_hand_data = self.load_left_hand_poses()
		right_hand_data = self.load_right_hand_poses()

		video_frames = self.load_frame_sequence(True)
		num_frames = len(video_frames)
		time_stamps = [x['timestamp'] for x in head_data['list']]
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.
		prev_ctr = 0

		if self.verbose:
			print('>>> Loading head and hand poses.')

		for i in range(0, len(video_frames)):
			head_pose = head_data['list'][i]
			left_hand_pose = left_hand_data['list'][i]
			right_hand_pose = right_hand_data['list'][i]

			file_name = video_frames[i].split('/')[-1]
			file_path = '/'.join(video_frames[i].split('/')[:-2])	#  Leave off NormalViewCameraFrames/filename.png
			time_stamp = time_stamps[i]
																	#  This function is called by the constructor, but it may also
																	#  be called independently. Therefore, self.frames may already exist
			if time_stamp not in self.frames:						#  and contain other data we do not want lost.
				self.frames[time_stamp] = Frame(file_name=file_name, time_stamp=time_stamp, path=file_path, width=self.width, height=self.height)

			angle = 2 * np.arccos(head_pose['pose']['rotation']['w'])
																	#  Convert quaternion -> Axis-Angle -> Rodrigues Angles -> Rotation Matrix
			x = head_pose['pose']['rotation']['x'] / np.sqrt(1.0 - head_pose['pose']['rotation']['w'] * head_pose['pose']['rotation']['w'])
			y = head_pose['pose']['rotation']['y'] / np.sqrt(1.0 - head_pose['pose']['rotation']['w'] * head_pose['pose']['rotation']['w'])
			z = head_pose['pose']['rotation']['z'] / np.sqrt(1.0 - head_pose['pose']['rotation']['w'] * head_pose['pose']['rotation']['w'])
			head_rot, _ = cv2.Rodrigues( np.array([x * angle, y * angle, z * angle], dtype='float32') )
			#  NOTICE!                         NEGATIVE-X
			self.frames[time_stamp].set_head_pose( (-head_pose['pose']['position']['x'], \
			                                         head_pose['pose']['position']['y'], \
			                                         head_pose['pose']['position']['z'], \
			                                         head_rot) )
			#  NOTICE!       NEGATIVE-X
			diff = np.array([-left_hand_pose['pose']['position']['x'] - self.frames[time_stamp].head_pose[0], \
			                  left_hand_pose['pose']['position']['y'] - self.frames[time_stamp].head_pose[1], \
			                  left_hand_pose['pose']['position']['z'] - self.frames[time_stamp].head_pose[2]])
			self.frames[time_stamp].set_left_hand_pose( (self.frames[time_stamp].head_pose[3][0].dot(diff), \
			                                             self.frames[time_stamp].head_pose[3][1].dot(diff), \
			                                             self.frames[time_stamp].head_pose[3][2].dot(diff), \
			                                             left_hand_pose['handState']) )
			#  NOTICE!                                          NEGATIVE-X
			self.frames[time_stamp].set_left_hand_global_pose( (-left_hand_pose['pose']['position']['x'], \
			                                                     left_hand_pose['pose']['position']['y'], \
			                                                     left_hand_pose['pose']['position']['z'],
			                                                     left_hand_pose['handState']) )
			#  NOTICE!       NEGATIVE-X
			diff = np.array([-right_hand_pose['pose']['position']['x'] - self.frames[time_stamp].head_pose[0], \
			                  right_hand_pose['pose']['position']['y'] - self.frames[time_stamp].head_pose[1], \
			                  right_hand_pose['pose']['position']['z'] - self.frames[time_stamp].head_pose[2]])
			self.frames[time_stamp].set_right_hand_pose( (self.frames[time_stamp].head_pose[3][0].dot(diff), \
			                                              self.frames[time_stamp].head_pose[3][1].dot(diff), \
			                                              self.frames[time_stamp].head_pose[3][2].dot(diff), \
			                                              right_hand_pose['handState']) )
			#  NOTICE!                                           NEGATIVE-X
			self.frames[time_stamp].set_right_hand_global_pose( (-right_hand_pose['pose']['position']['x'], \
			                                                      right_hand_pose['pose']['position']['y'], \
			                                                      right_hand_pose['pose']['position']['z'],
			                                                      right_hand_pose['handState']) )
			if self.verbose:
				if int(round(float(i) / float(num_frames) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(i) / float(num_frames) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(i) / float(num_frames) * 100.0))) + '%]')
					sys.stdout.flush()
		if self.verbose:
			print('')

		return

	#  As loaded from JSON, the sensor data for hands continue to be tracked even when the hands are not visible.
	#  Call this function to iterate over all frames and test whether the hand projects into the camera.
	#  If so, that pose for that hand remains. If not, that pose for that hand is set to None.
	#  (You can always call load_sensor_poses() again to restore all sensor-based hand poses.)
	def apply_camera_projection_shutoff(self):
		if self.verbose:
			print('>>> Applying projection cutoff to hand poses.')

		num_frames = len(self.frames)
		prev_ctr = 0
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.

		K = self.K()												#  Retrieve the enactment's camera.

		ctr = 0
		for time_stamp, frame in sorted(self.frames.items()):
			if frame.left_hand_pose is not None:
				x = np.array(frame.left_hand_pose[:3]).reshape(3, 1)
				p = np.dot(K, x)
				if p[2] != 0.0:
					p /= p[2]
					x = int(round(p[0][0]))							#  Round and discretize to pixels.
					y = int(round(p[1][0]))
					if x < 0 or x >= self.width or y < 0 or y >= self.height:
						frame.left_hand_pose = None					#  Zero out the local.
						frame.left_hand_global_pose = None			#  Zero out the global.

			if frame.right_hand_pose is not None:
				x = np.array(frame.right_hand_pose[:3]).reshape(3, 1)
				p = np.dot(K, x)
				if p[2] != 0.0:
					p /= p[2]
					x = int(round(p[0][0]))							#  Round and discretize to pixels.
					y = int(round(p[1][0]))
					if x < 0 or x >= self.width or y < 0 or y >= self.height:
						frame.right_hand_pose = None				#  Zero out the local.
						frame.right_hand_global_pose = None			#  Zero out the global.

			if self.verbose:
				if int(round(float(ctr) / float(num_frames) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(ctr) / float(num_frames) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(ctr) / float(num_frames) * 100.0))) + '%]')
					sys.stdout.flush()
			ctr += 1

		if self.verbose:
			print('')

		return

	#  Complement to write_hand_pose_file().
	def load_hand_pose_file(self, file_name):
		hand_poses = {}												#  key:time stamp ==> val:{key:'left'  ==> val:(x, y, z, state),
																	#                          key:'right' ==> val:(x, y, z, state)}
		fh = open(file_name, 'r')
		for line in fh.readlines():
			if line[0] != '#':
				arr = line.strip().split('\t')
				time_stamp = float(arr[0])
				frame_file = arr[1]
				label = arr[2]

				lh_vec = [float(x) for x in arr[3:9]]
				if 1.0 in lh_vec[3:]:
					lh_state = lh_vec[3:].index(1.0)
				else:
					lh_state = None

				rh_vec = [float(x) for x in arr[9:]]
				if 1.0 in rh_vec[3:]:
					rh_state = rh_vec[3:].index(1.0)
				else:
					rh_state = None

				hand_poses[time_stamp] = {}
				hand_poses[time_stamp]['left'] = (lh_vec[0], lh_vec[1], lh_vec[2], lh_state)
				hand_poses[time_stamp]['right'] = (rh_vec[0], rh_vec[1], rh_vec[2], rh_state)
		fh.close()

		for time_stamp, frame in self.frames.items():
			if time_stamp in hand_poses and hand_poses[time_stamp]['left'][3] is not None:
				frame.set_left_hand_pose( hand_poses[time_stamp]['left'] )
				frame.set_left_hand_global_pose( hand_poses[time_stamp]['left'] )
			else:
				frame.set_left_hand_pose(None)
				frame.set_left_hand_global_pose(None)

			if time_stamp in hand_poses and hand_poses[time_stamp]['right'][3] is not None:
				frame.set_right_hand_pose( hand_poses[time_stamp]['right'] )
				frame.set_right_hand_global_pose( hand_poses[time_stamp]['right'] )
			else:
				frame.set_right_hand_pose(None)
				frame.set_right_hand_global_pose(None)

		return

	#  Retrieve a list of unique action labels from this enactment (from JSON)
	def load_json_action_labels(self):
		fh = open(self.enactment_name + '/Labels.fvr', 'r')			#  Fetch action time stamps
		lines = ' '.join(fh.readlines())							#  These USED to be a single line; NOW they have carriage returns
		fh.close()
		labels = json.loads(lines)
		return labels

	#  Set all frame action labels according to the given file.
	#  (Complement of write_all_frame_labels() above)
	def load_all_frame_labels(self, filename):
		time_label_lookup = {}
		fh = open(filename, 'r')
		for line in fh.readlines():
			if line[0] != '#':
				arr = line.strip().split('\t')
				time_stamp = float(arr[0])
				label = arr[2]
				time_label_lookup[time_stamp] = label
		fh.close()

		for time_stamp, frame in self.frames.items():
			if time_stamp in time_label_lookup:
				frame.ground_truth_label = time_label_lookup[time_stamp]

		return

	#  Retrieve head poses: a valid datum must have a head pose
	def load_head_poses(self):
		fh = open(self.enactment_name + '/Users/' + self.user + '/Head.fvr', 'r')
		line = fh.readlines()[0]									#  Single line
		fh.close()
		headdata = json.loads(line)
		return headdata

	#  Retrieve left-hand poses: a valid datum must have a left-hand pose
	def load_left_hand_poses(self):
		fh = open(self.enactment_name + '/Users/' + self.user + '/LeftHand.fvr', 'r')
		line = fh.readlines()[0]									#  Single line
		fh.close()
		lefthanddata = json.loads(line)
		return lefthanddata

	#  Retrieve right-hand poses: a valid datum must have a right-hand pose
	def load_right_hand_poses(self):
		fh = open(self.enactment_name + '/Users/' + self.user + '/RightHand.fvr', 'r')
		line = fh.readlines()[0]									#  Single line
		fh.close()
		righthanddata = json.loads(line)
		return righthanddata

	#  Read from file, return a JSON
	def load_metadata(self):
		fh = open(self.enactment_name + '/metadata.fvr', 'r')
		line = fh.readlines()[0]
		fh.close()
		metadata = json.loads(line)
		return metadata

	#  Read from file, return a JSON
	def load_camera_intrinsics(self):
		fh = open(self.enactment_name + '/Users/' + self.user + '/POV/CameraIntrinsics.fvr', 'r')
		line = fh.readlines()[0]
		fh.close()
		camdata = json.loads(line)
		return camdata

	#  Build and return a K-matrix
	def K(self):
		Kmat = np.array([[self.focal_length, 0.0,              self.width  * 0.5], \
		                 [      0.0,        self.focal_length, self.height * 0.5], \
		                 [      0.0,         0.0,                 1.0           ]])
		return Kmat

	def load_colormap(self):
		fh = open(self.enactment_name + '/Users/' + self.user + '/POV/SubpropColorMap.fvr')
		line = fh.readlines()[0]
		fh.close()
		colormap = json.loads(line)
		return colormap

	#  Return a sorted list of frame file names
	def load_frame_sequence(self, full_path=False):
		stem = self.enactment_name + '/Users/' + self.user + '/POV/NormalViewCameraFrames/'
		if full_path:
			return [ stem + x for x in sorted([x for x in os.listdir(stem) if x.endswith('.png')], key=lambda x: int(x.split('_')[0])) ]
		return sorted(os.listdir(stem), key=lambda x: int(x.split('_')[0]))

	#  Return a sorted list of frame file names
	def load_color_sequence(self, full_path=False):
		stem = self.enactment_name + '/Users/' + self.user + '/POV/ColorMapCameraFrames/'
		if full_path:
			return [ stem + x for x in sorted([x for x in os.listdir(stem) if x.endswith('.png')], key=lambda x: int(x.split('_')[0])) ]
		return sorted(os.listdir(stem), key=lambda x: int(x.split('_')[0]))

	#  Return a sorted list of frame file names
	def load_depth_sequence(self, full_path=False):
		stem = self.enactment_name + '/Users/' + self.user + '/POV/DepthMapCameraFrames/'
		if full_path:
			return [ stem + x for x in sorted([x for x in os.listdir(stem) if x.endswith('.png')], key=lambda x: int(x.split('_')[0])) ]
		return sorted(os.listdir(stem), key=lambda x: int(x.split('_')[0]))
