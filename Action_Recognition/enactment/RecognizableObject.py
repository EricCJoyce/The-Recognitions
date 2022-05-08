import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
Recognizable Object, whether determined by ground-truth or by deep network detection.
'''
class RecognizableObject():
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
		self.colorsrc = None										#  A single RecognizableObject may contain several colors if it is a composite object.

		self.enabled = True											#  Let's allow ourselves the ability to toggle detections.
																	#  This will be helpful in deciding which are too small to bother about.
		self.epsilon = (0, 0, 0)

		if 'parent_frame' in kwargs:
			assert isinstance(kwargs['parent_frame'], str), 'Argument \'parent_frame\' passed to RecognizableObject must be a string.'
			self.parent_frame = kwargs['parent_frame']

		if 'instance_name' in kwargs:
			assert isinstance(kwargs['instance_name'], str), 'Argument \'instance_name\' passed to RecognizableObject must be a string.'
			self.instance_name = kwargs['instance_name']

		if 'object_name' in kwargs:
			assert isinstance(kwargs['object_name'], str), 'Argument \'object_name\' passed to RecognizableObject must be a string.'
			self.object_name = kwargs['object_name']

		if 'centroid' in kwargs:
			assert isinstance(kwargs['centroid'], tuple), 'Argument \'centroid\' passed to RecognizableObject must be a tuple.'
			self.centroid = kwargs['centroid']

		if 'detection_source' in kwargs:
			assert isinstance(kwargs['detection_source'], str), 'Argument \'detection_source\' passed to RecognizableObject must be a string.'
			self.detection_source = kwargs['detection_source']

		if 'mask_path' in kwargs:
			assert isinstance(kwargs['mask_path'], str), 'Argument \'mask_path\' passed to RecognizableObject must be a string.'
			self.mask_path = kwargs['mask_path']

		if 'bounding_box' in kwargs:
			assert isinstance(kwargs['bounding_box'], tuple) and \
			       len(kwargs['bounding_box']) == 4, 'Argument \'bounding_box\' passed to RecognizableObject must be a 4-tuple.'
			self.bounding_box = kwargs['bounding_box']

		if 'confidence' in kwargs:
			assert isinstance(kwargs['confidence'], float) and \
			       kwargs['confidence'] >= 0.0 and kwargs['confidence'] <= 1.0, 'Argument \'confidence\' passed to RecognizableObject must be a float in [0.0, 1.0].'
			self.confidence = kwargs['confidence']

		if 'colors' in kwargs:
			assert isinstance(kwargs['colors'], list) and \
			       [len(x) for x in kwargs['colors']].count(3) == len(kwargs['colors']), 'Argument \'colors\' passed to RecognizableObject must be a list of 3-tuples.'
			self.colormaps = kwargs['colors']

		if 'colorsrc' in kwargs:
			assert isinstance(kwargs['colorsrc'], str), 'Argument \'colorsrc\' passed to RecognizableObject must be a string.'
			self.colorsrc = kwargs['colorsrc']

	#  Return the 2D center (x, y) of this object, according either to its average (time-costly) or its bounding-box.
	def center(self, method='bbox'):
		assert isinstance(method, str) and (method == 'bbox' or method == 'avg'), 'Argument \'method\' passed to RecognizableObject.center() must be a string in {avg, bbox}.'

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
		assert isinstance(mode, str) and (mode == 'video' or mode == 'color' or mode == 'depth'), 'The argument \'mode\' in RecognizableObject.show() must be a string in {video, color, depth}.'
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
