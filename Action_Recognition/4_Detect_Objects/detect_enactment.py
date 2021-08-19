import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
import time

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)											# To find local version of the library

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

def main():
	params = getCommandLineParams()									#  Collect parameters
	if params['helpme'] or len(params['enactments']) == 0:
		usage()
		return

	if params['verbose']:
		print('')													#  Space sections out
		print('#####  Check')
	t1_start = time.process_time()
	check(params)													#  Run checks on the given enactments
	t1_stop = time.process_time()
	if params['verbose']:
		print('  Took ' + str(t1_stop - t1_start) + ' seconds')

	if params['verbose']:
		print('')													#  Space sections out
		print('#####  Poses')
	t1_start = time.process_time()
	build_poses(params)												#  Build *_poses.{full,shutoff}.txt files (and maybe a video and centipedes) for each enactment
	t1_stop = time.process_time()
	if params['verbose']:
		print('  Took ' + str(t1_stop - t1_start) + ' seconds')

	if params['verbose']:
		print('')													#  Space sections out
		print('#####  Loading trained classes')
	t1_start = time.process_time()
	classes = read_classes(params)
	t1_stop = time.process_time()
	if params['verbose']:
		print('  Took ' + str(t1_stop - t1_start) + ' seconds')

	if params['verbose']:
		print('')													#  Space sections out
		print('#####  Mask-RCNN detections')
	t1_start = time.process_time()
	build_maskrcnn_predictions(classes, params)						#  Build a *_detections.txt file and many mask pngs in /masks for each enactment
	t1_stop = time.process_time()
	if params['verbose']:
		print('  Took ' + str(t1_stop - t1_start) + ' seconds')

	return

#  For all enactments, create a "*_detections.txt" file that stores:
#  timestamp   filename   label   object-class-name   score   BBox-x1,BBox-y1;BBox-x2,BBox-y2   mask-filename
#  Files with no detections have only the first element, 'filename'
def build_maskrcnn_predictions(classes, params):
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")						#  Directory to save logs and trained model
	COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")	#  Local path to trained weights file
	if not os.path.exists(COCO_MODEL_PATH):							#  Download COCO trained weights from Releases if needed
		utils.download_trained_weights(COCO_MODEL_PATH)

	class FactualConfig(Config):									#  Derives from the base Config class and overrides values specific
																	#  to this dataset.
		NAME = "factual"											#  Give the configuration a recognizable name

		GPU_COUNT = params['gpus']									#  Train on params['gpus'] GPUs and 8 images per GPU. We can put multiple images on each
		IMAGES_PER_GPU = 8											#  GPU because the images are small. Batch size is 8 (GPUs * images/GPU).

		NUM_CLASSES = len(classes) + 1								#  Number of classes (the +1 includes background)

		IMAGE_MIN_DIM = 128											#  FactualVR video is params['imgw'] x params['imgh'], but we must reduce it
		IMAGE_MAX_DIM = 256											#  or this will take 10,000 years to train.
																	#  Use smaller anchors because our image and objects are small
		RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)					#  anchor side in pixels
																	#  Reduce training ROIs per image because the images are small and have
		TRAIN_ROIS_PER_IMAGE = 32									#  few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
		STEPS_PER_EPOCH = 100										#  Use a small epoch since the data is simple
		VALIDATION_STEPS = 5										#  use small validation steps since the epoch is small

	config = FactualConfig()
	config.display()

	#  Collects and organizes the FactualVR video dataset
	class FactualDataset(utils.Dataset):

		#  Fetch the requested number of images, 'count'
		def load_src(self, count, height, width):
			ctr = 1														#  Add classes
			for k in classes:
				self.add_class("factual", ctr, k)
				ctr += 1
			for i in range(count):										#  Add images
				srcfile = self.random_image()							#  Returns a file name
				self.add_image("factual", image_id=i, path=None, \
				               width=width, height=height, \
				               img_file=srcfile)

		#  Fetch the requested number of images, 'count'
		def load_set(self, set_files, height, width):
			ctr = 1														#  Add classes
			for k in classes:
				self.add_class("factual", ctr, k)
				ctr += 1
			i = 0
			for fname in set_files:										#  Add images
				self.add_image("factual", image_id=i, path=None, \
				               width=width, height=height, \
				               img_file=fname)
				i += 1

		#  'image_id' is an int:
		#  Fetch the image indicated at image_id and scale it down to the dimensions used for this network: 256 x 128
		#  For both 1920x1080 and 1280x720, 720/1280 = 1080/1920 = 0.5625 and 128/256 = 0.5.
		#  This is not ideal, but whatever.
		#  We can live with a bit of stretch as long as we stretch the mask, too.
		def load_image(self, image_id):
			info = self.image_info[image_id]							#  Retrieve this sample's information
			target_height = info['height']								#  Unpack height
			target_width = info['width']								#  Unpack width
																		#  The MASKS have ALPHA CHANNELS; the VIDEO does NOT
			img = cv2.imread(info['img_file'], cv2.IMREAD_COLOR)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img = cv2.resize(img, (target_width, target_height))		#  Scale down (and slightly stretch)

			return img

		#  image_id is an int
		def image_reference(self, image_id):
			"""Return the shapes data of the image."""
			info = self.image_info[image_id]
			if info["source"] == "shapes":
				return info["shapes"]
			else:
				super(self.__class__).image_reference(self, image_id)

	class InferenceConfig(FactualConfig):
		GPU_COUNT = params['gpus']
		IMAGES_PER_GPU = 1

	inference_config = InferenceConfig()
																	#  Get path to saved weights
	if params['model'] is None:										#  Either set a specific path or find last trained weights
																	#  Recreate the model in inference mode
		model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
		model_path = model.find_last()								#  model_path = os.path.join(ROOT_DIR, ".h5 file name here")
		model_name = model_path.split('/')[-1].split('.')[0]		#  /home/ejoyce/maskrcnn/logs/factual20201115T1913/mask_rcnn_factual_0007.h5
	else:
		model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir='.')
		model_path = params['model']
		model_name = params['model']

	if params['verbose']:											#  Load trained weights
		print('>>> Loading Mask-RCNN weights from ' + model_path)
	model.load_weights(model_path, by_name=True)

	for enactment in params['enactments']:							#  For each (REGULAR) enactment...
		if not os.path.exists(enactment + '_' + model_name + '_detections.txt') or params['force']:
			if params['verbose']:
				verbosestr = '** Scanning ' + enactment + ' **'
				print('*' * len(verbosestr))
				print(verbosestr)
				print('*' * len(verbosestr))
																	#  Does the folder [enactment]/maskrcnn/ exist?
			if os.path.exists(enactment + '/' + model_name):		#  If so, empty it
				shutil.rmtree(enactment + '/' + model_name)
			os.mkdir(enactment + '/' + model_name, mode=0o777)

			fh = open(enactment + '_poses.full.txt', 'r')			#  Load the *_poses.txt file for reference because
			lines = fh.readlines()									#  it has all timestamps and file names arranged temporally.
			fh.close()
																	#  Create the *_detections.txt file
			fh = open(enactment + '_' + model_name + '_detections.txt', 'w')
			fh.write('#  Detections performed by ' + model_path + '\n')
			fh.write('#  Trained for the following classes:\n')
			fh.write('#  ' + ' '.join(classes) + '\n')
			fh.write('#  Frames without matches have only the file name and a bunch of asterisks.\n')
			fh.write('#  Frames with several matches have one object per line in the following format:\n')
			fh.write('#  timestamp   filename   label   object-class-name   score   BBox-x1,BBox-y1;BBox-x2,BBox-y2   mask-filename\n')

			maskfilectr = 0											#  Count up the masks for this enactment

			for line in lines:
				if line[0] != '#':
					arr = line.strip().split('\t')					#  timestamp, label, image-file, LHx, LHy, LHz, LHstate, RHx, RHy, RHz, RHstate
					timestamp = arr[0]								#  string
					label = arr[1]									#  string
					filename = arr[2]								#  string

					img = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/' + filename, cv2.IMREAD_COLOR)
					img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
					results = model.detect([img], verbose=0)		#  Model makes prediction(s)
																	#  results[0]['masks']     = imgw x imgh x 1 bool matrix
																	#  results[0]['rois']      = each sub-array is b-box(y1, x1, y2, x2) upperL-lowerR
																	#  results[0]['class_ids'] = zero-index into 'classes'
																	#  results[0]['scores']    = each is confidence
					numDetections = len(results[0]['rois'])			#  Number of predictions

					if params['verbose']:
						print('>>> ' + enactment + ', ' + filename + ': ' + str(numDetections) + ' detections')

					if numDetections > 0:							#  At least one...
						for i in range(0, numDetections):			#  For each detection...
							mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
							for y in range(0, img.shape[0]):
								for x in range(0, img.shape[1]):
									if results[0]['masks'][y][x][i]:
										mask[y][x] = 255
																	#  Build the mask file path
							maskpath = enactment + '/' + model_name + '/mask_' + str(maskfilectr) + '.png'
							cv2.imwrite(maskpath, mask)				#  Save the mask
																	#  ZERO-INDEXED!
							cat_index = results[0]['class_ids'][i] - 1

							fh.write(timestamp + '\t' + filename + '\t' + label + '\t')
							if cat_index >= len(classes) or cat_index < 0:
								fh.write('OOB-ID:' + str(cat_index) + '\t')
								if params['verbose']:
									print('    WARNING: Out-of-bounds class prediction: ' + str(cat_index) + ' (zero-indexed) for ' + str(len(classes)) + ' classes!')
							else:
								fh.write(classes[ cat_index ] + '\t')
							fh.write(str(results[0]['scores'][i]) + '\t')
							fh.write(str(results[0]['rois'][i][1]) + ',' + str(results[0]['rois'][i][0]) + ';' + \
							         str(results[0]['rois'][i][3]) + ',' + str(results[0]['rois'][i][2]) + '\t')
							fh.write(maskpath + '\n')

							maskfilectr += 1						#  Increment counter
					else:											#  No detections...
						fh.write(timestamp + '\t' + filename + '\t' + label + '\t')
						fh.write('*' + '\t')						#  No object
						fh.write('*' + '\t')						#  No score
						fh.write('*' + '\t')						#  No bounding box
						fh.write('*' + '\n')						#  No mask

			fh.close()

			#########################################################  Making videos?
			if params['render']:
				if params['verbose']:
					print('  * Rendering video "' + enactment + '_' + model_name + '_detections.avi' + '"')
																	#  Read the file we just wrote
				fh = open(enactment + '_' + model_name + '_detections.txt', 'r')
				lines = fh.readlines()
				fh.close()

				frames = {}											#  One key per unique video file name
				for line in lines:
					if line[0] != '#':
						arr = line.strip().split('\t')
						filename = arr[1]
						if filename not in frames:
							frames[filename] = []

				for framename in frames.keys():						#  Second pass: collect each frame's masks, objects, and bounding-boxe strings
					for line in lines:
						if line[0] != '#':
							arr = line.strip().split('\t')
							timestamp = arr[0]
							filename = arr[1]
							label = arr[2]
							object_class_name = arr[3]
							score = arr[4]
							bboxstr = arr[5]
							maskfilename = arr[6]

							if filename == framename and object_class_name != '*':
								frames[framename].append( (object_class_name, maskfilename, bboxstr) )

				vid = cv2.VideoWriter(enactment + '_' + model_name + '_detections.avi', \
				                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
				                      10, \
				                      (params['imgw'], params['imgh']) )

				for framename, v in sorted(frames.items(), key=lambda x: int(x[0].split('_')[0])):
					img = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/' + framename, cv2.IMREAD_UNCHANGED)
					if img.shape[2] == 4:							#  Drop alpha channel
						img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
					maskcanvas = np.zeros((params['imgh'], params['imgw'], 3), dtype='uint8')

					for objdata in v:								#  For each (possibly zero) recognizable objects in the current frame
						object_class_name = objdata[0]				#  Object name: so we can look up the color
						maskfilename = objdata[1]					#  Mask file name: so we can overlay the mask for this object
																	#  Open the mask file
						mask = cv2.imread(maskfilename, cv2.IMREAD_UNCHANGED)
						mask[mask > 1] = 1							#  All things greater than 1 become 1
																	#  Extrude to four channels
						mask = mask[:, :, None] * np.ones(3, dtype='uint8')[None, None, :]
																	#  Convert this to a graphical overlay
						mask[:, :, 0] *= params['colors'][ object_class_name ][2]
						mask[:, :, 1] *= params['colors'][ object_class_name ][1]
						mask[:, :, 2] *= params['colors'][ object_class_name ][0]

						maskcanvas += mask							#  Add mask to mask accumulator
						maskcanvas[maskcanvas > 255] = 255			#  Clip accumulator to 255
																	#  Add mask accumulator to source frame
					img = cv2.addWeighted(img, 1.0, maskcanvas, 0.7, 0)
					img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)		#  Flatten alpha

					for objdata in v:
						object_class_name = objdata[0]				#  Object name: so we can look up the color
						bboxstr = objdata[2]						#  Bounding box: so we can draw the bounding box
						bbox = bboxstr.split(';')
						bbox = (int(bbox[0].split(',')[0]), int(bbox[0].split(',')[1]), int(bbox[1].split(',')[0]), int(bbox[1].split(',')[1]))
						cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (params['colors'][ object_class_name ][2], \
						                                                            params['colors'][ object_class_name ][1], \
						                                                            params['colors'][ object_class_name ][0]), 1)

					vid.write(img)

				vid.release()
		elif params['verbose']:
			print('*** ' + enactment + '_detections.txt already exists. Skipping.')
	return

#  Perhaps... dangerously simple: read the first line from the file "training-set.txt"
#  This of course assumes that the file with this name was used to train the version of Mask-RCNN you use to detect objects now!
def read_classes(params):
	fh = open('training-set.txt', 'r')
	lines = fh.readlines()
	fh.close()
	arr = lines[0].strip().split()

	if params['verbose']:
		print('>>> Classes retrieved from training-set roster:')
		for classname in arr[1:]:
			print('    ' + classname)

	return arr[1:]

#  For all enactments, create a "*_poses.txt" file that STORES HAND VECTORS IN THE HEAD FRAME
def build_poses(params):
	if params['render']:											#  Rendering centipedes? You'll need colors.
		actionColors = survey_action_colors(params)					#  Then build a color look-up table: action/label => (r, g, b)

	for enactment in params['enactments']:							#  For each enactment...
																	#  If *_poses.{full,shutoff}.txt files either do not exist or we are forcing...
		if not os.path.exists(enactment + '_poses.full.txt') or not os.path.exists(enactment + '_poses.shutoff.txt') or params['force']:
			if params['verbose']:
				verbosestr = '** Scanning ' + enactment + ' **'
				print('*' * len(verbosestr))
				print(verbosestr)
				print('*' * len(verbosestr))

			fh = open(enactment + '/Labels.fvr', 'r')				#  Fetch action time stamps
			lines = ' '.join(fh.readlines())						#  These USED to be a single line; NOW they have carriage returns
			fh.close()
			labels = json.loads(lines)
																	#  Retrieve head poses: a valid datum must have a head pose
			fh = open(enactment + '/Users/' + params['User'] + '/Head.fvr', 'r')
			line = fh.readlines()[0]								#  Single line
			fh.close()
			headdata = json.loads(line)
																	#  Retrieve left-hand poses: a valid datum must have a left-hand pose
			fh = open(enactment + '/Users/' + params['User'] + '/LeftHand.fvr', 'r')
			line = fh.readlines()[0]								#  Single line
			fh.close()
			lefthanddata = json.loads(line)
																	#  Retrieve right-hand poses: a valid datum must have a right-hand pose
			fh = open(enactment + '/Users/' + params['User'] + '/RightHand.fvr', 'r')
			line = fh.readlines()[0]								#  Single line
			fh.close()
			righthanddata = json.loads(line)

			fh = open(enactment + '/metadata.fvr', 'r')				#  Fetch enactment duration
			line = fh.readlines()[0]								#  Single line
			fh.close()
			metadata = json.loads(line)
			duration = metadata['duration']

			fh = open(enactment + '/Users/' + params['User'] + '/POV/CameraIntrinsics.fvr', 'r')
			line = fh.readlines()[0]
			fh.close()
			camdata = json.loads(line)
			fov = float(camdata['fov'])								#  We don't use camdata['focalLength'] anymore because IT'S IN MILLIMETERS
			if params['focal'] is None:								#  IN PIXELS!!!! ALSO: NOTE THAT THIS IS THE **VERTICAL** F.o.V. !!
				focalLength = params['imgh'] * 0.5 / np.tan(fov * np.pi / 180.0)
			else:
				focalLength = params['focal']
			K = np.array([[focalLength, 0.0, params['imgw'] * 0.5], \
			              [0.0, focalLength, params['imgh'] * 0.5], \
			              [0.0,         0.0,           1.0       ]])
			if params['verbose']:
				print('>>> Stated camera focal length: ' + str(camdata['focalLength']) + ' mm')
				print('    Computed camera focal length: ' + str(focalLength) + ' pixels')
				print('K = ')
				print(K)

			frames = [x for x in os.listdir(enactment + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/') if x.endswith('.png')]
			frames = sorted(frames, key=lambda x: int(x.split('_')[0]))
			if params['verbose']:
				print('    ' + str(len(frames)) + ' frames')

			#########################################################  Build timestamped poses: timestamp ==> head-left-right
																	#  key(float)    val(dict)
			samples = {}											#  timestamp ==> {} = [head]      ==> (x, y, z)
																	#                     [lefthand]  ==> (x, y, z)
																	#                     [righthand] ==> (x, y, z)
																	#                     [globalLH]  ==> (x, y, z)
																	#                     [globalRH]  ==> (x, y, z)
																	#                     [label]     ==> string or None
																	#                     [file]      ==> string

			for headpose in headdata['list']:						#  We will let the head determine what is valid data
				timestamp = headpose['timestamp']
				if timestamp not in samples:
					samples[timestamp] = {}
																	#  Convert quaternion -> Axis-Angle -> Rodrigues Angles -> Rotation Matrix
				angle = 2 * np.arccos(headpose['pose']['rotation']['w'])
				x = headpose['pose']['rotation']['x'] / np.sqrt(1.0 - headpose['pose']['rotation']['w'] * headpose['pose']['rotation']['w'])
				y = headpose['pose']['rotation']['y'] / np.sqrt(1.0 - headpose['pose']['rotation']['w'] * headpose['pose']['rotation']['w'])
				z = headpose['pose']['rotation']['z'] / np.sqrt(1.0 - headpose['pose']['rotation']['w'] * headpose['pose']['rotation']['w'])
				headRot, _ = cv2.Rodrigues( np.array([x * angle, y * angle, z * angle], dtype='float32') )

				#  NOTICE!                    NEGATIVE-X
				samples[timestamp]['head'] = (-headpose['pose']['position']['x'], \
				                               headpose['pose']['position']['y'], \
				                               headpose['pose']['position']['z'], \
				                               headRot)
				index = min(int(round((timestamp / duration) * float(len(frames)))), len(frames) - 1)
				samples[timestamp]['file'] = frames[ index ]
				samples[timestamp]['label'] = None

			for handpose in lefthanddata['list']:					#  Add a left-hand IFF there is a timestamp justified by a head pose
				timestamp = handpose['timestamp']
				if timestamp in samples:
					#  NOTICE!       NEGATIVE-X
					diff = np.array([-handpose['pose']['position']['x'] - samples[timestamp]['head'][0], \
					                  handpose['pose']['position']['y'] - samples[timestamp]['head'][1], \
					                  handpose['pose']['position']['z'] - samples[timestamp]['head'][2]])
					samples[timestamp]['lefthand'] = ( samples[timestamp]['head'][3][0].dot(diff), \
					                                   samples[timestamp]['head'][3][1].dot(diff), \
					                                   samples[timestamp]['head'][3][2].dot(diff), \
					                                   handpose['handState'] )
					#  NOTICE!                        NEGATIVE-X
					samples[timestamp]['globalLH'] = (-handpose['pose']['position']['x'], \
					                                   handpose['pose']['position']['y'], \
					                                   handpose['pose']['position']['z'])

			for handpose in righthanddata['list']:					#  Add a right-hand IFF there is a timestamp justified by a head pose
				timestamp = handpose['timestamp']
				if timestamp in samples:
					#  NOTICE!       NEGATIVE-X
					diff = np.array([-handpose['pose']['position']['x'] - samples[timestamp]['head'][0], \
					                  handpose['pose']['position']['y'] - samples[timestamp]['head'][1], \
					                  handpose['pose']['position']['z'] - samples[timestamp]['head'][2]])
					samples[timestamp]['righthand'] = ( samples[timestamp]['head'][3][0].dot(diff), \
					                                    samples[timestamp]['head'][3][1].dot(diff), \
					                                    samples[timestamp]['head'][3][2].dot(diff), \
					                                    handpose['handState'] )
					#  NOTICE!                        NEGATIVE-X
					samples[timestamp]['globalRH'] = (-handpose['pose']['position']['x'], \
					                                   handpose['pose']['position']['y'], \
					                                   handpose['pose']['position']['z'])

																	#  Make sure all parts of the skeleton are included
			samples = dict( [(k, v) for k, v in samples.items() if 'lefthand' in v and 'righthand' in v] )

			#########################################################  For all metadata labels, assign frames

			timeline = []											#  List of tuples(start, end, label)
			for action in labels['list']:
				timeline.append( [action['startTime'], action['endTime'], action['stepDescription']] )
			timeline = sorted(timeline, key=lambda x: x[0])			#  Sort temporally
			for i in range(0, len(timeline) - 1):					#  Examine all neighbors for conflicts
				if timeline[i][1] > timeline[i + 1][0]:				#  LABEL OVERLAP!
					if params['verbose']:
						print('    * Label conflict between ' + timeline[i][2] + ' and ' + timeline[i + 1][2])
																	#  ADJUST TIMES!
					mid = timeline[i + 1][0] - (timeline[i][1] - timeline[i + 1][0]) * 0.5
					timeline[i][1]     = mid
					timeline[i + 1][0] = mid

			for action in timeline:									#  Go through all actions labeled in the current enactment
				ctr = 0
				label = action[2]									#  Identify the label
				startTime = action[0]								#  Start and end times
				endTime = action[1]

				for timestamp, v in samples.items():				#  Find all the obvious fits
					if timestamp >= startTime and timestamp < endTime:
																	#  If this float falls between the other floats
						samples[timestamp]['label'] = label			#  then this frame is part of this action and receives this label
						ctr += 1

				if ctr == 0:										#  If there was no obvious fit, then find the nearest fit
					best = 0
					minerr = float('inf')
					keys = sorted(samples.keys())
					for i in range(0, len(keys)):
						err = np.fabs(startTime - keys[i])
						if err < minerr:
							best = i
							minerr = err
					samples[ keys[best] ]['label'] = label
					ctr += 1

				if params['verbose']:
					#print('    ' + label + ': ' + str(startTime) + ' to ' + str(endTime) + ' covers ' + str(ctr) + ' poses at ' + \
					#      ' '.join( [str(k) for k, v in samples.items() if v['label'] == label] ))
					print('    ' + label + ': ' + str(startTime) + ' to ' + str(endTime) + ' covers ' + str(ctr) + ' poses')

			#  By now we have samples[timestamp] ==> ['head'] = (x, y, z, R)
			#                                        ['lefthand'] = (x, y, z)
			#                                        ['righthand'] = (x, y, z)
			#                                        ['globalLH'] = (x, y, z)
			#                                        ['globalRH'] = (x, y, z)
			#                                        ['label'] = string or None
			#                                        ['file'] = string

			#########################################################  Write the intermediate head-files
			fh = open(enactment + '_poses.head.txt', 'w')
			fh.write('#  timestamp, label, image-file, head-x, head-y, head-z, R00, R01, R02, R10, R11, R12, R20, R21, R22\n')
			for k, v in sorted(samples.items()):
				fh.write(str(k) + '\t')
				if v['label'] is None:
					fh.write('*' + '\t')
				else:
					fh.write(v['label'] + '\t')
				fh.write(v['file'] + '\t')

				fh.write(str(v['head'][0]) + '\t' + str(v['head'][1]) + '\t' + str(v['head'][2]) + '\t')
				fh.write(str(v['head'][3][0][0]) + '\t' + str(v['head'][3][0][1]) + '\t' + str(v['head'][3][0][2]) + '\t')
				fh.write(str(v['head'][3][1][0]) + '\t' + str(v['head'][3][1][1]) + '\t' + str(v['head'][3][1][2]) + '\t')
				fh.write(str(v['head'][3][2][0]) + '\t' + str(v['head'][3][2][1]) + '\t' + str(v['head'][3][2][2]) + '\n')
			fh.close()

			#########################################################  Write the intermediate vector-files
			fh = open(enactment + '_poses.full.txt', 'w')
			fh.write('#  timestamp, label, image-file, LHx, LHy, LHz, LHstate_0, LHstate_1, LHstate_2, RHx, RHy, RHz, RHstate_0, RHstate_1, RHstate_2\n')
			fh.write('#  All hand positions are taken from JSON and made relative to the head in the head\'s frame.\n')
			fh.write('#  In this file, hand vectors are all recorded REGARDLESS OF WHETHER THE HAND PROJECTS INTO THE CAMERA!\n')
			for k, v in sorted(samples.items()):
				fh.write(str(k) + '\t')
				if v['label'] is None:
					fh.write('*' + '\t')
				else:
					fh.write(v['label'] + '\t')
				fh.write(v['file'] + '\t')

				fh.write(str(v['lefthand'][0])  + '\t' + str(v['lefthand'][1])  + '\t' + str(v['lefthand'][2])  + '\t')
				if v['lefthand'][3] == 0:
					fh.write('1\t0\t0\t')
				elif v['lefthand'][3] == 1:
					fh.write('0\t1\t0\t')
				elif v['lefthand'][3] == 2:
					fh.write('0\t0\t1\t')

				fh.write(str(v['righthand'][0]) + '\t' + str(v['righthand'][1]) + '\t' + str(v['righthand'][2]) + '\t')
				if v['righthand'][3] == 0:
					fh.write('1\t0\t0\n')
				elif v['righthand'][3] == 1:
					fh.write('0\t1\t0\n')
				elif v['righthand'][3] == 2:
					fh.write('0\t0\t1\n')
			fh.close()

			fh = open(enactment + '_poses.shutoff.txt', 'w')
			fh.write('#  timestamp, label, image-file, LHx, LHy, LHz, LHstate_0, LHstate_1, LHstate_2, RHx, RHy, RHz, RHstate_0, RHstate_1, RHstate_2\n')
			fh.write('#  All hand positions are taken from JSON and made relative to the head in the head\'s frame.\n')
			fh.write('#  In this file, hand vectors are set to zero if the hand does not project into the camera!\n')
			for k, v in sorted(samples.items()):
				fh.write(str(k) + '\t')
				if v['label'] is None:
					fh.write('*' + '\t')
				else:
					fh.write(v['label'] + '\t')
				fh.write(v['file'] + '\t')

				if projects_into_camera(v['globalLH'], v['head'][:3], v['head'][3], K, params) is not None:
					fh.write(str(v['lefthand'][0])  + '\t' + str(v['lefthand'][1])  + '\t' + str(v['lefthand'][2])  + '\t')
					if v['lefthand'][3] == 0:
						fh.write('1\t0\t0\t')
					elif v['lefthand'][3] == 1:
						fh.write('0\t1\t0\t')
					elif v['lefthand'][3] == 2:
						fh.write('0\t0\t1\t')
				else:
					fh.write('0.0\t0.0\t0.0\t0\t0\t0\t')

				if projects_into_camera(v['globalRH'], v['head'][:3], v['head'][3], K, params) is not None:
					fh.write(str(v['righthand'][0]) + '\t' + str(v['righthand'][1]) + '\t' + str(v['righthand'][2]) + '\t')
					if v['righthand'][3] == 0:
						fh.write('1\t0\t0\n')
					elif v['righthand'][3] == 1:
						fh.write('0\t1\t0\n')
					elif v['righthand'][3] == 2:
						fh.write('0\t0\t1\n')
				else:
					fh.write('0.0\t0.0\t0.0\t0\t0\t0\n')
			fh.close()

			#########################################################  Making centipedes and videos?
			if params['render']:
				#  For the 'skeleton' PLYs, there will be a single centipede, entirely connected through TIME.
				#    There will be no redundant vertices.
				#  For the 'action' centipedes, there will be one or more centipedes, each connected by ACTION.
				#    There will be redundant vertices if there are overlapping action labels.

				#####################################################  Write the left-hand/right-hand one in the head frame
				totalV = len(samples) * 3
				totalE = len(samples) - 1 + 2 * len(samples)

				fh = open(enactment + '.skeleton.headframe.ply', 'w')
				fh.write('ply' + '\n')
				fh.write('format ascii 1.0' + '\n')
				fh.write('comment Head = white' + '\n')
				fh.write('comment Left hand = green' + '\n')
				fh.write('comment Right hand = red' + '\n')
				fh.write('element vertex ' + str(totalV) + '\n')
				fh.write('property float x' + '\n')
				fh.write('property float y' + '\n')
				fh.write('property float z' + '\n')
				fh.write('property uchar red' + '\n')
				fh.write('property uchar green' + '\n')
				fh.write('property uchar blue' + '\n')
				fh.write('element edge ' + str(totalE) + '\n')
				fh.write('property int vertex1' + '\n')
				fh.write('property int vertex2' + '\n')
				fh.write('end_header' + '\n')

				lines = []
				ctr = 0
				for pose in [x[1] for x in sorted(samples.items(), key=lambda y: y[0])]:
					fh.write(str(pose['head'][0]) + ' ' + str(pose['head'][1]) + ' ' + str(pose['head'][2]) + ' 255 255 255' + '\n')
					fh.write(str(pose['head'][3][:, 0].dot(pose['lefthand'][:3]) + pose['head'][0]) + ' ' + \
					         str(pose['head'][3][:, 1].dot(pose['lefthand'][:3]) + pose['head'][1]) + ' ' + \
					         str(pose['head'][3][:, 2].dot(pose['lefthand'][:3]) + pose['head'][2]) + \
					         ' 0 255 0' + '\n')
					fh.write(str(pose['head'][3][:, 0].dot(pose['righthand'][:3]) + pose['head'][0]) + ' ' + \
					         str(pose['head'][3][:, 1].dot(pose['righthand'][:3]) + pose['head'][1]) + ' ' + \
					         str(pose['head'][3][:, 2].dot(pose['righthand'][:3]) + pose['head'][2]) + \
					         ' 255 0 0' + '\n')
					lines.append( (ctr, ctr + 1) )
					lines.append( (ctr, ctr + 2) )
					if ctr < totalV - 3:
						lines.append( (ctr, ctr + 3) )
					ctr += 3

				for line in lines:
					fh.write(str(line[0]) + ' ' + str(line[1]) + '\n')
				fh.close()

				#####################################################  Write the left-hand/right-hand one in the global frame
				totalV = len(samples) * 3
				totalE = len(samples) - 1 + 2 * len(samples)

				fh = open(enactment + '.skeleton.global.ply', 'w')
				fh.write('ply' + '\n')
				fh.write('format ascii 1.0' + '\n')
				fh.write('comment Head = white' + '\n')
				fh.write('comment Left hand = green' + '\n')
				fh.write('comment Right hand = red' + '\n')
				fh.write('element vertex ' + str(totalV) + '\n')
				fh.write('property float x' + '\n')
				fh.write('property float y' + '\n')
				fh.write('property float z' + '\n')
				fh.write('property uchar red' + '\n')
				fh.write('property uchar green' + '\n')
				fh.write('property uchar blue' + '\n')
				fh.write('element edge ' + str(totalE) + '\n')
				fh.write('property int vertex1' + '\n')
				fh.write('property int vertex2' + '\n')
				fh.write('end_header' + '\n')

				lines = []
				ctr = 0
				for pose in [x[1] for x in sorted(samples.items(), key=lambda y: y[0])]:
					fh.write(str(pose['head'][0]) + ' ' + str(pose['head'][1]) + ' ' + str(pose['head'][2]) + ' 255 255 255' + '\n')
					fh.write(str(pose['globalLH'][0]) + ' ' + \
					         str(pose['globalLH'][1]) + ' ' + \
					         str(pose['globalLH'][2]) + ' 0 255 0' + '\n')
					fh.write(str(pose['globalRH'][0]) + ' ' + \
					         str(pose['globalRH'][1]) + ' ' + \
					         str(pose['globalRH'][2]) + ' 255 0 0' + '\n')
					lines.append( (ctr, ctr + 1) )
					lines.append( (ctr, ctr + 2) )
					if ctr < totalV - 3:
						lines.append( (ctr, ctr + 3) )
					ctr += 3

				for line in lines:
					fh.write(str(line[0]) + ' ' + str(line[1]) + '\n')
				fh.close()

				#####################################################  Write the color-coded action one in the head frame
				samples_p = {}										#  Build a reverse dictionary:
				for v in samples.values():							#  key:label ==> val:temporal list of dict/frames
					if v['label'] is not None:
						samples_p[ v['label'] ] = []

				totalV = 0											#  samples_p[label] = [ {[timestamp]  = timestamp
				totalE = 0											#                        [head]       = (x, y, z)
																	#                        [lefthand]   = (x, y, z)
																	#                        [righthand]} = (x, y, z)
																	#                        [file]       = string,
																	#                       {[timestamp]  = timestamp
																	#                        [head]       = (x, y, z)
				for label in samples_p.keys():						#                        [lefthand]   = (x, y, z)
					if label is not None:							#                        [righthand]} = (x, y, z)
						for timestamp, v in samples.items():		#                        [file]       = string,
																	#
							if v['label'] == label:					#                           ... ,
																	#                       {[timestamp]  = timestamp
								samples_p[label].append( {} )		#                        [head]       = (x, y, z)
																	#                        [lefthand]   = (x, y, z)
																	#                        [righthand]} = (x, y, z)
																	#                        [file]       = string     ]
								samples_p[label][-1]['timestamp'] = timestamp
								samples_p[label][-1]['head'] = v['head']
								samples_p[label][-1]['lefthand'] = v['lefthand']
								samples_p[label][-1]['righthand'] = v['righthand']
								samples_p[label][-1]['globalLH'] = v['globalLH']
								samples_p[label][-1]['globalRH'] = v['globalRH']
								samples_p[label][-1]['file'] = v['file']
								totalV += 3
						if len(samples_p[label]) > 0:
							totalE += len(samples_p[label]) * 2 + len(samples_p[label]) - 1

				for timestamp, v in samples_p.items():				#  Sort temporally
					samples_p[timestamp] = sorted(v, key=lambda x: x['timestamp'])

				fh = open(enactment + '.actions.headframe.ply', 'w')
				fh.write('ply' + '\n')
				fh.write('format ascii 1.0' + '\n')
				for k, v in actionColors.items():
					fh.write('comment ' + k + ' = (' + ' '.join([str(x) for x in v]) + ')\n')
				fh.write('element vertex ' + str(totalV) + '\n')
				fh.write('property float x' + '\n')
				fh.write('property float y' + '\n')
				fh.write('property float z' + '\n')
				fh.write('property uchar red' + '\n')
				fh.write('property uchar green' + '\n')
				fh.write('property uchar blue' + '\n')
				fh.write('element edge ' + str(totalE) + '\n')
				fh.write('property int vertex1' + '\n')
				fh.write('property int vertex2' + '\n')
				fh.write('end_header' + '\n')

				lines = []
				ctroffset = 0
				for k, v in samples_p.items():						#  For every LABEL...
					ctr = 0
					act = k.split('(')[0]							#  ACT, derived from LABEL, determines COLOR
					for pose in v:
						fh.write(str(pose['head'][0]) + ' ' + str(pose['head'][1]) + ' ' + str(pose['head'][2]) + ' ' + ' '.join([str(x) for x in actionColors[act]]) + '\n')

						fh.write(str(pose['head'][3][:, 0].dot(pose['lefthand'][:3]) + pose['head'][0]) + ' ' + \
						         str(pose['head'][3][:, 1].dot(pose['lefthand'][:3]) + pose['head'][1]) + ' ' + \
						         str(pose['head'][3][:, 2].dot(pose['lefthand'][:3]) + pose['head'][2]) + ' ' + \
						         ' '.join([str(x) for x in actionColors[act]]) + ' ' + '\n')
						fh.write(str(pose['head'][3][:, 0].dot(pose['righthand'][:3]) + pose['head'][0]) + ' ' + \
						         str(pose['head'][3][:, 1].dot(pose['righthand'][:3]) + pose['head'][1]) + ' ' + \
						         str(pose['head'][3][:, 2].dot(pose['righthand'][:3]) + pose['head'][2]) + ' ' + \
						         ' '.join([str(x) for x in actionColors[act]]) + ' ' + '\n')
						lines.append( (ctroffset, ctroffset + 1) )
						lines.append( (ctroffset, ctroffset + 2) )
						if ctr > 0:
							lines.append( (ctroffset - 3, ctroffset) )
						ctroffset += 3
						ctr += 1

				for line in lines:
					fh.write(str(line[0]) + ' ' + str(line[1]) + '\n')
				fh.close()

				#####################################################  Make the color-coded action centipede in the global frame
				fh = open(enactment + '.actions.global.ply', 'w')
				fh.write('ply' + '\n')
				fh.write('format ascii 1.0' + '\n')
				for k, v in actionColors.items():
					fh.write('comment ' + k + ' = (' + ' '.join([str(x) for x in v]) + ')\n')
				fh.write('element vertex ' + str(totalV) + '\n')
				fh.write('property float x' + '\n')
				fh.write('property float y' + '\n')
				fh.write('property float z' + '\n')
				fh.write('property uchar red' + '\n')
				fh.write('property uchar green' + '\n')
				fh.write('property uchar blue' + '\n')
				fh.write('element edge ' + str(totalE) + '\n')
				fh.write('property int vertex1' + '\n')
				fh.write('property int vertex2' + '\n')
				fh.write('end_header' + '\n')

				lines = []
				ctroffset = 0
				for k, v in samples_p.items():						#  For every LABEL...
					ctr = 0
					act = k.split('(')[0]							#  ACT, derived from LABEL, determines COLOR
					for pose in v:
						fh.write(str(pose['head'][0]) + ' ' + str(pose['head'][1]) + ' ' + str(pose['head'][2]) + ' ' + ' '.join([str(x) for x in actionColors[act]]) + '\n')

						fh.write(str(pose['globalLH'][0]) + ' ' + \
						         str(pose['globalLH'][1]) + ' ' + \
						         str(pose['globalLH'][2]) + ' ' + \
						         ' '.join([str(x) for x in actionColors[act]]) + ' ' + '\n')
						fh.write(str(pose['globalRH'][0]) + ' ' + \
						         str(pose['globalRH'][1]) + ' ' + \
						         str(pose['globalRH'][2]) + ' ' + \
						         ' '.join([str(x) for x in actionColors[act]]) + ' ' + '\n')
						lines.append( (ctroffset, ctroffset + 1) )
						lines.append( (ctroffset, ctroffset + 2) )
						if ctr > 0:
							lines.append( (ctroffset - 3, ctroffset) )
						ctroffset += 3
						ctr += 1

				for line in lines:
					fh.write(str(line[0]) + ' ' + str(line[1]) + '\n')
				fh.close()

				#####################################################  Make videos
				vid = cv2.VideoWriter(enactment + '_annotated.avi', \
				                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
				                      10, \
				                      (params['imgw'], params['imgh']) )
				for k, v in sorted(samples.items()):
					img = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/' + v['file'], cv2.IMREAD_UNCHANGED)
					if img.shape[2] == 4:							#  Drop alpha channel
						img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
					if v['label'] is None:
																	#  Write the action
						cv2.putText(img, '<NEUTRAL>', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (255, 255, 255, 255), 3)
					else:
																	#  Write the action
						cv2.putText(img, v['label'], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (255, 255, 255, 255), 3)

																	#  Build left-hand string
					LHstr  = "{:.2f}".format(v['lefthand'][0]) + ' '
					LHstr += "{:.2f}".format(v['lefthand'][1]) + ' '
					LHstr += "{:.2f}".format(v['lefthand'][2]) + ' '
					LHstr += str(v['lefthand'][3])
																	#  Does left hand project into image?
					LHproj = projects_into_camera(v['globalLH'], v['head'][:3], v['head'][3], K, params)
					if LHproj is not None:							#  It projects: draw a green-for-left circle and write white text
						cv2.circle(img, (params['imgw'] - LHproj[0], params['imgh'] - LHproj[1]), 5, (0, 255, 0, 255), 3)
						cv2.putText(img, 'Left Hand: ' + LHstr, (10, params['imgh'] - 160), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (0, 255, 0, 255), 3)
					else:
																	#  It does NOT project: write black text
						cv2.putText(img, 'Left Hand: ' + LHstr, (10, params['imgh'] - 160), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (0, 128, 0, 255), 3)

																	#  Build right-hand string
					RHstr  = "{:.2f}".format(v['righthand'][0]) + ' '
					RHstr += "{:.2f}".format(v['righthand'][1]) + ' '
					RHstr += "{:.2f}".format(v['righthand'][2]) + ' '
					RHstr += str(v['righthand'][3])
																	#  Does right hand project into image?
					RHproj = projects_into_camera(v['globalRH'], v['head'][:3], v['head'][3], K, params)
					if RHproj is not None:							#  It projects: draw a red-for-right circle and write white text
						cv2.circle(img, (params['imgw'] - RHproj[0], params['imgh'] - RHproj[1]), 5, (0, 0, 255, 255), 3)
						cv2.putText(img, 'Right Hand: ' + RHstr, (10, params['imgh'] - 40), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (0, 0, 255, 255), 3)
					else:
																	#  It does NOT project: write black text
						cv2.putText(img, 'Right Hand: ' + RHstr, (10, params['imgh'] - 40), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (0, 0, 128, 255), 3)

					vid.write(img)									#  Write the frame to the current video

				vid.release()
		elif params['verbose']:
			print('*** ' + enactment + '_poses.{full,shutoff}.txt already exist. Skipping.')
	return

def projects_into_camera(handpos, headpos, headrot, K, params):
	world = np.array( [ [handpos[0]], \
	                    [handpos[1]], \
	                    [handpos[2]], \
	                    [ 1.0      ] ])
																	#  Construct hand location in homogeneous world-coordinates
																	#  Build the translation for the head
	T = np.concatenate((np.eye(3), np.array([[-headpos[0]], \
	                                         [-headpos[1]], \
	                                         [-headpos[2]]])), axis=1)
																	#  Project hand
	img = np.dot(K, np.dot(headrot, np.dot(T, world)))
	img /= img[2][0]

	x = int(round(img[0][0]))										#  Round and discretize to pixels
	y = int(round(img[1][0]))

	if x >= 0 and x < params['imgw'] and y >= 0 and y < params['imgh']:
		return (x, y)
	return None

#  Iterate over all enactments to build a master set of all labels/actions used.
#  Assign each a color.
#  This is relevant for building the centipedes.
def survey_action_colors(params):
	actionColors = {}												#  string ==> (r, g, b)

	for enactment in params['enactments']:							#  Survey actions across all enactments

		if params['verbose']:
			print('>>> Surveying actions in ' + enactment)

		fh = open(enactment + '/Labels.fvr', 'r')					#  Fetch action time stamps
		lines = ' '.join(fh.readlines())							#  These USED to be a single line; NOW they have carriage returns
		fh.close()
		labels = json.loads(lines)

		for action in labels['list']:
			act = action['stepDescription'].split('(')[0]
			actionColors[ act ] = True								#  True for now... once we've counted everything, these will be well-spaced colors

	if params['verbose']:
		print('>>> ' + str(len(actionColors)) + ' unique actions across all specified enactments')

																	#  Build colors for all enactments
	colors = np.array([ [int(x * 255)] for x in np.linspace(0.0, 1.0, len(actionColors)) ], np.uint8)
	colors = cv2.applyColorMap(colors, cv2.COLORMAP_JET)
	i = 0
	for k in actionColors.keys():
		actionColors[k] = [int(x) for x in colors[i][0]]
		actionColors[k].reverse()
		actionColors[k] = tuple(actionColors[k])
		i += 1

	return actionColors

#  Run what used to be 22feb21's check.py script and what used to be 22feb21's histogram_z.py script:
#  Survey depth ranges of all given enactments. Find the lowest low and the highest high.
#  Make sure all depth maps are single-channel images. Plot a histogram of all depths used in given enactments.
#  Render composite videos of all given enactments so that we can see at a glance whether the depth or color maps have been mismatched. It's happened before.
#  SKIP ALL OF THIS IF A LOG FILE ALREADY EXISTS
def check(params):
	if not os.path.exists('FVR_check.log') or params['force']:
		if params['verbose']:
			print('********************')
			print('** Running checks **')
			print('********************')

		depthRange = {}
		depthRange['min'] = float('inf')
		depthRange['max'] = float('-inf')
		for enactment in params['enactments']:
			fh = open(enactment + '/metadata.fvr', 'r')
			line = fh.readlines()[0]
			fh.close()
			metadata = json.loads(line)
			if params['verbose']:
				print('>>> ' + enactment + '/metadata.fvr depth range: [' + str(metadata['depthImageRange']['x']) + ', ' + str(metadata['depthImageRange']['y']) + ']')
			if metadata['depthImageRange']['x'] < depthRange['min']:
				depthRange['min'] = metadata['depthImageRange']['x']#  Save Z-map's minimum (in meters)
			if metadata['depthImageRange']['y'] > depthRange['max']:
				depthRange['max'] = metadata['depthImageRange']['y']#  Save Z-map's maximum (in meters)
		if params['verbose']:
			print('>>> Consensus depth range across all metadata files: [' + str(depthRange['min']) + ', ' + str(depthRange['max']) + ']')

		histogram = {}												#  Prepare the histogram--IN METERS
		for i in range(0, 256):
			d = (float(i) / 255.0) * (depthRange['max'] - depthRange['min']) + depthRange['min']
			histogram[d] = 0

		fh = open('FVR_check.log', 'w')

		for enactment in params['enactments']:						#  Make sure depth maps are grayscale/single-channel
			if params['verbose']:
				print('>>> Scanning ' + enactment + ' depth maps')
			for depthmapfilename in sorted(os.listdir(enactment + '/Users/' + params['User'] + '/POV/DepthMapCameraFrames/'), key=lambda x: int(x.split('_')[0])):
				depthmap = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/DepthMapCameraFrames/' + depthmapfilename, cv2.IMREAD_UNCHANGED)
				if len(depthmap.shape) != 2:
					fh.write(enactment + '/Users/' + params['User'] + '/POV/DepthMapCameraFrames/' + depthmapfilename + '\t' + 'is not single-channel' + '\n')
					if params['verbose']:
						print(enactment + '/Users/' + params['User'] + '/POV/DepthMapCameraFrames/' + depthmapfilename + ' is not single-channel')
					if depthmap.shape[3] == 4:						#  If it's RGB + alpha, drop the alpha
						depthmap = cv2.cvtColor(depthmap, cv2.COLOR_BGRA2BGR)
																	#  Convert RGB to gray
					depthmap = cv2.cvtColor(depthmap, cv2.COLOR_BGR2GRAY)
				histg = [int(x) for x in cv2.calcHist([depthmap], [0], None, [256], [0, 256])]
				for i in range(0, 256):
					d = (float(i) / 255.0) * (depthRange['max'] - depthRange['min']) + depthRange['min']
					histogram[d] += histg[i]
		if params['verbose']:
			print('')

		lo = 0														#  Discover least and greatest values used
		while lo < 256 and histogram[ (float(lo) / 255.0) * (depthRange['max'] - depthRange['min']) + depthRange['min'] ] == 0:
			lo += 1
		hi = 255
		while hi >= 0 and histogram[ (float(hi) / 255.0) * (depthRange['max'] - depthRange['min']) + depthRange['min'] ] == 0:
			hi -= 1
		histg = []
		for i in range(0, 256):
			d = (float(i) / 255.0) * (depthRange['max'] - depthRange['min']) + depthRange['min']
			histg.append( histogram[d] )
																	#  Save the histogram
		plt.bar([(float(i) / 255.0) * (depthRange['max'] - depthRange['min']) + depthRange['min'] for i in range(0, 256)], height=histg)
		plt.xlabel('Depth')
		plt.ylabel('Frequency')
		plt.title('Depths Used')
		plt.savefig('histogram.png')

		if params['verbose']:
			print('>>> Least depth encountered = ' + str((float(lo) / 255.0) * (depthRange['max'] - depthRange['min']) + depthRange['min']))
			print('>>> Greatest depth encountered = ' + str((float(hi) / 255.0) * (depthRange['max'] - depthRange['min']) + depthRange['min']))

		for enactment in params['enactments']:						#  Build composite videos so we can spot bogies
			if params['verbose']:
				print('>>> Building composite video for ' + enactment)
			vid = cv2.VideoWriter(enactment + '_composite.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (params['imgw'], params['imgh']))
			for framename in sorted(os.listdir(enactment + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/'), key=lambda x: int(x.split('_')[0])):

				normal_exists = os.path.exists(enactment + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/' + framename)
				segmap_exists = os.path.exists(enactment + '/Users/' + params['User'] + '/POV/ColorMapCameraFrames/' + framename)
				dmap_exists   = os.path.exists(enactment + '/Users/' + params['User'] + '/POV/DepthMapCameraFrames/' + framename)

				if normal_exists and segmap_exists and dmap_exists:

					frame = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/' + framename, cv2.IMREAD_UNCHANGED)
					if frame.shape[2] == 4:							#  Do these guys have alpha channels? I forget.
						frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

					seg_oly = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/ColorMapCameraFrames/' + framename, cv2.IMREAD_UNCHANGED)
					if seg_oly.shape[2] == 4:						#  Some of these guys have an alpha channel; some of these guys don't
						seg_oly = cv2.cvtColor(seg_oly, cv2.COLOR_BGRA2BGR)

					d_oly = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/DepthMapCameraFrames/' + framename, cv2.IMREAD_UNCHANGED)
					if len(d_oly.shape) != 2:
						d_oly = d_oly[:, :, 0]						#  Isolate single channel
					d_oly = cv2.cvtColor(d_oly, cv2.COLOR_GRAY2BGR)	#  Restore color and alpha channels so we can map it to the rest

					d_oly = cv2.addWeighted(d_oly, 1.0, seg_oly, 0.7, 0)

					frame = cv2.addWeighted(frame, 1.0, d_oly, 0.7, 0)

					frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)	#  Flatten alpha

					vid.write(frame)
				else:
					if not normal_exists:
						fh.write(enactment + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/' + framename + '\tdoes not exist\n')
					if not segmap_exists:
						fh.write(enactment + '/Users/' + params['User'] + '/POV/ColorMapCameraFrames/' + framename + '\tdoes not exist\n')
					if not dmap_exists:
						fh.write(enactment + '/Users/' + params['User'] + '/POV/DepthMapCameraFrames/' + framename + '\tdoes not exist\n')

			vid.release()

		fh.close()
	elif params['verbose']:
		print('*** FVR_check.log already exists. Skipping the checks.')

	return

def getCommandLineParams():
	params = {}
	params['enactments'] = []										#  List of file paths
	params['model'] = None
	params['force'] = False

	params['verbose'] = False
	params['helpme'] = False

	params['colors'] = None											#  Objects' colors file, if applicable
	params['render'] = False										#  Whether to render stuff.

	params['fontsize'] = 1											#  For rendering text to images and videos
	params['imgw'] = 1280											#  It used to be 1920, and I don't want to change a bunch of ints when it changes again
	params['imgh'] = 720											#  It used to be 1080, and I don't want to change a bunch of ints when it changes again
	params['focal'] = None											#  Can be used to override the focal length we otherwise derive from metadata
	params['User'] = 'vr1'											#  It used to be "admin", and I don't want to change a bunch of file paths when it changes again
	params['gpus'] = 1

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-e', '-model', '-f', '-render', '-colors', \
	         '-v', '-?', '-help', '--help', \
	         '-imgw', '-imgh', '-focal', '-User', '-gpus', '-fontsize']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-v':
				params['verbose'] = True
			elif sys.argv[i] == '-render':
				params['render'] = True
			elif sys.argv[i] == '-f':
				params['force'] = True
			elif sys.argv[i] == '-?' or sys.argv[i] == '-help' or sys.argv[i] == '--help':
				params['helpme'] = True
			else:
				argtarget = sys.argv[i]
		else:
			argval = sys.argv[i]

			if argtarget is not None:
				if argtarget == '-e':
					params['enactments'].append(argval)
				elif argtarget == '-model':
					params['model'] = argval
				elif argtarget == '-imgw':
					params['imgw'] = int(argval)
				elif argtarget == '-imgh':
					params['imgh'] = int(argval)
				elif argtarget == '-User':
					params['User'] = argval
				elif argtarget == '-fontsize':
					params['fontsize'] = float(argval)
				elif argtarget == '-gpus':
					params['gpus'] = int(argval)
				elif argtarget == '-colors':
					params['colors'] = {}
					fh = open(argval, 'r')
					for line in fh.readlines():
						arr = line.strip().split('\t')
						params['colors'][arr[0]] = [int(arr[1]), int(arr[2]), int(arr[3])]
					fh.close()
				elif argtarget == '-focal':
					params['focal'] = float(argval)

	if params['fontsize'] < 1:
		print('>>> INVALID DATA received for fontsize. Restoring default value.')
		params['fontsize'] = 1

	if params['imgw'] < 1 or params['imgh'] < 1:
		print('>>> INVALID DATA received for image dimensions. Restoring default values.')
		params['imgw'] = 1280
		params['imgh'] = 720

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('You\'ve received new or revised FactualVR enactments.')
	print('You may or may not have processed them yet...')
	print('  but you have a trained version of Mask-RCNN, and now you want to perform object-detection them.')
	print('')
	print('This script builds a Mask-RCNN detection lookup file for each enactment. Each row contains:')
	print('    <timestamp, image-filename, label, object-class-name, score, BBox-x1,BBox-y1;BBox-x2,BBox-y2, mask-filename>')
	print('These are derived by running Mask-RCNN on VR frames (NormalViewCameraFrames).')
	print('Frames without detections contain only the first element, image-filename.')
	print('')
	print('Usage:  python3.5 detect_enactment.py <parameters, preceded by flags>')
	print(' e.g.:  python3.5 detect_enactment.py -model mask_rcnn_factual_0028.h5 -e BackBreaker1 -e Enactment1 -e Enactment2 -e Enactment3 -e Enactment4 -e Enactment5 -e Enactment6 -e Enactment7 -e Enactment9 -e Enactment10 -e Enactment11 -e Enactment12 -e MainFeederBox1 -e Regulator1 -e Regulator2 -v -render -colors colors_train.txt')
	print('')
	print('Flags:  -e         MUST HAVE AT LEAST ONE: Path to a directory of raw enactment materials.')
	print('        -render    Generate illustrations (videos, 3D representations) for all given enactments.')
	print('        -v         Enable verbosity')
	print('        -?         Display this message')

if __name__ == '__main__':
	main()