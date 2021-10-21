import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)											#  To find local version of the library
																	#  Alternatively, you could make a link:
																	#  ln -s ~/maskrcnn/mrcnn ~/maskrcnn/samples/factualvr/mrcnn
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

import keras
from keras.callbacks import ModelCheckpoint
import tensorflow as tf												#  Prevent out of control memory hogging
from tensorflow.python.framework.config import list_physical_devices
from tensorflow.python.framework.config import set_memory_growth

'''
Mask-RCNN trains in two phases:
  Phase 1: the head weights.
           The idea is to freeze the weights that have already learned some rudimentary object qualities that will still be helpful for us.
           Textures, edges, patterns... the general qualities of "objectiness" remain untouched.
  Phase 2: all weights.
           Now blend the criteria learned for our specific use-case back into the rudimentary features.
           This bends the general qualities of "objectiness" to our task.

We define a Callback class for each phase so we can track progress separately.
'''
class BookmarkPhase1Callback(keras.callbacks.Callback):
	def on_train_begin(self, logs=None):
		return

	def on_train_end(self, logs=None):
		return

	def on_epoch_begin(self, epoch, logs=None):
		return

	def on_epoch_end(self, epoch, logs=None):
		if not os.path.exists('bookmark_ph1.txt'):					#  If it doesn't exist, create a bookmark.
			fh = open('bookmark_ph1.txt', 'w')
			fh.write('Epoch\tLoss\tVal.Loss\n')
			fh.close()

		if not os.path.exists('bookmark_ph1.log'):					#  If it doesn't exist, create a log.
			fh = open('bookmark_ph1.log', 'w')
			fh.write('Epoch\tLoss\tVal.Loss\n')
			fh.close()

		fh = open('bookmark_ph1.txt', 'r')							#  The bookmark is for quick checks:
		lines = fh.readlines()										#  how many epochs have we done, and what's the loss so far?
		fh.close()

		shutil.copy('bookmark_ph1.txt', 'bookmark_ph1.backup.txt')

		fh = open('bookmark_ph1.txt', 'w')
		fh.write('Epoch\tLoss\tVal.Loss\n')
		fh.write(str(epoch) + '\t' + str(logs['loss']) + '\t' + str(logs['val_loss']))
		fh.close()

		loss = []													#  Prepare to collect history for intermediate graph.
		val_loss = []

		shutil.copy('bookmark_ph1.log', 'bookmark_ph1.backup.log')	#  The log is for plotting after training: track the curve.

		fh = open('bookmark_ph1.log', 'w')
		linectr = 0
		for line in lines:
			fh.write(line)
			if linectr > 0:
				arr = line.strip().split('\t')
				loss.append(float(arr[1]))							#  At the same time, read them into our graphable lists.
				val_loss.append(float(arr[2]))
			linectr += 1
		fh.write(str(epoch) + '\t' + str(logs['loss']) + '\t' + str(logs['val_loss']) + '\n')
		loss.append(logs['loss'])									#  Add these to the graphable lists.
		val_loss.append(logs['val_loss'])
		fh.close()
																	#  Graph the intermediate loss.
		plt.plot(range(len(loss)), loss, 'bo', label='Training loss')
		plt.plot(range(len(loss)), val_loss, 'r', label='Validation loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig('latest-phase1.png')
		plt.clf()													#  Clear the graph.
																	#  Save the latest model.
		self.model.save_weights('maskrcnn.ph1.' + str(epoch) + '.h5')

		return

	def on_test_begin(self, logs=None):
		return

	def on_test_end(self, logs=None):
		return

	def on_predict_begin(self, logs=None):
		return

	def on_predict_end(self, logs=None):
		return

	def on_train_batch_begin(self, batch, logs=None):
		return

	def on_train_batch_end(self, batch, logs=None):
		return

	def on_test_batch_begin(self, batch, logs=None):
		return

	def on_test_batch_end(self, batch, logs=None):
		return

	def on_predict_batch_begin(self, batch, logs=None):
		return

	def on_predict_batch_end(self, batch, logs=None):
		return

class BookmarkPhase2Callback(keras.callbacks.Callback):
	def on_train_begin(self, logs=None):
		return

	def on_train_end(self, logs=None):
		return

	def on_epoch_begin(self, epoch, logs=None):
		return

	def on_epoch_end(self, epoch, logs=None):
		if not os.path.exists('bookmark_ph2.txt'):					#  If it doesn't exist, create a bookmark.
			fh = open('bookmark_ph2.txt', 'w')
			fh.write('Epoch\tLoss\tVal.Loss\n')
			fh.close()

		if not os.path.exists('bookmark_ph2.log'):					#  If it doesn't exist, create a log.
			fh = open('bookmark_ph2.log', 'w')
			fh.write('Epoch\tLoss\tVal.Loss\n')
			fh.close()

		fh = open('bookmark_ph2.txt', 'r')							#  The bookmark is for quick checks:
		lines = fh.readlines()										#  how many epochs have we done, and what's the loss so far?
		fh.close()

		shutil.copy('bookmark_ph2.txt', 'bookmark_ph2.backup.txt')

		fh = open('bookmark_ph2.txt', 'w')
		fh.write('Epoch\tLoss\tVal.Loss\n')
		fh.write(str(epoch) + '\t' + str(logs['loss']) + '\t' + str(logs['val_loss']))
		fh.close()

		loss = []													#  Prepare to collect history for intermediate graph.
		val_loss = []

		shutil.copy('bookmark_ph2.log', 'bookmark_ph2.backup.log')	#  The log is for plotting after training: track the curve.

		fh = open('bookmark_ph2.log', 'w')
		linectr = 0
		for line in lines:
			fh.write(line)
			if linectr > 0:
				arr = line.strip().split('\t')
				loss.append(float(arr[1]))							#  At the same time, read them into our graphable lists.
				val_loss.append(float(arr[2]))
			linectr += 1
		fh.write(str(epoch) + '\t' + str(logs['loss']) + '\t' + str(logs['val_loss']) + '\n')
		loss.append(logs['loss'])									#  Add these to the graphable lists.
		val_loss.append(logs['val_loss'])
		fh.close()
																	#  Save the latest model.
		self.model.save_weights('maskrcnn.ph2.' + str(epoch) + '.h5')
																	#  Graph the intermediate loss.
		plt.plot(range(len(loss)), loss, 'bo', label='Training loss')
		plt.plot(range(len(loss)), val_loss, 'r', label='Validation loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig('latest-phase2.png')
		plt.clf()													#  Clear the graph.

		return

	def on_test_begin(self, logs=None):
		return

	def on_test_end(self, logs=None):
		return

	def on_predict_begin(self, logs=None):
		return

	def on_predict_end(self, logs=None):
		return

	def on_train_batch_begin(self, batch, logs=None):
		return

	def on_train_batch_end(self, batch, logs=None):
		return

	def on_test_batch_begin(self, batch, logs=None):
		return

	def on_test_batch_end(self, batch, logs=None):
		return

	def on_predict_batch_begin(self, batch, logs=None):
		return

	def on_predict_batch_end(self, batch, logs=None):
		return

def main():
	params = get_command_line_params()								#  Collect parameters
	if params['helpme']:
		usage()
		return

	classes = {}													#  Initially a dictionary so we can treat it like a set.
	if len(params['enactments']) == 0:								#  No enactments listed? Use all *_props.txt that exist.
		for filename in [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('_props.txt')]:
			params['enactments'].append(filename.split('_')[0])

	for enactment in params['enactments']:							#  Scan all enactments and build a list of objects to learn.
		fh = open(enactment + '_props.txt', 'r')
		lines = fh.readlines()
		fh.close()
		for line in lines:
			if line[0] != '#':										#  timestamp
				arr = line.strip().split('\t')						#  image-filename
																	#  instance
				class_name = arr[3]									#  class
																	#  detection-source
																	#  confidence
																	#  bounding-box
																	#  mask-filename
																	#  3D-centroid-Avg
																	#  3D-centroid-BBox
				if class_name not in params['ignore']:
					classes[ class_name ] = True					#  Mark as present.

	classes = sorted([x for x in classes.keys()])					#  NOW make it a list.
	if len(classes) == 0:
		print('ERROR: No learnable classes!!')
		return

	if params['verbose']:											#  Skip a line; clear all that TensorFlow barf.
		print('')

	fh = open('mask_rcnn_factual.recognizable.objects', 'w')		#  Log the objects this succession of networks will learn.
	fh.write('#  Created for Mask-RCNN training session started at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
	fh.write('#  The accompanying succession of networks can recognize the following objects:\n')
	for object_name in classes:
		fh.write(object_name + '\n')
	fh.close()
																	#  Find all data for all classes for all enactments.
	train, valid = load_all_learnable_instances(classes, params)	#  'train' = key:class-name ==> val:[ (enactment, imgfile, maskpath), ... ]
																	#  'valid' = key:class-name ==> val:[ (enactment, imgfile, maskpath), ... ]
	devices = list_physical_devices()								#  Prevent memory hogging!!!
	for device in devices:
		if device.device_type == 'GPU':
			set_memory_growth(device, True)

	train_model(train, valid, classes, params)

	return

#  Iterate over all specified enactments and identify acceptable training samples.
def load_all_learnable_instances(classes, params):
	data = {}														#  key:class-name ==> val:[ (enactment, imgfile, maskpath), ... ]
	train = {}
	valid = {}
																	#  If we cannot find ready-made training and validation sets, generate them.
	if not os.path.exists('training-set.txt') or not os.path.exists('validation-set.txt') or params['force']:
		for enactment in params['enactments']:						#  Survey enactments directories
			if params['verbose']:
				print('>>> Scanning ' + enactment + ' for all instances of learnable objects...')

			fh = open(enactment + '_props.txt', 'r')
			lines = fh.readlines()
			for line in lines:
				if line[0] != '#':									#  timestamp
					arr = line.strip().split('\t')					#  image-filename
					imgfilename = arr[1]							#  instance
					classpresent = arr[3]							#  class
																	#  detection-source
																	#  confidence
																	#  bounding-box
					maskpath = arr[7]								#  mask-filename
																	#  3D-centroid-Avg
																	#  3D-centroid-BBox

					if classpresent in classes:						#  Here is an instance of one of the things we want to learn.
						mask = cv2.imread(maskpath, cv2.IMREAD_UNCHANGED)
						indices = np.where(mask == 255)
						if len(indices[0]) >= params['minpx']:		#  Is it large enough to be helpful?
							if classpresent not in data:
								data[classpresent] = []
							data[classpresent].append( (enactment, imgfilename, maskpath) )
			fh.close()

			for k, v in data.items():								#  Shuffle everything
				np.random.shuffle(data[k])							#  Remember: this shuffles IN PLACE!

		if params['balance']:										#  Balance the sets:
			least = float('inf')
			for k, v in data.items():								#  Find the least-represented class.
				if len(v) < least:
					least = len(v)
			for k, v in data.items():								#  Clamp all classes to that length.
				data[k] = v[:least]

		if params['clamp'] < float('inf'):							#  Maximum is [:inf]
			for k, v in data.items():								#  Clamp
				data[k] = v[:params['clamp'] + 1]					#  Minimum is [:2]

		for k, v in data.items():									#  Partition the sets
			m = int(round(float(len(v)) * params['train']))
			train[k] = data[k][:m]
			valid[k] = data[k][m:]

		fh = open('training-set.txt', 'w')							#  Write training set to file
		fh.write('#  ' + ' '.join(sorted(data.keys())) + '\n')
		fh.write('#  Learnable-object    Enactment    Image-file    Mask-path\n')
		for k, v in train.items():
			for vv in v:
				fh.write(k + '\t' + vv[0] + '\t' + vv[1] + '\t' + vv[2] + '\n')
		fh.close()

		fh = open('validation-set.txt', 'w')						#  Write validation set to file
		fh.write('#  ' + ' '.join(sorted(data.keys())) + '\n')
		fh.write('#  Learnable-object    Enactment    Image-file    Mask-path\n')
		for k, v in valid.items():
			for vv in v:
				fh.write(k + '\t' + vv[0] + '\t' + vv[1] + '\t' + vv[2] + '\n')
		fh.close()
	else:
		if params['verbose']:
			print('>>> Loading from "training-set.txt" and "validation-set.txt"...')

		fh = open('training-set.txt', 'r')
		lines = fh.readlines()
		for line in lines:
			if line[0] != '#':
				arr = line.strip().split('\t')
				classpresent = arr[0]
				enactment = arr[1]
				imgfilename = arr[2]
				maskpath = arr[3]
				if classpresent not in train:
					train[classpresent] = []
				train[classpresent].append( (enactment, imgfilename, maskpath) )
		fh.close()

		fh = open('validation-set.txt', 'r')
		lines = fh.readlines()
		for line in lines:
			if line[0] != '#':
				arr = line.strip().split('\t')
				classpresent = arr[0]
				enactment = arr[1]
				imgfilename = arr[2]
				maskpath = arr[3]
				if classpresent not in valid:
					valid[classpresent] = []
				valid[classpresent].append( (enactment, imgfilename, maskpath) )
		fh.close()

	if params['verbose']:
		print('    ' + str(sum([len(x) for x in train.values()]) + sum([len(x) for x in valid.values()])) + ' trainable samples, total')
		for classname in classes:
			if classname not in train:
				print('    WARNING!   No samples for ' + classname)
			else:
				print('    ' + str(len(train[classname])) + ' samples for ' + classname)

	return train, valid

#  The heavy lifting is done here.
def train_model(train, valid, classes, params):
	ROOT_DIR = os.path.abspath("../../")							# Root directory of the project
	sys.path.append(ROOT_DIR)										# To find local version of the library
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")						# Directory to save logs and trained model
	COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")	# Local path to trained weights file
	if not os.path.exists(COCO_MODEL_PATH):							# Download COCO trained weights from Releases if needed
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
		STEPS_PER_EPOCH = sum([len(x) for x in train.values()])		#  This used to be = 100 with note "Use a small epoch since the data is simple"
		VALIDATION_STEPS = 5										#  use small validation steps since the epoch is small

	#  Collects and organizes the FactualVR video dataset
	class FactualDataset(utils.Dataset):

		#  Fetch the requested number of images, 'count'
		def load_src(self, count, height, width):
			ctr = 1													#  Add classes
			for k in classes:
				self.add_class("factual", ctr, k)
				ctr += 1
			for i in range(count):									#  Add images
				srcfile = self.random_image()						#  Returns a file name
				self.add_image("factual", image_id=i, path=None, \
				               width=width, height=height, \
				               img_file=srcfile)

		#  Fetch the requested number of images, 'count'
		def load_set(self, set_files, height, width):
			ctr = 1													#  Add classes
			for k in classes:
				self.add_class("factual", ctr, k)
				ctr += 1
			i = 0
			for fname in set_files:									#  Add images
				self.add_image("factual", image_id=i, path=None, \
				               width=width, height=height, \
				               img_file=fname)
				i += 1

		#  'image_id' is an int:
		#  Fetch the image indicated at image_id and scale it down to the dimensions used for this network: 256 x 256
		#  We can live with a bit of squish as long as we squish the mask, too.
		def load_image(self, image_id):
			info = self.image_info[image_id]						#  Retrieve this sample's information
			target_height = info['height']							#  Unpack height
			target_width = info['width']							#  Unpack width
																	#  The MASKS have ALPHA CHANNELS; the VIDEO does NOT
			img = cv2.imread(info['img_file'], cv2.IMREAD_COLOR)	#  Images are already RGB[A]
			if img.shape[2] > 3:
				img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
			img = cv2.resize(img, (target_width, target_height))	#  Scale down (and squish)

			return img

		#  image_id is an int
		def image_reference(self, image_id):
			"""Return the shapes data of the image."""
			info = self.image_info[image_id]
			if info["source"] == "shapes":
				return info["shapes"]
			else:
				super(self.__class__).image_reference(self, image_id)

		#  image_id is an int
		#  End result of this method is a tuple:
		#    NumPy array of Bools
		#    List of indices into 'self.class_names' given the classname string
		def load_mask(self, image_id):
			info = self.image_info[image_id]						#  Retrieve this sample's information
			target_height = info['height']							#  Unpack height
			target_width = info['width']							#  Unpack width

			count = 0												#  Becomes the depth of the mask tensor

			filenameparse = info['img_file'].split('/')				#  Split up, e.g. "ArcSuit1/Users/vr1/POV/NormalViewCameraFrames/169_16.9.png"
			maskfiles = []											#  List of file paths to masks applicable to this image
			trueobjects = []										#  For which object this is a mask
			enactment = filenameparse[0]							#  Open, e.g., ArcSuit1_groundtruth.txt
			fh = open(enactment + '_groundtruth.txt', 'r')
			lines = fh.readlines()
			fh.close()
			for line in lines:
				if line[0] != '#':
					arr = line.strip().split('\t')
					enactmentgt_time = arr[0]
					enactmentgt_file = arr[1]
					enactmentgt_label = arr[2]
					enactmentgt_obj = arr[3]
					enactmentgt_conf = arr[4]
					enactmentgt_bboxstr = arr[5]
					enactmentgt_maskpath = arr[6]
																	#  Find all, e.g., for 169_16.9.png
					if enactmentgt_file == filenameparse[-1] and enactmentgt_obj in classes:
						maskfiles.append(enactmentgt_maskpath)		#  Which mask file this is
						trueobjects.append(enactmentgt_obj)			#  Class of this mask's object

			count = len(trueobjects)								#  Depth of buffer is number of classes in this image
			if count > 0:											#  Make sure *some*-thing is here
				class_ids = np.array( [self.class_names.index(name) for name in trueobjects] )
				mask = np.zeros([target_height, target_width, count], dtype=np.uint8)

				i = 0
				for maskfile in maskfiles:
					maskbuffer = cv2.imread(maskfile, cv2.IMREAD_UNCHANGED)
					maskbuffer = cv2.resize(maskbuffer, (target_width, target_height))
					mask[:, :, i] = maskbuffer
					i += 1
			else:													#  When sampling from enactments, we MIGHT draw an empty frame!
				class_ids = np.array( [] )
				mask = np.zeros([target_height, target_width, 1], dtype=np.uint8)

			#  REMEMBER THAT OCCLUSION BIT IN THE ORIGINAL CODE? IT USED np.logical_not AND np.logical_and.
			#  YEAH, YOU DON'T NEED THAT ANYMORE HERE BECAUSE OCCLUSION IS ALREADY HANDLED!

			return mask.astype(np.bool), class_ids.astype(np.int32)	#  Return scaled-down bool-map
																	#  and array of class-name indices

		#  Pull a random file (name) from either the training-set.txt list or the validation-set.txt list
		def random_image(self):
			unique_filenames = {}
			setlists = ['training-set.txt', 'validation-set.txt']
			fh = open(setlists[random.randint(0, 1)], 'r')
			lines = fh.readlines()
			fh.close()
			for line in lines:
				if line[0] != '#':
					arr = line.strip().split('\t')
					classname = arr[0]
					enactment = arr[1]
					imgfilename = arr[2]
					maskpath = arr[3]

					relfname = enactment + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/' + imgfilename

					if relfname not in unique_filenames:
						unique_filenames[relfname] = True

			unique_filenames = [x for x in unique_filenames.keys()]
			return unique_filenames[random.randint(0, len(unique_filenames) - 1)]

	config = FactualConfig()
	if params['verbose']:
		config.display()

	if params['verbose']:
		print('>>> Preparing data sets...')

	#################################################################
	##              Build training and validations sets            ##
	#################################################################

	dataset_train = FactualDataset()								# Training dataset
	T = []
	for k, v in train.items():
		for vv in v:
			T.append(vv[0] + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/' + vv[1])
	dataset_train.load_set(T, params['imgh'], params['imgw'])
	dataset_train.prepare()

	dataset_val = FactualDataset()									# Validation dataset
	V = []
	for k, v in valid.items():
		for vv in v:
			V.append(vv[0] + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/' + vv[1])
	dataset_val.load_set(V, params['imgh'], params['imgw'])
	dataset_val.prepare()

	if params['verbose']:
		print('    ...done')

	if params['colors'] is not None:								#  Build colors from classes
		fh = open(params['colors'], 'r')
		params['colors'] = {}
		for line in fh.readlines():
			arr = line.strip().split('\t')
			params['colors'][ arr[0] ] = (int(arr[1]), int(arr[2]), int(arr[3]))
		fh.close()
	else:															#  Generate random colors
		params['colors'] = {}
		for class_name in classes:
			params['colors'][class_name] = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

	sample_ctr = 1													# Load and display random samples
	image_ids = np.random.choice(dataset_train.image_ids, 10)
	for image_id in image_ids:
		image = dataset_train.load_image(image_id)
		mask, class_ids = dataset_train.load_mask(image_id)
		mask[mask > 1] = 1											#  All things greater than 1 become 1

		maskcanvas = np.zeros((config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], 3), dtype='uint8')
		#maskcanvas = np.zeros((params['imgh'], params['imgw'], 3), dtype='uint8')
		for i in range(0, mask.shape[2]):
			maskslice = mask[:, :, i]								#  Isolate the i-th slice of the mask-tensor
																	#  Extrude the slice to four channels
			maskslice = maskslice[:, :, None] * np.ones(3, dtype='uint8')[None, None, :]
																	#  Convert this to a graphical overlay
			maskslice[:, :, 0] *= params['colors'][ classes[class_ids[i] - 1] ][2]
			maskslice[:, :, 1] *= params['colors'][ classes[class_ids[i] - 1] ][1]
			maskslice[:, :, 2] *= params['colors'][ classes[class_ids[i] - 1] ][0]

			maskcanvas += maskslice									#  Add mask to mask accumulator
			maskcanvas[maskcanvas > 255] = 255						#  Clip accumulator to 255

		image = cv2.addWeighted(image, 1.0, maskcanvas, 0.7, 0)
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)				#  Flatten alpha

		cv2.imwrite('random_sample_' + str(sample_ctr) + '.png', image)
		sample_ctr += 1

																	#  Create model in training mode
	model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

	if params['init_with'] == "imagenet":							#  Which weights to start with?
		model.load_weights(model.get_imagenet_weights(), by_name=True)
	elif params['init_with'] == "coco":								#  Load weights trained on MS COCO, but skip layers that
																	#  are different due to the different number of classes
																	#  See README for instructions to download the COCO weights
		model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
	elif params['init_with'] == "last":								#  Load the last model you trained and continue training
		model.load_weights(model.find_last(), by_name=True)

	#################################################################
	#  Train the head branches.                                     #
	#  Passing layers="heads" freezes all layers except the head    #
	#  layers. You can also pass a regular expression to select     #
	#  which layers to train by name pattern.                       #
	#################################################################

	if params['verbose']:
		print('\n>>> Training phase 1 for ' + str(params['phase1epochs']) + ' epochs...')

	bookmark_phase1 = BookmarkPhase1Callback()
	callbacksList = [bookmark_phase1]
	model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, custom_callbacks=callbacksList, epochs=params['phase1epochs'], layers='heads')

	if params['verbose']:
		print('\n\t...done')

	#################################################################
	#  Fine tune all layers.                                        #
	#  Passing layers="all" trains all layers. You can also pass a  #
	#  regular expression to select which layers to train by name   #
	#  pattern.                                                     #
	#################################################################

	if params['verbose']:
		print('\n>>> Training phase 2 for ' + str(params['phase2epochs']) + ' epochs...')

	bookmark_phase2 = BookmarkPhase2Callback()
	callbacksList = [bookmark_phase2]
															#  Internally, epochs are counted cumulatively (stupid)
	model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, custom_callbacks=callbacksList, \
	            epochs=(params['phase1epochs'] + params['phase2epochs']), \
	            layers="all")

	if params['verbose']:
		print('\n\t...done')

	#################################################################
	#  Save weights.                                                #
	#  Typically not needed because callbacks save after every      #
	#  epoch.                                                       #
	#################################################################
	model_path = os.path.join(MODEL_DIR, "mask_rcnn_factual.h5")
	model.keras_model.save_weights(model_path)

	#################################################################
	#  Graph loss.                                                  #
	#################################################################
	loss = []
	val_loss = []
	if os.path.exists('bookmark_ph1.log'):
		fh = open('bookmark_ph1.log', 'r')
		lines = fh.readlines()[1:]
		fh.close()
		for line in lines:
			arr = line.strip().split('\t')
			loss.append(float(arr[1]))
			val_loss.append(float(arr[2]))
		plt.plot(range(len(loss)), loss, 'bo', label='Training loss')
		plt.plot(range(len(loss)), val_loss, 'r', label='Validation loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig('phase1.png')
		plt.clf()

	loss = []
	val_loss = []
	if os.path.exists('bookmark_ph2.log'):
		fh = open('bookmark_ph2.log', 'r')
		lines = fh.readlines()[1:]
		fh.close()
		for line in lines:
			arr = line.strip().split('\t')
			loss.append(float(arr[1]))
			val_loss.append(float(arr[2]))
		plt.plot(range(len(loss)), loss, 'bo', label='Training loss')
		plt.plot(range(len(loss)), val_loss, 'r', label='Validation loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig('phase2.png')

	return

def get_command_line_params():
	params = {}

	params['enactments'] = []
	params['train'] = 0.8											#  Default
	params['valid'] = 0.2											#  Default
	params['balance'] = False
	params['clamp'] = float('inf')
	params['force'] = False

	params['minpx'] = 100											#  Default

	params['phase1epochs'] = 1										#  Default
	params['phase2epochs'] = 2										#  Default
	params['init_with'] = 'coco'

	params['ignore'] = ['LeftHand', 'RightHand']
	params['colors'] = None

	params['imgw'] = 1280											#  It used to be 1920; let's avoid changing a bunch of ints when it changes again.
	params['imgh'] = 720											#  It used to be 1080; let's avoid changing a bunch of ints when it changes again.
	params['User'] = 'vr1'											#  It used to be "admin"; let's avoid changing a bunch of file paths when it changes again.
	params['gpus'] = 1

	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-e', '-train', '-valid', '-b', '-c', '-minpx', '-initW', '-ph1e', '-ph2e', '-x', \
	         '-f', '-colors', \
	         '-imgw', '-imgh', '-User', '-gpus', \
	         '-v', '-?', '-help', '--help']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-v':
				params['verbose'] = True
			elif sys.argv[i] == '-f':
				params['force'] = True
			elif sys.argv[i] == '-b':
				params['balance'] = True
			elif sys.argv[i] == '-?' or sys.argv[i] == '-help' or sys.argv[i] == '--help':
				params['helpme'] = True
			else:
				argtarget = sys.argv[i]
		else:
			argval = sys.argv[i]

			if argtarget is not None:
				if argtarget == '-e':
					params['enactments'].append(argval)
				elif argtarget == '-train':
					params['train'] = float(argval)
					params['valid'] = 1.0 - params['train']
				elif argtarget == '-valid':
					params['valid'] = float(argval)
					params['train'] = 1.0 - params['valid']
				elif argtarget == '-minpx':
					params['minpx'] = max(0, int(argval))
				elif argtarget == '-c':
					params['clamp'] = max(1, int(argval))
				elif argtarget == '-initW':
					params['init_with'] = argval
				elif argtarget == '-ph1e':							#  Following argument specifies phase 1 training epochs
					params['phase1epochs'] = int(argval)
				elif argtarget == '-ph2e':							#  Following argument specifies phase 2 training epochs
					params['phase2epochs'] = int(argval)
				elif argtarget == '-gpus':
					params['gpus'] = int(argval)
				elif argtarget == '-imgw':
					params['imgw'] = int(argval)
				elif argtarget == '-imgh':
					params['imgh'] = int(argval)
				elif argtarget == '-User':
					params['User'] = argval
				elif argtarget == '-colors':
					params['colors'] = argval
				elif argtarget == '-x':
					params['ignore'].append(argval)

	if params['init_with'] not in ['coco', 'imagenet', 'last']:
		print('WARNING: Invalid initial weights given; resorting to default.')
		params['init_with'] = 'coco'

	if params['phase1epochs'] <= 0 or params['phase2epochs'] <= 0:
		print('WARNING: Invalid number of epochs; resorting to defaults.')
		params['phase1epochs'] = 1
		params['phase2epochs'] = 2

	if params['train'] <= 0.0 or params['valid'] <= 0.0 or params['train'] + params['valid'] != 1.0:
		print('WARNING: Invalid training/validation allocation; resorting to defaults.')
		params['train'] = 0.8
		params['valid'] = 0.2

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('Train Mask RCNN on a FactualVR dataset.')
	print('')
	print('Usage:  python3 train_maskrcnn.py <parameters, preceded by flags>')
	print(' e.g.:  python3 train_maskrcnn.py -e BackBreaker1 -e Enactment1 -e Enactment2 -e Enactment3 -e Enactment4 -e Enactment5 -e Enactment6 -e Enactment7 -e Enactment9 -e Enactment10 -e Enactment11 -e Enactment12 -e MainFeederBox1 -e Regulator1 -e Regulator2 -x AuxiliaryFeederBox_Unknown -x BackBreaker_Unknown -x Disconnect_Unknown -x MainFeederBox_Unknown -x Regulator_Unknown -x SafetyPlank_Unknown -x TransferFeederBox_Unknown -ph1e 10 -ph2e 20 -v -minpx 200 -b -colors colors_train.txt')
	print('        nohup python3 train_maskrcnn.py -e BackBreaker1 -e Enactment1 -e Enactment2 -ph1e 10 -ph2e 40 &')
	print('        nohup python3 train_maskrcnn.py -e BackBreaker1 -e Enactment1 -e Enactment2 -ph1e 10 -ph2e 40 >/dev/null 2>&1 &')
	print('')
	print('Flags:  -e      Following argument is an enactment file to be used for training and/or validation data.')
	print('                If no enactments are listed, then every enactment that has a *_props.txt file will be used.')
	print('        -x      Specify a class-label to be excluded from learning. "LeftHand" and "RightHand" are omitted by default.')
	print('        -b      Balance the classes in the dataset. This means limiting all samples to the least represented class.')
	print('        -c      Clamp the maximum number of samples per class to this integer >= 1. (Default is infinity = no clamp.)')
	print('        -minpx  All object instances admitted to the data set must occupy at least this many pixels. Default is 100.')
	print('        -ph1e   Number of epochs for phase 1 training (only train the "heads".)')
	print('                Default is 1.')
	print('        -ph2e   Number of epochs for phase 2 training (fine-tune all network weights.)')
	print('                Default is 2.')
	print('        -train  In (0.0, 1.0). Portion of valid data to be allocated for training. Default is 0.8.')
	print('        -valid  In (0.0, 1.0). Portion of valid data to be allocated for validation. Default is 0.2.')
	print('        -v      Enable verbosity')
	print('        -?      Display this message')
	print('')
	print('Going to throw away some previous training efforts? Find them in ~/maskrcnn/logs/')
	return

if __name__ == '__main__':
	main()
