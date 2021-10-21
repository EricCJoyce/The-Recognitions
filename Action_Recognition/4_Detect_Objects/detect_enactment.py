from enactment import *

def main():
	params = get_command_line_params()								#  Collect parameters
	if params['helpme'] or len(params['enactments']) == 0:
		usage()
		return

	if params['verbose']:
		print('>>> Loading object-detection model "' + params['model'] + '".')
		print('')

	recognizable_objects = []										#  Prepare to read a list of recognizable objects.
	if 'mask_rcnn' in params['model']:
		fh_reference = open('mask_rcnn_factual.recognizable.objects', 'r')
		for line in fh_reference.readlines():
			if line[0] != '#':
				recognizable_objects.append( line.strip() )
		fh_reference.close()

		model, model_name = load_mask_rcnn(recognizable_objects, params)
	elif 'mobilenet' in params['model']:
		fh_reference = open('mobilenet_factual.recognizable.objects', 'r')
		for line in fh_reference.readlines():
			if line[0] != '#':
				recognizable_objects.append( line.strip() )
		fh_reference.close()

		model, model_name = load_mobilenet(params)
	else:
		print('ERROR: the specified network has no accompanying recognizable objects file (*_factual.recognizable.objects).')
		return

	if params['verbose']:
		print('>>> Performing object detections on the following enactments:')
		for enactment_name in params['enactments']:
			print('\t' + enactment_name)
		print('')

	timing = {}
	timing['image-open'] = []										#  Prepare to time image loading.
	timing['object-detection'] = []									#  Prepare to time object detections by a network.

	now = datetime.datetime.now()									#  Build a distinct substring so I don't accidentally overwrite results.
	file_timestamp = now.strftime("%d") + now.strftime("%m") + now.strftime("%Y")[-2:] + 'T' + now.strftime("%H:%M:%S").replace(':', '')
	fh = open('detect_enactment_' + file_timestamp + '.log', 'w')	#  Create a log file.

	for enactment_name in params['enactments']:
		maskfilectr = 0												#  Initialize mask counter for the current enactment.

		e = Enactment(enactment_name, enactment_file=enactment_name + '.enactment', user=params['User'], verbose=params['verbose'])
		e.recognizable_objects = recognizable_objects[:]			#  Copy recognizable objects into Enactment list.
		e.object_detection_source = params['model']
		directory_name = enactment_name + '/' + model_name
		if os.path.isdir(directory_name):							#  If 'directory_name' exists, delete it.
			shutil.rmtree(directory_name)
		os.mkdir(directory_name)									#  Create directory_name.

		frame_ctr = 0
		num_frames = e.num_frames()
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.
		prev_ctr = 0
		for time_stamp, frame in sorted(e.frames.items()):
			t1_start = time.process_time()							#  Start timer.
			img = cv2.imread(frame.fullpath('video'), cv2.IMREAD_COLOR)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			t1_stop = time.process_time()							#  Stop timer.
			timing['image-open'].append(t1_stop - t1_start)

			if 'mask_rcnn' in params['model']:						#  Perform detection using Mask-RCNN.
				t1_start = time.process_time()						#  Start timer.
				results = model.detect([img], verbose=0)			#  Model makes prediction(s)
				t1_stop = time.process_time()						#  Stop timer.
				timing['object-detection'].append(t1_stop - t1_start)

				for i in range(0, len(results[0]['rois'])):			#  For each detection...
					bbox = (results[0]['rois'][i][1], results[0]['rois'][i][0], \
					        results[0]['rois'][i][3], results[0]['rois'][i][2])
																	#  Write to 'mask'.
					mask = results[0]['masks'][:, :, i] * np.uint8(255)
					indices = np.where(mask > 0)					#  Mask-RCNN is capable of predicting bounding boxes empty of pixels!
					if len(indices[0]) > params['minpx']:
						confidence = float(results[0]['scores'][i])	#  Convert from numpy float32.
																	#  Build the mask file path.
						maskpath = enactment_name + '/' + model_name + '/mask_' + str(maskfilectr) + '.png'
						cv2.imwrite(maskpath, mask)					#  Save the mask.

						frame.detections.append( RObject(parent_frame=frame.fullpath('video'), \
						                                 object_name=recognizable_objects[ results[0]['class_ids'][i] - 1 ], \
						                                 detection_source=model_name, \
						                                 mask_path=maskpath, \
						                                 bounding_box=bbox, \
						                                 confidence=confidence) )

						maskfilectr += 1							#  Increment counter

			elif 'mobilenet' in params['model']:					#  Perform detection using MobileNet
				#######  LEFT OFF HERE !!! *** Add MobileNet!!! ***
				print('No....')

			if params['verbose']:
				if int(round(float(frame_ctr) / float(num_frames - 1) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(frame_ctr) / float(num_frames - 1) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(frame_ctr) / float(num_frames - 1) * 100.0))) + '%]')
					sys.stdout.flush()

			frame_ctr += 1

		e.render_detected(model_name)								#  Save detected objects to file.

		if params['render']:
			if params['colors'] is None:							#  No colors? Make random ones.
				e.random_colors()
			else:													#  Use given colors.
				e.load_color_map(params['colors'])
			e.render_annotated_video()

	fh.write('Avg. image opening and color converstion time\t' + str(np.mean(timing['image-open'])) + '\n')
	fh.write('Std.dev image opening and conversion time\t' + str(np.std(timing['image-open'])) + '\n\n')

	fh.write('Avg. object detection time (per frame, using ' + model_name + ')\t' + str(np.mean(timing['object-detection'])) + '\n')
	fh.write('Std.dev object detection time (per frame, using' + model_name + ')\t' + str(np.std(timing['object-detection'])) + '\n\n')

	fh.close()

	return

def load_mask_rcnn(recognizable_objects, params):
	# Root directory of the project
	#ROOT_DIR = os.path.abspath("../../")
	ROOT_DIR = os.path.abspath("./maskrcnn/")

	# Import Mask RCNN
	sys.path.append(ROOT_DIR)										# To find local version of the library

	from mrcnn.config import Config
	from mrcnn import utils
	import mrcnn.model as modellib
	from mrcnn import visualize
	from mrcnn.model import log

	MODEL_DIR = os.path.join(ROOT_DIR, "logs")						#  Directory to save logs and trained model
	COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")	#  Local path to trained weights file
	if not os.path.exists(COCO_MODEL_PATH):							#  Download COCO trained weights from Releases if needed
		utils.download_trained_weights(COCO_MODEL_PATH)

	class FactualConfig(Config):									#  Derives from the base Config class and overrides values specific
																	#  to this dataset.
		NAME = "factual"											#  Give the configuration a recognizable name

		GPU_COUNT = params['gpus']									#  Train on params['gpus'] GPUs and 8 images per GPU. We can put multiple images on each
		IMAGES_PER_GPU = 8											#  GPU because the images are small. Batch size is 8 (GPUs * images/GPU).

		NUM_CLASSES = len(recognizable_objects) + 1					#  Number of classes (the +1 includes background)

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
			for k in recognizable_objects:
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
			for k in recognizable_objects:
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
	model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir='.')
	model_path = params['model']
	model_name = params['model']

	if params['verbose']:											#  Load trained weights
		print('>>> Loading Mask-RCNN weights from ' + model_path)
	model.load_weights(model_path, by_name=True)

	return model, '.'.join(model_name.split('.')[:-1])

def load_mobilenet(params):
	model = None
	model_name = None
	return model, model_name

def get_command_line_params():
	params = {}
	params['enactments'] = []										#  List of file paths.
	params['model'] = None											#  Detection model.
	params['minpx'] = 400											#  Minimum number of pixels for something to be considered visible.
	params['colors'] = None											#  Optional color lookup table.
	params['render'] = False										#  Whether to render stuff.
	params['verbose'] = False
	params['helpme'] = False
	params['fontsize'] = 1											#  For rendering text to images and videos
	params['User'] = 'vr1'											#  It used to be "admin", and I don't want to change a bunch of file paths when it changes again
	params['gpus'] = 1

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-e', '-model', '-minpx', '-render', '-color', '-colors', '-gpus', \
	         '-v', '-?', '-help', '--help', \
	         '-User', '-fontsize']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-v':
				params['verbose'] = True
			elif sys.argv[i] == '-?' or sys.argv[i] == '-help' or sys.argv[i] == '--help':
				params['helpme'] = True
			elif sys.argv[i] == '-render':
				params['render'] = True
			else:
				argtarget = sys.argv[i]
		else:
			argval = sys.argv[i]

			if argtarget is not None:
				if argtarget == '-e':
					params['enactments'].append(argval)
				elif argtarget == '-model':
					params['model'] = argval
				elif argtarget == '-minpx':
					params['minpx'] = max(0, int(argval))
				elif argtarget == '-gpus':
					params['gpus'] = max(0, int(argval))
				elif argtarget == '-color' or argtarget == '-colors':
					params['colors'] = argval
				elif argtarget == '-User':
					params['User'] = argval
				elif argtarget == '-fontsize':
					params['fontsize'] = max(1.0, float(argval))

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('You\'ve received new or revised FactualVR enactments.')
	print('You want to check that they conform to specifications, equip yourself to spot any errors,')
	print('and finally to apply object combination and condition rules to derive recognizable objects')
	print('to be identified later in the pipeline.')
	print('')
	print('Usage:  python3 detect_enactment.py <parameters, preceded by flags>')
	print(' e.g.:  python3 detect_enactment.py -model mask_rcnn_factual_0028.h5 -e BackBreaker1 -e Enactment1 -v -render')
	print('')
	print('Flags:  -e       MUST HAVE AT LEAST ONE: Path to a directory of raw enactment materials: JSONs and color maps.')
	print('        -model   MUST HAVE EXACTLY ONE: Path to a model that can perform object recognition.')
	print('')
	print('        -render  Generate illustrations (videos, 3D representations) for all given enactments.')
	print('        -v       Enable verbosity')
	print('        -?       Display this message')

if __name__ == '__main__':
	main()
