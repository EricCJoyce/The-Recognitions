import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'							#  Suppress TensorFlow barf.
import tensorflow as tf
import time

from enactment import *
from object_detection.utils import label_map_util

def main():
	params = get_command_line_params()								#  Collect parameters.
	if params['helpme'] or len(params['enactments']) == 0 or params['model'] is None:
		usage()
		return

	timings = {}													#  Prepare to track run times.
	timings['image-open'] = []										#  Prepare to collect image-opening times.
	timings['detect'] = []											#  Prepare to collect detection times.

	now = datetime.datetime.now()									#  Build a distinct substring so I don't accidentally overwrite results.
	file_timestamp = now.strftime("%d") + now.strftime("%m") + now.strftime("%Y")[-2:] + 'T' + now.strftime("%H:%M:%S").replace(':', '')
	fh = open('detect_enactment_' + file_timestamp + '.log', 'w')	#  Create a log file.

	gpus = tf.config.experimental.list_physical_devices('GPU')		#  List all GPUs on this system.
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)			#  For each GPU, limit memory use.

	all_enactments_min_depth = float('inf')							#  Prepare to identify global depth map extrema.
	all_enactments_max_depth = float('-inf')
	dimensions = {}													#  key:enactment name ==> (width, height)
																	#  so we won't have to comb over these twice.
	#################################################################
	#  First pass on the given enactments: check frame sizes;       #
	#  check color channels; (optionally) render visual aides.      #
	#################################################################
	if params['verbose']:
		print('>>> Preparing the following enactments:')
		for enactment_name in params['enactments']:
			print('\t' + enactment_name)
		print('')

	for enactment_name in params['enactments']:						#  Go over each enactment and make sure they fit spec.
																	#  We deliberately leave out the kwarg width and height here:
																	#  we WANT the Enactment class to check every frame.
		e = Enactment(enactment_name, user=params['User'], verbose=params['verbose'])
		dimensions[enactment_name] = (e.width, e.height)			#  Save this enactment's dimensions.
																	#  Update global extrema.
		all_enactments_min_depth = min(all_enactments_min_depth, e.min_depth)
		all_enactments_max_depth = max(all_enactments_max_depth, e.max_depth)

		e.gt_label_super['fontsize'] = params['fontsize']			#  Set font sizes from this script's parameters.
		e.filename_super['fontsize'] = params['fontsize']
		e.LH_super['fontsize'] = params['fontsize']
		e.RH_super['fontsize'] = params['fontsize']

		color_maps = e.load_color_sequence(True)					#  Check every color map; make sure it matches the video dimensions.
		for color_map in color_maps:
			img = cv2.imread(color_map, cv2.IMREAD_UNCHANGED)
			if img.shape[0] != e.height or img.shape[1] != e.width:
				fh.write('ERROR: Color map "' + color_map + '" does not match the video frames shape in "' + enactment_name + '".\n')
				print('ERROR: Color map "' + color_map + '" does not match the video frames shape in "' + enactment_name + '".')

		depth_maps = e.load_depth_sequence(True)					#  Check every depth map; make sure it matches the video dimensions.
		for depth_map in depth_maps:
			img = cv2.imread(depth_map, cv2.IMREAD_UNCHANGED)
			if img.shape[0] != e.height or img.shape[1] != e.width:
				fh.write('ERROR: Depth map "' + depth_map + '" does not match the video frames shape in "' + enactment_name + '".\n')
				print('ERROR: Depth map "' + depth_map + '" does not match the video frames shape in "' + enactment_name + '".')
			if len(img.shape) > 2:
				fh.write('ERROR: Depth map "' + depth_map + '" in "' + enactment_name + '" has more channels than expected.\n')
				print('ERROR: Depth map "' + depth_map + '" in "' + enactment_name + '" has more channels than expected.')

		if params['render']:
			e.render_composite_video()								#  Make a composite video to make sure that these sets of frames align.
			#e.render_annotated_video()								#  Do not make an annotated video now; annotate with network detection overlays.

			e.render_skeleton_poses()								#  Render centipedes (headframe and global).
			e.render_action_poses()

	#################################################################
	#  Second pass on the given enactments: build depth histograms. #
	#################################################################
	if params['verbose']:
		print('>>> Surveying depths of the following enactments:')
		for enactment_name in params['enactments']:
			print('\t' + enactment_name)
		print('')

	histogram = {}													#  Build a depth histogram over all enactments.
	for i in range(0, 256):
		d = (float(i) / 255.0) * (all_enactments_max_depth - all_enactments_min_depth) + all_enactments_min_depth
		histogram[d] = 0

	for enactment_name in params['enactments']:
		e = Enactment(enactment_name, wh=dimensions[enactment_name], user=params['User'], verbose=params['verbose'])

		h = e.compute_depth_histogram()								#  Get a depth histogram for the current enactment.
		histg = []
		for i in range(0, 256):
			d = (float(i) / 255.0) * (e.max_depth - e.min_depth) + e.min_depth
			histg.append(h[d])
			if d not in histogram:
				histogram[d] = 0
			histogram[d] += h[d]
																	#  Save individual histograms
		plt.bar([(float(i) / 255.0) * (e.max_depth - e.min_depth) + e.min_depth for i in range(0, 256)], height=histg)
		plt.xlabel('Depth')
		plt.ylabel('Frequency')
		plt.title(enactment_name + ' Depth Histogram')
		plt.savefig(enactment_name + '-histogram.png')
		plt.clf()

	histg = []
	for i in range(0, 256):
		d = (float(i) / 255.0) * (all_enactments_max_depth - all_enactments_min_depth) + all_enactments_min_depth
		histg.append(histogram[d])
																	#  Save panoramic histogram
	plt.bar([(float(i) / 255.0) * (all_enactments_max_depth - all_enactments_min_depth) + all_enactments_min_depth for i in range(0, 256)], height=histg)
	plt.xlabel('Depth')
	plt.ylabel('Frequency')
	plt.title('Combined Depth Histogram')
	plt.savefig('total-histogram.png')
	plt.clf()

	#################################################################
	#  Final pass on the given enactments: apply network detection. #
	#################################################################
	if params['verbose']:
		print('')
		print('>>> Loading object-detection model "' + params['model'] + '".')
		print('')

	t1_start = time.process_time()									#  Start timer.
																	#  Load saved model and build detection function.
	detect_function = tf.saved_model.load(params['model'] + '/saved_model')
	t1_stop = time.process_time()									#  Stop timer.
	timings['load-model'] = t1_stop - t1_start
																	#  Retrieve the label lookup.
	label_path = '/'.join(params['model'].split('/')[:-2] + ['annotations', 'label_map.pbtxt'])
	recognizable_objects = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)

	model_name = params['model'].split('/')[-1]						#  Reference for this model.

	if params['verbose']:
		print('>>> Performing object detections on the following enactments:')
		for enactment_name in params['enactments']:
			print('\t' + enactment_name)
		print('')

	for enactment_name in params['enactments']:
		e = Enactment(enactment_name, wh=dimensions[enactment_name], user=params['User'], verbose=params['verbose'])
		e.recognizable_objects = [x[1]['name'] for x in sorted(recognizable_objects.items())]
		e.object_detection_source = params['model']

		maskfilectr = 0

		directory_name = enactment_name + '/' + model_name
		if os.path.isdir(directory_name):							#  If 'directory_name' exists, delete it.
			shutil.rmtree(directory_name)
		os.mkdir(directory_name)									#  Create directory_name.

		if params['verbose']:
			print('>>> Performing detections, frame by frame.')

		frame_ctr = 0
		num_frames = e.num_frames()
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.
		prev_ctr = 0
		for time_stamp, frame in sorted(e.frames.items()):
			t1_start = time.process_time()							#  Start timer.
			img = cv2.imread(frame.fullpath('video'), cv2.IMREAD_COLOR)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			imgt = tf.convert_to_tensor(img)
			input_tensor = imgt[tf.newaxis, ...]
			t1_stop = time.process_time()							#  Stop timer.
			timings['image-open'].append(t1_stop - t1_start)

			t1_start = time.process_time()							#  Start timer.
			detections = detect_function(input_tensor)				#  DETECT!
			t1_stop = time.process_time()							#  Stop timer.
			timings['detect'].append(t1_stop - t1_start)

			num_detections = int(detections.pop('num_detections'))
			detections = {key: val[0, :num_detections].numpy() for key, val in detections.items()}
			detections['num_detections'] = num_detections
			detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

			for i in range(0, detections['num_detections']):		#  SSD MOBILE-NET:
																	#    detection_classes, detection_multiclass_scores, detection_anchor_indices,
				detection_class = recognizable_objects[ detections['detection_classes'][i] ]['name']
				detection_box   = detections['detection_boxes'][i]	#    detection_boxes, raw_detection_boxes,
																	#    detection_scores, raw_detection_scores,
				detection_score = float(detections['detection_scores'][i])
																	#    num_detections
				bbox = ( int(round(detection_box[1] * e.width)), int(round(detection_box[0] * e.height)), \
				         int(round(detection_box[3] * e.width)), int(round(detection_box[2] * e.height)) )
				bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

				if detection_score >= params['score-threshold'] and bbox_area >= params['minpx']:
					mask = np.zeros((e.height, e.width), dtype=np.uint8)
																	#  For MOBILE-NET, mask is simply a bounding box.
					mask[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1] = np.uint(255)
																	#  Build the mask file path.
					maskpath = enactment_name + '/' + model_name + '/mask_' + str(maskfilectr) + '.png'
					cv2.imwrite(maskpath, mask)						#  Save the mask.
																	#  Add a new recognizable object.
					frame.detections.append( RObject(parent_frame=frame.fullpath('video'), \
					                                 object_name=detection_class, \
					                                 detection_source=model_name, \
					                                 mask_path=maskpath, \
					                                 bounding_box=bbox, \
					                                 confidence=detection_score) )

					maskfilectr += 1								#  Increment counter.

			if params['verbose']:
				if int(round(float(frame_ctr) / float(num_frames - 1) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(frame_ctr) / float(num_frames - 1) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(frame_ctr) / float(num_frames - 1) * 100.0))) + '%]')
					sys.stdout.flush()

			frame_ctr += 1

		if params['verbose']:
			print('')

		e.render_detected(model_name, params)						#  Save detected objects to file.

		if params['render']:
			if params['colors'] is None:							#  No colors? Make random ones.
				e.random_colors()
			else:													#  Use given colors.
				e.load_color_lookup(params['colors'])
			e.render_annotated_video()

	fh.write('Avg. image opening and color converstion time\t' + str(np.mean(timings['image-open'])) + '\n')
	fh.write('Std.dev image opening and conversion time\t' + str(np.std(timings['image-open'])) + '\n\n')

	fh.write('Avg. object detection time (per frame, using ' + model_name + ')\t' + str(np.mean(timings['detect'])) + '\n')
	fh.write('Std.dev object detection time (per frame, using' + model_name + ')\t' + str(np.std(timings['detect'])) + '\n\n')

	fh.close()

	return

def get_command_line_params():
	params = {}
	params['enactments'] = []										#  List of file paths.
	params['model'] = None											#  Detection model.
	params['score-threshold'] = 0.6									#  Detection score must be greater than this in order to register.
	params['minpx'] = 1												#  Minimum number of pixels for something to be considered visible.
	params['colors'] = None											#  Optional color lookup table.
	params['render'] = False										#  Whether to render stuff.
	params['verbose'] = False
	params['helpme'] = False
	params['fontsize'] = 1											#  For rendering text to images and videos
	params['User'] = 'vr1'											#  It used to be "admin", and I don't want to change a bunch of file paths when it changes again

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-e', '-model', '-th', '-minpx', '-render', '-color', '-colors', \
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
				elif argtarget == '-th':
					params['score-threshold'] = max(0.0, float(argval))
				elif argtarget == '-minpx':
					params['minpx'] = max(1, int(argval))
				elif argtarget == '-color' or argtarget == '-colors':
					params['colors'] = argval
				elif argtarget == '-User':
					params['User'] = argval
				elif argtarget == '-fontsize':
					params['fontsize'] = max(1.0, float(argval))

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('You\'ve received new or revised enactments.')
	print('You want to check that they conform to specifications, equip yourself to spot any errors,')
	print('and finally to apply a trained object-detection model to determine what is visible.')
	print('')
	print('Usage:  python3 detect_enactment.py <parameters, preceded by flags>')
	print(' e.g.:  python3 detect_enactment.py -model training/exported-models/ssd_mobilenet_640x640 -th 0.0 -e BackBreaker1 -e Enactment1 -v')
	print('')
	print('Flags:  -e       MUST HAVE AT LEAST ONE: Path to a directory of raw enactment materials: JSONs and color maps.')
	print('        -model   MUST HAVE EXACTLY ONE: Path to a model that can perform object recognition.')
	print('')
	print('        -th      Following real number in [0.0, 1.0] is the threshold, above which detections must score in order to count.')
	print('        -minpx   Following integer in [1, inf) is the minimum bounding box area in pixels for a detection to count.')
	print('')
	print('        -render  Generate illustrations (videos, 3D representations) for all given enactments.')
	print('        -v       Enable verbosity')
	print('        -?       Display this message')

if __name__ == '__main__':
	main()
