import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
import time

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

	fh = open('recognizable_objects.txt', 'w')						#  Log the objects the network will learn.
	fh.write('#  Dataset created at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
	fh.write('#  Made from the following enactments:\n')
	fh.write('#  ' + '\t'.join(params['enactments']) + '\n')
	fh.write('#  Network will learn to recognize the following objects:\n')
	for object_name in classes:
		fh.write(object_name + '\n')
	fh.close()

	build_dataset(classes, params)

	return

#  Iterate over all specified enactments and identify acceptable training samples.
def build_dataset(classes, params):
	object_lookup_images = {}										#  key: learnable-object-name ==> val: [list of images containing this learnable object]
	image_lookup_objects = {}										#  key: image-file-name ==> val: [ (class-name, enactment, dimensions,
																	#                                   maskpath, bboxstr, pixels-occupied), ... ]
	learnable_instances = {}										#  key: class-name ==> val: the number of instances we wish to admit
	counts = {}														#  key: class-name ==> val: number of instances actually collected
	counts_train = {}
	counts_valid = {}

	include_conditions = {}											#  key: enactment name ==> val: [ (timestamp, timestamp), (timestamp, timestamp), ... ]
	exclude_conditions = {}											#  key: enactment name ==> val: [ (timestamp, timestamp), (timestamp, timestamp), ... ]
	if params['conditions-file'] is not None:
		fh = open(params['conditions-file'], 'r')
		for line in fh.readlines():
			if line[0] != '#':
				arr = line.strip().split('\t')
				enactment = arr[0]
				command = arr[1]

				times = []
				i = 2
				while i < len(arr):
					time_a = float(arr[i])
					time_b = float(arr[i + 1])
					times.append( (time_a, time_b) )
					i += 2

				if command == 'INCLUDE':
					include_conditions[enactment] = times[:]
				elif command == 'EXCLUDE':
					exclude_conditions[enactment] = times[:]
		fh.close()

	for enactment in params['enactments']:							#  Survey enactments directories.
		if params['verbose']:
			if enactment in include_conditions:
				print('>>> Scanning ' + enactment + ' for all instances of learnable objects within given intervals...')
			elif enactment in exclude_conditions:
				print('>>> Scanning ' + enactment + ' for all instances of learnable objects, excluding given intervals...')
			else:
				print('>>> Scanning ' + enactment + ' for all instances of learnable objects...')

		fh = open(enactment + '_props.txt', 'r')
		lines = fh.readlines()
		for line in lines:
			if line[0] != '#':
				arr = line.strip().split('\t')
				timestamp = float(arr[0])							#  timestamp
				imgfilename = arr[1]								#  image-filename
																	#  instance
				classpresent = arr[3]								#  class
																	#  detection-source
																	#  confidence
				bbox_str = arr[6]									#  bounding-box
				maskpath = arr[7]									#  mask-filename
																	#  3D-centroid-Avg
																	#  3D-centroid-BBox
				p = False
				if enactment in include_conditions:					#  Scan inclusion intervals. If timestamp fits, accept.
					i = 0
					while i < len(include_conditions[enactment]):
						if timestamp >= include_conditions[enactment][i][0] and timestamp < include_conditions[enactment][i][1]:
							p = True
							break
						i += 1
				elif enactment in exclude_conditions:				#  Scan exclusion intervals. If timestamp does NOT fit, accept.
					i = 0
					while i < len(exclude_conditions[enactment]):
						if timestamp >= exclude_conditions[enactment][i][0] and timestamp < exclude_conditions[enactment][i][1]:
							break
						i += 1
					if i == len(exclude_conditions[enactment]):
						p = True
				else:												#  No conditions apply: accept.
					p = True

				if p:
					mask = cv2.imread(maskpath, cv2.IMREAD_UNCHANGED)
					dimensions = str(mask.shape[1]) + ',' + str(mask.shape[0])
					indices = np.where(mask == 255)
					pixels_occupied = len(indices[0])

					if imgfilename not in image_lookup_objects:
						image_lookup_objects[imgfilename] = []

					if classpresent not in params['ignore']:		#  As long as we're not outright ignoring this class, list it as a (collateral) candidate.
						image_lookup_objects[imgfilename].append( (classpresent, enactment, dimensions, maskpath, bbox_str, pixels_occupied) )

					if classpresent in classes:						#  Here is an instance of one of the things we want to learn.
						if pixels_occupied >= params['minpx']:		#  Is it large enough to be helpful?
							if classpresent not in object_lookup_images:
								object_lookup_images[classpresent] = []
							object_lookup_images[classpresent].append( imgfilename )

							if classpresent not in learnable_instances:
								learnable_instances[classpresent] = 0
							learnable_instances[classpresent] += 1

		fh.close()

	if params['balance']:											#  Balance the sets: reset 'learnable_instances'.
		least = float('inf')
		for k, v in object_lookup_images.items():					#  Find the least-represented class.
			if len(v) < least:
				least = len(v)
		for k in learnable_instances.keys():						#  Clamp all classes to that length.
			learnable_instances[k] = least

	if params['clamp'] < float('inf'):								#  Clamp: reset 'learnable_instances'.
		for k in learnable_instances.keys():						#  Maximum is [:inf].
			learnable_instances[k] = max(params['clamp'], 2)		#  Minimum is [:2].

	used = {}														#  key: image-file-name ==> val: True
	train = []														#  List of tuples: (class-name, enactment, dimensions, maskpath, bboxstr)
	valid = []														#  List of tuples: (class-name, enactment, dimensions, maskpath, bboxstr)

	for classname in classes:
		if classname not in counts:									#  Initialize actual counts.
			counts[classname] = 0
		if classname not in counts_train:
			counts_train[classname] = 0
		if classname not in counts_valid:
			counts_valid[classname] = 0
																	#  Start from the least populous class because it is the most fragile.
	for classname, samples in sorted(object_lookup_images.items(), key=lambda x: len(x[1])):
		filtered_samples = list(dict.fromkeys(samples))				#  Filter out duplicates.
																	#  Filter out anything that has already been allocated to one of the sets.
		filtered_samples = [x for x in filtered_samples if x not in used]

		if len(filtered_samples) > 0:
																	#  Ensure we have at least one!
			train_set_size = min( int(round(float(len(filtered_samples)) * params['train'])), learnable_instances[classname] - 1 )

			np.random.shuffle(filtered_samples)						#  Shuffle IN PLACE!

			allocate_for_train = filtered_samples[:train_set_size]	#  Image file names.
			allocate_for_valid = filtered_samples[train_set_size:]

			for image in allocate_for_train:
				if params['include-collateral']:
					for learnable_instance in image_lookup_objects[image]:
																	#  Omit pixels occupied.
						train.append( tuple([image] + list(learnable_instance[:5])) )
						counts[ learnable_instance[0] ] += 1
						counts_train[ learnable_instance[0] ] += 1
				else:
					for learnable_instance in [x for x in image_lookup_objects[image] if x[5] >= params['minpx']]:
																	#  Omit pixels occupied.
						train.append( tuple([image] + list(learnable_instance[:5])) )
						counts[ learnable_instance[0] ] += 1
						counts_train[ learnable_instance[0] ] += 1

				used[image] = True									#  Mark the image as used.

			for image in allocate_for_valid:
				if params['include-collateral']:
					for learnable_instance in image_lookup_objects[image]:
																	#  Omit pixels occupied.
						valid.append( tuple([image] + list(learnable_instance[:5])) )
						counts[ learnable_instance[0] ] += 1
						counts_valid[ learnable_instance[0] ] += 1
				else:
					for learnable_instance in [x for x in image_lookup_objects[image] if x[5] >= params['minpx']]:
																	#  Omit pixels occupied.
						valid.append( tuple([image] + list(learnable_instance[:5])) )
						counts[ learnable_instance[0] ] += 1
						counts_valid[ learnable_instance[0] ] += 1

				used[image] = True									#  Mark the image as used.

	if params['verbose']:
		for classname in classes:
			if counts[ classname ] > 0:
				print('      ' + str(counts[ classname ]) + ' samples for ' + classname + \
				          ':\ttrain: ' + str(counts_train[ classname ]) + \
				          ',\tvalidate: ' + str(counts_valid[ classname ]))
			else:
				print('  WARNING!   No samples for ' + classname)

	dataset_timestamp = time.strftime('%l:%M%p %Z on %b %d, %Y')	#  Should be the same for training and validation set files.

	fh = open('training-set.txt', 'w')								#  Write training set to file
	fh.write('#  Training set for Object Recognition, created ' + dataset_timestamp + '\n')
	fh.write('#  ' + ' '.join(sorted(classes)) + '\n')
	fh.write('#  ' + ' '.join(sys.argv) + '\n')
	fh.write('#  Image-file    Dimensions    Enactment    Learnable-object    B-Box    Mask-path\n')
	for sample in train:
		imagefilename = sample[0]
		classpresent  = sample[1]
		enactment     = sample[2]
		dimensions    = sample[3]
		maskpath      = sample[4]
		bbox_str      = sample[5]
		fh.write(imagefilename + '\t' + dimensions + '\t' + enactment + '\t' + classpresent + '\t' + bbox_str + '\t' + maskpath + '\n')
	fh.close()

	fh = open('validation-set.txt', 'w')							#  Write validation set to file
	fh.write('#  Validation set for Object Recognition, created ' + dataset_timestamp + '\n')
	fh.write('#  ' + ' '.join(sorted(classes)) + '\n')
	fh.write('#  ' + ' '.join(sys.argv) + '\n')
	fh.write('#  Image-file    Dimensions    Enactment    Learnable-object    B-Box    Mask-path\n')
	for sample in valid:
		imagefilename = sample[0]
		classpresent  = sample[1]
		enactment     = sample[2]
		dimensions    = sample[3]
		maskpath      = sample[4]
		bbox_str      = sample[5]
		fh.write(imagefilename + '\t' + dimensions + '\t' + enactment + '\t' + classpresent + '\t' + bbox_str + '\t' + maskpath + '\n')
	fh.close()

	return

def get_command_line_params():
	params = {}

	params['enactments'] = []
	params['train'] = 0.8											#  Default
	params['valid'] = 0.2											#  Default
	params['balance'] = False
	params['clamp'] = float('inf')
	params['include-collateral'] = False							#  Default
	params['conditions-file'] = None								#  By default, use no conditions file.

	params['minpx'] = 100											#  Default

	params['ignore'] = ['LeftHand', 'RightHand']
	params['colors'] = None

	params['imgw'] = 1280											#  It used to be 1920; let's avoid changing a bunch of ints when it changes again.
	params['imgh'] = 720											#  It used to be 1080; let's avoid changing a bunch of ints when it changes again.
	params['User'] = 'vr1'											#  It used to be "admin"; let's avoid changing a bunch of file paths when it changes again.

	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-e', '-train', '-valid', '-b', '-c', '-minpx', '-coll', '-cond', '-x', '-colors', \
	         '-imgw', '-imgh', '-User', \
	         '-v', '-?', '-help', '--help']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-v':
				params['verbose'] = True
			elif sys.argv[i] == '-b':
				params['balance'] = True
			elif sys.argv[i] == '-coll':
				params['include-collateral'] = True
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
				elif argtarget == '-cond':
					params['conditions-file'] = argval
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

	if params['train'] <= 0.0 or params['valid'] <= 0.0 or params['train'] + params['valid'] != 1.0:
		print('WARNING: Invalid training/validation allocation; resorting to defaults.')
		params['train'] = 0.8
		params['valid'] = 0.2

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('Build a dataset from FactualVR enactments in preparation for training an object-detection network.')
	print('NOTE: several objects typically appear in a single frame. This script attempts to balance training')
	print('      and validation sets according to the given parameters, but must allocate frames entirely to')
	print('      one set or the other. We do not want to incur false false positives!')
	print('')
	print('Usage:  python3 build_object_recog_dataset.py <parameters, preceded by flags>')
	print(' e.g.   ')
	print('        python3 build_object_recog_dataset.py -e BackBreaker1 -e Enactment1 -e Enactment2 -e Enactment3 -e Enactment4 -e Enactment5 -e Enactment6 -e Enactment7 -e Enactment9 -e Enactment10 -e MainFeederBox1 -e Regulator1 -e Regulator2 -v -minpx 400')
	print(' e.g.   Drop intermediate-state objects, balance sets:')
	print('        python3 build_object_recog_dataset.py -e BackBreaker1 -e Enactment1 -e Enactment2 -e Enactment3 -e Enactment4 -e Enactment5 -e Enactment6 -e Enactment7 -e Enactment9 -e Enactment10 -e MainFeederBox1 -e Regulator1 -e Regulator2 -x AuxiliaryFeederBox_Unknown -x BackBreaker_Unknown -x Disconnect_Unknown -x MainFeederBox_Unknown -x Regulator_Unknown -x SafetyPlank_Unknown -x TransferFeederBox_Unknown -v -minpx 200 -b')
	print(' e.g.   Run in background:')
	print('        nohup python3 build_object_recog_dataset.py -e BackBreaker1 -e Enactment1 -e Enactment2 &')
	print('        nohup python3 build_object_recog_dataset.py -e BackBreaker1 -e Enactment1 -e Enactment2 >/dev/null 2>&1 &')
	print('')
	print('Flags:  -e      Following argument is an enactment file to be used for training and/or validation data.')
	print('                If no enactments are listed, then every enactment that has a *_props.txt file will be used.')
	print('        -x      Specify a class-label to be excluded from learning. "LeftHand" and "RightHand" are omitted by default.')
	print('')
	print('        -b      Balance the classes in the dataset. This means limiting all samples to the least represented class.')
	print('        -c      Clamp the maximum number of samples per class to this integer >= 1. (Default is infinity = no clamp.)')
	print('        -coll   Include collateral learnable objects. Default is False; passing this flag makes it True.')
	print('                The minimum number of pixels determines what is a "learnable object." An image useful for training an')
	print('                object of the requisite size might also include an object of inadequate size. If this parameter is set')
	print('                to True, then such collateral instances will be included in the dataset despite their being too small.')
	print('        -cond   Following argument is the path to a "conditions file" used to impose conditions on the use of')
	print('                given enactments. By default, no such file is expected.')
	print('')
	print('        -minpx  All object instances admitted to the data set must occupy at least this many pixels.')
	print('                Default is 100, and pixels are counted in the actual object mask--not the bounding box area.')
	print('        -train  In (0.0, 1.0). Portion of valid data to be allocated for training. Default is 0.8.')
	print('        -valid  In (0.0, 1.0). Portion of valid data to be allocated for validation. Default is 0.2.')
	print('')
	print('        -v      Enable verbosity')
	print('        -?      Display this message')
	return

if __name__ == '__main__':
	main()
