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
	data = {}														#  key:class-name ==> val:[ (enactment, imgfile, maskpath), ... ]
	train = {}
	valid = {}

	for enactment in params['enactments']:							#  Survey enactments directories
		if params['verbose']:
			print('>>> Scanning ' + enactment + ' for all instances of learnable objects...')

		fh = open(enactment + '_props.txt', 'r')
		lines = fh.readlines()
		for line in lines:
			if line[0] != '#':										#  timestamp
				arr = line.strip().split('\t')						#  image-filename
				imgfilename = arr[1]								#  instance
				classpresent = arr[3]								#  class
																	#  detection-source
																	#  confidence
				bbox_str = arr[6]									#  bounding-box
				maskpath = arr[7]									#  mask-filename
																	#  3D-centroid-Avg
																	#  3D-centroid-BBox

				if classpresent in classes:							#  Here is an instance of one of the things we want to learn.
					mask = cv2.imread(maskpath, cv2.IMREAD_UNCHANGED)
					indices = np.where(mask == 255)
					if len(indices[0]) >= params['minpx']:			#  Is it large enough to be helpful?
						if classpresent not in data:
							data[classpresent] = []
						data[classpresent].append( (enactment, imgfilename, maskpath, bbox_str) )
		fh.close()

	for k, v in data.items():										#  Shuffle everything
		np.random.shuffle(data[k])									#  Remember: this shuffles IN PLACE!

	if params['balance']:											#  Balance the sets:
		least = float('inf')
		for k, v in data.items():									#  Find the least-represented class.
			if len(v) < least:
				least = len(v)
		for k, v in data.items():									#  Clamp all classes to that length.
			data[k] = v[:least]

	if params['clamp'] < float('inf'):								#  Maximum is [:inf]
		for k, v in data.items():									#  Clamp
			data[k] = v[:params['clamp'] + 1]						#  Minimum is [:2]

	for k, v in data.items():										#  Partition the sets
		m = int(round(float(len(v)) * params['train']))
		train[k] = data[k][:m]
		valid[k] = data[k][m:]

	fh = open('training-set.txt', 'w')								#  Write training set to file
	fh.write('#  ' + ' '.join(sorted(data.keys())) + '\n')
	fh.write('#  Learnable-object    Enactment    Image-file    Mask-path    B-Box\n')
	for k, v in train.items():
		for vv in v:
			fh.write(k + '\t' + vv[0] + '\t' + vv[1] + '\t' + vv[2] + '\t' + vv[3] + '\n')
	fh.close()

	fh = open('validation-set.txt', 'w')							#  Write validation set to file
	fh.write('#  ' + ' '.join(sorted(data.keys())) + '\n')
	fh.write('#  Learnable-object    Enactment    Image-file    Mask-path    B-Box\n')
	for k, v in valid.items():
		for vv in v:
			fh.write(k + '\t' + vv[0] + '\t' + vv[1] + '\t' + vv[2] + '\t' + vv[3] + '\n')
	fh.close()

	if params['verbose']:
		print('    ' + str(sum([len(x) for x in train.values()]) + sum([len(x) for x in valid.values()])) + ' trainable samples, total')
		for classname in classes:
			if classname not in train:
				print('  WARNING!   No samples for ' + classname)
			else:
				print('      ' + str(len(train[classname])) + ' samples for ' + classname)

	return

def get_command_line_params():
	params = {}

	params['enactments'] = []
	params['train'] = 0.8											#  Default
	params['valid'] = 0.2											#  Default
	params['balance'] = False
	params['clamp'] = float('inf')

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
	flags = ['-e', '-train', '-valid', '-b', '-c', '-minpx', '-x', '-colors', \
	         '-imgw', '-imgh', '-User', \
	         '-v', '-?', '-help', '--help']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-v':
				params['verbose'] = True
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
	print('')
	print('Usage:  python3 build_object_recog_dataset.py <parameters, preceded by flags>')
	print(' e.g.   Drop intermediate-state objects:')
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
	print('')
	print('        -minpx  All object instances admitted to the data set must occupy at least this many pixels. Default is 100.')
	print('')
	print('        -train  In (0.0, 1.0). Portion of valid data to be allocated for training. Default is 0.8.')
	print('        -valid  In (0.0, 1.0). Portion of valid data to be allocated for validation. Default is 0.2.')
	print('')
	print('        -v      Enable verbosity')
	print('        -?      Display this message')
	print('')
	print('Going to throw away some previous training efforts? Find them in ~/maskrcnn/logs/')
	return

if __name__ == '__main__':
	main()
