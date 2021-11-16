import os
from shutil import copyfile
import subprocess
import sys
import time

def main():
	params = get_command_line_params()								#  Collect parameters
	if params['helpme']:
		usage()
		return

	train = {}														#  key: image file ==> val: [ (Learnable-object, B-Box), ... ]
	valid = {}														#  key: image file ==> val: [ (Learnable-object, B-Box), ... ]

	train_dataset_mask_file = params['train'].split('.')[0] + '.dataset-mask'
	valid_dataset_mask_file = params['valid'].split('.')[0] + '.dataset-mask'
	train_dataset_mask = {}											#  key: image file ==> val: [ (Learnable-object, B-Box, bool), ... ]
	valid_dataset_mask = {}											#  key: image file ==> val: [ (Learnable-object, B-Box, bool), ... ]

	common_timestamp = time.strftime('%l:%M%p %Z on %b %d, %Y')		#  Used in two tracking files.

	#################################################################
	#  Check for or create the dataset masks.                       #
	#################################################################
	if not os.path.exists(train_dataset_mask_file):
		fh = open(train_dataset_mask_file, 'w')
		fh.write('#  Training set mask file created for "' + params['train'] + '" at ' + common_timestamp + '.\n')
		fh.close()
	else:
		fh = open(train_dataset_mask_file, 'r')
		for line in fh.readlines():
			if line[0] != '#':
				arr = line.strip().split('\t')
				imagefile = arr[0]
				learnable_object = arr[1]
				bbox_str = arr[2]
				active = arr[3]

				if imagefile not in train_dataset_mask:
					train_dataset_mask[imagefile] = []
				train_dataset_mask[imagefile].append( (learnable_object, bbox_str) )
		fh.close()

	if not os.path.exists(valid_dataset_mask_file):
		fh = open(valid_dataset_mask_file, 'w')
		fh.write('#  Validation set mask file created for "' + params['valid'] + '" at ' + common_timestamp + '.\n')
		fh.close()
	else:
		fh = open(valid_dataset_mask_file, 'r')
		for line in fh.readlines():
			if line[0] != '#':
				arr = line.strip().split('\t')
				imagefile = arr[0]
				learnable_object = arr[1]
				bbox_str = arr[2]
				active = arr[3]

				if imagefile not in valid_dataset_mask:
					valid_dataset_mask[imagefile] = []
				valid_dataset_mask[imagefile].append( (learnable_object, bbox_str) )
		fh.close()

	#################################################################
	#  Build the work set, excluding anything that's already been   #
	#  ruled on.                                                    #
	#################################################################
	fh = open(params['train'], 'r')									#  Read everything in the training set.
	for line in fh.readlines():
		if line[0] != '#':
			arr = line.strip().split('\t')
			imagefile = arr[0]
			dimensions = arr[1]
			enactment = arr[2]
			learnable_object = arr[3]
			bbox_str = arr[4]
			bbox_parse = bbox_str.split(';')
			bbox = [x for x in bbox_parse[0].split(',')] + [x for x in bbox_parse[1].split(',')]
			maskfile = arr[5]

			if imagefile not in train:
				train[imagefile] = []

			if (imagefile not in train_dataset_mask) or (imagefile in train_dataset_mask and (learnable_object, bbox_str) in train_dataset_mask[imagefile]):
				train[imagefile].append( (learnable_object, bbox) )
	fh.close()

	fh = open(params['valid'], 'r')									#  Read everything in the validation set.
	for line in fh.readlines():
		if line[0] != '#':
			arr = line.strip().split('\t')
			imagefile = arr[0]
			dimensions = arr[1]
			enactment = arr[2]
			learnable_object = arr[3]
			bbox_str = arr[4]
			bbox_parse = bbox_str.split(';')
			bbox = [x for x in bbox_parse[0].split(',')] + [x for x in bbox_parse[1].split(',')]
			maskfile = arr[5]

			if imagefile not in valid:
				valid[imagefile] = []

			if (imagefile not in valid_dataset_mask) or (imagefile in valid_dataset_mask and (learnable_object, bbox_str) in valid_dataset_mask[imagefile]):
				valid[imagefile].append( (learnable_object, bbox) )
	fh.close()

	for k, v in train.items():										#  Do the training set.
		args = ['./clicker', '-img', k]
		for vv in v:
			args.append(vv[0])										#  Append the learnable object.
			args += vv[1]											#  Append the bbox.
		output = str(subprocess.getoutput(' '.join(args)))			#  Collect output.
		output = list(output)										#  Listify output.

		addition = ''												#  Accumulate an addition to the dataset mask file.
		for i in range(0, len(output)):
			if output[i] == '0':
				addition += k + '\t' + v[i][0] + '\t' + v[i][1][0] + ',' + v[i][1][1] + ';' + v[i][1][2] + ',' + v[i][1][3] + '\t' + 'DROP' + '\n'
			elif output[i] == '1':
				addition += k + '\t' + v[i][0] + '\t' + v[i][1][0] + ',' + v[i][1][1] + ';' + v[i][1][2] + ',' + v[i][1][3] + '\t' + 'ACTIVE' + '\n'
																	#  Make a back up copy before updating.
		copyfile(params['train'].split('.')[0] + '.dataset-mask', params['train'].split('.')[0] + '.backup.dataset-mask')
		fh = open(params['train'].split('.')[0] + '.dataset-mask', 'r')
		lines = fh.readlines()
		fh.close()
																	#  Re-write and add the addition.
		fh = open(params['train'].split('.')[0] + '.dataset-mask', 'w')
		for line in lines:
			fh.write(line)
		fh.write(addition)
		fh.close()

	for k, v in valid.items():										#  Do the validation set.
		args = ['./clicker', '-img', k]
		for vv in v:
			args.append(vv[0])										#  Append the learnable object.
			args += vv[1]											#  Append the bbox.
		output = str(subprocess.getoutput(' '.join(args)))			#  Collect output.
		output = list(output)										#  Listify output.

		addition = ''												#  Accumulate an addition to the dataset mask file.
		for i in range(0, len(output)):
			if output[i] == '0':
				addition += k + '\t' + v[i][0] + '\t' + v[i][1][0] + ',' + v[i][1][1] + ';' + v[i][1][2] + ',' + v[i][1][3] + '\t' + 'DROP' + '\n'
			elif output[i] == '1':
				addition += k + '\t' + v[i][0] + '\t' + v[i][1][0] + ',' + v[i][1][1] + ';' + v[i][1][2] + ',' + v[i][1][3] + '\t' + 'ACTIVE' + '\n'
																	#  Make a back up copy before updating.
		copyfile(params['valid'].split('.')[0] + '.dataset-mask', params['valid'].split('.')[0] + '.backup.dataset-mask')
		fh = open(params['valid'].split('.')[0] + '.dataset-mask', 'r')
		lines = fh.readlines()
		fh.close()
																	#  Re-write and add the addition.
		fh = open(params['valid'].split('.')[0] + '.dataset-mask', 'w')
		for line in lines:
			fh.write(line)
		fh.write(addition)
		fh.close()

	return

def get_command_line_params():
	params = {}

	params['train'] = 'training-set.txt'							#  Default
	params['valid'] = 'validation-set.txt'							#  Default

	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-t', '-v', '-?', '-help', '--help']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-?' or sys.argv[i] == '-help' or sys.argv[i] == '--help':
				params['helpme'] = True
			else:
				argtarget = sys.argv[i]
		else:
			argval = sys.argv[i]

			if argtarget is not None:
				if argtarget == '-t':
					params['train'] = argval
				elif argtarget == '-v':
					params['valid'] = argval

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('Use an interactive sub-program to clean up the dataset.')
	print('This is likely to be long, tedious work, so we\'ll want to save our progress.')
	print('That\'s why this script creates and/or consults *.dataset-mask files for the given')
	print('training and validation files. (Meaning, if you use the defaults "training-set.txt"')
	print('and "validation-set.txt" you\'ll get "training-set.dataset-mask" and')
	print('"validation-set.dataset-mask" to track your progress.')
	print('(When you want to quit, click the terminal window from which you launched this script')
	print(' and force quit the script. The pop-up window will disappear.)')
	print('')
	print('Usage:  python3 refine_object_recog_dataset.py <parameters, preceded by flags>')
	print(' e.g.   ')
	print('        python3 refine_object_recog_dataset.py -t training-set.txt -v validation-set.txt')
	print('')
	print('Flags:  -t   Following string is the file path to the training set file created by build_object_recog_dataset.py.')
	print('             If this argument is omitted, then this script expects to find "training-set.txt" in the current directory.')
	print('        -v   Following string is the file path to the validation set file created by build_object_recog_dataset.py.')
	print('             If this argument is omitted, then this script expects to find "validation-set.txt" in the current directory.')
	print('')
	print('        -?   Display this message')
	return

if __name__ == '__main__':
	main()
