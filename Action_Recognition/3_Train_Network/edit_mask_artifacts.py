from enactment import *
import os
import subprocess
import sys

def main():
	params = get_command_line_params()								#  Collect parameters
	if params['enactment'] is None or params['instance'] is None or params['helpme']:
		usage()
		return

	if params['wh'] is not None:
		e = Enactment(params['enactment'], wh=params['wh'], verbose=params['verbose'])
	else:
		e = Enactment(params['enactment'], verbose=params['verbose'])

	e.load_parsed_objects()

	fh = open(e.enactment_name + '.editlist', 'w')					#  Save edits to an edit list.

	detections = e.list_detections( (lambda detection: detection.instance_name == params['instance']) )
	for detection in detections:
		args = ['./edit_artifacts', '-mask', detection.mask_path, '-img', detection.parent_frame]
		output = str(subprocess.getoutput(' '.join(args)))			#  Collect output.
		fh.write(detection.mask_path + '\t' + output + '\n')
		if params['verbose']:
			print(detection.mask_path + '\t' + output)

	fh.close()

	return

def get_command_line_params():
	params = {}

	params['enactment'] = None
	params['instance'] = None

	params['wh'] = (1280, 720)
	params['verbose'] = False

	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-e', '-i', '-v', '-?', '-help', '--help']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-?' or sys.argv[i] == '-help' or sys.argv[i] == '--help':
				params['helpme'] = True
			elif sys.argv[i] == '-v':
				params['verbose'] = True
			else:
				argtarget = sys.argv[i]
		else:
			argval = sys.argv[i]

			if argtarget is not None:
				if argtarget == '-e':
					params['enactment'] = argval
				elif argtarget == '-i':
					params['instance'] = argval

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('Use an interactive sub-program to clean up the mask artifacts.')
	print('You must specify an enactment and an instance within that enactment.')
	print('This script will then iterate over all masks and frames containing this instance')
	print('and allow you to inspect/edit mask artifacts for that instance.')
	print('')
	print('Usage:  python3 edit_mask_artifacts.py <parameters, preceded by flags>')
	print(' e.g.   ')
	print('        python3 edit_mask_artifacts.py -e BackBreaker1 -i Target1_BackBreaker_4007')
	print('')
	print('Flags:  -e   REQUIRED: Following string is the name of an enactment.')
	print('        -i   REQUIRED: Following string is the name of an object instance.')
	print('')
	print('        -?   Display this message')
	return

if __name__ == '__main__':
	main()
