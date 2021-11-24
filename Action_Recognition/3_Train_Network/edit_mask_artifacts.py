from enactment import *
import os
import shutil
import subprocess
import sys
import time

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

	edit_list = {}													#  Save edits to an edit list.
	editlist_filename = e.enactment_name + '.' + params['instance'] + '.editlist'

	if os.path.exists(editlist_filename):							#  Resuming previous work?
		fh = open(editlist_filename, 'r')
		for line in fh.readlines():
			if line[0] != '#':
				arr = line.strip().split('\t')
				edit_list[ arr[0] ] = arr[1]
		fh.close()

	detections = e.list_detections( (lambda detection: detection.instance_name == params['instance']) )
	for detection in detections:
		if detection.mask_path not in edit_list:					#  Do not redo previous work.
			args = ['./edit_artifacts', '-mask', detection.mask_path, '-img', detection.parent_frame]
			output = str(subprocess.getoutput(' '.join(args)))		#  Collect output.

			edit_list[detection.mask_path] = output					#  Save to table.
			update_edit_file(editlist_filename, edit_list, params)	#  Write to file.

			if params['verbose']:
				print(detection.mask_path + '\t' + output)
	return

def update_edit_file(editlist_filename, edit_list, params):
	if os.path.exists(editlist_filename):							#  Back up the existing copy.
		shutil.copy(editlist_filename, editlist_filename + '.backup')

	fh = open(editlist_filename, 'w')
	fh.write('#  Edit list for enactment "' + params['enactment'] + '", instance "' + params['instance'] + '"\n')
	fh.write('#  Created ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
																	#  Yes, I made this a float because in cases when I had
																	#  to manually split a mask, I made the daughter mask *.5.
																	#  e.g. Enactment1/GT/mask_1.png
																	#       Enactment1/GT/mask_1.5.png
	for k, v in sorted(edit_list.items(), key=lambda x: float('.'.join(x[0].split('_')[-1].split('.')[:-1]))):
		fh.write(k + '\t' + v + '\n')								#  Write to file.

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
