from enactment import *
import numpy as np
import os
import sys

def main():
	params = get_command_line_params()
	if params['helpme'] or len(params['enactments']) == 0:
		usage()
		return

	if params['verbose']:
		print('>>> Enactment sources:')
		for filename in params['enactments']:
			print('    ' + filename)
		if params['relabel-file'] is not None:
			print('>>> Relabeling from "' + params['relabel-file'] + '"')
		print('')
																	#  Show which enactments contain which actions.
	labels = {}														#  key: label ==> val: [filename, filename, ...]

	maxenactmentlen = 0												#  For the print-out, find the greatest enactment-name length.
	for filename in params['enactments']:
		if len(filename) > maxenactmentlen:
			maxenactmentlen = len(filename)

		if os.path.exists(filename + '.enactment'):
			e = Enactment(filename, enactment_file=filename + '.enactment', verbose=params['verbose'])
		else:
			e = Enactment(filename, verbose=params['verbose'])

		if params['relabel-file'] is not None:
			e.relabel_from_file(params['relabel-file'])

		for label in e.labels():									#  Fill in dictionary.
			if label != '*' and label not in labels:
				labels[label] = []

		for i in range(0, e.num_actions()):							#  Add instances to dictionary.
			action_frames = e.action_frameset(i)
			if len(action_frames) >= params['minimum-duration']:
				labels[ e.actions[i].label ].append( filename )

	maxlabellen = 0													#  For the print-out, find the greatest label length.
	for k, v in sorted(labels.items()):
		if len(k) > maxlabellen:
			maxlabellen = len(k)

	for k, v in sorted(labels.items()):
		print(k + ' '*(maxlabellen - len(k) + 2))					#  Print the label.
		present_in_enactments = np.unique(v)						#  Find all unique enactments containing this label.
		for enactment_name in sorted(present_in_enactments):
			print(' '*(len(k)) + ' '*(maxlabellen - len(k) + 2) + enactment_name + ':' + ' '*(maxenactmentlen - len(enactment_name) + 2) + str(v.count(enactment_name)))
		print(' '*(len(k)) + ' '*(maxlabellen - len(k) + 2) + 'TOTAL:' + ' '*(maxenactmentlen - 3) + str(len(v)) + ' instances')

	return

def get_command_line_params():
	params = {}
	params['enactments'] = []										#  List of filenames
	params['relabel-file'] = None
	params['minimum-duration'] = 1									#  By default, no minimum duration on actions.
	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-e', '-relabel', '-min', \
	         '-v', '-?', '-help', '--help']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-v':
				params['verbose'] = True
			elif sys.argv[i] == '-?' or sys.argv[i] == '-help' or sys.argv[i] == '--help':
				params['helpme'] = True
			else:
				argtarget = sys.argv[i]
		else:
			argval = sys.argv[i]

			if argtarget is not None:
				if argtarget == '-e':
					params['enactments'].append(argval)
				elif argtarget == '-relabel':
					params['relabel-file'] = argval
				elif argtarget == '-min':
					params['minimum-duration'] = max(1, int(argval))

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('Scan the given enactments and report all unique action labels.')
	print('')
	print('Usage:  python3 survey_labels.py <parameters, preceded by flags>')
	print('')
	print('e.g.:  python3 survey_labels.py -e BackBreaker1 -e Enactment1 -e Enactment2 -e Enactment3 -e Enactment4 -e Enactment5 -e Enactment6 -e Enactment9 -e Enactment10 -e Enactment11 -e Enactment12 -e MainFeederBox1 -e Regulator1 -e Regulator2 -relabel relabels.txt -v')
	print('')
	print('Flags:  -e        Following argument is an enactment file to include in the survey.')
	print('                  You must have at least one.')
	print('')
	print('        -relabel  Following argument is the path to a relabeling file, allowing you to rename actions at runtime.')
	print('        -min      Following int > 0 is the minimum allowable duration for an action.')
	print('                  By default, actions of all lengths are counted.')
	print('')
	print('        -v        Enable verbosity')
	print('        -?        Display this message')
	return

if __name__ == '__main__':
	main()
