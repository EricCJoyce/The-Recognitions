import numpy as np
import sys

def main():
	params = getCommandLineParams()
	if params['helpme'] or len(params['enactments']) == 0:
		usage()
		return

	if params['verbose']:
		print('SOURCES:')
		for filename in params['enactments']:
			print('\t' + filename)
		print('')

	#################################################################  Load, merge, clean.

	object_names = survey_raw_actions(params)

	seqs, vector_len = join_sequences(params)
	params['vector_len'] = vector_len								#  Pack the vector-length into params

	actions, seqs = clean_sequences(seqs, params)

	X = []
	y = []

	for action, seqdata in actions.items():							#  key is an action label;
		indices = seqdata['seq_indices'][:]							#  val is a dictionary: seq_indices, frame_ctr

		for i in range(0, len(indices)):							#  For every training-set sequence, seqs[indices[i]]...
			if params['window'] < float('inf'):						#  Use window and stride
				if params['window'] < float('inf'):					#  Finite stride
					for fr_head_index in range(0, len(seqs[indices[i]]) - params['window'], params['stride']):
						seq = []
						for fr_ctr in range(0, params['window']):
							seq.append( seqs[indices[i]][fr_head_index + fr_ctr]['vec'] )
						X.append( seq )
						y.append(action)
				else:												#  Infinite stride: only read the window once
					seq = []
					for fr_ctr in range(0, min(len(seqs[indices[i]]), params['window'])):
						seq.append( seqs[indices[i]][fr_ctr]['vec'] )
					X.append( seq )
					y.append(action)
			else:													#  Use the whole sequence
				seq = []
				for frame in seqs[indices[i]]:
					seq.append( frame['vec'] )						#  Build the sequence
				X.append( seq )										#  Append the sequence
				y.append(action)

	print('\nThe given enactments and settings will yield a database containing the following ' + str(len(actions)) + ' actions:')
	maxstrlen = 0
	for action in actions.keys():
		if len(action) > maxstrlen:
			maxstrlen = len(action)

	if params['window'] == float('inf') and params['stride'] == float('inf'):
		for action, seqdata in sorted(actions.items()):
			print('\t' + action + ' '*(maxstrlen - len(action)) + '    ' + str(len(seqdata['seq_indices'])) + ' sequences')
	else:
		for action, seqdata in sorted(actions.items()):
			print('\t' + action + ' '*(maxstrlen - len(action)) + '    ' + str(len([x for x in y if x == action])) + ' ' + str(params['window']) + '-frame snippets')

	params['least_Gaussian_value'] = 0.0							#  Add these to the parameters dictonary; why not?
	params['greatest_Gaussian_value'] = 0.0

	print('\n')
	for action, seqdata in sorted(actions.items()):					#  key:action label ==> val:{ key:'seq_indices' ==> list of indices into 'seqs',
																	#                             key:'frame_ctr'   ==> number of total frames for this label }
		action_profile = {}											#  key:index into object_names ==> count of non-zero vector values
		frame_denom = 0
		for seq_index in seqdata['seq_indices']:
			for frame in seqs[seq_index]:
				for i in range(0, len(frame['vec'][12:])):
					if frame['vec'][12:][i] < params['least_Gaussian_value']:
						params['least_Gaussian_value'] = frame['vec'][12:][i]

					if frame['vec'][12:][i] > params['greatest_Gaussian_value']:
						params['greatest_Gaussian_value'] = frame['vec'][12:][i]

					if frame['vec'][12:][i] > 0.0:
						if i not in action_profile:
							action_profile[i] = 0
						action_profile[i] += 1

				frame_denom += 1

		for k, v in action_profile.items():							#  k:index into object_names ==> v:count of non-zero vector values
			if v == frame_denom:
				print(object_names[k] + ' is non-zero in every frame of ' + action)

	return

#  Perform drops and filter out any sequences that are too short
def clean_sequences(seqs, params):
	clean_actions = {}												#  Reset. This still counts AND tracks:
																	#  key: string ==> val: {seq_indices:[int, int, int], frame_ctr:int}

	if params['verbose']:											#  Tell user which labels will be dropped
		print('*\tSequences with fewer than ' + str(params['minlen']) + ' frames will be dropped')
		print('*\t<NEURTAL> will be dropped')
		print('')

	clean_seqs = []													#  List of Lists of Dicts(key: timestamp ==> val: {filename, (re)label, vector})
																	#  [ [{}, {}, {}], [{}, {}], [{}, {}, ..., {}], ..., [{}, {}, {}] ]
																	#    sequence      sequence  sequence       ^         sequence
																	#                                           |
																	#                                        frame
	for i in range(0, len(seqs)):
		if len(seqs[i]) >= params['minlen']:						#  If this is longer than or equal to the minimum, keep it.
			action = seqs[i][0]['label']
			if action != '*':
				if action not in clean_actions:
					clean_actions[action] = {}
					clean_actions[action]['seq_indices'] = []
					clean_actions[action]['frame_ctr'] = 0
				clean_actions[action]['seq_indices'].append( len(clean_seqs) )
				clean_seqs.append( [] )								#  Add a sequence
				for frame in seqs[i]:								#  Add all frames
					clean_seqs[-1].append( {} )
					clean_seqs[-1][-1]['timestamp'] = frame['timestamp']
					clean_seqs[-1][-1]['file'] = frame['file']
					clean_seqs[-1][-1]['frame'] = frame['frame']
					clean_seqs[-1][-1]['label'] = frame['label']
					clean_seqs[-1][-1]['vec'] = frame['vec'][:]

				clean_actions[action]['frame_ctr'] += len(seqs[i])

	seqs = clean_seqs												#  Copy back
	del clean_seqs													#  Destroy the temp

	actions = dict(clean_actions.items())							#  Rebuild dictionary

	if params['verbose']:
		print('FILTERED ACTIONS:')
		for k, v in sorted(actions.items()):
			if k == '*':
				print('\t<NEURTAL>: ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
			else:
				print('\t' + k + ': ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
		print('')

	return actions, seqs

#  Read all sequences. Perform any joins/relabelings.
def join_sequences(params):
	reverse_joins = {}
	for k, v in params['joins'].items():
		for vv in v:
			if vv not in reverse_joins:
				reverse_joins[vv] = k
	if params['verbose']:											#  Tell user which labels will be joined/merged
		for k, v in params['joins'].items():
			for vv in v:
				print('$\t"' + vv + '" will be re-labeled as "' + k + '"')
		print('')

	seqs = []														#  List of Lists of Dicts(key: timestamp ==> val: {filename, (re)label, vector})
																	#  [ [{}, {}, {}], [{}, {}], [{}, {}, ..., {}], ..., [{}, {}, {}] ]
																	#    sequence      sequence  sequence       ^         sequence
																	#                                           |
																	#                                        frame
	actions = {}													#  This will count AND track:
																	#  key: string ==> val: {seq_indices:[int, int, int], frame_ctr:int}
	vector_len = 0													#  I'd also like to know how long a vector is

	for filename in params['enactments']:
		fh = open(filename + '.enactment', 'r')
		lines = fh.readlines()
		fh.close()
		currentLabel = None
		for line in lines:
			if line[0] != '#':
				arr = line.strip().split('\t')
				timestamp = float(arr[0])
				srcfile = arr[1]									#  Frame file name (old data set does not have these)
				label = arr[2]
				vec = [float(x) for x in arr[3:]]
				if vector_len == 0:									#  Save vector length
					vector_len = len(vec)

				if label in reverse_joins:
					label = reverse_joins[label]

				if label not in actions:
					actions[label] = {}
					actions[label]['seq_indices'] = []
					actions[label]['frame_ctr'] = 0

				if label != currentLabel:
					currentLabel = label
					actions[currentLabel]['seq_indices'].append( len(seqs) )
					seqs.append( [] )

				actions[currentLabel]['frame_ctr'] += 1

				seqs[-1].append( {} )
				seqs[-1][-1]['timestamp'] = timestamp
				seqs[-1][-1]['file'] = filename
				seqs[-1][-1]['frame'] = srcfile
				seqs[-1][-1]['label'] = currentLabel
				seqs[-1][-1]['vec'] = vec[:]

	if params['verbose']:
		print('MERGED ACTIONS:')
		for k, v in sorted(actions.items()):
			if k == '*':
				print('\t<NEURTAL>: ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
			else:
				print('\t' + k + ': ' + str(v['frame_ctr']) + ' frames total, ' + str(len(v['seq_indices'])) + ' sequences: [' + ' '.join([str(x) for x in v['seq_indices']]) + ']')
		print('')

	return seqs, vector_len

#  Survey the set of all actions in the given enactments.
def survey_raw_actions(params):
	actions = {}													#  key: string ==> val: {seq_ctr:int, frame_ctr:int}
	object_names = []
	reading_classes = False

	for filename in params['enactments']:
		fh = open(filename + '.enactment', 'r')
		lines = fh.readlines()
		fh.close()
		currentLabel = None
		for line in lines:
			if line[0] != '#':
				arr = line.strip().split('\t')
				label = arr[2]

				if label not in actions:
					actions[label] = {}
					actions[label]['seq_ctr'] = 0
					actions[label]['frame_ctr'] = 0

				if label != currentLabel:
					currentLabel = label
					actions[currentLabel]['seq_ctr'] += 1

				actions[currentLabel]['frame_ctr'] += 1
			elif 'CLASSES:' in line and not reading_classes:
				reading_classes = True
			elif reading_classes:
				object_names = line[5:].strip().split()
				reading_classes = False

	if params['verbose']:
		print('RAW DEFINED ACTIONS:')
		for k, v in sorted(actions.items()):
			if k == '*':
				print('\t<NEURTAL>: ' + str(v['frame_ctr']) + ' total frames, ' + str(v['seq_ctr']) + ' sequences')
			else:
				print('\t' + k + ': ' + str(v['frame_ctr']) + ' total frames, ' + str(v['seq_ctr']) + ' sequences')
		print('')

		print('OBJECTS SUB-VECTOR:')
		for name in object_names:
			print('\t' + name)

	return object_names

def getCommandLineParams():
	params = {}
	params['enactments'] = []										#  List of filenames
	params['joins'] = {}											#  key: new label; val: [old labels]
	params['minlen'] = 2											#  Smallest length of a sequence deemed acceptable for the dataset

	params['window'] = float('inf')									#  By default, "infinite" windows cover entire sequences
	params['stride'] = float('inf')									#  By default, advance the window to infinity (past the end of the sequence)

	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-e', '-joinfile', '-minlen', \
	         '-window', '-stride', \
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

				elif argtarget == '-joinfile':
					fh = open(argval, 'r')
					lines = fh.readlines()
					fh.close()
					for line in lines:
						if line[0] != '#':
							arr = line.strip().split('\t')
							newlabel = arr[0]
							oldlabel = arr[1]
							if newlabel not in params['joins']:
								params['joins'][newlabel] = []
							if oldlabel not in params['joins'][newlabel]:
								params['joins'][newlabel].append(oldlabel)

				elif argtarget == '-minlen':
					params['minlen'] = int(argval)

				elif argtarget == '-window':
					if argval == 'inf':
						params['window'] = float('inf')
					else:
						params['window'] = int(argval)
				elif argtarget == '-stride':
					if argval == 'inf':
						params['stride'] = float('inf')
					else:
						params['stride'] = int(argval)

	if params['window'] <= 0 or params['stride'] < 1:
		print('WARNING: Invalid values received for window size and/or stride.')
		print('         Resorting to defaults.')
		params['window'] = float('inf')
		params['stride'] = float('inf')

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('Scan the given Factual enactments and report which object signals are reliably non-zero during which actions.')
	print('')
	print('Usage:  python3.5 survey_enactments.py <parameters, preceded by flags>')
	print('')
	print('e.g.:  Use Ground-Truth. Divide everything.')
	print('       python3.5 survey_enactments.py -e BackBreaker1 -e Enactment1 -e Enactment2 -e Enactment3 -e Enactment4 -e Enactment5 -e Enactment6 -e Enactment7 -e Enactment9 -e Enactment10 -e MainFeederBox1 -e Regulator1 -e Regulator2 -v -joinfile joins_05jul21.txt -window 10 -stride 2')
	print('')
	print('Flags:  -e         Following argument is an enactment file from which actions will be added to the database.')
	print('        -joinfile  Following argument is the path to a relabeling guide file.')
	print('        -minlen    Following argument is the minimum acceptable length for a sequence. Default is 2.')
	print('')
	print('        -window    Following argument is the size of a (hypothetical) snippet window. Default is inf = use full sequences.')
	print('        -stride    Following argument is the stride of the (hypothetical) snippet window. Default is inf = use the window once.')
	print('')
	print('        -v         Enable verbosity')
	print('        -?         Display this message')
	return

if __name__ == '__main__':
	main()