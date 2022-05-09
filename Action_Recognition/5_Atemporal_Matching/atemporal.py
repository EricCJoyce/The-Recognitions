import sys

from enactment import *
from classifier import *

def main():
	params = get_command_line_params()								#  Collect parameters
	if params['helpme'] or (len(params['training']) == 0 and len(params['divide']) == 0) or \
	                       (len(params['test']) == 0 and len(params['divide']) == 0):
		usage()
		return

	atemporal = AtemporalClassifier(window_size=params['window'], stride=params['stride'], \
	                                train=params['training'], divide=params['divide'], test=params['test'], \
	                                conf_func=params['confidence-function'], threshold=params['threshold'], \
	                                dtw_diagonal=params['dtw-diagonal'], \
	                                isotonic_file=params['isotonic-file'], conditions_file=params['conditions-file'], \
	                                hand_schema=params['hand-schema'], hands_coeff=params['hands-coeff'], props_coeff=params['props-coeff'], \
	                                train_portion=params['train-portion'], test_portion=params['test-portion'], \
	                                minimum_length=params['minimum-length'], shuffle=params['shuffle'], \
	                                render=params['render'], verbose=params['verbose'])

	if params['color-file'] is not None:
		atemporal.load_color_map(params['color-file'])

	if params['relabel-file'] is not None:
		atemporal.relabel_from_file(params['relabel-file'])

	if params['split-file'] is not None:
		atemporal.load_data_split(params['split-file'])

	atemporal.commit()												#  Chop the allocated actions up into snippets.

	stats = atemporal.classify()									#  Do it.
	time_stamp = atemporal.time_stamp()								#  Save a time stamp to unify our outputs.

	M = atemporal.confusion_matrix(stats['_tests'])					#  Collect results into a confusion matrix.
	acc = float(M.trace()) / float(M.sum())							#  Toal accuracy.
	trace = M.trace()												#  Total correct.
	total = M.sum()													#  Total snippets.

	atemporal.write_confusion_matrix(stats['_tests'], time_stamp)	#  Write confidence files.
	atemporal.write_confidences(stats['_conf'], time_stamp)
	atemporal.write_results(stats, time_stamp)
	atemporal.write_timing(time_stamp)

	print('Accuracy: ' + str(acc))									#  Output results.
	print('Correct:  ' + str(trace))
	print('Total:    ' + str(total))

	return

def get_command_line_params():
	params = {}
	params['training'] = []											#  List of enactment names to allocate entirely to the training set.
	params['test'] = []												#  List of enactment names to allocate entirely to the test set.
	params['divide'] = []											#  List of enactment names to allocate to training and testing in the given proportion.

	params['train-portion'] = 0.8									#  Portion of divide set to allocate to training.
	params['test-portion'] = 0.2									#  Portion of divide set to allocate to testing.

	params['confidence-function'] = 'sum2'							#  Confidence function.
	params['threshold'] = 0.0										#  By default, no threshold.
	params['isotonic-file'] = None									#  Isotonic mapping file.
	params['conditions-file'] = None								#  Cutoff conditions file.

	params['split-file'] = None										#  Load a data-set allocation from file.
	params['relabel-file'] = None									#  Relabel actions according to this lookup table.

	params['window'] = 10											#  Length of the sliding window.
	params['stride'] = 2											#  Stride of the slide.

	params['hand-schema'] = 'left-right'							#  Hand subvector encoding.
	params['hands-coeff'] = 1.0										#  Coefficient for the hands subvector (excluding the one-hot components.)
	params['props-coeff'] = 1.0										#  Coefficient for the props subvector.

	params['minimum-length'] = 2
	params['shuffle'] = False

	params['dtw-diagonal'] = 2.0									#  The Classifier defaults to 2.0 anyway.

	params['color-file'] = None										#  Recognizable-object color look up table.
	params['render'] = False										#  Rendering, yes or no?

	params['verbose'] = False
	params['helpme'] = False

	params['fontsize'] = 1											#  For rendering text to images and videos
	params['imgw'] = 1280
	params['imgh'] = 720
	params['User'] = 'vr1'											#  It used to be "admin", and I don't want to change a bunch of file paths when it changes again

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-t', '-v', '-d', '-tPor', '-vPor', '-split', '-splits', '-window', '-stride', \
	         '-conf', '-th', '-iso', '-cond', '-minlen', '-shuffle', '-dtw', \
	         '-schema', '-hand', '-hands', '-prop', '-props', '-relabel', '-color', '-colors', \
	         '-render', '-V', '-?', '-help', '--help', \
	         '-User', '-imgw', '-imgh', '-fontsize']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-V':
				params['verbose'] = True
			elif sys.argv[i] == '-?' or sys.argv[i] == '-help' or sys.argv[i] == '--help':
				params['helpme'] = True
			elif sys.argv[i] == '-render':
				params['render'] = True
			elif sys.argv[i] == '-shuffle':
				params['shuffle'] = True
			else:
				argtarget = sys.argv[i]
		else:
			argval = sys.argv[i]

			if argtarget is not None:
				if argtarget == '-t':
					params['training'].append(argval)
				elif argtarget == '-v':
					params['test'].append(argval)
				elif argtarget == '-d':
					params['divide'].append(argval)

				elif argtarget == '-window':
					params['window'] = max(1, int(argval))
				elif argtarget == '-stride':
					params['stride'] = max(1, int(argval))
				elif argtarget == '-minlen':
					params['minimum-length'] = max(0, int(argval))

				elif argtarget == '-tPor':
					params['train-portion'] = max(0.0, min(1.0, float(argval)))
					params['test-portion'] = 1.0 - params['train-portion']
				elif argtarget == '-vPor':
					params['test-portion'] = max(0.0, min(1.0, float(argval)))
					params['train-portion'] = 1.0 - params['test-portion']
				elif argtarget == '-split' or argtarget == '-splits':
					params['split-file'] = argval
				elif argtarget == '-color' or argtarget == '-colors':
					params['color-file'] = argval
				elif argtarget == '-relabel':
					params['relabel-file'] = argval
				elif argtarget == '-iso':
					params['isotonic-file'] = argval
				elif argtarget == '-cond':
					params['conditions-file'] = argval

				elif argtarget == '-conf':
					params['confidence-function'] = argval
				elif argtarget == '-th':
					params['threshold'] = float(argval)
				elif argtarget == '-dtw':
					params['dtw-diagonal'] = float(argval)

				elif argtarget == '-schema':
					params['hand-schema'] = argval
				elif argtarget == '-hand' or argtarget == '-hands':
					params['hands-coeff'] = float(argval)
				elif argtarget == '-prop' or argtarget == '-props':
					params['props-coeff'] = float(argval)

				elif argtarget == '-imgw':
					params['imgw'] = max(1, int(argval))
				elif argtarget == '-imgh':
					params['imgh'] = max(1, int(argval))
				elif argtarget == '-User':
					params['User'] = argval
				elif argtarget == '-fontsize':
					params['fontsize'] = float(argval)

	if params['train-portion'] <= 0.0 or params['train-portion'] >= 1.0 or \
	   params['test-portion']  <= 0.0 or params['test-portion']  >= 1.0:
		print('>>> INVALID DATA received for training and test portions. Restoring default value.')
		params['train-portion'] = 0.8
		params['test-portion'] = 0.2

	if params['fontsize'] < 1:
		print('>>> INVALID DATA received for fontsize. Restoring default value.')
		params['fontsize'] = 1

	return params

#  Explain usage of this script and its options to the user.
def usage():
	c = Classifier()												#  Just get its 'confidence_function_names'.

	print('Perform atemporal classification using the given enactments in the specified arrangement between training and test sets.')
	print('This script writes results to files organized by time stamp.')
	print('')
	print('Usage:  python3 atemporal.py <parameters, preceded by flags>')
	print(' e.g.:  python3 atemporal.py -d BackBreaker1 -d Enactment1 -d Enactment2 -d Enactment3 -d Enactment4 -d Enactment5 -d Enactment6 -d Enactment9 -d Enactment10 -d MainFeederBox1 -d Regulator1 -d Regulator2 -split data-split-04oct21.txt -V')
	print(' e.g.:  python3 atemporal.py -t BackBreaker1 -t Enactment1 -t Enactment2 -t Enactment3 -t Enactment4 -t Enactment5 -t Enactment6 -t Enactment9 -t Enactment10 -t MainFeederBox1 -t Regulator1 -t Regulator2 -v Enactment11 -v Enactment12 -split data-split-1_09sep21.txt -colors colors_gt.txt -V -render')
	print('')
	print('Flags:  -t        Following argument is the filepath to an enactment to be allocated entirely to the training set.')
	print('        -v        Following argument is the filepath to an enactment to be allocated entirely to the test set.')
	print('        -d        Following argument is the filepath to an enactment to be allocated split across both training and test sets')
	print('                  according to training-set portion and test-set portion, which can be controlled using the flags below.')
	print('        -tPor     Following real number in (0.0, 1.0) determines the training-set portion of divided enactments. Default is 0.8.')
	print('        -vPor     Following real number in (0.0, 1.0) determines the test-set portion of divided enactments. Default is 0.2.')
	print('        -split    Following argument is the filepath to a data-split file.')
	print('')
	print('        -window   Following integer > 0 is the length of the sliding window used to chop actions into snippets. Default is 10.')
	print('        -stride   Following integer > 0 is the stride of the sliding window. Default is 2.')
	print('')
	print('        -hands    Following real number is the coefficient for the hands subvector. Default is 1.0.')
	print('        -props    Following real number is the coefficient for the props subvector. Default is 1.0.')
	print('')
	print('        -conf     Following string indicates which confidence function to use.')
	print('                  Must be in {' + ', '.join(c.confidence_function_names) + '}. Default is "sum2".')
	print('        -th       Following real number is the confidence/probability threshold. Default is 0.0.')
	print('        -dtw      Following real number is the weight to give to diagonal moves in the DTW cost matrix. Default is 2.0.')
	print('')
	print('        -relabel  Following argument is the filepath to a relabeling table.')
	print('        -iso      Following argument is the filepath to an isotonic mapping table.')
	print('        -cond     Following argument is the filepath to a cutoff conditions table.')
	print('')
	print('        -color    Used when rendering, the following argument is filepath to a color lookup table. If no color table is')
	print('                  provided, then random colors for all objects will be generated.')
	print('        -render   Enable rendering.')
	print('')
	print('        -V        Enable verbosity.')
	print('        -?        Display this message.')
	return

if __name__ == '__main__':
	main()
