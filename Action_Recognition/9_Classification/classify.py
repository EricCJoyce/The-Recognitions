from classifier import *
import shutil

def main():
	params = get_command_line_params()								#  Collect parameters.
	if params['helpme'] or len(params['enactments']) == 0 or params['database'] is None:
		usage()
		return

	if params['load-from'] is not None:
		for enactment in params['enactments']:
			enactment_file_name = enactment + '.enactment'
			enactment_src_dir = params['load-from']
			enactment_dst_dir = './'
			if params['verbose']:
				print('>>> Copying ' + enactment_src_dir + '/' + enactment_file_name + ' to ' + enactment_dst_dir + enactment_file_name)
			shutil.copy(enactment_src_dir + '/' + enactment_file_name, \
			            enactment_dst_dir + enactment_file_name)

	temporal = TemporalClassifier(rolling_buffer_length=params['rolling-buffer-length'], \
	                              rolling_buffer_stride=params['rolling-buffer-stride'], \
	                              temporal_buffer_length=params['temporal-buffer-length'], \
	                              temporal_buffer_stride=params['temporal-buffer-stride'], \
	                              db_file=params['database'], \
	                              relabel=params['relabel-file'], \
	                              conf_func=params['conf-function'], \
	                              threshold=params['threshold'], \
	                              hand_schema=params['hand-schema'], \
	                              props_coeff=params['props-coeff'], \
	                              hands_one_hot_coeff=params['one-hot-coeff'], \
	                              inputs=params['enactments'], \
	                              min_bbox=params['minimum-pixels'], \
	                              detection_confidence=params['detection-threshold'], \
	                              isotonic_file=params['map-conf-prob'], \
	                              verbose=params['verbose'])

	for label in params['hidden-labels']:							#  Hide all hidden labels.
		temporal.hide_label(label)

	stats = temporal.classify(params['detection-model'], params['skip-unfair'])

	if params['verbose']:
		M = temporal.confusion_matrix(stats['_tests'])
		print('Confusion Matrix Trace: ' + str(M.trace()))
		print('Confusion Matrix Sum:   ' + str(M.sum()))
		print('Accuracy:               ' + str(M.trace() / M.sum()))

	temporal.write_confusion_matrix(stats['_tests'], params['result-string'], 'train')
	temporal.write_results(stats, params['result-string'])
	temporal.write_confidences(stats['_conf'], params['result-string'])
	temporal.write_probabilities(stats['_prob'], params['result-string'])

	return

def get_command_line_params():
	params = {}
	params['enactments'] = []										#  Names of enactments on which to perform action recognition.
	params['database'] = None										#  The database file to use.
	params['detection-model'] = None								#  Default to ground-truth.
	params['conf-function'] = 'sum2'								#  The default confidence function.
	params['map-conf-prob'] = None									#  Default to using confidence scores.
	params['threshold'] = 0.0										#  Default to 0.0 threshold.
	params['detection-threshold'] = 0.0								#  Default to 0.0 threshold.
	params['minimum-pixels'] = 1									#  Default to 1 pixel threshold.
	params['relabel-file'] = None									#  No relabeling by default.

	params['rolling-buffer-length'] = 10
	params['rolling-buffer-stride'] = 2
	params['temporal-buffer-length'] = 3
	params['temporal-buffer-stride'] = 1
	params['hand-schema'] = 'strong-hand'
	params['hand-coeff'] = 1.0
	params['one-hot-coeff'] = 6.0
	params['props-coeff'] = 9.0
	params['hidden-labels'] = []

	params['result-string'] = None									#  Default to the timestamp.
	params['load-from'] = None										#  Directory from which enactment files should be loaded into the working directory at runtime.
	params['skip-unfair'] = False

	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-e', '-db', '-model', \
	         '-conf', '-th', '-map', '-detth', '-minpx', '-relabel', '-hide', \
	         '-lddir', '-id', '-fair', \
	         '-v', '-?', '-help', '--help']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-fair':
				params['skip-unfair'] = True
			elif sys.argv[i] == '-v':
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
				elif argtarget == '-db':
					params['database'] = argval
				elif argtarget == '-model':
					params['detection-model'] = argval
				elif argtarget == '-th':
					params['threshold'] = min(1.0, max(0.0, float(argval)))
				elif argtarget == '-detth':
					params['detection-threshold'] = min(1.0, max(0.0, float(argval)))
				elif argtarget == '-conf':
					params['conf-function'] = argval
				elif argtarget == '-minpx':
					params['minimum-pixels'] = max(1, int(argval))
				elif argtarget == '-map':
					params['map-conf-prob'] = argval
				elif argtarget == '-relabel':
					params['relabel-file'] = argval
				elif argtarget == '-hide':
					params['hidden-labels'].append(argval)
				elif argtarget == '-lddir':
					params['load-from'] = argval
				elif argtarget == '-id':
					params['result-string'] = argval

	return params

#  Explain usage of this script and its options to the user.
def usage():
	c = Classifier()
	print('Run the classifier on the given enactments.')
	print('This script produces several files outlining the system\'s performance.')
	print('')
	print('Usage:  python3 classify.py <parameters, preceded by flags>')
	print(' e.g.:  python3 classify.py -e Enactment11 -e Enactment12 -model training/exported-models/ssd_mobilenet_640x640 -db 10f-split2-stride2-MobileNet0.2-train.db -map MobileNet-th0.2.isomap -relabel relabels.txt -detth 0.2 -lddir MobileNet-th0.2 -id MobileNet-th0.2 -v')
	print(' e.g.:  python3 classify.py -e Enactment11 -e Enactment12 -db 10f-split2-stride2-GT-train.db -map GT.isomap -relabel relabels.txt -lddir GT -id GT -hide Read\\ \\(C.\\ Panel\\) -v')
	print('')
	print('Flags:  -e        Following argument is the name of an enactment on which to perform classification.')
	print('                  Must have at least one.')
	print('        -db       Following argument is the path to a database file.')
	print('                  REQUIRED.')
	print('        -model    Following argument is the path to a trained object detector.')
	print('                  If this argument is not provided, then ground-truth objects will be used.')
	print('        -map      Following argument is the path to a file describing probability bins to which confidence scores are mapped.')
	print('                  If this argument is not provided, then confidence scores will be used in place of probabilities.')
	print('                  This will function but is not advised.')
	print('        -conf     Following string in {' + ', '.join(c.confidence_function_names) + '} indicates which confidence function to use.')
	print('                  Default is "sum2". Be carefule when changing this; the confidence function should match that which was')
	print('                  used to compute the probability bins during isotonic regression.')
	print('        -th       Following real number in [0.0, 1.0] is the threshold to use when classifying actions.')
	print('                  The default is 0.0.')
	print('        -relabel  Following argument is the path to a relabeling file, allowing you to rename actions at runtime.')
	print('        -detth    Following real number in [0.0, 1.0] is the detection score threshold to use when recognizing objects.')
	print('                  The default is 0.0.')
	print('        -minpx    Following integer > 0 is the pixel area minimum to use when recognizing objects.')
	print('                  The default is 1.')
	print('        -hide     Following string (escaped where necessary) is a label the system should "hide".')
	print('                  Hidden labels are in the database, and can be recognized; but when they are chosen as the system prediction,.')
	print('                  hidden labels are changed to no-votes.')
	print('        -v        Enable verbosity.')
	print('        -?        Display this message.')
	return

if __name__ == '__main__':
	main()
