from classifier import *
from mlp import *
import shutil
import sys

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
	                              dtw_diagonal=params['dtw-diagonal'], \
	                              conditions=params['conditions-files'], \
	                              inputs=params['enactments'], \
	                              min_bbox=params['minimum-pixels'], \
	                              detection_confidence=params['detection-threshold'], \
	                              use_detection_source=params['detection-source'], \
	                              isotonic_file=params['map-conf-prob'], \
	                              verbose=params['verbose'])

	if params['config-file'] is not None:							#  Got a config file?
		temporal.load_config_file(params['config-file'])

	if params['proxy-file'] is not None:							#  Given a proxy file?
		temporal.load_proxies_from_file(params['proxy-file'])

	for label in params['hidden-labels']:							#  Hide all hidden labels.
		temporal.hide_label(label)

	'''
	temporal.render = True
	temporal.render_modes = ['smooth']
	temporal.load_color_map('colors.txt')
	stats = temporal.simulated_classify(False)
	'''
	stats = temporal.classify(params['detection-model'], params['skip-unfair'])

	if params['verbose']:
		M = temporal.confusion_matrix(stats['_tests'])
		print('Confusion Matrix Trace: ' + str(M.trace()))
		print('Confusion Matrix Sum:   ' + str(M.sum()))
		print('Accuracy:               ' + str(M.trace() / M.sum()))

	temporal.write_confusion_matrix(stats['_tests'], params['result-string'], 'train')
	temporal.write_results(stats, params['result-string'])
	temporal.write_matching_costs(stats['_costs'], params['result-string'])
	temporal.write_confidences(stats['_conf'], params['result-string'])
	temporal.write_probabilities(stats['_prob'], params['result-string'])

	return

def get_command_line_params():
	params = {}
	params['enactments'] = []										#  Names of enactments on which to perform action recognition.
	params['database'] = None										#  The database file to use.
	params['detection-model'] = None								#  Default to ground-truth.
	params['config-file'] = None									#  No default config file.

	params['conf-function'] = 'sum2'								#  The default confidence function.
	params['map-conf-prob'] = None									#  Default to using confidence scores.
	params['threshold'] = 0.0										#  Default to 0.0 threshold.
	params['detection-threshold'] = 0.0								#  Default to 0.0 threshold.
	params['detection-source'] = None								#  Default to ground-truth detections.
	params['minimum-pixels'] = 1									#  Default to 1 pixel threshold.
	params['relabel-file'] = None									#  No relabeling by default.
	params['proxy-file'] = None

	params['rolling-buffer-length'] = 10
	params['rolling-buffer-stride'] = 2
	params['temporal-buffer-length'] = 3
	params['temporal-buffer-stride'] = 1
	params['hand-schema'] = 'strong-hand'
	params['hand-coeff'] = 1.0
	#params['one-hot-coeff'] = 6.0
	params['one-hot-coeff'] = 1.0
	#params['props-coeff'] = 9.0
	params['props-coeff'] = 1.0
	params['dtw-diagonal'] = 2.0									#  Classifier defaults to 2.0 anyway.
	params['dtw-L'] = 2												#  Classifier defaults to 2 anyway.
	params['hidden-labels'] = []
	params['conditions-files'] = None

	params['result-string'] = None									#  Default to the timestamp.
	params['load-from'] = None										#  Directory from which enactment files should be loaded into the working directory at runtime.
	params['skip-unfair'] = False

	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-e', '-db', '-model', '-config', \
	         '-conf', '-th', '-map', '-detth', '-detsrc', '-minpx', '-relabel', \
	         '-hands', '-handc', '-props', '-onehot', \
	         '-dtwd', '-dtwl', '-hide', '-proxy', '-cond', \
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
				elif argtarget == '-config':
					params['config-file'] = argval
				elif argtarget == '-th':
					params['threshold'] = min(1.0, max(0.0, float(argval)))
				elif argtarget == '-detth':
					params['detection-threshold'] = min(1.0, max(0.0, float(argval)))
				elif argtarget == '-detsrc':
					params['detection-source'] = argval
				elif argtarget == '-conf':
					params['conf-function'] = argval
				elif argtarget == '-minpx':
					params['minimum-pixels'] = max(1, int(argval))

				elif argtarget == '-map':
					params['map-conf-prob'] = argval

				elif argtarget == '-dtwd':
					params['dtw-diagonal'] = float(argval)
				elif argtarget == '-dtwl':
					params['dtw-L'] = int(argval)

				elif argtarget == '-relabel':
					params['relabel-file'] = argval

				elif argtarget == '-hands':
					params['hand-schema'] = argval

				elif argtarget == '-handc':
					params['hand-coeff'] = float(argval)
				elif argtarget == '-props':
					params['props-coeff'] = float(argval)
				elif argtarget == '-onehot':
					params['one-hot-coeff'] = float(argval)

				elif argtarget == '-hide':
					params['hidden-labels'].append(argval)

				elif argtarget == '-proxy':
					params['proxy-file'] = argval

				elif argtarget == '-cond':
					if params['conditions-files'] is None:
						params['conditions-files'] = []
					params['conditions-files'].append(argval)
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
	print(' e.g.:  python3 classify.py -e Enactment11 -e Enactment12 -db 10f-split2-stride2-GT-train.db -map GT-sum2.isomap -relabel relabels.txt -lddir GT -id GT -hide Read\\ \\(C.\\ Panel\\) -v')
	print(' e.g.:  python3 classify.py -e Enactment11 -e Enactment12 -db 10f-split2-stride2-GT-train.db -map GT-sum2.isomap -relabel relabels.txt -cond hands.conditions -lddir GT -id GT -v')
	print(' e.g.:  python3 classify.py -e Enactment11 -e Enactment12 -db 10f-split2-stride2-GT-train.db -config ConfidenceMLP.config -relabel relabels.txt -lddir GT/g0 -id GT -v')
	print(' e.g.:  python3 classify.py -e Enactment11 -e Enactment12 -db 10f-s2-g3-w-GT.db -config IsomapLookup.config -relabel relabels.txt -lddir GT/g3 -id GT -v')
	print(' e.g.:  python3 classify.py -e Enactment11 -e Enactment12 -db 10f-split2-stride2-GT-train.db -map GT-sum2.isomap -relabel relabels.txt -lddir GT -id GT -hide Read\\ \\(C.\\ Panel\\) -proxy NOTES/VI/proxy.txt -v')
	print('')
	print('Flags:  -e        Following argument is the name of an enactment on which to perform classification.')
	print('                  Must have at least one.')
	print('        -db       Following argument is the path to a database file.')
	print('                  REQUIRED.')
	print('        -model    Following argument is the path to a trained object detector.')
	print('                  If this argument is not provided, then ground-truth objects will be used.')
	print('        -config   Following argument is the path to a config file.')
	print('                  The default configuration is to convert confidence scores to probabilities without isotonic lookup or an MLP.')
	print('        -map      Following argument is the path to a file describing probability bins to which confidence scores are mapped.')
	print('                  If this argument is not provided, then confidence scores will be used in place of probabilities.')
	print('                  This will function but is not advised.')
	print('        -conf     Following string in {' + ', '.join(c.confidence_function_names) + '} indicates which confidence function to use.')
	print('                  Default is "sum2". Be carefule when changing this; the confidence function should match that which was')
	print('                  used to compute the probability bins during isotonic regression.')
	print('        -th       Following real number in [0.0, 1.0] is the threshold to use when classifying actions.')
	print('                  The default is 0.0.')
	print('        -hands    Following string in {' + ', '.join(c.hand_schema_names) + '} indicates the hand-encoding scheme the classifier should use.')
	print('                  The default is \'strong-hand\'.')
	print('        -handc    Following real number is the coefficient applied to the hands subvector.')
	print('                  The default is 1.0.')
	print('        -props    Following real number is the coefficient applied to the props subvector.')
	print('                  The default is 1.0.')
	print('        -onehot   Following real number is the coefficient applied to the hands one-hot subvector.')
	print('                  The default is 1.0.')
	print('        -relabel  Following argument is the path to a relabeling file, allowing you to rename actions at runtime.')
	print('        -detth    Following real number in [0.0, 1.0] is the detection score threshold to use when recognizing objects.')
	print('                  The default is 0.0.')
	print('        -detsrc   Following string is the detection source to use when reading object detections from file.')
	print('                  e.g. ssd_mobilenet_640x640-th0.2 will read from "<enactment>_ssd_mobilenet_640x640-th0.2_detections.txt".')
	print('                  If this argument is left unspecified, object detection defaults to ground truth, expecting to find "<enactment>_props.txt".')
	print('        -minpx    Following integer > 0 is the pixel area minimum to use when recognizing objects.')
	print('                  The default is 1.')
	print('        -dtwd     Following real number is the weight to give to diagonal moves in the DTW cost matrix. Default is 2.0.')
	print('        -dtwl     Following integer in [1, 2] indicates which distance computes the DTW cost matrix. Default is 2, meaning L2 distance.')
	print('        -cond     Following string is the path to a conditions file to give the classifier.')
	print('                  You may supply more than one, but each one must be preceded by this flag.')
	print('        -hide     Following string (escaped where necessary) is a label the system should "hide".')
	print('                  Hidden labels are in the database, and can be recognized; but when they are chosen as the system prediction,.')
	print('                  hidden labels are changed to no-votes.')
	print('        -proxy    Following argument is a path to a file containing tab-separated proxy and target labels.')
	print('        -v        Enable verbosity.')
	print('        -?        Display this message.')
	return

if __name__ == '__main__':
	main()
