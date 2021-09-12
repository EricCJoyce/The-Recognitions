import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
import sys

def main():
	params = get_command_line_params()								#  Collect parameters
	if params['helpme'] or params['src'] is None:
		usage()
		return

	fh = open(params['src'], 'r')
	lines = fh.readlines()
	fh.close()

	X = []															#  Confidences
	y = []															#  1 if we correctly matched; 0 else

	for line in lines:
		if line[0] != '#':
			arr = line.strip().split('\t')
			conf = float(arr[0])
			pred = arr[1]
			gt = arr[2]
			X.append(conf)											#  Add confidence
			if pred == gt:											#  Add correctness:
				y.append(1)											#    right
			else:
				y.append(0)											#    wrong

	stem = params['src'].split('-')[-1].split('.')[0]				#  Save the stem

	X.reverse()														#  Make increasing
	y.reverse()														#  (This, too)

	iso_reg = IsotonicRegression(out_of_bounds='clip').fit(X, y)	#  Fit

	fake_X = np.linspace(0.0, X[-1], num=1000)
	fake_y = iso_reg.predict(fake_X)
	pred_y = iso_reg.predict(X)

	plt.plot(fake_X, fake_y, 'b')
	plt.plot(X, pred_y, 'bo')
	plt.xlabel('Confidence')
	plt.ylabel('Probability')
	plt.savefig('isoreg-' + stem + '.png')

	buckets = {}													#  key:isotonic probability ==> val:{key:'min' ==> val:confidence,
																	#                                    key:'max' ==> val:confidence}
	for i in range(0, len(X)):
		if pred_y[i] not in buckets:
			buckets[pred_y[i]] = {}
			buckets[pred_y[i]]['min'] = float('inf')
			buckets[pred_y[i]]['max'] = float('-inf')
	for i in range(0, len(X)):
		if i == 0:
			buckets[pred_y[i]]['min'] = float('-inf')
			buckets[pred_y[i]]['max'] = max(buckets[pred_y[i]]['max'], X[i])
		elif i == len(X) - 1:
			buckets[pred_y[i]]['min'] = min(buckets[pred_y[i]]['min'], X[i])
			buckets[pred_y[i]]['max'] = float('inf')
		else:
			buckets[pred_y[i]]['min'] = min(buckets[pred_y[i]]['min'], X[i])
			buckets[pred_y[i]]['max'] = max(buckets[pred_y[i]]['max'], X[i + 1])

	fh = open(stem + '.isoreg', 'w')
	fh.write('#  Step-Lower-Bound    Step-Upper-Bound    Probability\n')
	fh.write('#  First lower bound and last upper bound are "*", which stands for "unbounded."\n')
	for k, v in sorted(buckets.items()):
		if v['min'] == float('-inf'):
			fh.write('*\t')
		else:
			fh.write(str(v['min']) + '\t')

		if v['max'] == float('inf'):
			fh.write('*\t')
		else:
			fh.write(str(v['max']) + '\t')

		fh.write(str(k) + '\n')

	fh.close()

	return

def get_command_line_params():
	params = {}
	params['src'] = None											#  File path to a "confidences" file output by a Classifier.
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-src', '-?', '-help', '--help']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-?' or sys.argv[i] == '-help' or sys.argv[i] == '--help':
				params['helpme'] = True
			else:
				argtarget = sys.argv[i]
		else:
			argval = sys.argv[i]

			if argtarget is not None:
				if argtarget == '-src':
					params['src'] = argval

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('Perform isotonic regression on confidence scores listed in the given source file.')
	print('Plot the isotonic curve and write discretized "buckets" to file.')
	print('')
	print('Usage:  python3.5 isoreg.py <parameters, preceded by flags>')
	print(' e.g.:  python3.5 isoreg.py -src confidences-winners-110921T003434.txt')
	print('')
	print('Flags:  -src  Following argument is the filepath to a "confidences" file written by one of the Classifiers.')
	print('        -?    Display this message.')

if __name__ == '__main__':
	main()