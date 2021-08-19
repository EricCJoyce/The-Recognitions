import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
import sys

def main():
	if len(sys.argv) > 1:
		flag = sys.argv[1]
	else:
		flag = 'winners'

	if flag == 'all':
		fh = open('conf-pred-gt-all.txt', 'r')
	else:
		fh = open('conf-pred-gt.txt', 'r')
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
	if flag == 'all':
		plt.savefig('isoreg-ALL.png')
	else:
		plt.savefig('isoreg.png')

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

	if flag == 'all':
		fh = open('isoreg-ALL.txt', 'w')
	else:
		fh = open('isoreg.txt', 'w')
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

if __name__ == '__main__':
	main()