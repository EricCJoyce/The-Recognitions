import numpy as np
import os
import sys
import struct
import subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'							#  Suppress TensorFlow barf.

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model

def main():
	params = get_command_line_params()								#  Collect parameters.
	if params['helpme'] or len(params['source-epoch']) == 0:
		usage()
		return

	for filename in sorted([x for x in os.listdir('.') if x.endswith('.pb')], key=lambda x: int(x.split('.')[1])):
		filename_arr = filename.split('.')
		dirname = filename_arr[1]
		dirnum = dirname.lstrip('0')								#  Strip leading zeroes

		if int(dirnum) in params['source-epoch'] and not os.path.exists(dirnum):
			if params['verbose']:
				print('>>> Extracting weights for ' + filename)

			os.mkdir(dirnum)

			conv2dctr = 0
			densectr = 0
			model = load_model(filename)

			for i in range(0, len(model.layers)):

				if isinstance(model.layers[i], keras.layers.Conv2D):
					weights = conv2d_to_weights(model.layers[i])	#  Each in 'weights' is a filter

					for weightArr in weights:
						fh = open(dirnum + '/Conv2D-' + str(conv2dctr) + '.weights', 'wb')
						packstr = '<' + 'd'*len(weightArr)
						fh.write(struct.pack(packstr, *weightArr))
						fh.close()

						conv2dctr += 1

				elif isinstance(model.layers[i], keras.layers.Dense):
					weights = dense_to_weights(model.layers[i])
					layername = model.layers[i].name

					fh = open(dirnum + '/Dense-' + str(densectr) + '.weights', 'wb')

					packstr = '<' + 'd'*len(weights)
					fh.write(struct.pack(packstr, *weights))
					fh.close()

					densectr += 1

			args = ['./build_mlp_model', dirnum]
			comp_proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			out = comp_proc.stdout.decode('utf-8')
			err = comp_proc.stderr.decode('utf-8')

			if params['verbose']:
				print('>>> Built Neuron-C model "mlp-' + dirnum + '.nn"')

			fh = open('dataset.txt', 'r')

			if params['verbose']:
				print('>>> Performing consistency checks')

			ctr = 1
			lines = [line for line in fh.readlines() if line[0] != '#']
			for line in lines:
				arr = line.strip().split('\t')
				if not check_pb_nn_outputs(model, 'mlp-' + dirnum + '.nn', arr[4:-2], params):
					print('ERROR: *.nn and *.pb outputs differ by more than ' + str(params['epsilon']))
					fh.close()
					return
				if params['verbose']:
					print('    Test ' + str(ctr) + '/' + str(len(lines)) + ' passed')
				ctr += 1
			fh.close()

	return

def check_pb_nn_outputs(model, nn_file, x_str_vec, params):
	passed = True

	args = ['./run_mlp', nn_file] + x_str_vec
	comp_proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out = comp_proc.stdout.decode('utf-8')
	err = comp_proc.stderr.decode('utf-8')
	y_hat_nn = [float(x) for x in out.split()]

	x_vec = np.array([[float(x) for x in x_str_vec]])				#  [[I] [only] [worry]], [[that] [you] [don't] [have] [enough] [brackets]].
	y_hat_pb = model.predict( x_vec )
	y_hat_pb = y_hat_pb[0]

	for i in range(0, len(y_hat_nn)):
		passed = passed and abs(y_hat_pb[i] - y_hat_nn[i]) < params['epsilon']

	return passed

#  Write layer weights to file in ROW-MAJOR ORDER so our C program can read them into the model
def dense_to_weights(layer):
	ret = []

	w = layer.get_weights()
	width = len(w[1])												#  Number of units
	height = len(w[0])												#  Number of inputs (excl. bias)

	for hctr in range(0, height):									#  This is the row-major read
		for wctr in range(0, width):
			ret.append(w[0][hctr][wctr])

	for wctr in range(0, width):
		ret.append(w[1][wctr])

	return ret

#  Return a list of lists of weights.
#  Each can be written to a buffer and passed as weights to a C Conv2DLayer.
def conv2d_to_weights(layer):
	ret = []

	w = layer.get_weights()
	filterW = len(w[0][0])
	filterH = len(w[0])
	numFilters = len(w[1])

	for fctr in range(0, numFilters):
		ret.append( [] )
		for hctr in range(0, filterH):
			for wctr in range(0, filterW):
				ret[-1].append( w[0][hctr][wctr][0][fctr])
		ret[-1].append(w[1][fctr])

	return ret

def get_command_line_params():
	params = {}
	params['source-epoch'] = []

	params['epsilon'] = 0.00001
	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-src', '-v', '-?', '-help', '--help']

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
				if argtarget == '-src':
					params['source-epoch'].append( int(argval) )

	return params

def usage():
	return

if __name__ == '__main__':
	main()
