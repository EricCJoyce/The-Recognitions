import subprocess
import sys

def main():
	params = get_command_line_params()								#  Collect parameters.
	if params['helpme'] or params['num-classes'] is None:
		usage()
		return

	build_model_code(params)
	build_run_code(params)
	build_makefile(params)

	if params['verbose']:
		print('>>> Making executables')
	args = ['make']
	output = subprocess.check_output(args)

	return

def build_model_code(params):
	dense_map = {}													#  key: layer index ==> val: number of units
	fh = open('Network-map.txt', 'r')
	for line in fh.readlines():
		if line[0] != '#' and len(line) > 1:
			arr = line.strip().split('\t')
			layer_index = int(arr[0])
			num_units = int(arr[1])
			dense_map[ layer_index ] = num_units
	fh.close()

	if params['verbose']:
		print('>>> Building "build_mlp_model.py"')

	fh_py = open('build_mlp_model.py', 'w')							#  Automatically generate explicit Python code to build the network.
	fh_py.write('import os\n')
	fh_py.write('os.environ[\'TF_CPP_MIN_LOG_LEVEL\'] = \'2\'							#  Suppress TensorFlow barf.\n')
	fh_py.write('import tensorflow as tf\n')
	fh_py.write('import tensorflow.keras as keras\n')
	fh_py.write('from tensorflow.keras import models\n')
	fh_py.write('from tensorflow.keras import optimizers\n')
	fh_py.write('from tensorflow.keras.layers import Input, Dense, Dropout\n')
	fh_py.write('from tensorflow.keras.callbacks import ModelCheckpoint\n')
	fh_py.write('\n')
	fh_py.write('def build_model(params):\n')
	fh_py.write('\tinput_layer = Input(shape=(' + str(params['num-classes']) + ', ))\n')

	if params['verbose']:
		print('>>> Building "build_mlp_model.c"')

	fh_c = open('build_mlp_model.c', 'w')							#  Automatically generate C code to build the Neuron-C network.
	fh_c.write('#include "neuron.h"\n')
	fh_c.write('\n')
	fh_c.write('unsigned int readWeights(char*, double**);\n')
	fh_c.write('\n')
	fh_c.write('int main(int argc, char* argv[])\n')
	fh_c.write('  {\n')
	fh_c.write('    NeuralNet* nn;\n')
	fh_c.write('    double* w = NULL;\n')
	fh_c.write('    char buffer[256];\n')
	fh_c.write('    unsigned char len;\n')
	fh_c.write('    unsigned char i;\n')
	fh_c.write('    unsigned int j;\n')
	fh_c.write('\n')
	fh_c.write('    init_NN(&nn, ' + str(params['num-classes']) + ');                                               //  Initialize for input ' + str(params['num-classes']) + '-vec\n')
	fh_c.write('\n')

	fh_c.write('    /******************************************************************************/\n')
	fh_c.write('    /***************************************************************    D E N S E */\n')

	#################################################################  Dense
	for ctr in range(0, len(dense_map)):
		if ctr == 0:
			fh_c.write('\n')
			fh_c.write('    add_Dense(' + str(params['num-classes']) + ', ' + str(dense_map[ctr]) + ', nn);                                        //  Add dense layer (DENSE_ARRAY, ' + str(ctr) + ')\n')
			fh_c.write('    setName_Dense("Dense-' + str(ctr) + '", nn->denselayers);                      //  Name the ' + str(ctr) + '-th dense layer\n')
			fh_c.write('    len = sprintf(buffer, "%s/Dense-' + str(ctr) + '.weights", argv[1]);\n')
			fh_c.write('    buffer[len] = \'\\0\';\n')
			fh_c.write('    readWeights(buffer, &w);\n')
			fh_c.write('    setW_Dense(w, nn->denselayers);\n')
			fh_c.write('    for(j = 0; j < ' + str(dense_map[ctr]) + '; j++)\n')
			fh_c.write('      setF_i_Dense(RELU, j, nn->denselayers);     //  Set activation function for weight[0][i]\n')
			fh_c.write('    free(w);\n')

			fh_py.write('\tdense' + str(dense_map[ctr]) + ' = Dense(' + str(dense_map[ctr]) + ', activation=\'relu\', name=\'dense' + str(dense_map[ctr]) + '\')(input_layer)\n')

		else:
			fh_c.write('\n')
			fh_c.write('    add_Dense(' + str(dense_map[ctr - 1]) + ', ' + str(dense_map[ctr]) + ', nn);                                        //  Add dense layer (DENSE_ARRAY, ' + str(ctr) + ')\n')
			fh_c.write('    setName_Dense("Dense-' + str(dense_map[ctr]) + '", nn->denselayers + ' + str(ctr) + ');                  //  Name the ' + str(ctr) + '-th dense layer\n')
			fh_c.write('    len = sprintf(buffer, "%s/Dense-' + str(ctr) + '.weights", argv[1]);\n')
			fh_c.write('    buffer[len] = \'\\0\';\n')
			fh_c.write('    readWeights(buffer, &w);\n')
			fh_c.write('    setW_Dense(w, nn->denselayers + ' + str(ctr) + ');\n')
			fh_c.write('    for(j = 0; j < ' + str(dense_map[ctr]) + '; j++)            //  Set activation function for weight[' + str(ctr) + '][i]\n')
			fh_c.write('      setF_i_Dense(RELU, j, nn->denselayers + ' + str(ctr) + ');\n')
			fh_c.write('    free(w);\n')

			fh_py.write('\tdense' + str(dense_map[ctr]) + ' = Dense(' + str(dense_map[ctr]) + ', activation=\'relu\', name=\'dense' + str(dense_map[ctr]) + '\')(dense' + str(dense_map[ctr - 1]) + ')\n')
																	#  The final output unit.
	fh_py.write('\tdense_final = Dense(' + str(params['num-classes'] + 1) + ', activation=\'softmax\', name=\'dense_final\')(dense'+str(dense_map[ctr]) + ')\n')

	fh_c.write('\n')
	fh_c.write('    add_Dense(' + str(dense_map[ctr]) + ', ' + str(params['num-classes'] + 1) + ', nn);                                        //  Add final dense layer (DENSE_ARRAY, ' + str(len(dense_map)) + ')\n')
	fh_c.write('    setName_Dense("Dense-Final", nn->denselayers + ' + str(len(dense_map)) + ');                  //  Name the last dense layer\n')
	fh_c.write('    for(i = 0; i < ' + str(params['num-classes'] + 1) + '; i++)\n')
	fh_c.write('      setF_i_Dense(SOFTMAX, i, nn->denselayers + ' + str(len(dense_map)) + ');     //  Set output layer\'s activation function to softmax\n')
	fh_c.write('    len = sprintf(buffer, "%s/Dense-' + str(len(dense_map)) + '.weights", argv[1]);\n')
	fh_c.write('    buffer[len] = \'\\0\';\n')
	fh_c.write('    readWeights(buffer, &w);\n')
	fh_c.write('    setW_Dense(w, nn->denselayers + ' + str(len(dense_map)) + ');\n')
	fh_c.write('    free(w);\n')
	fh_c.write('\n')
	fh_c.write('    /******************************************************************************/\n')
	fh_c.write('\n')
	fh_c.write('    if(!linkLayers(INPUT_ARRAY, 0, 0, ' + str(params['num-classes']) + ', DENSE_ARRAY, 0, nn))     //  Connect input to dense[0]\n')
	fh_c.write('      printf(">>>                Link[0] failed\\n");\n')
	fh_c.write('\n')
	for ctr in range(0, len(dense_map)):
		fh_c.write('    if(!linkLayers(DENSE_ARRAY, ' + str(ctr) + ', 0, ' + str(dense_map[ctr]) + ', DENSE_ARRAY, ' + str(ctr + 1) + ', nn))     //  Connect dense[' + str(ctr) + '] to dense[' + str(ctr + 1) + ']\n')
		fh_c.write('      printf(">>>                Link[' + str(ctr + 1) + '] failed\\n");\n')
	fh_c.write('\n')
	fh_c.write('    sortEdges(nn);\n')
	fh_c.write('    printEdgeList(nn);\n')
	fh_c.write('    printf("\\n\\n");\n')
	fh_c.write('    len = sprintf(buffer, "MLP for action classification");\n')
	fh_c.write('    i = 0;\n')
	fh_c.write('    while(i < len && i < COMMSTR_LEN)\n')
	fh_c.write('      {\n')
	fh_c.write('        nn->comment[i] = buffer[i];\n')
	fh_c.write('        i++;\n')
	fh_c.write('      }\n')
	fh_c.write('    while(i < COMMSTR_LEN)\n')
	fh_c.write('      {\n')
	fh_c.write('        nn->comment[i] = \'\\0\';\n')
	fh_c.write('        i++;\n')
	fh_c.write('      }\n')
	fh_c.write('    print_NN(nn);\n')
	fh_c.write('\n')
	fh_c.write('    len = sprintf(buffer, "mlp-%s.nn", argv[1]);\n')
	fh_c.write('    buffer[len] = \'\\0\';\n')
	fh_c.write('\n')
	fh_c.write('    write_NN(buffer, nn);\n')
	fh_c.write('    free_NN(nn);\n')
	fh_c.write('\n')
	fh_c.write('    return 0;\n')
	fh_c.write('  }\n')
	fh_c.write('\n')
	fh_c.write('/* Open the given file, read its weights into the given \'buffer,\' and return the length of \'buffer.\' */\n')
	fh_c.write('unsigned int readWeights(char* filename, double** buffer)\n')
	fh_c.write('  {\n')
	fh_c.write('    FILE* fp;\n')
	fh_c.write('    unsigned int len = 0;\n')
	fh_c.write('    double x;\n')
	fh_c.write('\n')
	fh_c.write('    printf("Reading %s:", filename);\n')
	fh_c.write('\n')
	fh_c.write('    if((fp = fopen(filename, "rb")) == NULL)\n')
	fh_c.write('      {\n')
	fh_c.write('        printf("ERROR: Unable to open file\\n");\n')
	fh_c.write('        exit(1);\n')
	fh_c.write('      }\n')
	fh_c.write('    fseek(fp, 0, SEEK_SET);                                         //  Rewind\n')
	fh_c.write('    while(!feof(fp))\n')
	fh_c.write('      {\n')
	fh_c.write('        if(fread(&x, sizeof(double), 1, fp) == 1)\n')
	fh_c.write('          {\n')
	fh_c.write('            if(++len == 1)\n')
	fh_c.write('              {\n')
	fh_c.write('                if(((*buffer) = (double*)malloc(sizeof(double))) == NULL)\n')
	fh_c.write('                  {\n')
	fh_c.write('                    printf("ERROR: Unable to malloc buffer\\n");\n')
	fh_c.write('                    exit(1);\n')
	fh_c.write('                  }\n')
	fh_c.write('              }\n')
	fh_c.write('            else\n')
	fh_c.write('              {\n')
	fh_c.write('                if(((*buffer) = (double*)realloc((*buffer), len * sizeof(double))) == NULL)\n')
	fh_c.write('                  {\n')
	fh_c.write('                    printf("ERROR: Unable to realloc buffer\\n");\n')
	fh_c.write('                    exit(1);\n')
	fh_c.write('                  }\n')
	fh_c.write('              }\n')
	fh_c.write('            (*buffer)[len - 1] = x;\n')
	fh_c.write('          }\n')
	fh_c.write('      }\n')
	fh_c.write('    printf(" %d weights\\n", len);\n')
	fh_c.write('    fclose(fp);                                                     //  Close the file\n')
	fh_c.write('\n')
	fh_c.write('    return len;\n')
	fh_c.write('  }\n')
	fh_c.close()													#  Done building C code.

	fh_py.write('\tmodel = models.Model( [ input_layer ], [ dense_final ] )\n')
																	#  These settings apply if you decide to let networks learn from tournament transcripts.
	fh_py.write('\tmodel.compile(optimizer=keras.optimizers.SGD(learning_rate=params[\'learning-rate\'], momentum=params[\'momentum\'], nesterov=True), loss=\'categorical_crossentropy\', metrics=[\'acc\'])\n')
	fh_py.write('\treturn model\n')
	fh_py.close()													#  Done building Python code.

	return

def build_run_code(params):
	if params['verbose']:
		print('>>> Building "run_mlp.c"')

	if params['training-source'] is not None:
		fh = open(params['training-source'], 'r')
		lines = [line for line in fh.readlines() if line[0] != '#']
		sample_string = ' '.join(lines[0].strip().split('\t')[3:-2])
		fh.close()
	else:
		sample_string = ' '.join([str(1.0 / float(params['num-classes'])) for i in range(0, params['num-classes'])])

	fh = open('run_mlp.c', 'w')
	fh.write('#include "neuron.h"\n')
	fh.write('\n')
	fh.write('/*\n')
	fh.write('  Eric C. Joyce, Stevens Institute of Technology, 2022\n')
	fh.write('\n')
	fh.write('  Convert argv input into an array of doubles.\n')
	fh.write('  Input this floating-point buffer to the neural network.\n')
	fh.write('  Print the evaluation.\n')
	fh.write('\n')
	fh.write('  gcc -c -Wall run_mlp.c\n')
	fh.write('  gfortran -o run_mlp run_mlp.o cblas_LINUX.a libblas.a\n')
	fh.write('\n')
	fh.write('  ./run_mlp mlp.nn ' + sample_string + '\n')
	fh.write('*/\n')
	fh.write('\n')
	fh.write('int main(int argc, char* argv[])\n')
	fh.write('  {\n')
	fh.write('    NeuralNet* nn;\n')
	fh.write('    double* x;\n')
	fh.write('    double* out;\n')
	fh.write('    unsigned int i;\n')
	fh.write('\n')
	fh.write('    init_NN(&nn, ' + str(params['num-classes']) + ');                                               //  Initialize\n')
	fh.write('    load_NN(argv[1], nn);                                           //  Load the network\n')
	fh.write('\n')
	fh.write('    if((x = (double*)malloc(' + str(params['num-classes']) + ' * sizeof(double))) == NULL)\n')
	fh.write('      {\n')
	fh.write('        free_NN(nn);\n')
	fh.write('        return 1;\n')
	fh.write('      }\n')
	fh.write('    for(i = 0; i < ' + str(params['num-classes']) + '; i++)                                         //  Convert to floating point\n')
	fh.write('      x[i] = (double)atof(argv[2 + i]);\n')
	fh.write('\n')
	fh.write('    run_NN(x, nn, &out);                                            //  Run the network\n')
	fh.write('    for(i = 0; i < ' + str(params['num-classes'] + 1) + '; i++)\n')
	fh.write('      {\n')
	fh.write('        if(i < ' + str(params['num-classes']) + ')\n')
	fh.write('          printf("%f ", out[i]);\n')
	fh.write('        else\n')
	fh.write('          printf("%f", out[i]);\n')
	fh.write('      }\n')
	fh.write('\n')
	fh.write('    free_NN(nn);\n')
	fh.write('    free(out);\n')
	fh.write('    free(x);\n')
	fh.write('\n')
	fh.write('    return 0;\n')
	fh.write('  }\n')

	return

def build_makefile(params):
	if params['verbose']:
		print('>>> Building "Makefile"')

	fh = open('Makefile', 'w')
	fh.write('all: build_mlp_model.o build_mlp_model run_mlp.o run_mlp\n')
	fh.write('.PHONY: all\n')
	fh.write('\n')
	fh.write('build_mlp_model: build_mlp_model.o\n')
	fh.write('\tgfortran -o build_mlp_model build_mlp_model.o ./neuron/cblas_LINUX.a ./neuron/libblas.a `mysql_config --libs`\n')
	fh.write('\n')
	fh.write('build_mlp_model.o: ./neuron/neuron.h build_mlp_model.c\n')
	fh.write('\tgcc -Wall -I ./neuron -c -lpthread -O3 `mysql_config --cflags` build_mlp_model.c\n')
	fh.write('\n')
	fh.write('run_mlp: run_mlp.o\n')
	fh.write('\tgfortran -o run_mlp run_mlp.o ./neuron/cblas_LINUX.a ./neuron/libblas.a `mysql_config --libs`\n')
	fh.write('\n')
	fh.write('run_mlp.o: ./neuron/neuron.h run_mlp.c\n')
	fh.write('\tgcc -Wall -I ./neuron -c -lpthread -O3 `mysql_config --cflags` run_mlp.c\n')

	fh.close()
	return

def get_command_line_params():
	params = {}
	params['num-classes'] = None
	params['training-source'] = None
	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-n', '-src', '-v', '-?', '-help', '--help']
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
				if argtarget == '-n':								#  Keep it above 1
					params['num-classes'] = max(2, int(argval))
				elif argtarget == '-src':
					params['training-source'] = argval

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('Generate model-building code in C and Python according to the model specs.')
	print('')
	print('Usage:  python3 build_model_code.py <parameters, preceded by flags>')
	print(' e.g.:  python3 build_model_code.py -n 25 -v')
	print('')
	print('Flags:  -n   REQUIRED. The number of classes (actions).')
	print('        -v   Enable verbosity')
	print('        -?   Display this message')

if __name__ == '__main__':
	main()
