import os
import shutil
import sys

def main():
	params = get_command_line_params()								#  Collect parameters
	if params['helpme'] or not os.path.exists('training-set.txt') or not os.path.exists('validation-set.txt'):
		usage()
		return

	if os.path.exists('./training/images/train'):
		print('ERROR: a directory named \'./training/images/train\' already exists.')
		return
	if os.path.exists('./training/images/test'):
		print('ERROR: a directory named \'./training/images/test\' already exists.')
		return

	os.makedirs('./training/images/train')							#  Make directories.
	os.makedirs('./training/images/test')

	if params['verbose']:
		print('>>> Created folders \'./training/images/train\' and \'./training/images/test\'')

	train = {}														#  key: (image file path, w, h) ==> val: [ (object-name, bbox),
	test = {}														#                                          (object-name, bbox), ...]

	if params['verbose']:
		print('>>> Scanning training-set.txt...')

	fh = open('training-set.txt', 'r')								#  Scan the training set file...
	for line in fh.readlines():
		if line[0] != '#':
			arr = line.strip().split('\t')
			img_path = arr[0]
			dimensions_str = arr[1]
			width = int(dimensions_str.split(',')[0])
			height = int(dimensions_str.split(',')[1])
			recognizable_object = arr[3]
			bbox_str = arr[4]

			if (img_path, width, height) not in train:
				train[ (img_path, width, height) ] = []
			train[ (img_path, width, height) ].append( (recognizable_object, bbox_str) )
	fh.close()

	if params['verbose']:
		print('>>> Scanning validation-set.txt...')

	fh = open('validation-set.txt', 'r')							#  Scan the validation set file...
	for line in fh.readlines():
		if line[0] != '#':
			arr = line.strip().split('\t')
			img_path = arr[0]
			dimensions_str = arr[1]
			width = int(dimensions_str.split(',')[0])
			height = int(dimensions_str.split(',')[1])
			recognizable_object = arr[3]
			bbox_str = arr[4]

			if (img_path, width, height) not in test:
				test[ (img_path, width, height) ] = []
			test[ (img_path, width, height) ].append( (recognizable_object, bbox_str) )
	fh.close()

	if params['verbose']:
		print('')
		print('>>> Populating \'./training/images/train\'...')

	train_ctr = 1
	for imgpath_w_h, annotations in train.items():					#  Clone images and build XMLs.
		imgpath = imgpath_w_h[0]
		width   = imgpath_w_h[1]
		height  = imgpath_w_h[2]
		img_ext = imgpath.split('.')[-1]

		xml_imgfilename = imgpath.split('/')[-1]
		xml_filename = './training/images/train/' + str(train_ctr) + '.xml'

		fh_xml = open(xml_filename, 'w')
		fh_xml.write('<annotation>\n')
		fh_xml.write('\t<folder>train</folder>\n')
		fh_xml.write('\t<filename>' + xml_imgfilename + '</filename>\n')
		fh_xml.write('\t<path>' + imagefilename + '</path>\n')
		fh_xml.write('\t<souce>\n')
		fh_xml.write('\t\t<database>Unknown</database>\n')
		fh_xml.write('\t</souce>\n')
		fh_xml.write('\t<size>\n')
		fh_xml.write('\t\t<width>' + str(width) + '</width>\n')
		fh_xml.write('\t\t<height>' + str(height) + '</height>\n')
		fh_xml.write('\t\t<depth>' + str(params['imgd']) + '</depth>\n')
		fh_xml.write('\t</size>\n')
		fh_xml.write('\t<segmented>0</segmented>\n')

		if params['verbose']:
			print('    ' + imgpath + ': ' + str(len(annotations)) + ' annotated objects.')

		for recog_obj in annotations:
			recog_obj_label = recog_obj[0]
			xmin            = recog_obj[1].split(';')[0].split(',')[0]
			ymin            = recog_obj[1].split(';')[0].split(',')[1]
			xmax            = recog_obj[1].split(';')[1].split(',')[0]
			ymax            = recog_obj[1].split(';')[1].split(',')[1]

			fh_xml.write('\t<object>\n')
			fh_xml.write('\t\t<name>' + recog_obj_label + '</name>\n')
			fh_xml.write('\t\t<pose>Unspecified</pose>\n')
			fh_xml.write('\t\t<truncated>0</truncated>\n')
			fh_xml.write('\t\t<difficult>0</difficult>\n')
			fh_xml.write('\t\t<bndbox>\n')
			fh_xml.write('\t\t\t<xmin>' + xmin + '</xmin>\n')
			fh_xml.write('\t\t\t<ymin>' + ymin + '</ymin>\n')
			fh_xml.write('\t\t\t<xmax>' + xmax + '</xmax>\n')
			fh_xml.write('\t\t\t<ymax>' + ymax + '</ymax>\n')
			fh_xml.write('\t\t</bndbox>\n')
			fh_xml.write('\t</object>\n')
		fh_xml.write('</annotation>')
		fh_xml.close()

		shutil.copyfile(imgpath, './training/images/train/' + str(train_ctr) + '.' + img_ext)

		train_ctr += 1

	if params['verbose']:
		print('')
		print('>>> Populating \'./training/images/test\'...')

	test_ctr = 1
	for imgpath_w_h, annotations in test.items():					#  Clone images and build XMLs.
		imgpath = imgpath_w_h[0]
		width   = imgpath_w_h[1]
		height  = imgpath_w_h[2]
		img_ext = imgpath.split('.')[-1]

		xml_imgfilename = imgpath.split('/')[-1]
		xml_filename = './training/images/test/' + str(test_ctr) + '.xml'

		fh_xml = open(xml_filename, 'w')
		fh_xml.write('<annotation>\n')
		fh_xml.write('\t<folder>test</folder>\n')
		fh_xml.write('\t<filename>' + xml_imgfilename + '</filename>\n')
		fh_xml.write('\t<path>' + imagefilename + '</path>\n')
		fh_xml.write('\t<souce>\n')
		fh_xml.write('\t\t<database>Unknown</database>\n')
		fh_xml.write('\t</souce>\n')
		fh_xml.write('\t<size>\n')
		fh_xml.write('\t\t<width>' + str(width) + '</width>\n')
		fh_xml.write('\t\t<height>' + str(height) + '</height>\n')
		fh_xml.write('\t\t<depth>' + str(params['imgd']) + '</depth>\n')
		fh_xml.write('\t</size>\n')
		fh_xml.write('\t<segmented>0</segmented>\n')

		if params['verbose']:
			print('    ' + imgpath + ': ' + str(len(annotations)) + ' annotated objects.')

		for recog_obj in annotations:
			recog_obj_label = recog_obj[0]
			xmin            = recog_obj[1].split(';')[0].split(',')[0]
			ymin            = recog_obj[1].split(';')[0].split(',')[1]
			xmax            = recog_obj[1].split(';')[1].split(',')[0]
			ymax            = recog_obj[1].split(';')[1].split(',')[1]

			fh_xml.write('\t<object>\n')
			fh_xml.write('\t\t<name>' + recog_obj_label + '</name>\n')
			fh_xml.write('\t\t<pose>Unspecified</pose>\n')
			fh_xml.write('\t\t<truncated>0</truncated>\n')
			fh_xml.write('\t\t<difficult>0</difficult>\n')
			fh_xml.write('\t\t<bndbox>\n')
			fh_xml.write('\t\t\t<xmin>' + xmin + '</xmin>\n')
			fh_xml.write('\t\t\t<ymin>' + ymin + '</ymin>\n')
			fh_xml.write('\t\t\t<xmax>' + xmax + '</xmax>\n')
			fh_xml.write('\t\t\t<ymax>' + ymax + '</ymax>\n')
			fh_xml.write('\t\t</bndbox>\n')
			fh_xml.write('\t</object>\n')
		fh_xml.write('</annotation>')
		fh_xml.close()

		shutil.copyfile(imgpath, './training/images/test/' + str(test_ctr) + '.' + img_ext)

		test_ctr += 1

	return

def get_command_line_params():
	params = {}

	params['imgd'] = 3												#  We expect three channels: red, green, blue.

	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set.
																	#  Permissible setting flags.
	flags = ['-v', '-?', '-help', '--help']
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
				if argtarget == '-imgd':
					params['imgd'] = max(1, int(argval))			#  Has to be at least one channel.

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('Convert the training and validation set files to formats expected by the TensorFlow Object-Detection API.')
	print('This means creating two folders in the current working directory:')
	print('  \'training/images/train\' and \'training/images/test\'.')
	print('This script expects to find \'training-set.txt\' and \'validation-set.txt\'.')
	print('It also expects to find the enactments referenced by these documents.')
	print('This script will copy all images into the respective folders and generate one annotation XML per image.')
	print('')
	print('Usage:  python3 convert_object_recog_dataset.py <parameters, preceded by flags>')
	print(' e.g.   ')
	print('        python3 convert_object_recog_dataset.py -v')
	print('')
	print('Flags:  -v      Enable verbosity')
	print('        -?      Display this message')
	return

if __name__ == '__main__':
	main()
