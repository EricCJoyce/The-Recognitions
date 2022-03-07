import os
import shutil
import sys

def main():
	params = get_command_line_params()								#  Collect parameters.
	if params['helpme'] or len(params['enactments']) == 0 or params['source-suffix'] is None:
		usage()
		return

	for enactment_name in params['enactments']:
		detection_src_file = enactment_name + params['source-suffix'] + '_detections.txt'
		detection_directory_src = enactment_name + params['source-suffix']
		index = detection_directory_src.index('_')
		detection_directory_src = list(detection_directory_src)
		detection_directory_src[index] = '/'
		detection_directory_src = ''.join(detection_directory_src)

		detection_dst_file = enactment_name + params['source-suffix'].split('-')[0] + '-th' + '{0:.2f}'.format(params['score-threshold']) + '_detections.txt'
		detection_directory_dst = enactment_name + params['source-suffix'].split('-')[0] + '-th' + '{0:.2f}'.format(params['score-threshold'])
		index = detection_directory_dst.index('_')
		detection_directory_dst = list(detection_directory_dst)
		detection_directory_dst[index] = '/'
		detection_directory_dst = ''.join(detection_directory_dst)

		fh = open(detection_src_file, 'r')
		lines = fh.readlines()
		fh.close()

		if params['verbose']:
			print(enactment_name + ': ' + detection_src_file + ', ' + detection_directory_src)
			print(' '*len(enactment_name) + ' --> ' + detection_dst_file + ', ' + detection_directory_dst)

		fh = open(detection_dst_file, 'w')
		mask_ctr = 0
		os.makedirs(detection_directory_dst)

		for line in lines:
			if line[0] == '#':
				fh.write(line)
			else:
				arr = line.strip().split('\t')
				timestamp        = arr[0]
				imgfilename      = arr[1]
				instance         = arr[2]
				classname        = arr[3]
				detection_src    = arr[4]
				confidence       = arr[5]
				bounding_box_str = arr[6]
				maskfilename     = arr[7]
				centroid_3d_avg  = arr[8]
				centroid_3d_bbox = arr[9]

				bounding_box_arr = bounding_box_str.split(';')
				bounding_box_upper_left = [int(x) for x in bounding_box_arr[0].split(',')]
				bounding_box_lower_right = [int(x) for x in bounding_box_arr[1].split(',')]
				bounding_box_width = bounding_box_lower_right[0] - bounding_box_upper_left[0]
				bounding_box_height = bounding_box_lower_right[1] - bounding_box_upper_left[1]
				bounding_box_area = bounding_box_width * bounding_box_height

				if float(confidence) >= params['score-threshold'] and bounding_box_area >= params['minpx']:
					new_maskfilename = detection_directory_dst + '/mask_' + str(mask_ctr) + '.png'
					shutil.copyfile(maskfilename, new_maskfilename)
					fh.write(timestamp + '\t' + imgfilename + '\t' + instance + '\t' + classname + '\t' + detection_src + '\t' + confidence + '\t' + \
					         bounding_box_str + '\t' + new_maskfilename + '\t' + centroid_3d_avg + '\t' + centroid_3d_bbox + '\n')
					mask_ctr += 1

		fh.close()

	return

def get_command_line_params():
	params = {}
	params['enactments'] = []										#  List of file paths.
	params['source-suffix'] = None
	params['score-threshold'] = 0.6									#  Detection score must be greater than this in order to register.
	params['minpx'] = 1												#  Minimum number of pixels for something to be considered visible.
	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-e', '-src', '-th', '-minpx', \
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
				elif argtarget == '-src':
					params['source-suffix'] = argval
				elif argtarget == '-th':
					params['score-threshold'] = max(0.0, float(argval))
				elif argtarget == '-minpx':
					params['minpx'] = max(1, int(argval))

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('This is a time-saver: as you try to gague which detection model threshold works best')
	print('for your application, you can set a detection threshold to zero, capture everything, and')
	print('then derive new, more discerning detections.')
	print('')
	print('Usage:  python3 filter_detections.py <parameters, preceded by flags>')
	print(' e.g.:  python3 filter_detections.py -th 0.3 -src _ssd_mobilenet_640x640-th0.0 -e BackBreaker1 -e Enactment1 -e Enactment2 -e Enactment3 -e Enactment4 -e Enactment5 -e Enactment6 -e Enactment7 -e Enactment9 -e Enactment10 -e Enactment11 -e Enactment12 -e MainFeederBox1 -e Regulator1 -e Regulator2 -v')
	print('')
	print('Flags:  -e       MUST HAVE AT LEAST ONE: Path to a directory of raw enactment materials, including the source/detection you wish to filter.')
	print('        -src     MUST HAVE EXACTLY ONE: Path to a model that can perform object recognition.')
	print('')
	print('        -th      Following real number in [0.0, 1.0] is the threshold, above which detections must score in order to count.')
	print('        -minpx   Following integer in [1, inf) is the minimum bounding box area in pixels for a detection to count.')
	print('')
	print('        -v       Enable verbosity')
	print('        -?       Display this message')

if __name__ == '__main__':
	main()
