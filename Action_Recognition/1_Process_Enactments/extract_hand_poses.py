from enactment import *

def main():
	params = get_command_line_params()								#  Collect parameters
	if params['helpme'] or len(params['enactments']) == 0:
		usage()
		return

	for enactment_name in params['enactments']:
		if params['verbose']:
			print('\n>>> Reading ' + enactment_name + '\n')

		e = Enactment(enactment_name, enactment_file=enactment_name + '_props.txt', verbose=params['verbose'])
																	#  Remember: the Enactment constructor already loads sensor hand poses.
		if params['pose-source'] == 'sensor':
			e.write_hand_pose_file(enactment_name + '.sensor')

			e.apply_camera_projection_shutoff()
			e.write_hand_pose_file(enactment_name + '.sensor-shutoff')
		elif params['pose-source'] == 'IK':
			e.load_parsed_objects(props_file=enactment_name + '_props.txt', centroid_src='avg')
			e.compute_IK_poses()
			e.write_hand_pose_file(enactment_name + '.IK-avg')

			e.load_parsed_objects(props_file=enactment_name + '_props.txt', centroid_src='bbox')
			e.compute_IK_poses()
			e.write_hand_pose_file(enactment_name + '.IK-bbox')

	return

def get_command_line_params():
	params = {}
	params['enactments'] = []										#  List of file paths
	params['pose-source'] = None
	acceptable_pose_sources = ['IK', 'sensor']
	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-e', '-src', '-v', '-?', '-help', '--help']
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
				elif argtarget == '-src' and argval in acceptable_pose_sources:
					params['pose-source'] = argval

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('You\'ve already processed one or more enactments and now want to extract the hand poses from them.')
	print('You would do this if you want to use the centroids of the Inverse-Kinematic hands without also')
	print('using ground-truth object detection.')
	print('Write extracted hand poses to a separate file, then integrate this file with network detections')
	print('in the script "detect_enactment.py".')
	print('')
	print('Usage:  python3 extract_hand_poses.py <parameters, preceded by flags>')
	print(' e.g.:  python3 extract_hand_poses.py -src IK-bbox -e Enactment11 -e Enactment12 -v')
	print('')
	print('Flags:  -e    MUST HAVE AT LEAST ONE: Path to a directory of raw enactment materials: JSONs and color maps.')
	print('        -src  Following string must be in {IK, sensor}.')
	print('')
	print('        -v    Enable verbosity')
	print('        -?    Display this message')

if __name__ == '__main__':
	main()
