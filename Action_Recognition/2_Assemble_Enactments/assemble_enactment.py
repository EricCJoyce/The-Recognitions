from enactment import *

def main():
	params = get_command_line_params()								#  Collect parameters
	if params['helpme']:
		usage()
		return

	for enactment_name in params['enactments']:						#  Write an *.enactment file for each of the given enactments.
		if params['imgw'] is not None and params['imgh'] is not None:
			e = Enactment(enactment_name, wh=(params['imgw'], params['imgh']), user=params['User'], verbose=params['verbose'])
		else:
			e = Enactment(enactment_name, user=params['User'], verbose=params['verbose'])

		e.gt_label_super['fontsize'] = params['fontsize']			#  Set font sizes from this script's parameters.
		e.filename_super['fontsize'] = params['fontsize']
		e.LH_super['fontsize'] = params['fontsize']
		e.RH_super['fontsize'] = params['fontsize']

		if params['colors-file'] is not None:						#  Were we given a color look-up file?
			e.load_color_map(params['colors-file'])

		if params['pose-src'] == 'IK-bbox':							#  Pick your pose.
			e.load_parsed_objects(None, 'bbox')
			e.compute_IK_poses()
		elif params['pose-src'] == 'IK-avg':
			e.load_parsed_objects(None, 'avg')
			e.compute_IK_poses()
		elif params['pose-src'] == 'sensor-shutoff':
			e.load_parsed_objects(None, 'bbox')
			e.apply_camera_projection_shutoff()
		else:
			e.load_parsed_objects(None, 'bbox')

		g = Gaussian(mu=params['gaussian']['mu'], \
		             sigma_gaze=params['gaussian']['sigma-gaze'], \
		             sigma_hand=params['gaussian']['sigma-hand'])

		if params['min-pixels'] > 0:								#  Kill scrub detections.
			e.disable_detections( (lambda detection: detection.get_mask_area() < params['min-pixels']) )

																	#  Do the following disablings AFTER computing IK poses!
		e.disable_detections( (lambda detection: detection.object_name == 'RightHand') )
		e.disable_detections( (lambda detection: detection.object_name == 'LeftHand') )

		e.write_text_enactment_file(g)								#  Create the file we want.

		if params['render']:										#  Rendering?
			e.render_gaussian_weighted_video(g)

	return

def get_command_line_params():
	params = {}
	params['enactments'] = []										#  List of file paths
	params['gaussian'] = {}											#  key:mu         ==> val:(x, y, z)
	params['gaussian']['mu'] = (0.0, 0.0, 0.0)						#  key:sigma-gaze ==> val:(x, y, z)
	params['gaussian']['sigma-gaze'] = (2.0, 1.5, 3.0)				#  key:sigma-hand ==> val:(x, y, z)
	params['gaussian']['sigma-hand'] = (0.5, 0.5, 0.5)

	params['pose-src'] = 'IK-bbox'									#  By default, use the bounding-box center of the IK hands.
																	#  (I don't like sensor data: they are WAY too noisy!)
	params['min-pixels'] = 400										#  I decided this was a good default minimum.
	params['colors-file'] = None
	params['render'] = False										#  Whether to render stuff.

	params['verbose'] = False
	params['helpme'] = False

	params['fontsize'] = 1											#  For rendering text to images and videos
	params['imgw'] = 1280
	params['imgh'] = 720
	params['User'] = 'vr1'											#  It used to be "admin", and I don't want to change a bunch of file paths when it changes again

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-e', \
	         '-mu', '-sigHead', '-sigHand', \
	         '-render', '-color', '-colors', '-minpx', '-pose', \
	         '-v', '-?', '-help', '--help', \
	         '-User', '-imgw', '-imgh', '-fontsize']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-v':
				params['verbose'] = True
			elif sys.argv[i] == '-?' or sys.argv[i] == '-help' or sys.argv[i] == '--help':
				params['helpme'] = True
			elif sys.argv[i] == '-render':
				params['render'] = True
			else:
				argtarget = sys.argv[i]
		else:
			argval = sys.argv[i]

			if argtarget is not None:
				if argtarget == '-e':
					params['enactments'].append(argval)

				elif argtarget == '-mu':
					params['gaussian']['mu'] = (float(argval), params['gaussian']['mu'][1], params['gaussian']['mu'][2])
					argtarget = '-muY'
				elif argtarget == '-muY':
					params['gaussian']['mu'] = (params['gaussian']['mu'][0], float(argval), params['gaussian']['mu'][2])
					argtarget = '-muZ'
				elif argtarget == '-muZ':
					params['gaussian']['mu'] = (params['gaussian']['mu'][0], params['gaussian']['mu'][1], float(argval))

				elif argtarget == '-sigHead':
					params['gaussian']['sigma-gaze'] = (float(argval), params['gaussian']['sigma-gaze'][1], params['gaussian']['sigma-gaze'][2])
					argtarget = '-sigHeadY'
				elif argtarget == '-sigHeadY':
					params['gaussian']['sigma-gaze'] = (params['gaussian']['sigma-gaze'][0], float(argval), params['gaussian']['sigma-gaze'][2])
					argtarget = '-sigHeadZ'
				elif argtarget == '-sigHeadZ':
					params['gaussian']['sigma-gaze'] = (params['gaussian']['sigma-gaze'][0], params['gaussian']['sigma-gaze'][1], float(argval))

				elif argtarget == '-sigHand':
					params['gaussian']['sigma-hand'] = (float(argval), params['gaussian']['sigma-hand'][1], params['gaussian']['sigma-hand'][2])
					argtarget = '-sigHandY'
				elif argtarget == '-sigHandY':
					params['gaussian']['sigma-hand'] = (params['gaussian']['sigma-hand'][0], float(argval), params['gaussian']['sigma-hand'][2])
					argtarget = '-sigHandZ'
				elif argtarget == '-sigHandZ':
					params['gaussian']['sigma-hand'] = (params['gaussian']['sigma-hand'][0], params['gaussian']['sigma-hand'][1], float(argval))

				elif argtarget == '-minpx':
					params['min-pixels'] = max(1, abs(int(argval)))
				elif argtarget == '-color' or argtarget == '-colors':
					params['colors-file'] = argval
				elif argtarget == '-pose':
					params['pose-src'] = argval

				elif argtarget == '-imgw':
					params['imgw'] = max(1, int(argval))
				elif argtarget == '-imgh':
					params['imgh'] = max(1, int(argval))
				elif argtarget == '-User':
					params['User'] = argval
				elif argtarget == '-fontsize':
					params['fontsize'] = float(argval)

	if params['pose-src'] not in ['IK-bbox', 'IK-avg', 'sensor-shutoff', 'sensor']:
		print('>>> INVALID DATA received for pose source. Restoring default value.')
		params['pose-src'] = 1

	if params['fontsize'] < 1:
		print('>>> INVALID DATA received for fontsize. Restoring default value.')
		params['fontsize'] = 1

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('Enactments have been processed and now need to be "interpreted."')
	print('That is, we know what is in each frame but must weigh objects\' presences by a Gaussian')
	print('"attention cone." This determines the relevance of objects with which users interact.')
	print('')
	print('Usage:  python3 assemble_enactment.py <parameters, preceded by flags>')
	print(' e.g.:  python3 assemble_enactment.py -e Enactment11 -e Enactment12 -v -render')
	print('')
	print('Flags:  -e        Following argument is path to a directory of raw enactment materials: JSONs and color maps.')
	print('        -mu       Following three arguments are the three components of the Gaussian\'s mu for the gaze.')
	print('        -sigHead  Following three arguments are the three components of the Gaussian\'s sigma for the gaze.')
	print('        -sigHand  Following three arguments are the three components of the Gaussian\'s sigma for the hands.')
	print('        -minpx    Following argument is the minimum number of pixels an object needs to occupy to be admitted')
	print('                  to the *.enactment file. The default is 400.')
	print('        -pose     Following argument is a string in {IK-bbox, IK-avg, sensor-shutoff, sensor}.')
	print('                  This indicates what the source for hand poses should be. The default is "IK-bbox", meaning')
	print('                  hand poses are derived from the bounding-box center of the IK hands. If a hand is not')
	print('                  visible, then that hand\'s subvector is zero. The idea is that this mimics a HoloLens.')
	print('                  The "sensor" alternatives take hand data from the hand paddle sensors. The "shutoff" option')
	print('                  mimics the HoloLens assumption that if hands are not seen, then their poses are not registered.')
	print('        -color    Following argument is the path to a text file itemizing which recognizable objects should')
	print('                  receive which color overlays (Only matters if you are rendering.) If no file is specified,')
	print('                  then colors will be generated randomly.')
	print('        -render   Generate videos for all given enactments.')
	print('        -v        Enable verbosity')
	print('        -?        Display this message')

if __name__ == '__main__':
	main()
