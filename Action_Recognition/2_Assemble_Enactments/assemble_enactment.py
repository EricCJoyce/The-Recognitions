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

		#############################################################
		#  Enactments initialize with JSON sensor data by default.  #
		#                                                           #
		#  Determine the enactment's source for object data:        #
		#    ground-truth (*_props.txt) or                          #
		#    network (e.g. *_ssd_mobilenet_640x640_detections.txt)  #
		#  This is enactment_name + params['parse-suffix'].         #
		#                                                           #
		#  Do we want to use the action labels provided in the JSON #
		#  or in some other source?                                 #
		#  This is params['action-label-source'].                   #
		#  If it is None, then use JSON. Otherwise, for each        #
		#  enactment look for *<params['action-label-source']>      #
		#                                                           #
		#  When handling objects, do we call their centroid the     #
		#  average of all pixels in the mask or the bounding-box    #
		#  center? (Applies to both objects and IK hands.)          #
		#  This is params['centroid-mode'].                         #
		#                                                           #
		#  Do we want hand poses from the sensors (noisy!) or from  #
		#  the IK hands?                                            #
		#  This is params['hand-pose-mode'].                        #
		#                                                           #
		#  If sensor data: are we shutting off the signal when      #
		#  hands are not visible?                                   #
		#  If IK: are they available? And are we using the pixel    #
		#  average or the bounding-box average?                     #
		#                                                           #
		#  Where will hand poses come from?                         #
		#  That is params['hand-pose-src-file'].                    #
		#############################################################
																	#  Read objects--possibly including GT hands.
		e.load_parsed_objects(enactment_name + params['parse-suffix'], params['centroid-mode'])

		if params['action-label-source'] is not None:				#  If we are taking action labels from a source other than the JSON files.
			e.load_all_frame_labels(enactment_name + params['action-label-source'])

		if params['hand-pose-src-file'] is None:					#  No external hand-pose source file:
																	#  then consider whether we will use sensor poses or IK poses.
			if params['hand-pose-mode'] == 'IK-bbox' or params['hand-pose-mode'] == 'IK-avg':
				e.compute_IK_poses()								#  Centroid-mode handles the rest.
			elif params['hand-pose-mode'] == 'sensor-shutoff':
				e.apply_camera_projection_shutoff()
		else:														#  We are we using a hand-pose source file.
			e.load_hand_pose_file(enactment_name + params['hand-pose-src-file'])

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
			if params['colors-file'] is not None:					#  Were we given a color look-up file?
				e.load_color_lookup(params['colors-file'])

			#e.render_gaussian_weighted_video(g)
			e.render_annotated_video()

	return

def get_command_line_params():
	params = {}
	params['enactments'] = []										#  List of file paths
	params['gaussian'] = {}											#  key:mu         ==> val:(x, y, z)
	params['gaussian']['mu'] = (0.0, 0.0, 0.0)						#  key:sigma-gaze ==> val:(x, y, z)
	params['gaussian']['sigma-gaze'] = (2.0, 1.5, 3.0)				#  key:sigma-hand ==> val:(x, y, z)
	params['gaussian']['sigma-hand'] = (0.5, 0.5, 0.5)

	params['parse-suffix'] = '_props.txt'							#  Suffix of source of object detections--possibly including hands.
	params['action-label-source'] = None							#  By default, use the given JSON action labels.
	params['centroid-mode'] = 'bbox'
	params['hand-pose-mode'] = 'sensor-shutoff'
	params['hand-pose-src-file'] = None

	params['min-pixels'] = 1										#  By default, admit everything.
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
	         '-suffix', '-action', '-centroid', '-handmode', '-handsrc', \
	         '-render', '-color', '-colors', '-minpx', \
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

				elif argtarget == '-suffix':
					params['parse-suffix'] = argval
				elif argtarget == '-action':
					params['action-label-source'] = argval
				elif argtarget == '-centroid':
					params['centroid-mode'] = argval
				elif argtarget == '-handmode':
					params['hand-pose-mode'] = argval
				elif argtarget == '-handsrc':
					params['hand-pose-src-file'] = argval

				elif argtarget == '-imgw':
					params['imgw'] = max(1, int(argval))
				elif argtarget == '-imgh':
					params['imgh'] = max(1, int(argval))
				elif argtarget == '-User':
					params['User'] = argval
				elif argtarget == '-fontsize':
					params['fontsize'] = float(argval)

	if params['hand-pose-mode'] not in ['IK-bbox', 'IK-avg', 'sensor-shutoff', 'sensor']:
		print('>>> INVALID DATA received for hand-pose mode. Restoring default value.')
		params['hand-pose-mode'] = 'sensor-shutoff'

	if params['centroid-mode'] not in ['bbox', 'avg']:
		print('>>> INVALID DATA received for centroid mode. Restoring default value.')
		params['centroid-mode'] = 'bbox'

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
	print(' e.g.:  python3 assemble_enactment.py -e Enactment1 -suffix _ssd_mobilenet_640x640_detections.txt -handsrc .IK-bbox.handposes -v -render')
	print('')
	print('Flags:  -e        Following argument is path to a directory of raw enactment materials: JSONs and color maps.')
	print('        -mu       Following three arguments are the three components of the Gaussian\'s mu for the gaze.')
	print('        -sigHead  Following three arguments are the three components of the Gaussian\'s sigma for the gaze.')
	print('        -sigHand  Following three arguments are the three components of the Gaussian\'s sigma for the hands.')
	print('        -minpx    Following argument is the minimum number of pixels an object needs to occupy to be admitted')
	print('                  to the *.enactment file. The default is 1.')
	print('')
	print('        -suffix   Following string is the suffix of the file to be used as object-detection source.')
	print('                  By default, this is "_props.txt", meaning the script expects to use ground-truth detections')
	print('                  unless directed elsewhere.')
	print('        -action   If given, the string following this flag is the file extension containing label lookups for')
	print('                  each enactment. When omitted, enactments take action labels from their JSON files (or an override')
	print('                  file "labels.txt" found inside the enactment directory.)')
	print('        -centroid Following string in {avg, bbox} indicates how an object\'s centroid is computed.')
	print('                  Default is "bbox".')
	print('        -handmode Following string in {IK-bbox, IK-avg, sensor-shutoff, sensor} indicates how hand poses')
	print('                  are to be computed. Default is "sensor-shutoff".')
	print('        -handsrc  Following string is a suffix for special files containing only hand poses.')
	print('                  If invoked, the poses in these files overwrite the enactments\' poses.')
	print('                  You would derive a hand-pose file from the script "extract_hand_poses.py" and, say,')
	print('                  apply these poses to object-detections made using a network (that was not trained to')
	print('                  detect hands by itself.)')
	print('')
	print('        -color    Following argument is the path to a text file itemizing which recognizable objects should')
	print('                  receive which color overlays (Only matters if you are rendering.) If no file is specified,')
	print('                  then colors will be generated randomly.')
	print('        -render   Generate videos for all given enactments.')
	print('        -v        Enable verbosity')
	print('        -?        Display this message')

if __name__ == '__main__':
	main()
