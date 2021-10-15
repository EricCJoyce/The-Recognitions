from enactment import *

def main():
	params = get_command_line_params()								#  Collect parameters
	if params['helpme'] or len(params['enactments']) == 0 or params['rules'] is None:
		usage()
		return

	now = datetime.datetime.now()									#  Build a distinct substring so I don't accidentally overwrite results.
	file_timestamp = now.strftime("%d") + now.strftime("%m") + now.strftime("%Y")[-2:] + 'T' + now.strftime("%H:%M:%S").replace(':', '')
	fh = open('process_enactment_' + file_timestamp + '.log', 'w')	#  Create a log file.

	all_enactments_min_depth = float('inf')							#  Prepare to identify global depth map extrema.
	all_enactments_max_depth = float('-inf')
	dimensions = {}													#  key:enactment name ==> (width, height)
																	#  so we won't have to comb over these twice.
	#################################################################
	#  First pass on the given enactments: check frame sizes;       #
	#  check color channels; (optionally) render visual aides.      #
	#################################################################
	if params['verbose']:
		print('>>> Preparing the following enactments:')
		for enactment_name in params['enactments']:
			print('\t' + enactment_name)
		print('')

	for enactment_name in params['enactments']:						#  Go over each enactment and make sure they fit spec.
																	#  We deliberately leave out the kwarg width and height here:
																	#  we WANT the Enactment class to check every frame.
		e = Enactment(enactment_name, user=params['User'], verbose=params['verbose'])
		dimensions[enactment_name] = (e.width, e.height)			#  Save this enactment's dimensions.
																	#  Update global extrema.
		all_enactments_min_depth = min(all_enactments_min_depth, e.min_depth)
		all_enactments_max_depth = max(all_enactments_max_depth, e.max_depth)

		e.gt_label_super['fontsize'] = params['fontsize']			#  Set font sizes from this script's parameters.
		e.filename_super['fontsize'] = params['fontsize']
		e.LH_super['fontsize'] = params['fontsize']
		e.RH_super['fontsize'] = params['fontsize']

		color_maps = e.load_color_sequence(True)					#  Check every color map; make sure it matches the video dimensions.
		for color_map in color_maps:
			img = cv2.imread(color_map, cv2.IMREAD_UNCHANGED)
			if img.shape[0] != e.height or img.shape[1] != e.width:
				fh.write('ERROR: Color map "' + color_map + '" does not match the video frames shape in "' + enactment_name + '".\n')
				print('ERROR: Color map "' + color_map + '" does not match the video frames shape in "' + enactment_name + '".')

		depth_maps = e.load_depth_sequence(True)					#  Check every depth map; make sure it matches the video dimensions.
		for depth_map in depth_maps:
			img = cv2.imread(depth_map, cv2.IMREAD_UNCHANGED)
			if img.shape[0] != e.height or img.shape[1] != e.width:
				fh.write('ERROR: Depth map "' + depth_map + '" does not match the video frames shape in "' + enactment_name + '".\n')
				print('ERROR: Depth map "' + depth_map + '" does not match the video frames shape in "' + enactment_name + '".')
			if len(img.shape) > 2:
				fh.write('ERROR: Depth map "' + depth_map + '" in "' + enactment_name + '" has more channels than expected.\n')
				print('ERROR: Depth map "' + depth_map + '" in "' + enactment_name + '" has more channels than expected.')

		if params['render']:
			e.render_composite_video()								#  Make a composite video to make sure that these sets of frames align.
			e.render_annotated_video()								#  Make an annotated video; it's helpful to see.

			e.render_skeleton_poses()								#  Render centipedes (headframe and global).
			e.render_action_poses()

	#################################################################
	#  Second pass on the given enactments: build depth histograms. #
	#################################################################
	if params['verbose']:
		print('>>> Surveying depths of the following enactments:')
		for enactment_name in params['enactments']:
			print('\t' + enactment_name)
		print('')

	histogram = {}													#  Build a depth histogram over all enactments.
	for i in range(0, 256):
		d = (float(i) / 255.0) * (all_enactments_max_depth - all_enactments_min_depth) + all_enactments_min_depth
		histogram[d] = 0

	for enactment_name in params['enactments']:
		e = Enactment(enactment_name, wh=dimensions[enactment_name], user=params['User'], verbose=params['verbose'])

		h = e.compute_depth_histogram()								#  Get a depth histogram for the current enactment.
		histg = []
		for i in range(0, 256):
			d = (float(i) / 255.0) * (e.max_depth - e.min_depth) + e.min_depth
			histg.append(h[d])
			if d not in histogram:
				histogram[d] = 0
			histogram[d] += h[d]
																	#  Save individual histograms
		plt.bar([(float(i) / 255.0) * (e.max_depth - e.min_depth) + e.min_depth for i in range(0, 256)], height=histg)
		plt.xlabel('Depth')
		plt.ylabel('Frequency')
		plt.title(enactment_name + ' Depth Histogram')
		plt.savefig(enactment_name + '-histogram.png')

	histg = []
	for i in range(0, 256):
		d = (float(i) / 255.0) * (all_enactments_max_depth - all_enactments_min_depth) + all_enactments_min_depth
		histg.append(histogram[d])
																	#  Save panoramic histogram
	plt.bar([(float(i) / 255.0) * (all_enactments_max_depth - all_enactments_min_depth) + all_enactments_min_depth for i in range(0, 256)], height=histg)
	plt.xlabel('Depth')
	plt.ylabel('Frequency')
	plt.title('Combined Depth Histogram')
	plt.savefig('total-histogram.png')

	#################################################################
	#  Final pass on the given enactments: apply rules; build       #
	#  lookup table; render masks.                                  #
	#################################################################
	if params['verbose']:
		print('>>> Applying "' + params['rules'] + '" to the following enactments:')
		for enactment_name in params['enactments']:
			print('\t' + enactment_name)
		print('')

	for enactment_name in params['enactments']:
		e = Enactment(enactment_name, wh=dimensions[enactment_name], user=params['User'], verbose=params['verbose'])

		e.logical_parse(params['rules'])							#  Apply rules; parse objects.
		e.render_parsed()											#  Save parsed objects' masks.

	fh.close()

	return

def get_command_line_params():
	params = {}
	params['enactments'] = []										#  List of file paths
	params['rules'] = None											#  Rules file
	params['render'] = False										#  Whether to render stuff.
	params['verbose'] = False
	params['helpme'] = False
	params['fontsize'] = 1											#  For rendering text to images and videos
	params['User'] = 'vr1'											#  It used to be "admin", and I don't want to change a bunch of file paths when it changes again

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-e', '-r', '-render', \
	         '-v', '-?', '-help', '--help', \
	         '-User', '-fontsize']
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
				elif argtarget == '-r':
					params['rules'] = argval
				elif argtarget == '-User':
					params['User'] = argval
				elif argtarget == '-fontsize':
					params['fontsize'] = float(argval)

	if params['fontsize'] < 1:
		print('>>> INVALID DATA received for fontsize. Restoring default value.')
		params['fontsize'] = 1

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('You\'ve received new or revised FactualVR enactments.')
	print('You want to check that they conform to specifications, equip yourself to spot any errors,')
	print('and finally to apply object combination and condition rules to derive recognizable objects')
	print('to be identified later in the pipeline.')
	print('')
	print('Usage:  python3 process_enactment.py <parameters, preceded by flags>')
	print(' e.g.:  python3 process_enactment.py -r substation.rules -e Enactment11 -e Enactment12 -v -render')
	print('')
	print('Flags:  -e       MUST HAVE AT LEAST ONE: Path to a directory of raw enactment materials: JSONs and color maps.')
	print('        -r       MUST HAVE EXACTLY ONE: Path to a rules file that tells the parser which objects are which.')
	print('')
	print('        -render  Generate illustrations (videos, 3D representations) for all given enactments.')
	print('        -v       Enable verbosity')
	print('        -?       Display this message')

if __name__ == '__main__':
	main()
