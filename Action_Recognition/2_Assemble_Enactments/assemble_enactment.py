import cv2
import json
import numpy as np
import os
import sys
import time

def main():
	params = getCommandLineParams()									#  Collect parameters
	if params['helpme'] or len(params['enactments']) == 0:
		usage()
		return
																	#  Build the flip matrix, once
	flip = np.array([[-1.0,  0.0, 0.0], \
	                 [ 0.0, -1.0, 0.0], \
	                 [ 0.0,  0.0, 1.0]], dtype='float64')

	ignore = ['LeftHand', 'RightHand']								#  These have been permitted in the *_groundtruth.txt files
																	#  but should not be considered now.
	classnames = {}													#  To become an alphabetically sorted list of (unique) strings
	for enactment in params['enactments']:
		#  Open your choice of OBJECT SOURCE: {*_groundtruth.txt,
		#                                      *_mask_rcnn_factual_0799_detections.txt}
		if os.path.exists(enactment + '_' + params['obj-src'] + '_detections.txt'):
			fh = open(enactment + '_' + params['obj-src'] + '_detections.txt', 'r')
		else:
			fh = open(enactment + '_groundtruth.txt', 'r')
		lines = fh.readlines()										#  Read in objects visible and their corresponding masks
		fh.close()
		for line in lines:
			if line[0] != '#':
				arr = line.strip().split('\t')

				timestamp = arr[0]									#  Kept as a string
				filename = arr[1]									#  string
				label = arr[2]										#  string
				object_class_name = arr[3]							#  string
				if object_class_name != '*':
					score = float(arr[4])							#  float
					bboxstr = arr[5]								#  string
					maskpath = arr[6]								#  string
																	#  Only do this work if something is there
				if object_class_name != '*' and object_class_name not in ignore:
					classnames[object_class_name] = True

	classnames = sorted(classnames.keys())							#  We only care about a unique set of keys
	if params['verbose']:
		print('ALL CLASSES from ALL ENACTMENTS:')
		for classname in classnames:
			print('\t' + classname)
		print('Prop sub-vector length = ' + str(len(classnames)))

	#################################################################
	##                  For all given enactments                   ##
	#################################################################

	for enactment in params['enactments']:
		if params['verbose']:
			print('')
			verbosestr = 'Building enactment file for ' + enactment
			print(verbosestr)
			print('=' * len(verbosestr))

		t1_start = time.process_time()								#  Start timer

		frames = {}													#  key:filename      string
																	#  val:['timestamp'] kept as a string
																	#      ['label']     string
																	#      ['left']      6-vec float
																	#      ['right']     6-vec float
																	#      ['objs']      [ (object, maskfile, bbox, centroid-xyz),
																	#                      (object, maskfile, bbox, centroid-xyz),
																	#                         ...
																	#                      (object, maskfile, bbox, centroid-xyz) ]
		#############################################################
		#  Open your choice                                         #
		#  of POSE SOURCE: {*_IKposes.bbox.shutoff.txt,             #
		#                   *_IKposes.cntr.shutoff.txt,             #
		#                   *_poses.full.txt,                       #
		#                   *_poses.shutoff.txt}                    #
		#############################################################
		if params['pose-src'] == 'ikbbox':
			fh = open(enactment + '_IKposes.bbox.shutoff.txt', 'r')
			if params['verbose']:
				print('>>> Reading hand poses from I.K. bounding-box file.')
		elif params['pose-src'] == 'ikcntr':
			fh = open(enactment + '_IKposes.cntr.shutoff.txt', 'r')
			if params['verbose']:
				print('>>> Reading hand poses from I.K. averaged pixel file.')
		elif params['pose-src'] == 'shutoff':
			fh = open(enactment + '_poses.shutoff.txt', 'r')
			if params['verbose']:
				print('>>> Reading hand poses from VR controller file with shutoffs.')
		else:
			fh = open(enactment + '_poses.full.txt', 'r')
			if params['verbose']:
				print('>>> Reading hand poses from VR controller file without shutoffs. Hands are considered "visible" in every frame.')

		lines = fh.readlines()										#  Read in poses
		fh.close()
		for line in lines:
			if line[0] != '#':
				arr = line.strip().split('\t')

				timestamp = arr[0]
				label = arr[1]
				imagefile = arr[2]

				LHx = float(arr[3])
				LHy = float(arr[4])
				LHz = float(arr[5])
				LH0 = int(arr[6])
				LH1 = int(arr[7])
				LH2 = int(arr[8])

				RHx = float(arr[9])
				RHy = float(arr[10])
				RHz = float(arr[11])
				RH0 = int(arr[12])
				RH1 = int(arr[13])
				RH2 = int(arr[14])

				frames[imagefile] = {}
				frames[imagefile]['timestamp'] = timestamp
				frames[imagefile]['label'] = label
				frames[imagefile]['left'] = (LHx, LHy, LHz, LH0, LH1, LH2)
				frames[imagefile]['right'] = (RHx, RHy, RHz, RH0, RH1, RH2)
				frames[imagefile]['objs'] = []

		#############################################################
		##               Read camera and depth data                ##
		#############################################################
		fh = open(enactment + '/Users/' + params['User'] + '/POV/CameraIntrinsics.fvr', 'r')
		line = fh.readlines()[0]
		fh.close()
		camdata = json.loads(line)
		fov = float(camdata['fov'])									#  We don't use camdata['focalLength'] anymore because IT'S IN MILLIMETERS
		if params['focal'] is None:									#  IN PIXELS!!!!
			focalLength = params['imgh'] * 0.5 / np.tan(fov * np.pi / 180.0)
		else:
			focalLength = params['focal']
		K = np.array([[focalLength, 0.0, params['imgw'] * 0.5], \
		              [0.0, focalLength, params['imgh'] * 0.5], \
		              [0.0,         0.0,           1.0       ]])
		K_inv = np.linalg.inv(K)									#  Build inverse K-matrix
		if params['verbose']:
			print('>>> Stated camera focal length: ' + str(camdata['focalLength']) + ' mm')
			print('    Computed camera focal length: ' + str(focalLength) + ' pixels')
			print('K = ')
			print(K)

		fh = open(enactment + '/metadata.fvr', 'r')
		line = fh.readlines()[0]
		fh.close()
		metadata = json.loads(line)
		depthRange = {}
		depthRange['min'] = metadata['depthImageRange']['x']		#  Save Z-map's minimum (in meters)
		depthRange['max'] = metadata['depthImageRange']['y']		#  Save Z-map's maximum (in meters)
		if params['verbose']:
			print('>>> Stated depth range: [' + str(depthRange['min']) + ', ' + str(depthRange['max']) + ']')

		#############################################################
		#  Open your choice of                                      #
		#  OBJECT SOURCE: {*_groundtruth.txt,                       #
		#                  *_mask_rcnn_factual_0799_detections.txt} #
		#############################################################
		if os.path.exists(enactment + '_' + params['obj-src'] + '_detections.txt'):
			fh = open(enactment + '_' + params['obj-src'] + '_detections.txt', 'r')
			if params['verbose']:
				print('>>> Reading object detections recorded by "' + params['obj-src'] + '".')
		else:
			fh = open(enactment + '_groundtruth.txt', 'r')
			if params['verbose']:
				print('>>> Reading object detections from ground truth.')
		lines = fh.readlines()										#  Read in objects visible and their corresponding masks
		fh.close()
		for line in lines:
			if line[0] != '#':
				arr = line.strip().split('\t')

				timestamp = arr[0]									#  Kept as a string
				filename = arr[1]									#  string
				label = arr[2]										#  string
				object_class_name = arr[3]							#  string
				score = arr[4]										#  float
				bboxstr = arr[5]									#  string
				maskpath = arr[6]									#  string
																	#  Only do this work if something is there
				if object_class_name != '*' and object_class_name not in ignore:
					score = float(score)							#  Convert to float

					bbox = bboxstr.split(';')
					bbox = ( int(bbox[0].split(',')[0]), int(bbox[0].split(',')[1]), int(bbox[1].split(',')[0]), int(bbox[1].split(',')[1]) )
					#################################################  16may21: why are there blank masks in MRCNN output!??
					#################################################  It detected something--what?--with no pixels!???!!?!??!
					mask = cv2.imread(maskpath, cv2.IMREAD_UNCHANGED)
					indices = np.where(mask == 255)

					if len(indices[0]) >= params['minimum-pixels']:	#  Does this object occupy the minimal amount of pixels?
																	#  Use the bounding box to compute the centroid
						if params['centroid-mode'] == 'bbox' or len(indices[0]) == 0:
							x = int(round(float(bbox[0] + bbox[2]) * 0.5))
							y = int(round(float(bbox[1] + bbox[3]) * 0.5))
						else:										#  Use the object mask to compute the centroid
							x = int(round(np.mean(indices[1])))
							y = int(round(np.mean(indices[0])))
						depthmap = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/DepthMapCameraFrames/' + filename, cv2.IMREAD_UNCHANGED)
						z = depthRange['min'] + (float(depthmap[y][x]) / 255.0) * float(depthRange['max'] - depthRange['min'])
						if z < depthRange['max']:					#  Do NOT consider points that are (effectively) "infinitely" far away
																	#  Build a 3D point
							centroid = np.dot(K_inv, np.array([x, y, 1.0]))
							centroid *= z							#  Scale by known depth from the camera/head (at [0, 0, 0])

							pt = np.dot(flip, centroid)				#  Flip dots

							#pt_x = frames[timestamp]['head'][3][:, 0].dot(pt)
							#pt_y = frames[timestamp]['head'][3][:, 1].dot(pt)
							#pt_z = frames[timestamp]['head'][3][:, 2].dot(pt)
																	#  Append an object-name-and-3D-centroid
							#frames[timestamp]['objs'].append( (object_class_name, (pt_x, pt_y, pt_z)) )
							frames[filename]['objs'].append( (object_class_name, maskpath, bbox, (pt[0], pt[1], pt[2]), (x, y)) )

																	#  ELSE: The centroid is (effectively) "infinitely" far away

		#############################################################  Write to file
		if params['verbose']:
			print('>>> Writing ' + enactment + '.enactment')
		fh = open(enactment + '.enactment', 'w')
		fh.write('#  Enactment vectors derived from FactualVR enactment materials.\n')
		fh.write('#\n')
		if params['pose-src'] == 'ikbbox':
			fh.write('#  This file used IK hand poses and hand bounding-box centroids.\n')
			fh.write('#  Hand vectors drop out to [0.0, 0.0, 0.0, 0, 0, 0] when the hand is not visible.\n')
		elif params['pose-src'] == 'ikcntr':
			fh.write('#  This file used IK hand poses and average hand-pixel centroids.\n')
			fh.write('#  Hand vectors drop out to [0.0, 0.0, 0.0, 0, 0, 0] when the hand is not visible.\n')
		elif params['pose-src'] == 'shutoff':
			fh.write('#  This file used controller-based hand poses.\n')
			fh.write('#  Hand vectors drop out to [0.0, 0.0, 0.0, 0, 0, 0] when the hand centroid does not project into the camera.\n')
		else:
			fh.write('#  This file used controller-based hand poses.\n')
			fh.write('#  Hand vectors remain regardless of whether the hand centroid projects into the camera.\n')
		if os.path.exists(enactment + '_' + params['obj-src'] + '_detections.txt'):
			fh.write('#  This file used object detections made by Mask-RCNN network "' + params['obj-src'] + '".\n')
		else:
			fh.write('#  This file used ground-truth object detections.\n')
		fh.write('#\n')
		fh.write('#  Minimum number of pixels for an object to count as present: ' + str(params['minimum-pixels']) + '\n')
		fh.write('#\n')
		fh.write('#  Camera focal length: ' + str(focalLength) + ' px\n')
		fh.write('#\n')
		fh.write('#  GAUSSIAN:\n')
		fh.write('#    Mu_gaze_x = ' + str(params['muX']) + '\n')
		fh.write('#    Mu_gaze_y = ' + str(params['muY']) + '\n')
		fh.write('#    Mu_gaze_z = ' + str(params['muZ']) + '\n')
		fh.write('#    Sig_gaze_x = ' + str(params['sigX_gaze']) + '\n')
		fh.write('#    Sig_gaze_y = ' + str(params['sigY_gaze']) + '\n')
		fh.write('#    Sig_gaze_z = ' + str(params['sigZ_gaze']) + '\n')
		fh.write('#    Sig_hand_x = ' + str(params['sigX_hand']) + '\n')
		fh.write('#    Sig_hand_y = ' + str(params['sigY_hand']) + '\n')
		fh.write('#    Sig_hand_z = ' + str(params['sigZ_hand']) + '\n')
		fh.write('#\n')
		fh.write('#  CLASSES:\n')
		fh.write('#    ' + ' '.join(classnames) + '\n')
		fh.write('#\n')
		objstr = ''
		for i in range(0, len(classnames)):
			objstr += '   Obj' + str(i)
		fh.write('#  timestamp   filename   label   LHx   LHy   LHz   LH0   LH1   LH2   RHx   RHy   RHz   RH0   RH1   RH2' + objstr + '\n')
		tuple_ctr = 0												#  I'd like to know how many rows/tuples were involved in this writing.
		for k, v in sorted(frames.items(), key=lambda x: int(x[0].split('_')[0])):
			fh.write(v['timestamp'] + '\t')							#  Write the timestamp
			fh.write(k + '\t')										#  Write the file name
			fh.write(v['label'] + '\t')								#  Write the action label
			fh.write('\t'.join([str(x) for x in v['left']]) + '\t')	#  Write the left-hand sub-vector
			fh.write('\t'.join([str(x) for x in v['right']]) + '\t')#  Write the right-hand sub-vector

			if 1 in v['left'][3:]:									#  Left hand is "live"
				LHcentroid = v['left'][:3]
			else:
				LHcentroid = None
			if 1 in v['right'][3:]:									#  Right hand is "live"
				RHcentroid = v['right'][:3]
			else:
				RHcentroid = None
																	#  Initialize an object sub-vector
			objvec = [0.0 for i in range(0, len(classnames))]
			tuple_ctr += len(v['objs'])
			for obj in v['objs']:									#  For every (object-name, maskpath, bbox, 3D-centroid, 2D-centroid) tuple for this frame...
				g = weigh(obj[3], LHcentroid, RHcentroid, params)	#  Weigh centroids by 3D Gaussian
				if g > objvec[ classnames.index(obj[0]) ]:			#  If it's greater than what was there before, update it
					objvec[ classnames.index(obj[0]) ] = g

			fh.write('\t'.join([str(x) for x in objvec]) + '\n')	#  Write the object sub-vector

		fh.close()

		t1_stop = time.process_time()								#  Stop timer

		if params['verbose']:
			print('    Writing ' + enactment + '.enactment took ' + str(t1_stop - t1_start) + ' seconds ' + \
			      'for ' + str(len(frames)) + ' frames, ' + str(tuple_ctr) + ' tuples')

		#############################################################  Rendering video?
		if params['render']:
			if params['verbose']:
				if params['obj-src'] == 'gt':
					print('>>> Rendering ' + enactment + '.avi')
				else:
					print('>>> Rendering ' + enactment + '_' + params['obj-src'] + '.avi')

			t1_start = time.process_time()							#  Start timer

			if params['obj-src'] == 'gt':
				vid = cv2.VideoWriter(enactment + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (params['imgw'], params['imgh']) )
			else:
				vid = cv2.VideoWriter(enactment + '_' + params['obj-src'] + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (params['imgw'], params['imgh']) )

			#########################################################
			#           For each frame in rendered enactment        #
			#########################################################

			for k, v in sorted(frames.items(), key=lambda x: int(x[0].split('_')[0])):
				img = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/' + k, cv2.IMREAD_UNCHANGED)
				if img.shape[2] == 4:								#  Drop alpha channel
					img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

				#####################################################
				#                Add masks/overlays                 #
				#####################################################

				maskcanvas = np.zeros((params['imgh'], params['imgw'], 3), dtype='uint8')
				for obj in v['objs']:								#  For every (object-name, maskpath, bbox, 3D-centroid, 2D-centroid) tuple for this frame...
					g = weigh(obj[3], LHcentroid, RHcentroid, params)
					mask = cv2.imread(obj[1], cv2.IMREAD_UNCHANGED)	#  Open the mask file
					mask[mask > 1] = 1								#  All things greater than 1 become 1
																	#  Extrude to four channels
					mask = mask[:, :, None] * np.ones(3, dtype='uint8')[None, None, :]
																	#  Convert this to a (weighted) graphical overlay
					mask[:, :, 0] *= int(round(params['colors'][ obj[0] ][2] * g))
					mask[:, :, 1] *= int(round(params['colors'][ obj[0] ][1] * g))
					mask[:, :, 2] *= int(round(params['colors'][ obj[0] ][0] * g))
																	#  Draw an unweighted bounding box
					cv2.rectangle(mask, (obj[2][0], obj[2][1]), (obj[2][2], obj[2][3]), (params['colors'][ obj[0] ][2], \
					                                                                     params['colors'][ obj[0] ][1], \
					                                                                     params['colors'][ obj[0] ][0]), 1)

					maskcanvas += mask								#  Add mask to mask accumulator
					maskcanvas[maskcanvas > 255] = 255				#  Clip accumulator to 255

				img = cv2.addWeighted(img, 1.0, maskcanvas, 0.7, 0)	#  Add mask accumulator to source frame
				img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)			#  Flatten alpha

				for obj in v['objs']:								#  Again, for every (object-name, maskpath, bbox, 3D-centroid, 2D-centroid) tuple for this frame...
					cv2.circle(img, obj[4], 5, (params['colors'][ obj[0] ][2], \
					                            params['colors'][ obj[0] ][1], \
					                            params['colors'][ obj[0] ][0]), 3)

				#####################################################  Add text overlays and hand-projections
				if v['label'] == '*':								#  Write <NEUTRAL>
					cv2.putText(img, '<NEUTRAL>', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (255, 255, 255, 255), 3)
				else:												#  Write the action
					cv2.putText(img, v['label'], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (255, 255, 255, 255), 3)
				if 1 in v['left'][3:]:								#  Left hand is "live"
					LHcentroid = v['left'][:3]
					LHstate = v['left'][3:].index(1)
				else:
					LHcentroid = None
				if 1 in v['right'][3:]:								#  Right hand is "live"
					RHcentroid = v['right'][:3]
					RHstate = v['right'][3:].index(1)
				else:
					RHcentroid = None
				if 1 in v['left'][3:]:								#  Write LH sub-vector and project LH into camera
					LHproj = np.dot(K, np.array(v['left'][:3]))
					LHproj /= LHproj[2]
					x = int(round(LHproj[0]))
					y = int(round(LHproj[1]))
					cv2.circle(img, (params['imgw'] - x, params['imgh'] - y), 5, (0, 255, 0, 255), 3)
					cv2.putText(img, 'Left Hand: ' + ' '.join(["{:.2f}".format(val) for val in LHcentroid]) + ' ' + str(LHstate), \
					            (10, params['imgh'] - 160), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (0, 255, 0, 255), 3)
				if 1 in v['right'][3:]:								#  Write RH sub-vector and project RH into camera
					RHproj = np.dot(K, np.array(v['right'][:3]))
					RHproj /= RHproj[2]
					x = int(round(RHproj[0]))
					y = int(round(RHproj[1]))
					cv2.circle(img, (params['imgw'] - x, params['imgh'] - y), 5, (0, 0, 255, 255), 3)
					cv2.putText(img, 'Right Hand: ' + ' '.join(["{:.2f}".format(val) for val in RHcentroid]) + ' ' + str(RHstate), \
					            (10, params['imgh'] - 40), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (0, 0, 255, 255), 3)

				vid.write(img)										#  Write the frame to the current video

			vid.release()

			t1_stop = time.process_time()							#  Stop timer

			if params['verbose']:
				print('    Rendering ' + enactment + '.avi took ' + str(t1_stop - t1_start) + ' seconds')

	return

#  obj_centroid is a Numpy array
#  LH_centroid is a Numpy array or None
#  RH_centroid is a Numpy array or None
def weigh(obj_centroid, LH_centroid, RH_centroid, params):
	x = np.array([obj_centroid[0] - params['muX'], \
	              obj_centroid[1] - params['muY'], \
	              obj_centroid[2] - params['muZ']])
	C = np.array([[params['sigX_gaze'] * params['sigX_gaze'], 0.0, 0.0], \
	              [0.0, params['sigY_gaze'] * params['sigY_gaze'], 0.0], \
	              [0.0, 0.0, params['sigZ_gaze'] * params['sigZ_gaze']]])
	C_inv = np.linalg.inv(C)
	f_head = np.exp(-0.5 * np.dot(np.dot(x.T, C_inv), x))

	if LH_centroid is not None:
		x = np.array([obj_centroid[0] - LH_centroid[0], \
		              obj_centroid[1] - LH_centroid[1], \
		              obj_centroid[2] - LH_centroid[2]])
		C = np.array([[params['sigX_hand'] * params['sigX_hand'], 0.0, 0.0], \
		              [0.0, params['sigY_hand'] * params['sigY_hand'], 0.0], \
		              [0.0, 0.0, params['sigZ_hand'] * params['sigZ_hand']]])
		C_inv = np.linalg.inv(C)
		f_Lhand = np.exp(-0.5 * np.dot(np.dot(x.T, C_inv), x))
	else:
		f_Lhand = 0.0

	if RH_centroid is not None:
		x = np.array([obj_centroid[0] - RH_centroid[0], \
		              obj_centroid[1] - RH_centroid[1], \
		              obj_centroid[2] - RH_centroid[2]])
		C = np.array([[params['sigX_hand'] * params['sigX_hand'], 0.0, 0.0], \
		              [0.0, params['sigY_hand'] * params['sigY_hand'], 0.0], \
		              [0.0, 0.0, params['sigZ_hand'] * params['sigZ_hand']]])
		C_inv = np.linalg.inv(C)
		f_Rhand = np.exp(-0.5 * np.dot(np.dot(x.T, C_inv), x))
	else:
		f_Rhand = 0.0

	return max(f_head, f_Lhand, f_Rhand)

def getCommandLineParams():
	params = {}
	params['enactments'] = []										#  List of file paths
	params['verbose'] = False
	params['helpme'] = False

	params['minimum-pixels'] = 40									#  Minimum number of pixels necessary for an object to count as present

	params['render'] = False										#  Whether to render stuff.
	params['colors'] = None

	params['muX'] = 0.0												#  Centered on the head
	params['muY'] = 0.0
	params['muZ'] = 0.0

	params['sigX_gaze'] = 2.0										#  IN METERS
	params['sigY_gaze'] = 1.5
	params['sigZ_gaze'] = 3.0

	#  No mus for hands: their centers are hand-centroids
	params['sigX_hand'] = 0.5										#  IN METERS
	params['sigY_hand'] = 0.5
	params['sigZ_hand'] = 0.5

	params['pose-src'] = 'ikcntr'
	params['obj-src'] = 'gt'
	params['centroid-mode'] = 'avg'
	params['focal'] = None											#  Focal-length override

	params['imgw'] = 1280											#  It used to be 1920, and I don't want to change a bunch of ints when it changes again
	params['imgh'] = 720											#  It used to be 1080, and I don't want to change a bunch of ints when it changes again
	params['focal'] = None											#  Can be used to override the focal length we otherwise derive from metadata
	params['fontsize'] = 1											#  For rendering text to images and videos
	params['User'] = 'vr1'											#  It used to be "admin", and I don't want to change a bunch of file paths when it changes again

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-e', \
	         '-render', '-color', '-colors', '-minpx', \
	         '-mu', '-sigHead', '-sigHand', '-focal', '-pose', '-obj', '-centroid', \
	         '-v', '-?', '-help', '--help', \
	         '-imgw', '-imgh', '-User', '-fontsize']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-v':
				params['verbose'] = True
			elif sys.argv[i] == '-f':
				params['force'] = True
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
					params['muX'] = float(argval)
					argtarget = '-muY'
				elif argtarget == '-muY':
					params['muY'] = float(argval)
					argtarget = '-muZ'
				elif argtarget == '-muZ':
					params['muZ'] = float(argval)

				elif argtarget == '-sigHead':
					params['sigX_gaze'] = float(argval)
					argtarget = '-sigHeadY'
				elif argtarget == '-sigHeadY':
					params['sigY_gaze'] = float(argval)
					argtarget = '-sigHeadZ'
				elif argtarget == '-sigHeadZ':
					params['sigZ_gaze'] = float(argval)

				elif argtarget == '-sigHand':
					params['sigX_hand'] = float(argval)
					argtarget = '-sigHandY'
				elif argtarget == '-sigHandY':
					params['sigY_hand'] = float(argval)
					argtarget = '-sigHandZ'
				elif argtarget == '-sigHandZ':
					params['sigZ_hand'] = float(argval)

				elif argtarget == '-minpx':
					params['minimum-pixels'] = max(1, abs(int(argval)))

				elif argtarget == '-focal':
					params['focal'] = float(argval)
				elif argtarget == '-pose':
					params['pose-src'] = argval
				elif argtarget == '-obj':
					params['obj-src'] = argval
				elif argtarget == '-centroid':
					params['centroid-mode'] = argval
				elif argtarget == '-color' or argtarget == '-colors':
					params['colors'] = {}
					fh = open(argval, 'r')
					for line in fh.readlines():
						if line[0] != '#':
							arr = line.strip().split('\t')
							params['colors'][arr[0]] = [int(arr[1]), int(arr[2]), int(arr[3])]
					fh.close()

				elif argtarget == '-imgw':
					params['imgw'] = int(argval)
				elif argtarget == '-imgh':
					params['imgh'] = int(argval)
				elif argtarget == '-User':
					params['User'] = argval
				elif argtarget == '-fontsize':
					params['fontsize'] = float(argval)

	if params['imgw'] < 1 or params['imgh'] < 1:
		print('>>> INVALID DATA received for image dimensions. Restoring default values.')
		params['imgw'] = 1280
		params['imgh'] = 720

	if params['pose-src'] not in ['ikbbox', 'ikcntr', 'shutoff', 'full']:
		print('>>> INVALID DATA received for pose source. Restoring default value.')
		params['pose-src'] = 'ikcntr'

	if params['obj-src'] != 'gt':
		found = True
		for enactment in params['enactments']:
			if not os.path.exists(enactment + '_' + params['obj-src'] + '_detections.txt'):
				print('>>> INVALID DATA: file not found "' + enactment + '_' + params['obj-src'] + '_detections.txt".')
				found = False

		if not found:
			print('>>> INVALID DATA received for object source. Restoring default value.')
			params['obj-src'] = 'gt'

	if params['centroid-mode'] not in ['bbox', 'avg']:
		print('>>> INVALID DATA received for centroid source. Restoring default value.')
		params['centroid-mode'] = 'avg'

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('You\'ve run "process_enactment.py" and produced a ton of source files.')
	print('Now you want to make classifier-ready or DTW-ready files from these materials.')
	print('')
	print('This script produces an *.enactment file for each given enactment.')
	print('')
	print('Usage:  python3.5 assemble_enactment.py <parameters, preceded by flags>')
	print(' e.g.:  python3.5 assemble_enactment.py -e BackBreaker1 -e Enactment1 -e Enactment2 -e Enactment3 -e Enactment4 -e Enactment5 -e Enactment6 -e Enactment7 -e Enactment9 -e Enactment10 -e Enactment11 -e Enactment12 -e MainFeederBox1 -e Regulator1 -e Regulator2 -minpx 100 -v -render -colors colors_gt.txt')
	print(' e.g.:  python3.5 assemble_enactment.py -e BackBreaker1 -e Enactment1 -e Enactment2 -e Enactment3 -e Enactment4 -e Enactment5 -e Enactment6 -e Enactment7 -e Enactment9 -e Enactment10 -e Enactment11 -e Enactment12 -e MainFeederBox1 -e Regulator1 -e Regulator2 -minpx 200 -obj mask_rcnn_factual_0028.h5 -v -render -colors colors_train.txt')
	print(' e.g.:  python3.5 assemble_enactment.py -e Enactment11 -e Enactment12 -obj mask_rcnn_factual_0028.h5 -v')
	print('')
	print('Flags:  -e          MUST HAVE AT LEAST ONE: Path to a directory of raw enactment materials: JSONs and color maps.')
	print('        -mu         Following three arguments are the X, Y, and Z coordinates of the Gaussian peak.')
	print('                    Defaults are 0.0, 0.0, and 0.0.')
	print('        -sigHead    Following three arguments are the standard deviations for the gaze along X, Y, and Z axes.')
	print('                    Defaults are 2.0, 1.5, and 3.0.')
	print('        -sigHand    Following three arguments are the standard deviations for the hands along X, Y, and Z axes.')
	print('                    Defaults are 0.5, 0.5, and 0.5.')
	print('        -minpx      Following argument sets the minimum number of pixels an object must occupy to count as present.')
	print('                    Default is 40.')
	print('        -focal      Following argument will override the focal length computed from metadata.')
	print('        -render     Generate illustrations (videos, 3D representations) for all given enactments.')
	print('        -v          Enable verbosity')
	print('        -?          Display this message')

if __name__ == '__main__':
	main()