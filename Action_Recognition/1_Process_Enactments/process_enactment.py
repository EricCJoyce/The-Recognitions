import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
import time

def main():
	params = getCommandLineParams()									#  Collect parameters
	if params['helpme'] or len(params['enactments']) == 0 or params['rules'] is None:
		usage()
		return

	if params['verbose']:
		print('')													#  Space sections out
		print('#####  Check')
	t1_start = time.process_time()
	check(params)													#  Run checks on the given enactments
	t1_stop = time.process_time()
	if params['verbose']:
		print('  Took ' + str(t1_stop - t1_start) + ' seconds')

	if params['verbose']:
		print('')													#  Space sections out
		print('#####  Poses')
	t1_start = time.process_time()
	build_poses(params)												#  Build *_poses.{full,shutoff}.txt files (and maybe a video and centipedes) for each enactment
	t1_stop = time.process_time()
	if params['verbose']:
		print('  Took ' + str(t1_stop - t1_start) + ' seconds')

	if params['verbose']:
		print('')													#  Space sections out
		print('#####  Loading rules')
	t1_start = time.process_time()
	rules = load_rules(params)										#  Build the pan-enactments file parse.txt
	base_classes = compute_base_classes(rules)
	t1_stop = time.process_time()
	if params['verbose']:
		print('  Took ' + str(t1_stop - t1_start) + ' seconds')

	if params['verbose']:
		print('')													#  Space sections out
		print('#####  Sub-props')
	t1_start = time.process_time()
	classes = build_subprops(base_classes, rules, params)			#  Build a *_subprops.txt file
	t1_stop = time.process_time()
	if params['verbose']:
		print('  Took ' + str(t1_stop - t1_start) + ' seconds')

	if params['verbose']:
		print('')													#  Space sections out
		print('#####  Ground-truth')
	t1_start = time.process_time()
	build_groundtruths(params, classes, rules)						#  Build a *_groundtruth.txt file and many mask pngs in /gt for each enactment
	t1_stop = time.process_time()
	if params['verbose']:
		print('  Took ' + str(t1_stop - t1_start) + ' seconds')

	if params['verbose']:
		print('')													#  Space sections out
		print('#####  Poses from IK hands')
	t1_start = time.process_time()
	build_IKposes(params)											#  Build a *_IKposes.shutoff.txt files (and maybe a video and centipedes) for each enactment
	t1_stop = time.process_time()
	if params['verbose']:
		print('  Took ' + str(t1_stop - t1_start) + ' seconds')

	return

#  Build hand data on the assumption that the centroids of the visible (rendered from the IK puppet) hands
#  are the hands' true positions.
def build_IKposes(params):
	for enactment in params['enactments']:							#  For each enactment...

		fh = open(enactment + '/Users/' + params['User'] + '/POV/CameraIntrinsics.fvr', 'r')
		line = fh.readlines()[0]
		fh.close()
		camdata = json.loads(line)
		fov = float(camdata['fov'])									#  We don't use camdata['focalLength'] anymore because IT'S IN MILLIMETERS
		if params['focal'] is None:									#  IN PIXELS!!!! ALSO: NOTE THAT THIS IS THE **VERTICAL** F.o.V. !!
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
																	#  Build the flip matrix
		flip = np.array([[-1.0,  0.0, 0.0], \
		                 [ 0.0, -1.0, 0.0], \
		                 [ 0.0,  0.0, 1.0]], dtype='float64')

		fh = open(enactment + '/metadata.fvr', 'r')
		line = fh.readlines()[0]
		fh.close()
		metadata = json.loads(line)
		depthRange = {}
		depthRange['min'] = metadata['depthImageRange']['x']		#  Save Z-map's minimum (in meters)
		depthRange['max'] = metadata['depthImageRange']['y']		#  Save Z-map's maximum (in meters)
		if params['verbose']:
			print('>>> Stated depth range: [' + str(depthRange['min']) + ', ' + str(depthRange['max']) + ']')

																	#  Retrieve head poses: a valid datum must have a head pose
		fh = open(enactment + '/Users/' + params['User'] + '/Head.fvr', 'r')
		line = fh.readlines()[0]									#  Single line
		fh.close()
		headdata = json.loads(line)
																	#  If *_poses.shutoff.txt files either do not exist or we are forcing...
		if not os.path.exists(enactment + '_IKposes.bbox.shutoff.txt') or \
		   not os.path.exists(enactment + '_IKposes.cntr.shutoff.txt') or \
		   params['force']:
			if params['verbose']:
				verbosestr = '** Scanning ' + enactment + ' **'
				print('*' * len(verbosestr))
				print(verbosestr)
				print('*' * len(verbosestr))

			fh = open(enactment + '_poses.full.txt', 'r')			#  Load the full-poses file: structure the timeline to be built here
			lines = fh.readlines()
			fh.close()

			poses = {}												#  Key: string-file-name
																	#  Val: {timestamp, label, left-json, right-json,
																	#        head-x, head-y, head-z, head-R,
																	#        left-bbox, right-bbox, left-avg, right-avg}
			for line in lines:
				if line[0] != '#':
					arr = line.strip().split('\t')

					timestamp = arr[0]								#  leave it as a string
					label = arr[1]									#  string
					imagefilename = arr[2]							#  string

					lefthand_x = float(arr[3])						#  float, float, float, int, int, int
					lefthand_y = float(arr[4])
					lefthand_z = float(arr[5])
					lefthand_0 = int(arr[6])
					lefthand_1 = int(arr[7])
					lefthand_2 = int(arr[8])

					righthand_x = float(arr[9])						#  float, float, float, int, int, int
					righthand_y = float(arr[10])
					righthand_z = float(arr[11])
					righthand_0 = int(arr[12])
					righthand_1 = int(arr[13])
					righthand_2 = int(arr[14])

					poses[imagefilename] = {}
					poses[imagefilename]['timestamp'] = timestamp
					poses[imagefilename]['label'] = label
					poses[imagefilename]['left-json'] = (lefthand_x, lefthand_y, lefthand_z, lefthand_0, lefthand_1, lefthand_2)
					poses[imagefilename]['right-json'] = (righthand_x, righthand_y, righthand_z, righthand_0, righthand_1, righthand_2)

			fh = open(enactment + '_poses.head.txt', 'r')
			lines = fh.readlines()
			fh.close()
			for line in lines:
				if line[0] != '#':
					arr = line.strip().split('\t')
					timestamp = arr[0]
					label = arr[1]
					imagefilename = arr[2]
					x = float(arr[3])
					y = float(arr[4])
					z = float(arr[5])
					R = np.array([[float(arr[6]),  float(arr[7]),  float(arr[8]) ], \
					              [float(arr[9]),  float(arr[10]), float(arr[11])], \
					              [float(arr[12]), float(arr[13]), float(arr[14])]])

					poses[imagefilename]['head-x'] = x
					poses[imagefilename]['head-y'] = y
					poses[imagefilename]['head-z'] = z
					poses[imagefilename]['head-R'] = R

			fh = open(enactment + '_groundtruth.txt', 'r')
			lines = fh.readlines()
			fh.close()
			for line in lines:
				if line[0] != '#':
					arr = line.strip().split('\t')
					timestamp = arr[0]								#  leave it as a string
					imagefilename = arr[1]							#  string
					label = arr[2]									#  string
					object_class_name = arr[3]						#  string
					score = float(arr[4])							#  float
					bboxstr = arr[5]								#  string
					maskfilename = arr[6]							#  includes enactment name, like enactment/gt/mask_0.png

					if object_class_name == 'LeftHand' or object_class_name == 'RightHand':
						mask = cv2.imread(maskfilename, cv2.IMREAD_UNCHANGED)
						dmap = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/DepthMapCameraFrames/' + imagefilename, cv2.IMREAD_UNCHANGED)
						indices = np.where(mask == 255)				#  Compute the bounding box for this mask
						upperleft_x = np.min(indices[1])
						upperleft_y = np.min(indices[0])
						lowerright_x = np.max(indices[1])
						lowerright_y = np.max(indices[0])

						bboxcenter = (int(round((upperleft_x + lowerright_x) * 0.5)), int(round((upperleft_y + lowerright_y) * 0.5)))
						avgcenter = (int(round( np.mean(indices[1]) )), int(round( np.mean(indices[0]) )))

						if object_class_name == 'LeftHand':
																	#  In meters
							z = lookupDepth(dmap, bboxcenter[0], bboxcenter[1], depthRange)
							centroid = np.dot(K_inv, np.array([bboxcenter[0], bboxcenter[1], 1.0]))
							centroid *= z							#  Scale by known depth (meters from head)
							pt = np.dot(flip, centroid)				#  Flip dots
							poses[imagefilename]['left-bbox'] = ( pt[0], pt[1], pt[2], \
							                                      poses[imagefilename]['left-json'][3], \
							                                      poses[imagefilename]['left-json'][4], \
							                                      poses[imagefilename]['left-json'][5] )

							z = lookupDepth(dmap, avgcenter[0], avgcenter[1], depthRange)
							centroid = np.dot(K_inv, np.array([avgcenter[0], avgcenter[1], 1.0]))
							centroid *= z							#  Scale by known depth (meters from head)
							pt = np.dot(flip, centroid)				#  Flip dots
							poses[imagefilename]['left-avg'] = ( pt[0], pt[1], pt[2], \
							                                     poses[imagefilename]['left-json'][3], \
							                                     poses[imagefilename]['left-json'][4], \
							                                     poses[imagefilename]['left-json'][5] )

						elif object_class_name == 'RightHand':
																	#  In meters
							z = lookupDepth(dmap, bboxcenter[0], bboxcenter[1], depthRange)
							centroid = np.dot(K_inv, np.array([bboxcenter[0], bboxcenter[1], 1.0]))
							centroid *= z							#  Scale by known depth (meters from head)
							pt = np.dot(flip, centroid)				#  Flip dots
							poses[imagefilename]['right-bbox'] = ( pt[0], pt[1], pt[2], \
							                                       poses[imagefilename]['right-json'][3], \
							                                       poses[imagefilename]['right-json'][4], \
							                                       poses[imagefilename]['right-json'][5] )

							z = lookupDepth(dmap, avgcenter[0], avgcenter[1], depthRange)
							centroid = np.dot(K_inv, np.array([avgcenter[0], avgcenter[1], 1.0]))
							centroid *= z							#  Scale by known depth (meters from head)
							pt = np.dot(flip, centroid)				#  Flip dots
							poses[imagefilename]['right-avg'] = ( pt[0], pt[1], pt[2], \
							                                      poses[imagefilename]['right-json'][3], \
							                                      poses[imagefilename]['right-json'][4], \
							                                      poses[imagefilename]['right-json'][5] )

			fh = open(enactment + '_IKposes.bbox.shutoff.txt', 'w')
			fh.write('#  timestamp, label, image-file, LHx, LHy, LHz, LHstate_0, LHstate_1, LHstate_2, RHx, RHy, RHz, RHstate_0, RHstate_1, RHstate_2\n')
			fh.write('#  All hand positions are computed using the hands\' bounding box centroids and made relative to the head in the head\'s frame.\n')
			fh.write('#  In this file, hand vectors are set to zero if the hand does not project into the camera!\n')
			fh.write('#  Obviously, since the hand centroids come from the hand segmentation masks.\n')
			for k, v in sorted(poses.items(), key=lambda x: int(x[0].split('_')[0])):
				fh.write(v['timestamp'] + '\t')
				fh.write(v['label'] + '\t')
				fh.write(k + '\t')
				if 'left-bbox' in v:
					fh.write('\t'.join([str(x) for x in v['left-bbox']]) + '\t')
				else:
					fh.write('\t'.join(['0.0', '0.0', '0.0', '0', '0', '0']) + '\t')
				if 'right-bbox' in v:
					fh.write('\t'.join([str(x) for x in v['right-bbox']]) + '\n')
				else:
					fh.write('\t'.join(['0.0', '0.0', '0.0', '0', '0', '0']) + '\n')
			fh.close()

			fh = open(enactment + '_IKposes.cntr.shutoff.txt', 'w')
			fh.write('#  timestamp, label, image-file, LHx, LHy, LHz, LHstate_0, LHstate_1, LHstate_2, RHx, RHy, RHz, RHstate_0, RHstate_1, RHstate_2\n')
			fh.write('#  All hand positions are computed using the hands\' bounding box centroids and made relative to the head in the head\'s frame.\n')
			fh.write('#  In this file, hand vectors are set to zero if the hand does not project into the camera!\n')
			fh.write('#  Obviously, since the hand centroids come from the hand segmentation masks.\n')
			for k, v in sorted(poses.items(), key=lambda x: int(x[0].split('_')[0])):
				fh.write(v['timestamp'] + '\t')
				fh.write(v['label'] + '\t')
				fh.write(k + '\t')
				if 'left-avg' in v:
					fh.write('\t'.join([str(x) for x in v['left-avg']]) + '\t')
				else:
					fh.write('\t'.join(['0.0', '0.0', '0.0', '0', '0', '0']) + '\t')
				if 'right-avg' in v:
					fh.write('\t'.join([str(x) for x in v['right-avg']]) + '\n')
				else:
					fh.write('\t'.join(['0.0', '0.0', '0.0', '0', '0', '0']) + '\n')
			fh.close()

			#########################################################  Making centipedes and videos?
			if params['render']:

				#  For the 'skeleton' PLYs, there will be a single centipede, entirely connected through TIME.
				#    There will be no redundant vertices.

				#####################################################  Write the left-hand/right-hand one using BBox centroids
				totalV = len([x for x in poses.values() if 'left-bbox' in x]) + \
				         len([x for x in poses.values() if 'right-bbox' in x]) + \
				         len(poses)
				totalE = len([x for x in poses.values() if 'left-bbox' in x]) + \
				         len([x for x in poses.values() if 'right-bbox' in x]) + \
				         len(poses) - 1

				fh = open(enactment + '.IK.bbox.ply', 'w')
				fh.write('ply' + '\n')
				fh.write('format ascii 1.0' + '\n')
				fh.write('comment Centipede made from hand bounding-box centroids.' + '\n')
				fh.write('comment If a hand was not visible in a given frame, it will not appear here.' + '\n')
				fh.write('comment Head = white' + '\n')
				fh.write('comment Left hand = green' + '\n')
				fh.write('comment Right hand = red' + '\n')
				fh.write('element vertex ' + str(totalV) + '\n')
				fh.write('property float x' + '\n')
				fh.write('property float y' + '\n')
				fh.write('property float z' + '\n')
				fh.write('property uchar red' + '\n')
				fh.write('property uchar green' + '\n')
				fh.write('property uchar blue' + '\n')
				fh.write('element edge ' + str(totalE) + '\n')
				fh.write('property int vertex1' + '\n')
				fh.write('property int vertex2' + '\n')
				fh.write('end_header' + '\n')

				lines = []
				head_vertex_indices = []
				ctr = 0
				for k, v in sorted(poses.items(), key=lambda x: int(x[0].split('_')[0])):
																	#  Every pose will have a head
					fh.write(str(v['head-x']) + ' ' + str(v['head-y']) + ' ' + str(v['head-z']) + ' 255 255 255' + '\n')
					head_vertex_indices.append(ctr)					#  Save index of this head
					ctr += 1										#  Increment counter

					if 'left-bbox' in v:							#  Pose has left hand?
						fh.write(str(v['head-R'][:, 0].dot(np.array(v['left-bbox'][:3])) + v['head-x']) + ' ' + \
						         str(v['head-R'][:, 1].dot(np.array(v['left-bbox'][:3])) + v['head-y']) + ' ' + \
						         str(v['head-R'][:, 2].dot(np.array(v['left-bbox'][:3])) + v['head-z']) + \
						         ' 0 255 0' + '\n')
						lines.append((ctr, head_vertex_indices[-1]))#  Draw a line from the LH to the head
						ctr += 1									#  Increment counter

					if 'right-bbox' in v:							#  Pose has right hand?
						fh.write(str(v['head-R'][:, 0].dot(np.array(v['right-bbox'][:3])) + v['head-x']) + ' ' + \
						         str(v['head-R'][:, 1].dot(np.array(v['right-bbox'][:3])) + v['head-y']) + ' ' + \
						         str(v['head-R'][:, 2].dot(np.array(v['right-bbox'][:3])) + v['head-z']) + \
						         ' 255 0 0' + '\n')
						lines.append((ctr, head_vertex_indices[-1]))#  Draw a line from the RH to the head
						ctr += 1									#  Increment counter

				for i in range(1, len(head_vertex_indices)):
					lines.append( (head_vertex_indices[i - 1], head_vertex_indices[i]) )

				for line in lines:
					fh.write(str(line[0]) + ' ' + str(line[1]) + '\n')

				fh.close()

				#####################################################  Write the left-hand/right-hand one using BBox centroids
				totalV = len([x for x in poses.values() if 'left-avg' in x]) + \
				         len([x for x in poses.values() if 'right-avg' in x]) + \
				         len(poses)
				totalE = len([x for x in poses.values() if 'left-avg' in x]) + \
				         len([x for x in poses.values() if 'right-avg' in x]) + \
				         len(poses) - 1

				fh = open(enactment + '.IK.avg.ply', 'w')
				fh.write('ply' + '\n')
				fh.write('format ascii 1.0' + '\n')
				fh.write('comment Centipede made from hand means.' + '\n')
				fh.write('comment If a hand was not visible in a given frame, it will not appear here.' + '\n')
				fh.write('comment Head = white' + '\n')
				fh.write('comment Left hand = green' + '\n')
				fh.write('comment Right hand = red' + '\n')
				fh.write('element vertex ' + str(totalV) + '\n')
				fh.write('property float x' + '\n')
				fh.write('property float y' + '\n')
				fh.write('property float z' + '\n')
				fh.write('property uchar red' + '\n')
				fh.write('property uchar green' + '\n')
				fh.write('property uchar blue' + '\n')
				fh.write('element edge ' + str(totalE) + '\n')
				fh.write('property int vertex1' + '\n')
				fh.write('property int vertex2' + '\n')
				fh.write('end_header' + '\n')

				lines = []
				head_vertex_indices = []
				ctr = 0
				for k, v in sorted(poses.items(), key=lambda x: int(x[0].split('_')[0])):
																	#  Every pose will have a head
					fh.write(str(v['head-x']) + ' ' + str(v['head-y']) + ' ' + str(v['head-z']) + ' 255 255 255' + '\n')
					head_vertex_indices.append(ctr)					#  Save index of this head
					ctr += 1										#  Increment counter

					if 'left-avg' in v:								#  Pose has left hand?
						fh.write(str(v['head-R'][:, 0].dot(np.array(v['left-avg'][:3])) + v['head-x']) + ' ' + \
						         str(v['head-R'][:, 1].dot(np.array(v['left-avg'][:3])) + v['head-y']) + ' ' + \
						         str(v['head-R'][:, 2].dot(np.array(v['left-avg'][:3])) + v['head-z']) + \
						         ' 0 255 0' + '\n')
						lines.append((ctr, head_vertex_indices[-1]))#  Draw a line from the LH to the head
						ctr += 1									#  Increment counter

					if 'right-avg' in v:							#  Pose has right hand?
						fh.write(str(v['head-R'][:, 0].dot(np.array(v['right-avg'][:3])) + v['head-x']) + ' ' + \
						         str(v['head-R'][:, 1].dot(np.array(v['right-avg'][:3])) + v['head-y']) + ' ' + \
						         str(v['head-R'][:, 2].dot(np.array(v['right-avg'][:3])) + v['head-z']) + \
						         ' 255 0 0' + '\n')
						lines.append((ctr, head_vertex_indices[-1]))#  Draw a line from the RH to the head
						ctr += 1									#  Increment counter

				for i in range(1, len(head_vertex_indices)):
					lines.append( (head_vertex_indices[i - 1], head_vertex_indices[i]) )

				for line in lines:
					fh.write(str(line[0]) + ' ' + str(line[1]) + '\n')

				fh.close()

				#####################################################  Make videos
				if params['verbose']:
					print('>>> Rendering video with hand bounding-box centroids projected')

				vid = cv2.VideoWriter(enactment + '_annotated-bbox.avi', \
				                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
				                      10, \
				                      (params['imgw'], params['imgh']) )
				for k, v in sorted(poses.items(), key=lambda x: int(x[0].split('_')[0])):
					img = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/' + k, cv2.IMREAD_UNCHANGED)
					if img.shape[2] == 4:							#  Drop alpha channel
						img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
					if v['label'] == '*':
																	#  Write the action
						cv2.putText(img, '<NEUTRAL>', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (255, 255, 255, 255), 3)
					else:
																	#  Write the action
						cv2.putText(img, v['label'], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (255, 255, 255, 255), 3)

					if 'left-bbox' in v:
																	#  Build left-hand string
						LHstr  = "{:.2f}".format(v['left-bbox'][0]) + ' '
						LHstr += "{:.2f}".format(v['left-bbox'][1]) + ' '
						LHstr += "{:.2f}".format(v['left-bbox'][2]) + ' '
						if v['left-bbox'][3] == 1:
							LHstr += '0'
						elif v['left-bbox'][4] == 1:
							LHstr += '1'
						elif v['left-bbox'][5] == 1:
							LHstr += '2'
																	#  Project left hand into camera
						LHproj = np.dot(K, np.array(v['left-bbox'][:3]))
						LHproj /= LHproj[2]
						x = int(round(LHproj[0]))
						y = int(round(LHproj[1]))
						cv2.circle(img, (params['imgw'] - x, params['imgh'] - y), 5, (0, 255, 0, 255), 3)
						cv2.putText(img, 'Left Hand: ' + LHstr, (10, params['imgh'] - 160), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (0, 255, 0, 255), 3)

					if 'right-bbox' in v:
																	#  Build right-hand string
						RHstr  = "{:.2f}".format(v['right-bbox'][0]) + ' '
						RHstr += "{:.2f}".format(v['right-bbox'][1]) + ' '
						RHstr += "{:.2f}".format(v['right-bbox'][2]) + ' '
						if v['right-bbox'][3] == 1:
							RHstr += '0'
						elif v['right-bbox'][4] == 1:
							RHstr += '1'
						elif v['right-bbox'][5] == 1:
							RHstr += '2'
																	#  Project right hand into camera
						RHproj = np.dot(K, np.array(v['right-bbox'][:3]))
						RHproj /= RHproj[2]
						x = int(round(RHproj[0]))
						y = int(round(RHproj[1]))
						cv2.circle(img, (params['imgw'] - x, params['imgh'] - y), 5, (0, 0, 255, 255), 3)
						cv2.putText(img, 'Right Hand: ' + RHstr, (10, params['imgh'] - 40), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (0, 0, 255, 255), 3)

					vid.write(img)									#  Write the frame to the current video

				vid.release()

				if params['verbose']:
					print('>>> Rendering video with hand average pixels projected')

				vid = cv2.VideoWriter(enactment + '_annotated-avg.avi', \
				                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
				                      10, \
				                      (params['imgw'], params['imgh']) )
				for k, v in sorted(poses.items(), key=lambda x: int(x[0].split('_')[0])):
					img = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/' + k, cv2.IMREAD_UNCHANGED)
					if img.shape[2] == 4:							#  Drop alpha channel
						img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
					if v['label'] == '*':
																	#  Write the action
						cv2.putText(img, '<NEUTRAL>', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (255, 255, 255, 255), 3)
					else:
																	#  Write the action
						cv2.putText(img, v['label'], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (255, 255, 255, 255), 3)

					if 'left-avg' in v:
																	#  Build left-hand string
						LHstr  = "{:.2f}".format(v['left-avg'][0]) + ' '
						LHstr += "{:.2f}".format(v['left-avg'][1]) + ' '
						LHstr += "{:.2f}".format(v['left-avg'][2]) + ' '
						if v['left-avg'][3] == 1:
							LHstr += '0'
						elif v['left-avg'][4] == 1:
							LHstr += '1'
						elif v['left-avg'][5] == 1:
							LHstr += '2'
																	#  Project left hand into camera
						LHproj = np.dot(K, np.array(v['left-avg'][:3]))
						LHproj /= LHproj[2]
						x = int(round(LHproj[0]))
						y = int(round(LHproj[1]))
						cv2.circle(img, (params['imgw'] - x, params['imgh'] - y), 5, (0, 255, 0, 255), 3)
						cv2.putText(img, 'Left Hand: ' + LHstr, (10, params['imgh'] - 160), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (0, 255, 0, 255), 3)

					if 'right-avg' in v:
																	#  Build right-hand string
						RHstr  = "{:.2f}".format(v['right-avg'][0]) + ' '
						RHstr += "{:.2f}".format(v['right-avg'][1]) + ' '
						RHstr += "{:.2f}".format(v['right-avg'][2]) + ' '
						if v['right-avg'][3] == 1:
							RHstr += '0'
						elif v['right-avg'][4] == 1:
							RHstr += '1'
						elif v['right-avg'][5] == 1:
							RHstr += '2'
																	#  Project right hand into camera
						RHproj = np.dot(K, np.array(v['right-avg'][:3]))
						RHproj /= RHproj[2]
						x = int(round(RHproj[0]))
						y = int(round(RHproj[1]))
						cv2.circle(img, (params['imgw'] - x, params['imgh'] - y), 5, (0, 0, 255, 255), 3)
						cv2.putText(img, 'Right Hand: ' + RHstr, (10, params['imgh'] - 40), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (0, 0, 255, 255), 3)

					vid.write(img)									#  Write the frame to the current video

				vid.release()

	return

#  'img' has already been opened
def lookupDepth(img, x, y, depthRange):
	return depthRange['min'] + (float(img[y][x]) / 255.0) * (depthRange['max'] - depthRange['min'])

#  For all enactments, create a "*_groundtruth.txt" file that stores:
#  timestamp   filename   label   object-class-name   score   BBox-x1,BBox-y1;BBox-x2,BBox-y2   mask-filename
#  Files with no recognizable objects (in list 'classes') have only the first element, 'filename'
def build_groundtruths(params, classes, rules):
	epsilon = (0, 0, 0)

	if params['colors'] is None:
		params['colors'] = {}
		for obj in classes:
			params['colors'][obj] = [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]

	for enactment in params['enactments']:							#  For each enactment...
		if not os.path.exists(enactment + '_groundtruth.txt') or params['force']:
			if params['verbose']:
				verbosestr = '** Scanning ' + enactment + ' **'
				print('*' * len(verbosestr))
				print(verbosestr)
				print('*' * len(verbosestr))
																	#  Does the folder [enactment]/gt/ exist?
			if os.path.exists(enactment + '/gt'):					#  If so, empty it
				shutil.rmtree(enactment + '/gt')
			os.mkdir(enactment + '/gt', mode=0o777)

			fh = open(enactment + '_subprops.txt', 'r')				#  Load the *_subprops.txt file for reference
			lines = fh.readlines()
			fh.close()

			fh = open(enactment + '_groundtruth.txt', 'w')			#  Create the *_groundtruth.txt file
			fh.write('#  timestamp   filename   label   object-class-name   score   BBox-x1,BBox-y1;BBox-x2,BBox-y2   mask-filename\n')
			frames = {}												#  key:timestamp ==> val:{['label']     = string
																	#                         ['file']      = string
																	#                         ['instances'] = {instance-name ==> [class-name, (rgb), (rgb), ..., (rgb)],
																	#                                          instance-name ==> [class-name, (rgb), (rgb), ..., (rgb)],
																	#                                             ...
																	#                                          instance-name ==> [class-name, (rgb), (rgb), ..., (rgb)]
																	#                                         }
																	#                        }

			for line in lines:										#  First pass: build dictionary structure
				if line[0] != '#':
					arr = line.strip().split('\t')					#  time, label, filename, rgb, part, instance, class
					timestamp = arr[0]								#  string
					label = arr[1]									#  string
					filename = arr[2]								#  string
					instancename = arr[5]							#  string
					classname = arr[6]								#  string

					if timestamp not in frames:
						frames[timestamp] = {}
						frames[timestamp]['label'] = label
						frames[timestamp]['file'] = filename
						frames[timestamp]['instances'] = {}

					if instancename != '*' and instancename not in frames[timestamp]['instances']:
						frames[timestamp]['instances'][instancename] = [classname]

			for line in lines:										#  Second pass: fill in all instances' colors
				if line[0] != '#':
					arr = line.strip().split('\t')					#  time, label, filename, rgb, part, instance, class
					timestamp = arr[0]								#  string
					if arr[3] != '*':								#  (int, int, int) (avoid converting N/As)
						rgb = tuple( [int(x) for x in arr[3].split()] )
					instancename = arr[5]							#  string

					if instancename != '*' and rgb not in frames[timestamp]['instances'][instancename]:
						frames[timestamp]['instances'][instancename].append( rgb )

			mask_ctr = 0
			for k, v in sorted(frames.items(), key=lambda x: float(x[0])):
				if len(v['instances']) > 0:							#  If this timestamp has any recognizable objects...
					if params['verbose']:
						print('>>> Building ' + str(len(v['instances'])) + ' ground-truth masks for ' + enactment + ', ' + k)
					for instancename in v['instances'].keys():		#  go through each object and make a mask for that object
						img = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/ColorMapCameraFrames/' + v['file'], cv2.IMREAD_UNCHANGED)
						if img.shape[2] > 3:
							img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
																	#  Prep a mask for v['instances'][instancename],
																	#  which is a v['instances'][instancename][0]
						mask = np.zeros((params['imgh'], params['imgw']))

						for clr_ctr in range(1, len(v['instances'][instancename])):
																	#  Bloody OpenCV, man...
							color = (v['instances'][instancename][clr_ctr][2], \
							         v['instances'][instancename][clr_ctr][1], \
							         v['instances'][instancename][clr_ctr][0])
							epsilonHi = tuple(map(lambda i, j: i + j, color, epsilon))
							epsilonLo = tuple(map(lambda i, j: i - j, color, epsilon))
							indices = np.where(np.all(img >= epsilonLo, axis=-1) & np.all(img <= epsilonHi, axis=-1))
							if len(indices[0]) > 0:					#  We were able to match this color
								for i in range(0, len(indices[0])):
									mask[indices[0][i]][indices[1][i]] = 255

						indices = np.where(mask == 255)				#  Compute the bounding box for this mask
						upperleft_x = np.min(indices[1])
						upperleft_y = np.min(indices[0])
						lowerright_x = np.max(indices[1])
						lowerright_y = np.max(indices[0])
																	#  Build the bounding-box string
						bboxstr  = str(upperleft_x) + ',' + str(upperleft_y) + ';'
						bboxstr += str(lowerright_x) + ',' + str(lowerright_y)
						cv2.imwrite(enactment + '/gt/mask_' + str(mask_ctr) + '.png', mask)

						fh.write(k + '\t')							#  At time k
						fh.write(v['file'] + '\t')					#  we see image v['file'],
						fh.write(v['label'] + '\t')					#  which is part of action-label v['label'].
																	#  In image v['file'] can be seen an instance of v['instances'][instancename][0]
						fh.write(v['instances'][instancename][0] + '\t')
						fh.write('1.0' + '\t')						#  with utmost confidence.
						fh.write(bboxstr + '\t')					#  This is the bounding box for this recognizable object.
																	#  This mask lives at this address.
						fh.write(enactment + '/gt/mask_' + str(mask_ctr) + '.png' + '\n')
						mask_ctr += 1
				else:
					if params['verbose']:
						print('>>> No ground-truth masks for ' + enactment + ', ' + k)
					fh.write(k + '\t')								#  At time k
					fh.write(v['file'] + '\t')						#  we see image v['file'],
					fh.write(v['label'] + '\t')						#  which is part of action-label v['label'].
					fh.write('*' + '\t')							#  Here we see nothing.
					fh.write('1.0' + '\t')							#  I am certain of that.
					fh.write('*' + '\t')							#  There is no bounding box to speak of.
					fh.write('*' + '\n')							#  And there is no mask.

			fh.close()

			if params['render']:									#  Making videos?
				if params['verbose']:
					print('\n>>> Rendering ground-truth video for ' + enactment)

				fh = open(enactment + '_groundtruth.txt', 'r')		#  Read the file we just wrote
				lines = fh.readlines()
				fh.close()

				frames = {}											#  One key per unique video file name
				for line in lines:
					if line[0] != '#':
						arr = line.strip().split('\t')
						filename = arr[1]
						if filename not in frames:
							frames[filename] = []					#  Will be a list of tuples:(object-class-name, maskfilename, bbox-string)

				for framename in frames.keys():						#  Second pass: collect each frame's masks, objects, and bounding-boxe strings
					for line in lines:
						if line[0] != '#':
							arr = line.strip().split('\t')
							timestamp = arr[0]
							filename = arr[1]
							label = arr[2]
							object_class_name = arr[3]
							score = arr[4]
							bboxstr = arr[5]
							maskfilename = arr[6]

							if filename == framename and object_class_name != '*' and object_class_name not in rules['PERMIT']:
								frames[framename].append( (object_class_name, maskfilename, bboxstr) )

				vid = cv2.VideoWriter(enactment + '_groundtruth.avi', \
				                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
				                      10, \
				                      (params['imgw'], params['imgh']) )

				for framename, v in sorted(frames.items(), key=lambda x: int(x[0].split('_')[0])):
					img = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/' + framename, cv2.IMREAD_UNCHANGED)
					if img.shape[2] == 4:							#  Drop alpha channel
						img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
					maskcanvas = np.zeros((params['imgh'], params['imgw'], 3), dtype='uint8')

					for objdata in v:								#  For each (possibly zero) recognizable objects in the current frame
						object_class_name = objdata[0]				#  Object name: so we can look up the color
						maskfilename = objdata[1]					#  Mask file name: so we can overlay the mask for this object
																	#  Open the mask file
						mask = cv2.imread(maskfilename, cv2.IMREAD_UNCHANGED)
						mask[mask > 1] = 1							#  All things greater than 1 become 1
																	#  Extrude to three channels
						mask = mask[:, :, None] * np.ones(3, dtype='uint8')[None, None, :]
																	#  Convert this to a graphical overlay:
						mask[:, :, 0] *= params['colors'][ object_class_name ][2]
						mask[:, :, 1] *= params['colors'][ object_class_name ][1]
						mask[:, :, 2] *= params['colors'][ object_class_name ][0]

						maskcanvas += mask							#  Add mask to mask accumulator
						maskcanvas[maskcanvas > 255] = 255			#  Clip accumulator to 255
																	#  Add mask accumulator to source frame
					img = cv2.addWeighted(img, 1.0, maskcanvas, 0.7, 0)
					img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)		#  Flatten alpha

					for objdata in v:
						object_class_name = objdata[0]				#  Object name: so we can look up the color
						bboxstr = objdata[2]						#  Bounding box: so we can draw the bounding box
						bbox = bboxstr.split(';')
						bbox = (int(bbox[0].split(',')[0]), int(bbox[0].split(',')[1]), int(bbox[1].split(',')[0]), int(bbox[1].split(',')[1]))
						cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (params['colors'][ object_class_name ][2], \
						                                                            params['colors'][ object_class_name ][1], \
						                                                            params['colors'][ object_class_name ][0]), 1)

					vid.write(img)

				vid.release()
		elif params['verbose']:
			print('*** ' + enactment + '_groundtruth.txt already exists. Skipping.')
	return

#  For all enactments, create a "*_subprops.txt" file that stores:
#  timestamp   label   colormap-filename   r-g-b   part-name   instance-name   class-name
#  If the part does not belong to an instance or to a class, asterisks will fill these columns.
#  The idea is that we can construct ground-truth masks by looking up the colormap-filename and
#  assembling everything of the same instance into a single mask.
#
#  Reminder: the 'rules' dictionary looks like this:
#    rules['IGNORE']:  key:string ==> True
#    rules['PERMIT']:  key:string ==> True
#    rules['DEF']:     key:instance-name ==> val:class-name
#    rules['COLL']:    key:(part-name, part-name, ...) ==> val:(label, label, ..., class-name)
#    rules['COND']:    key:pseudo-class-name ==> val:{key:(part, attribute, value) ==> val:class-name
#                                                     key:(part, attribute, value) ==> val:class-name
#                                                     key:(part, attribute, value) ==> val:class-name
#                                                            ... }
def build_subprops(base_classes, rules, params):
	epsilon = (0, 0, 0)

	for enactment in params['enactments']:							#  For each enactment...

		if not os.path.exists(enactment + '_subprops.txt') or params['force']:
			if params['verbose']:
				verbosestr = '** Scanning ' + enactment + ' **'
				print('*' * len(verbosestr))
				print(verbosestr)
				print('*' * len(verbosestr))

			fh = open(enactment + '/Users/' + params['User'] + '/POV/SubpropColorMap.fvr')
			line = fh.readlines()[0]
			fh.close()
			colormap = json.loads(line)

			part_lookup_colorAndLink = {}							#  e.g. key:Target1_AuxiliaryFeederBox_Auxiliary.Jaw1
																	#         ==>  val:{rgb:(255, 205, 0),
																	#                   link:OpenClose0/Users/Subprops/Target1_AuxiliaryFeederBox_Auxiliary.Jaw1.fvr}
			color_lookup_part = {}									#  e.g. key:(255, 205, 0) ==> Target1_AuxiliaryFeederBox_Auxiliary.Jaw1

			if params['verbose']:
				print('>>> Parsing colors from SubpropColorMap.fvr')

			for colorkeyed_item in colormap['list']:				#  Iterate over all color-keyed items
				part_lookup_colorAndLink[ colorkeyed_item['label'] ] = {}
				part_lookup_colorAndLink[ colorkeyed_item['label'] ]['rgb'] = (colorkeyed_item['color']['r'], colorkeyed_item['color']['g'], colorkeyed_item['color']['b'])
				if colorkeyed_item['label'] not in rules['IGNORE']:	#  Ignore what we ignore
					color_lookup_part[ (colorkeyed_item['color']['r'], colorkeyed_item['color']['g'], colorkeyed_item['color']['b']) ] = colorkeyed_item['label']
				if len(colorkeyed_item['metadataPath']) > 0:
					part_lookup_colorAndLink[ colorkeyed_item['label'] ]['link'] = enactment + '/' + colorkeyed_item['metadataPath']
				else:
					part_lookup_colorAndLink[ colorkeyed_item['label'] ]['link'] = None

			#  Iterate over all PARTS:
			#    Does the part have a '.' in it? Then look up the instance name among 'base_classes': Add part to that instance's (all instances) list.
			#    Does the part NOT have a '.' in it? Then look up the part name (whole) among 'base_classes': Add part (same as instance) to instance's list.
			for k, v in part_lookup_colorAndLink.items():			#  Target1_AuxiliaryFeederBox_Auxiliary.Jaw1 ==> {'rgb', 'link'}
				whole_name = k
				if '.' in k:
					instance_name = k.split('.')[0]
				else:
					instance_name = k

				for class_k, class_v in base_classes.items():		#  ControlPanel ==> {'Spare_ControlPanel_Main': [],
																	#                    'Target1_ControlPanel_Main': [],
																	#                    'Target2_ControlPanel_Main': []}
					for instance_k, instance_v in class_v.items():
						if (instance_k == instance_name or whole_name == instance_k) and k not in instance_v:
							instance_v.append(k)

			#  Iterate over all COLLECT rules:
			#    fill in their parts
			for k, v in rules['COLL'].items():						#  key:(part-name, part-name, ...) ==> val:(label, label, ..., instance-name)
				instance_name = v[-1]
				for class_k, class_v in base_classes.items():		#  key:class-name ==> val:{key:instance-name ==> val:[part-names, ... ]}
					if instance_name in class_v:
						for part_name in k:
							class_v[instance_name].append(part_name)

			#  Expanded classes cover conditional class assignment
			expanded_classes = {}
			for k, v in base_classes.items():
				expanded_classes[k] = {}
				for kk, vv in v.items():
					expanded_classes[k][kk] = vv[:]
			for k, v in rules['COND'].items():
				#  COND_rules:
				#  e.g. MainFeederBox ==> ('Door', 'hingeStatus', int(0)) ==> MainFeederBox_Closed
				#                         ('Door', 'hingeStatus', int(1)) ==> MainFeederBox_Open
				#                         ('Door', 'hingeStatus', int(2)) ==> MainFeederBox_Unknown

				#  classes:
				#  e.g. SafetyPlank ==> 'Spare_TransferFeederBox_Transfer.SafetyPlank'   ==> []
				#                       'Target1_TransferFeederBox_Transfer.SafetyPlank' ==> []
				#                       'Target2_TransferFeederBox_Transfer.SafetyPlank' ==> []
				save_dict = {}
				for kk, vv in expanded_classes[k].items():
					save_dict[kk] = vv

				del expanded_classes[k]

				for cond_class_name in v.values():
					expanded_classes[cond_class_name] = save_dict

			#  Unaffected by conditionals
			part_lookup_instances = {}								#  e.g. key:Spare_AuxiliaryFeederBox_Auxiliary.Connector1
																	#       val:[Spare_AuxiliaryFeederBox_Auxiliary, Disconnect0]
			for class_name, inst_parts_dict in base_classes.items():#  AuxiliaryFeederBox ==> {Spare_AuxiliaryFeederBox_Auxiliary ==> [],
																	#                          Target1_AuxiliaryFeederBox_Auxiliary ==> [],
																	#                          Target2_AuxiliaryFeederBox_Auxiliary ==> []}
				for k, v in inst_parts_dict.items():				#
					for vv in v:
						if vv not in part_lookup_instances:
							part_lookup_instances[vv] = []
						part_lookup_instances[vv].append(k)

			#  Iterate over all PARTS:
			#    Consult all classes ==> instances and find the unique instance in which this part appears
			instance_lookup_classes = {}
			for k, v in expanded_classes.items():					#  class-name ==> instance ==> [parts]
																	#                 instance ==> [parts] ...
				for kk, vv in v.items():							#  instance-name ==> [parts]
					if kk not in instance_lookup_classes:
						instance_lookup_classes[kk] = []
					instance_lookup_classes[kk].append(k)			#  instance-name ==> [all classes to which this instance CAN belong]

			instance_lookup_pseudoclass = {}
			for k, v in base_classes.items():						#  class-name ==> instance ==> [parts]
																	#                 instance ==> [parts] ...
				for kk, vv in v.items():							#  instance-name ==> [parts]
					instance_lookup_pseudoclass[kk] = k				#  instance-name ==> pseudo-class to which this instance belongs
																	#  (Pseudo-classes may yet be subject to conditionals)

			if params['verbose']:
				print('    PARTS: ')								#  Print all parts: r g b    part-name
				ctr = 1												#  (Note: this will include things ignored and "permitted")
				for k, v in sorted(part_lookup_colorAndLink.items()):
					print('\t' + str(ctr) + ':\t' + ' '.join([str(x) for x in v['rgb']]) + '\t' + k)
					ctr += 1

				print('    INSTANCES: ')							#  Print all instances: instance-name    class-name OR class-name class-name class-name...
				ctr = 1												#                       followed by all colors that make up this instance
																	#  instance_lookup_classes{} = instance-name ==> [class, class, ... ]
				for k, v in sorted(instance_lookup_classes.items()):
					print('\t' + str(ctr) + ':\t' + k + '\t' + ' '.join(v))

					colorstr = '\t\t'								#  All colors for all parts of this instance
					for k2, v2 in base_classes.items():				#  class-name ==> {instance ==> [parts]; instance ==> [parts]; ... }
						for k3, v3 in v2.items():					#  instance-name ==> [parts]
							if k3 == k:
								for coloritem in v3:
									colorstr += '  (' + str(part_lookup_colorAndLink[coloritem]['rgb'][0]) + ' ' + \
									                    str(part_lookup_colorAndLink[coloritem]['rgb'][1]) + ' ' + \
									                    str(part_lookup_colorAndLink[coloritem]['rgb'][2]) + ')'
					print(colorstr)
					ctr += 1

				print('    TYPES: ')								#  Print all classes: class-anme    instance instance instance...
				ctr = 1
				for k, v in sorted(expanded_classes.items()):
					print('\t' + str(ctr) + ':\t' + k + '\t' + ' '.join(v.keys()))
					ctr += 1

			fh = open(enactment + '_poses.full.txt', 'r')			#  Read the intermediate (full) "poses" file.
																	#  It indicates which .png files we care about.
			lines = [line.strip() for line in fh.readlines() if line[0] != '#']
			fh.close()

			fh = open(enactment + '_subprops.txt', 'w')				#  Create a new intermediate file for this enactment.
																	#  Build the file header
			fh.write('#  PARTS\t' + ' '.join( sorted(part_lookup_colorAndLink.keys()) ) + '\n')
			fh.write('#  INSTANCES\t' + ' '.join( sorted(instance_lookup_classes.keys()) ) + '\n')
			fh.write('#  TYPES\t' + ' '.join( sorted(expanded_classes.keys()) ) + '\n')
			fh.write('#  timestamp   label   image-filename   r-g-b   part   instance   class\n')

			for line in lines:										#  For every image that belongs to this enactment...
				arr = line.strip().split('\t')
				timestamp = arr[0]									#  The timestamp is [0]
				label = arr[1]										#  The label is [1]
				colormapfilename = arr[2]							#  The file name is [2]
																	#  23feb21: SOME of these have alpha channels.
																	#           And now SOME do not.
				t1_start = time.process_time()
				img = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/ColorMapCameraFrames/' + colormapfilename, cv2.IMREAD_UNCHANGED)
				if img.shape[2] > 3:
					img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

				found_something = False
				for k, part_name in color_lookup_part.items():		#  For every color of which we are aware (related to a part or permitted)...
					r = k[0]
					g = k[1]
					b = k[2]

					color = (b, g, r)								#  Bloody OpenCV, man...
					epsilonHi = tuple(map(lambda i, j: i + j, color, epsilon))
					epsilonLo = tuple(map(lambda i, j: i - j, color, epsilon))
					indices = np.where(np.all(img >= epsilonLo, axis=-1) & np.all(img <= epsilonHi, axis=-1))
					if len(indices[0]) > 0:							#  We were able to match this color
						found_something = True

						if part_name in rules['PERMIT']:			#  This is a special case: do not bother looking for components
							fh.write(timestamp + '\t')				#  Write timestamp
							fh.write(label + '\t')					#  Write action-label
							fh.write(colormapfilename + '\t')		#  Write filename and R G B
							fh.write(str(r) + ' ' + str(g) + ' ' + str(b) + '\t')
							fh.write(part_name + '\t')				#  There will always be at least a part-name
							fh.write(part_name + '\t')				#  Stands for instance name
							fh.write(part_name + '\n')				#  Stands for class name
						else:
							writestr = ''							#  Covers all instances for this part
																	#  Retrieve a list of all instances this part comprises
							instances = part_lookup_instances[part_name]

							#  A PART MAY BELONG TO SEVERAL INSTANCES, so FOR EACH INSTANCE...
							#  And each instance may be subject to condition, affecting the class-label assigned to this part.

							for instance_name in instances:
																	#  Only ONE label in the pseudo-class lookup--though these may be subject to condition
								pseudo_class = instance_lookup_pseudoclass[instance_name]

								if pseudo_class in rules['COND']:	#  CLASS ASSIGNMENT IS SUBJECT TO CONDITION!

									#  Find the first test to pass...
									assigned_class = None			#  Find the condition that determines 'pseudo-class-name's assigned class

									#  rules['COND']:	key:pseudo-class-name ==> val:{key:(part, attribute, value) ==> val:class-name
									#                                                  key:(part, attribute, value) ==> val:class-name
									#                                                  key:(part, attribute, value) ==> val:class-name
									#                                                         ... }

									#  rules['COND'][pseudo_class] = key:(part, attribute, value) ==> val:class-name
									#                                key:(part, attribute, value) ==> val:class-name
									#                                key:(part, attribute, value) ==> val:class-name
									#                          e.g.  key:(*, hingeStatus, 0) ==> val:SafetyPlank_Closed
									#                          e.g.  key:(Door, hingeStatus, 0) ==> val:MainFeederBox_Closed
									#                          e.g.  key:(Jaw, hingeStatus, 0) ==> val:Disconnect_Closed
																	#  Check all conditional rules
									for cond_rule, resultant_class in rules['COND'][pseudo_class].items():
										determining_component = cond_rule[0]
										determining_attribute = cond_rule[1]
										determining_value     = cond_rule[2]
																	#  None stands in for '*', which is a reference to "self"
										if determining_component is None:
											determining_component = part_name
										else:						#  The determining component is given--but may still be artificial
																	#  Check in COLL rules: a moniker will have been assigned
											#  Is the pseudo-class a collection (COLL)?
											defined_among_collections = False
											for coll_parts, coll_labels in rules['COLL'].items():
												if coll_labels[-1] == instance_name:
													defined_among_collections = True
													i = 0
													while i < len(coll_parts) and coll_labels[i] != determining_component:
														i += 1
													determining_component = coll_parts[i]
													break

											if not defined_among_collections:
												determining_component = instance_name + '.' + determining_component

																	#  Open the JSON
										jsonfh = open(part_lookup_colorAndLink[determining_component]['link'], 'r')
										line = jsonfh.readlines()[0]#  Convert string to JSON
										jsonfh.close()				#  Find the string inside the JSON
										component_history = json.loads(line)
																	#  Make that a JSON, too.... what the hell?
										unpacked_string = json.loads(component_history['frames'])
										i = 0
										j = 0
										while j < len(unpacked_string['list']):
											if unpacked_string['list'][j]['timestamp'] == float(timestamp):
												val = unpacked_string['list'][j][determining_attribute]
												if val == determining_value:
													assigned_class = resultant_class
													#print('* IN ' + colormapfilename + '\t' + part_name + '\t-->\t' + assigned_class)
												break
											j += 1
										i += 1

									if assigned_class is not None:
										writestr += timestamp + '\t'#  Write timestamp
										writestr += label + '\t'	#  Write action-label
																	#  Write filename
										writestr += colormapfilename + '\t'
																	#  Write R G B
										writestr += str(r) + ' ' + str(g) + ' ' + str(b) + '\t'
										writestr += part_name + '\t'
										writestr += instance_name + '\t'
										writestr += assigned_class + '\n'
								else:								#  Class assignment is NOT subject to condition
									writestr += timestamp + '\t'	#  Write timestamp
									writestr += label + '\t'		#  Write action-label
																	#  Write filename
									writestr += colormapfilename + '\t'
																	#  Write R G B
									writestr += str(r) + ' ' + str(g) + ' ' + str(b) + '\t'
									writestr += part_name + '\t'
									writestr += instance_name + '\t'
									writestr += pseudo_class + '\n'

							fh.write(writestr)

						if params['verbose']:						#  Show me what we got
							print('>>>      ' + enactment + ', ' + colormapfilename + ' in action ' + label + ': ' + color_lookup_part[ (r, g, b) ] + ' is visible')

				if not found_something:
																	#                  No rgb       No part      No instance  No class
					fh.write(timestamp + '\t' + label + '\t' + colormapfilename + '\t' + '*' + '\t' + '*' + '\t' + '*' + '\t' + '*' + '\n')
					if params['verbose']:
						print('>>>      ' + enactment + ', ' + colormapfilename + ' in action ' + label + ': NOTHING is visible')
				t1_stop = time.process_time()
				if params['verbose']:
					print('    Frame took ' + str(t1_stop - t1_start) + ' seconds')

			fh.close()
																	#  Prepare a list of conditional-inclusive classes to return
			expanded_classes = [x for x in sorted(expanded_classes.keys())]
		else:
			expanded_classes = []									#  Just fetch the keys to return
			fh = open(enactment + '_subprops.txt', 'r')
			lines = fh.readlines()
			fh.close()
			i = 0
			while i < len(lines) and lines[i][:9] != '#  TYPES\t':
				i += 1
			line = lines[i].strip().split('\t')[1]
			arr = line.split()
			for class_name in arr:
				expanded_classes.append(class_name)
			if params['verbose']:
				print('*** ' + enactment + '_subprops.txt already exists. Skipping.')

	return expanded_classes											#  Return a list of all class-labels including conditionals

def compute_base_classes(rules):
	classes = {}													#  Lookup table of classes, instances, and each instance's parts-strings:
																	#  key:class-name ==> val:{ [instance-name] ==> [ part-name, part-name, ... ]
																	#                           [instance-name] ==> [ part-name, part-name, ... ]
																	#                              ...
																	#                         }
	for k, v in rules['DEF'].items():
		if v not in classes:
			classes[v] = {}
		classes[v][k] = []

	return classes

#  Load the given rules to all given enactments and determine the classes/objects/props to be parsed.
#  The returned 'classes' is a dictionary: key:(class-name, instance) ==> val:[part-string, part-string, ... ]
#  The returned 'permitted' is a list: most probably [LeftHand, RightHand]
#  The returned 'ignore' is a list: most probably [Environment, Head]
def load_rules(params):
	IGNORE_rules = {}												#  Things we do NOT wish to consider
	PERMIT_rules = {}												#  Things we wish to consider but which are not classes

	#  For all colored items, if it contains a '.' then initially assume that it is a PART of the INSTANCE to the LEFT
	#  These assumptions may or may not be overridden.
	#  An instance IS NEVER a class-name: they all have something prefixed like Target1_ DeezNuts_, etc.

	DEF_rules = {}													#  key:instance-name ==> val:class-name

	COLL_rules = {}													#  key:(part-name, part-name, ...) ==> val:(label, label, ..., instance-name)

	COND_rules = {}													#  key:pseudo-class-name ==> val:{key:(part, attribute, value) ==> val:class-name
																	#                                 key:(part, attribute, value) ==> val:class-name
																	#                                 key:(part, attribute, value) ==> val:class-name
																	#                                                ... }

	if params['verbose']:
		print('>>> Loading rules file ' + params['rules'])

	fh = open(params['rules'], 'r')									#  Load object-parsing rules
	lines = fh.readlines()
	fh.close()
	for line in lines:
		if line[0] != '#' and len(line.strip()) > 0:				#  Skip comments and empties
			arr = line.strip().split('\t')

			if arr[0] == 'IGNORE':									#  Ignore the Environment
				IGNORE_rules[ arr[1] ] = True

			elif arr[0] == 'PERMIT':								#  Track e.g. the hands but do not consider them "props" per se.
				PERMIT_rules[ arr[1] ] = True

			elif arr[0] == 'DEFINE':								#  e.g. Global_SubstationTag_Yellow2 is a YellowTag
				DEF_rules[ arr[1] ] = arr[2]						#  DEF_rules['Global_SubstationTag_Yellow2'] ==> 'YellowTag'

			elif arr[0] == 'COND':									#  e.g. Spare_AuxiliaryFeederBox_Auxiliary is an AuxiliaryFeederBox_Closed
																	#       if component Door has hingeStatus == int(0)
				if arr[1] not in COND_rules:
					COND_rules[ arr[1] ] = {}						#  COND_rules['AuxiliaryFeederBox'][ ( 'Door', 'hingeStatus', int(0) ) ] ==>
																	#                                                 'AuxiliaryFeederBox_Closed'
				if arr[3] == '*':
					defining_component = None
				else:
					defining_component = arr[3]

				defining_attribute = arr[4]

				if arr[5] == 'int':
					defining_value = int(arr[6])
				elif arr[5] == 'float':
					defining_value = float(arr[6])
				elif arr[5] == 'str':
					defining_value = arr[6]

				COND_rules[ arr[1] ][ (defining_component, defining_attribute, defining_value) ] = arr[2]

			elif arr[0] == 'COLLECT':								#  e.g. Target1_TransferFeederBox_Transfer.Connector1 is known internally as Contact
																	#       Target1_TransferFeederBox_Transfer.Jaw1	is known internally as  Jaw
																	#       in the instance Disconnect19
				instance_name = arr[-1]
				rule = arr[1:-1]									#  COLL_rules[ ( Target2_TransferFeederBox_Transfer.Connector1,
				keys = []											#                Target2_TransferFeederBox_Transfer.Jaw1         ) ]
				labels = []											#   ==> (Contact, Jaw, Disconnect31)
				for i in range(0, len(rule), 3):
					keys.append(rule[i])
					labels.append(rule[i + 2])
				keys = tuple(keys)
				labels = tuple(labels + [instance_name])

				COLL_rules[ keys ] = labels

	rules = {}
	rules['IGNORE'] = IGNORE_rules
	rules['PERMIT'] = PERMIT_rules
	rules['DEF'] = DEF_rules
	rules['COLL'] = COLL_rules
	rules['COND'] = COND_rules

	return rules

#  For all enactments, create a "*_poses.txt" file that STORES HAND VECTORS IN THE HEAD FRAME
def build_poses(params):
	if params['render']:											#  Rendering centipedes? You'll need colors.
		actionColors = survey_action_colors(params)					#  Then build a color look-up table: action/label => (r, g, b)

	for enactment in params['enactments']:							#  For each enactment...
																	#  If *_poses.{full,shutoff}.txt files either do not exist or we are forcing...
		if not os.path.exists(enactment + '_poses.full.txt') or not os.path.exists(enactment + '_poses.shutoff.txt') or params['force']:
			if params['verbose']:
				verbosestr = '** Scanning ' + enactment + ' **'
				print('*' * len(verbosestr))
				print(verbosestr)
				print('*' * len(verbosestr))

			fh = open(enactment + '/Labels.fvr', 'r')				#  Fetch action time stamps
			lines = ' '.join(fh.readlines())						#  These USED to be a single line; NOW they have carriage returns
			fh.close()
			labels = json.loads(lines)
																	#  Retrieve head poses: a valid datum must have a head pose
			fh = open(enactment + '/Users/' + params['User'] + '/Head.fvr', 'r')
			line = fh.readlines()[0]								#  Single line
			fh.close()
			headdata = json.loads(line)
																	#  Retrieve left-hand poses: a valid datum must have a left-hand pose
			fh = open(enactment + '/Users/' + params['User'] + '/LeftHand.fvr', 'r')
			line = fh.readlines()[0]								#  Single line
			fh.close()
			lefthanddata = json.loads(line)
																	#  Retrieve right-hand poses: a valid datum must have a right-hand pose
			fh = open(enactment + '/Users/' + params['User'] + '/RightHand.fvr', 'r')
			line = fh.readlines()[0]								#  Single line
			fh.close()
			righthanddata = json.loads(line)

			fh = open(enactment + '/metadata.fvr', 'r')				#  Fetch enactment duration
			line = fh.readlines()[0]								#  Single line
			fh.close()
			metadata = json.loads(line)
			duration = metadata['duration']

			fh = open(enactment + '/Users/' + params['User'] + '/POV/CameraIntrinsics.fvr', 'r')
			line = fh.readlines()[0]
			fh.close()
			camdata = json.loads(line)
			fov = float(camdata['fov'])								#  We don't use camdata['focalLength'] anymore because IT'S IN MILLIMETERS
			if params['focal'] is None:								#  IN PIXELS!!!! ALSO: NOTE THAT THIS IS THE **VERTICAL** F.o.V. !!
				focalLength = params['imgh'] * 0.5 / np.tan(fov * np.pi / 180.0)
			else:
				focalLength = params['focal']
			K = np.array([[focalLength, 0.0, params['imgw'] * 0.5], \
			              [0.0, focalLength, params['imgh'] * 0.5], \
			              [0.0,         0.0,           1.0       ]])
			if params['verbose']:
				print('>>> Stated camera focal length: ' + str(camdata['focalLength']) + ' mm')
				print('    Computed camera focal length: ' + str(focalLength) + ' pixels')
				print('K = ')
				print(K)

			frames = [x for x in os.listdir(enactment + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/') if x.endswith('.png')]
			frames = sorted(frames, key=lambda x: int(x.split('_')[0]))
			if params['verbose']:
				print('    ' + str(len(frames)) + ' frames')

			#########################################################  Build timestamped poses: timestamp ==> head-left-right
																	#  key(float)    val(dict)
			samples = {}											#  timestamp ==> {} = [head]      ==> (x, y, z)
																	#                     [lefthand]  ==> (x, y, z)
																	#                     [righthand] ==> (x, y, z)
																	#                     [globalLH]  ==> (x, y, z)
																	#                     [globalRH]  ==> (x, y, z)
																	#                     [label]     ==> string or None
																	#                     [file]      ==> string

			for headpose in headdata['list']:						#  We will let the head determine what is valid data
				timestamp = headpose['timestamp']
				if timestamp not in samples:
					samples[timestamp] = {}
																	#  Convert quaternion -> Axis-Angle -> Rodrigues Angles -> Rotation Matrix
				angle = 2 * np.arccos(headpose['pose']['rotation']['w'])
				x = headpose['pose']['rotation']['x'] / np.sqrt(1.0 - headpose['pose']['rotation']['w'] * headpose['pose']['rotation']['w'])
				y = headpose['pose']['rotation']['y'] / np.sqrt(1.0 - headpose['pose']['rotation']['w'] * headpose['pose']['rotation']['w'])
				z = headpose['pose']['rotation']['z'] / np.sqrt(1.0 - headpose['pose']['rotation']['w'] * headpose['pose']['rotation']['w'])
				headRot, _ = cv2.Rodrigues( np.array([x * angle, y * angle, z * angle], dtype='float32') )

				#  NOTICE!                    NEGATIVE-X
				samples[timestamp]['head'] = (-headpose['pose']['position']['x'], \
				                               headpose['pose']['position']['y'], \
				                               headpose['pose']['position']['z'], \
				                               headRot)
				index = min(int(round((timestamp / duration) * float(len(frames)))), len(frames) - 1)
				samples[timestamp]['file'] = frames[ index ]
				samples[timestamp]['label'] = None

			for handpose in lefthanddata['list']:					#  Add a left-hand IFF there is a timestamp justified by a head pose
				timestamp = handpose['timestamp']
				if timestamp in samples:
					#  NOTICE!       NEGATIVE-X
					diff = np.array([-handpose['pose']['position']['x'] - samples[timestamp]['head'][0], \
					                  handpose['pose']['position']['y'] - samples[timestamp]['head'][1], \
					                  handpose['pose']['position']['z'] - samples[timestamp]['head'][2]])
					samples[timestamp]['lefthand'] = ( samples[timestamp]['head'][3][0].dot(diff), \
					                                   samples[timestamp]['head'][3][1].dot(diff), \
					                                   samples[timestamp]['head'][3][2].dot(diff), \
					                                   handpose['handState'] )
					#  NOTICE!                        NEGATIVE-X
					samples[timestamp]['globalLH'] = (-handpose['pose']['position']['x'], \
					                                   handpose['pose']['position']['y'], \
					                                   handpose['pose']['position']['z'])

			for handpose in righthanddata['list']:					#  Add a right-hand IFF there is a timestamp justified by a head pose
				timestamp = handpose['timestamp']
				if timestamp in samples:
					#  NOTICE!       NEGATIVE-X
					diff = np.array([-handpose['pose']['position']['x'] - samples[timestamp]['head'][0], \
					                  handpose['pose']['position']['y'] - samples[timestamp]['head'][1], \
					                  handpose['pose']['position']['z'] - samples[timestamp]['head'][2]])
					samples[timestamp]['righthand'] = ( samples[timestamp]['head'][3][0].dot(diff), \
					                                    samples[timestamp]['head'][3][1].dot(diff), \
					                                    samples[timestamp]['head'][3][2].dot(diff), \
					                                    handpose['handState'] )
					#  NOTICE!                        NEGATIVE-X
					samples[timestamp]['globalRH'] = (-handpose['pose']['position']['x'], \
					                                   handpose['pose']['position']['y'], \
					                                   handpose['pose']['position']['z'])

																	#  Make sure all parts of the skeleton are included
			samples = dict( [(k, v) for k, v in samples.items() if 'lefthand' in v and 'righthand' in v] )

			#########################################################  For all metadata labels, assign frames

			timeline = []											#  List of tuples(start, end, label)
			for action in labels['list']:
				timeline.append( [action['startTime'], action['endTime'], action['stepDescription']] )
			timeline = sorted(timeline, key=lambda x: x[0])			#  Sort temporally
			for i in range(0, len(timeline) - 1):					#  Examine all neighbors for conflicts
				if timeline[i][1] > timeline[i + 1][0]:				#  LABEL OVERLAP!
					if params['verbose']:
						print('    * Label conflict between ' + timeline[i][2] + ' and ' + timeline[i + 1][2])
																	#  ADJUST TIMES!
					mid = timeline[i + 1][0] - (timeline[i][1] - timeline[i + 1][0]) * 0.5
					timeline[i][1]     = mid
					timeline[i + 1][0] = mid

			for action in timeline:									#  Go through all actions labeled in the current enactment
				ctr = 0
				label = action[2]									#  Identify the label
				startTime = action[0]								#  Start and end times
				endTime = action[1]

				for timestamp, v in samples.items():				#  Find all the obvious fits
					if timestamp >= startTime and timestamp < endTime:
																	#  If this float falls between the other floats
						samples[timestamp]['label'] = label			#  then this frame is part of this action and receives this label
						ctr += 1

				if ctr == 0:										#  If there was no obvious fit, then find the nearest fit
					best = 0
					minerr = float('inf')
					keys = sorted(samples.keys())
					for i in range(0, len(keys)):
						err = np.fabs(startTime - keys[i])
						if err < minerr:
							best = i
							minerr = err
					samples[ keys[best] ]['label'] = label
					ctr += 1

				if params['verbose']:
					print('    ' + label + ': ' + str(startTime) + ' to ' + str(endTime) + ' covers ' + str(ctr) + ' poses')

			#  By now we have samples[timestamp] ==> ['head'] = (x, y, z, R)
			#                                        ['lefthand'] = (x, y, z)
			#                                        ['righthand'] = (x, y, z)
			#                                        ['globalLH'] = (x, y, z)
			#                                        ['globalRH'] = (x, y, z)
			#                                        ['label'] = string or None
			#                                        ['file'] = string

			#########################################################  Write the intermediate head-files
			fh = open(enactment + '_poses.head.txt', 'w')
			fh.write('#  timestamp, label, image-file, head-x, head-y, head-z, R00, R01, R02, R10, R11, R12, R20, R21, R22\n')
			for k, v in sorted(samples.items()):
				fh.write(str(k) + '\t')
				if v['label'] is None:
					fh.write('*' + '\t')
				else:
					fh.write(v['label'] + '\t')
				fh.write(v['file'] + '\t')

				fh.write(str(v['head'][0]) + '\t' + str(v['head'][1]) + '\t' + str(v['head'][2]) + '\t')
				fh.write(str(v['head'][3][0][0]) + '\t' + str(v['head'][3][0][1]) + '\t' + str(v['head'][3][0][2]) + '\t')
				fh.write(str(v['head'][3][1][0]) + '\t' + str(v['head'][3][1][1]) + '\t' + str(v['head'][3][1][2]) + '\t')
				fh.write(str(v['head'][3][2][0]) + '\t' + str(v['head'][3][2][1]) + '\t' + str(v['head'][3][2][2]) + '\n')
			fh.close()

			#########################################################  Write the intermediate vector-files
			fh = open(enactment + '_poses.full.txt', 'w')
			fh.write('#  timestamp, label, image-file, LHx, LHy, LHz, LHstate_0, LHstate_1, LHstate_2, RHx, RHy, RHz, RHstate_0, RHstate_1, RHstate_2\n')
			fh.write('#  All hand positions are taken from JSON and made relative to the head in the head\'s frame.\n')
			fh.write('#  In this file, hand vectors are all recorded REGARDLESS OF WHETHER THE HAND PROJECTS INTO THE CAMERA!\n')
			for k, v in sorted(samples.items()):
				fh.write(str(k) + '\t')
				if v['label'] is None:
					fh.write('*' + '\t')
				else:
					fh.write(v['label'] + '\t')
				fh.write(v['file'] + '\t')

				fh.write(str(v['lefthand'][0])  + '\t' + str(v['lefthand'][1])  + '\t' + str(v['lefthand'][2])  + '\t')
				if v['lefthand'][3] == 0:
					fh.write('1\t0\t0\t')
				elif v['lefthand'][3] == 1:
					fh.write('0\t1\t0\t')
				elif v['lefthand'][3] == 2:
					fh.write('0\t0\t1\t')

				fh.write(str(v['righthand'][0]) + '\t' + str(v['righthand'][1]) + '\t' + str(v['righthand'][2]) + '\t')
				if v['righthand'][3] == 0:
					fh.write('1\t0\t0\n')
				elif v['righthand'][3] == 1:
					fh.write('0\t1\t0\n')
				elif v['righthand'][3] == 2:
					fh.write('0\t0\t1\n')
			fh.close()

			fh = open(enactment + '_poses.shutoff.txt', 'w')
			fh.write('#  timestamp, label, image-file, LHx, LHy, LHz, LHstate_0, LHstate_1, LHstate_2, RHx, RHy, RHz, RHstate_0, RHstate_1, RHstate_2\n')
			fh.write('#  All hand positions are taken from JSON and made relative to the head in the head\'s frame.\n')
			fh.write('#  In this file, hand vectors are set to zero if the hand does not project into the camera!\n')
			for k, v in sorted(samples.items()):
				fh.write(str(k) + '\t')
				if v['label'] is None:
					fh.write('*' + '\t')
				else:
					fh.write(v['label'] + '\t')
				fh.write(v['file'] + '\t')

				if projects_into_camera(v['globalLH'], v['head'][:3], v['head'][3], K, params) is not None:
					fh.write(str(v['lefthand'][0])  + '\t' + str(v['lefthand'][1])  + '\t' + str(v['lefthand'][2])  + '\t')
					if v['lefthand'][3] == 0:
						fh.write('1\t0\t0\t')
					elif v['lefthand'][3] == 1:
						fh.write('0\t1\t0\t')
					elif v['lefthand'][3] == 2:
						fh.write('0\t0\t1\t')
				else:
					fh.write('0.0\t0.0\t0.0\t0\t0\t0\t')

				if projects_into_camera(v['globalRH'], v['head'][:3], v['head'][3], K, params) is not None:
					fh.write(str(v['righthand'][0]) + '\t' + str(v['righthand'][1]) + '\t' + str(v['righthand'][2]) + '\t')
					if v['righthand'][3] == 0:
						fh.write('1\t0\t0\n')
					elif v['righthand'][3] == 1:
						fh.write('0\t1\t0\n')
					elif v['righthand'][3] == 2:
						fh.write('0\t0\t1\n')
				else:
					fh.write('0.0\t0.0\t0.0\t0\t0\t0\n')
			fh.close()

			#########################################################  Making centipedes and videos?
			if params['render']:
				#  For the 'skeleton' PLYs, there will be a single centipede, entirely connected through TIME.
				#    There will be no redundant vertices.
				#  For the 'action' centipedes, there will be one or more centipedes, each connected by ACTION.
				#    There will be redundant vertices if there are overlapping action labels.

				#####################################################  Write the left-hand/right-hand one in the head frame
				totalV = len(samples) * 3
				totalE = len(samples) - 1 + 2 * len(samples)

				fh = open(enactment + '.skeleton.headframe.ply', 'w')
				fh.write('ply' + '\n')
				fh.write('format ascii 1.0' + '\n')
				fh.write('comment Head = white' + '\n')
				fh.write('comment Left hand = green' + '\n')
				fh.write('comment Right hand = red' + '\n')
				fh.write('element vertex ' + str(totalV) + '\n')
				fh.write('property float x' + '\n')
				fh.write('property float y' + '\n')
				fh.write('property float z' + '\n')
				fh.write('property uchar red' + '\n')
				fh.write('property uchar green' + '\n')
				fh.write('property uchar blue' + '\n')
				fh.write('element edge ' + str(totalE) + '\n')
				fh.write('property int vertex1' + '\n')
				fh.write('property int vertex2' + '\n')
				fh.write('end_header' + '\n')

				lines = []
				ctr = 0
				for pose in [x[1] for x in sorted(samples.items(), key=lambda y: y[0])]:
					fh.write(str(pose['head'][0]) + ' ' + str(pose['head'][1]) + ' ' + str(pose['head'][2]) + ' 255 255 255' + '\n')
					fh.write(str(pose['head'][3][:, 0].dot(pose['lefthand'][:3]) + pose['head'][0]) + ' ' + \
					         str(pose['head'][3][:, 1].dot(pose['lefthand'][:3]) + pose['head'][1]) + ' ' + \
					         str(pose['head'][3][:, 2].dot(pose['lefthand'][:3]) + pose['head'][2]) + \
					         ' 0 255 0' + '\n')
					fh.write(str(pose['head'][3][:, 0].dot(pose['righthand'][:3]) + pose['head'][0]) + ' ' + \
					         str(pose['head'][3][:, 1].dot(pose['righthand'][:3]) + pose['head'][1]) + ' ' + \
					         str(pose['head'][3][:, 2].dot(pose['righthand'][:3]) + pose['head'][2]) + \
					         ' 255 0 0' + '\n')
					lines.append( (ctr, ctr + 1) )
					lines.append( (ctr, ctr + 2) )
					if ctr < totalV - 3:
						lines.append( (ctr, ctr + 3) )
					ctr += 3

				for line in lines:
					fh.write(str(line[0]) + ' ' + str(line[1]) + '\n')
				fh.close()

				#####################################################  Write the left-hand/right-hand one in the global frame
				totalV = len(samples) * 3
				totalE = len(samples) - 1 + 2 * len(samples)

				fh = open(enactment + '.skeleton.global.ply', 'w')
				fh.write('ply' + '\n')
				fh.write('format ascii 1.0' + '\n')
				fh.write('comment Head = white' + '\n')
				fh.write('comment Left hand = green' + '\n')
				fh.write('comment Right hand = red' + '\n')
				fh.write('element vertex ' + str(totalV) + '\n')
				fh.write('property float x' + '\n')
				fh.write('property float y' + '\n')
				fh.write('property float z' + '\n')
				fh.write('property uchar red' + '\n')
				fh.write('property uchar green' + '\n')
				fh.write('property uchar blue' + '\n')
				fh.write('element edge ' + str(totalE) + '\n')
				fh.write('property int vertex1' + '\n')
				fh.write('property int vertex2' + '\n')
				fh.write('end_header' + '\n')

				lines = []
				ctr = 0
				for pose in [x[1] for x in sorted(samples.items(), key=lambda y: y[0])]:
					fh.write(str(pose['head'][0]) + ' ' + str(pose['head'][1]) + ' ' + str(pose['head'][2]) + ' 255 255 255' + '\n')
					fh.write(str(pose['globalLH'][0]) + ' ' + \
					         str(pose['globalLH'][1]) + ' ' + \
					         str(pose['globalLH'][2]) + ' 0 255 0' + '\n')
					fh.write(str(pose['globalRH'][0]) + ' ' + \
					         str(pose['globalRH'][1]) + ' ' + \
					         str(pose['globalRH'][2]) + ' 255 0 0' + '\n')
					lines.append( (ctr, ctr + 1) )
					lines.append( (ctr, ctr + 2) )
					if ctr < totalV - 3:
						lines.append( (ctr, ctr + 3) )
					ctr += 3

				for line in lines:
					fh.write(str(line[0]) + ' ' + str(line[1]) + '\n')
				fh.close()

				#####################################################  Write the color-coded action one in the head frame
				samples_p = {}										#  Build a reverse dictionary:
				for v in samples.values():							#  key:label ==> val:temporal list of dict/frames
					if v['label'] is not None:
						samples_p[ v['label'] ] = []

				totalV = 0											#  samples_p[label] = [ {[timestamp]  = timestamp
				totalE = 0											#                        [head]       = (x, y, z)
																	#                        [lefthand]   = (x, y, z)
																	#                        [righthand]} = (x, y, z)
																	#                        [file]       = string,
																	#                       {[timestamp]  = timestamp
																	#                        [head]       = (x, y, z)
				for label in samples_p.keys():						#                        [lefthand]   = (x, y, z)
					if label is not None:							#                        [righthand]} = (x, y, z)
						for timestamp, v in samples.items():		#                        [file]       = string,
																	#
							if v['label'] == label:					#                           ... ,
																	#                       {[timestamp]  = timestamp
								samples_p[label].append( {} )		#                        [head]       = (x, y, z)
																	#                        [lefthand]   = (x, y, z)
																	#                        [righthand]} = (x, y, z)
																	#                        [file]       = string     ]
								samples_p[label][-1]['timestamp'] = timestamp
								samples_p[label][-1]['head'] = v['head']
								samples_p[label][-1]['lefthand'] = v['lefthand']
								samples_p[label][-1]['righthand'] = v['righthand']
								samples_p[label][-1]['globalLH'] = v['globalLH']
								samples_p[label][-1]['globalRH'] = v['globalRH']
								samples_p[label][-1]['file'] = v['file']
								totalV += 3
						if len(samples_p[label]) > 0:
							totalE += len(samples_p[label]) * 2 + len(samples_p[label]) - 1

				for timestamp, v in samples_p.items():				#  Sort temporally
					samples_p[timestamp] = sorted(v, key=lambda x: x['timestamp'])

				fh = open(enactment + '.actions.headframe.ply', 'w')
				fh.write('ply' + '\n')
				fh.write('format ascii 1.0' + '\n')
				for k, v in actionColors.items():
					fh.write('comment ' + k + ' = (' + ' '.join([str(x) for x in v]) + ')\n')
				fh.write('element vertex ' + str(totalV) + '\n')
				fh.write('property float x' + '\n')
				fh.write('property float y' + '\n')
				fh.write('property float z' + '\n')
				fh.write('property uchar red' + '\n')
				fh.write('property uchar green' + '\n')
				fh.write('property uchar blue' + '\n')
				fh.write('element edge ' + str(totalE) + '\n')
				fh.write('property int vertex1' + '\n')
				fh.write('property int vertex2' + '\n')
				fh.write('end_header' + '\n')

				lines = []
				ctroffset = 0
				for k, v in samples_p.items():						#  For every LABEL...
					ctr = 0
					act = k.split('(')[0]							#  ACT, derived from LABEL, determines COLOR
					for pose in v:
						fh.write(str(pose['head'][0]) + ' ' + str(pose['head'][1]) + ' ' + str(pose['head'][2]) + ' ' + ' '.join([str(x) for x in actionColors[act]]) + '\n')

						fh.write(str(pose['head'][3][:, 0].dot(pose['lefthand'][:3]) + pose['head'][0]) + ' ' + \
						         str(pose['head'][3][:, 1].dot(pose['lefthand'][:3]) + pose['head'][1]) + ' ' + \
						         str(pose['head'][3][:, 2].dot(pose['lefthand'][:3]) + pose['head'][2]) + ' ' + \
						         ' '.join([str(x) for x in actionColors[act]]) + ' ' + '\n')
						fh.write(str(pose['head'][3][:, 0].dot(pose['righthand'][:3]) + pose['head'][0]) + ' ' + \
						         str(pose['head'][3][:, 1].dot(pose['righthand'][:3]) + pose['head'][1]) + ' ' + \
						         str(pose['head'][3][:, 2].dot(pose['righthand'][:3]) + pose['head'][2]) + ' ' + \
						         ' '.join([str(x) for x in actionColors[act]]) + ' ' + '\n')
						lines.append( (ctroffset, ctroffset + 1) )
						lines.append( (ctroffset, ctroffset + 2) )
						if ctr > 0:
							lines.append( (ctroffset - 3, ctroffset) )
						ctroffset += 3
						ctr += 1

				for line in lines:
					fh.write(str(line[0]) + ' ' + str(line[1]) + '\n')
				fh.close()

				#####################################################  Make the color-coded action centipede in the global frame
				fh = open(enactment + '.actions.global.ply', 'w')
				fh.write('ply' + '\n')
				fh.write('format ascii 1.0' + '\n')
				for k, v in actionColors.items():
					fh.write('comment ' + k + ' = (' + ' '.join([str(x) for x in v]) + ')\n')
				fh.write('element vertex ' + str(totalV) + '\n')
				fh.write('property float x' + '\n')
				fh.write('property float y' + '\n')
				fh.write('property float z' + '\n')
				fh.write('property uchar red' + '\n')
				fh.write('property uchar green' + '\n')
				fh.write('property uchar blue' + '\n')
				fh.write('element edge ' + str(totalE) + '\n')
				fh.write('property int vertex1' + '\n')
				fh.write('property int vertex2' + '\n')
				fh.write('end_header' + '\n')

				lines = []
				ctroffset = 0
				for k, v in samples_p.items():						#  For every LABEL...
					ctr = 0
					act = k.split('(')[0]							#  ACT, derived from LABEL, determines COLOR
					for pose in v:
						fh.write(str(pose['head'][0]) + ' ' + str(pose['head'][1]) + ' ' + str(pose['head'][2]) + ' ' + ' '.join([str(x) for x in actionColors[act]]) + '\n')

						fh.write(str(pose['globalLH'][0]) + ' ' + \
						         str(pose['globalLH'][1]) + ' ' + \
						         str(pose['globalLH'][2]) + ' ' + \
						         ' '.join([str(x) for x in actionColors[act]]) + ' ' + '\n')
						fh.write(str(pose['globalRH'][0]) + ' ' + \
						         str(pose['globalRH'][1]) + ' ' + \
						         str(pose['globalRH'][2]) + ' ' + \
						         ' '.join([str(x) for x in actionColors[act]]) + ' ' + '\n')
						lines.append( (ctroffset, ctroffset + 1) )
						lines.append( (ctroffset, ctroffset + 2) )
						if ctr > 0:
							lines.append( (ctroffset - 3, ctroffset) )
						ctroffset += 3
						ctr += 1

				for line in lines:
					fh.write(str(line[0]) + ' ' + str(line[1]) + '\n')
				fh.close()

				#####################################################  Make videos
				vid = cv2.VideoWriter(enactment + '_annotated.avi', \
				                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
				                      10, \
				                      (params['imgw'], params['imgh']) )
				for k, v in sorted(samples.items()):
					img = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/' + v['file'], cv2.IMREAD_UNCHANGED)
					if img.shape[2] == 4:							#  Drop alpha channel
						img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
					if v['label'] is None:
																	#  Write the action
						cv2.putText(img, '<NEUTRAL>', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (255, 255, 255, 255), 3)
					else:
																	#  Write the action
						cv2.putText(img, v['label'], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (255, 255, 255, 255), 3)

																	#  Build left-hand string
					LHstr  = "{:.2f}".format(v['lefthand'][0]) + ' '
					LHstr += "{:.2f}".format(v['lefthand'][1]) + ' '
					LHstr += "{:.2f}".format(v['lefthand'][2]) + ' '
					LHstr += str(v['lefthand'][3])
																	#  Does left hand project into image?
					LHproj = projects_into_camera(v['globalLH'], v['head'][:3], v['head'][3], K, params)
					if LHproj is not None:							#  It projects: draw a green-for-left circle and write white text
						cv2.circle(img, (params['imgw'] - LHproj[0], params['imgh'] - LHproj[1]), 5, (0, 255, 0, 255), 3)
						cv2.putText(img, 'Left Hand: ' + LHstr, (10, params['imgh'] - 160), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (0, 255, 0, 255), 3)
					else:
																	#  It does NOT project: write black text
						cv2.putText(img, 'Left Hand: ' + LHstr, (10, params['imgh'] - 160), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (0, 128, 0, 255), 3)

																	#  Build right-hand string
					RHstr  = "{:.2f}".format(v['righthand'][0]) + ' '
					RHstr += "{:.2f}".format(v['righthand'][1]) + ' '
					RHstr += "{:.2f}".format(v['righthand'][2]) + ' '
					RHstr += str(v['righthand'][3])
																	#  Does right hand project into image?
					RHproj = projects_into_camera(v['globalRH'], v['head'][:3], v['head'][3], K, params)
					if RHproj is not None:							#  It projects: draw a red-for-right circle and write white text
						cv2.circle(img, (params['imgw'] - RHproj[0], params['imgh'] - RHproj[1]), 5, (0, 0, 255, 255), 3)
						cv2.putText(img, 'Right Hand: ' + RHstr, (10, params['imgh'] - 40), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (0, 0, 255, 255), 3)
					else:
																	#  It does NOT project: write black text
						cv2.putText(img, 'Right Hand: ' + RHstr, (10, params['imgh'] - 40), cv2.FONT_HERSHEY_SIMPLEX, params['fontsize'], (0, 0, 128, 255), 3)

					vid.write(img)									#  Write the frame to the current video

				vid.release()
		elif params['verbose']:
			print('*** ' + enactment + '_poses.{full,shutoff}.txt already exist. Skipping.')
	return

def projects_into_camera(handpos, headpos, headrot, K, params):
	world = np.array( [ [handpos[0]], \
	                    [handpos[1]], \
	                    [handpos[2]], \
	                    [ 1.0      ] ])
																	#  Construct hand location in homogeneous world-coordinates
																	#  Build the translation for the head
	T = np.concatenate((np.eye(3), np.array([[-headpos[0]], \
	                                         [-headpos[1]], \
	                                         [-headpos[2]]])), axis=1)
																	#  Project hand
	img = np.dot(K, np.dot(headrot, np.dot(T, world)))
	img /= img[2][0]

	x = int(round(img[0][0]))										#  Round and discretize to pixels
	y = int(round(img[1][0]))

	if x >= 0 and x < params['imgw'] and y >= 0 and y < params['imgh']:
		return (x, y)
	return None

#  Iterate over all enactments to build a master set of all labels/actions used.
#  Assign each a color.
#  This is relevant for building the centipedes.
def survey_action_colors(params):
	actionColors = {}												#  string ==> (r, g, b)

	for enactment in params['enactments']:							#  Survey actions across all enactments

		if params['verbose']:
			print('>>> Surveying actions in ' + enactment)

		fh = open(enactment + '/Labels.fvr', 'r')					#  Fetch action time stamps
		lines = ' '.join(fh.readlines())							#  These USED to be a single line; NOW they have carriage returns
		fh.close()
		labels = json.loads(lines)

		for action in labels['list']:
			act = action['stepDescription'].split('(')[0]
			actionColors[ act ] = True								#  True for now... once we've counted everything, these will be well-spaced colors

	if params['verbose']:
		print('>>> ' + str(len(actionColors)) + ' unique actions across all specified enactments')

																	#  Build colors for all enactments
	colors = np.array([ [int(x * 255)] for x in np.linspace(0.0, 1.0, len(actionColors)) ], np.uint8)
	colors = cv2.applyColorMap(colors, cv2.COLORMAP_JET)
	i = 0
	for k in actionColors.keys():
		actionColors[k] = [int(x) for x in colors[i][0]]
		actionColors[k].reverse()
		actionColors[k] = tuple(actionColors[k])
		i += 1

	return actionColors

#  Run what used to be 22feb21's check.py script and what used to be 22feb21's histogram_z.py script:
#  Survey depth ranges of all given enactments. Find the lowest low and the highest high.
#  Make sure all depth maps are single-channel images. Plot a histogram of all depths used in given enactments.
#  Render composite videos of all given enactments so that we can see at a glance whether the depth or color maps have been mismatched. It's happened before.
#  SKIP ALL OF THIS IF A LOG FILE ALREADY EXISTS
def check(params):
	if not os.path.exists('FVR_check.log') or params['force']:
		if params['verbose']:
			print('********************')
			print('** Running checks **')
			print('********************')

		depthRange = {}
		depthRange['min'] = float('inf')
		depthRange['max'] = float('-inf')
		for enactment in params['enactments']:
			fh = open(enactment + '/metadata.fvr', 'r')
			line = fh.readlines()[0]
			fh.close()
			metadata = json.loads(line)
			if params['verbose']:
				print('>>> ' + enactment + '/metadata.fvr depth range: [' + str(metadata['depthImageRange']['x']) + ', ' + str(metadata['depthImageRange']['y']) + ']')
			if metadata['depthImageRange']['x'] < depthRange['min']:
				depthRange['min'] = metadata['depthImageRange']['x']#  Save Z-map's minimum (in meters)
			if metadata['depthImageRange']['y'] > depthRange['max']:
				depthRange['max'] = metadata['depthImageRange']['y']#  Save Z-map's maximum (in meters)
		if params['verbose']:
			print('>>> Consensus depth range across all metadata files: [' + str(depthRange['min']) + ', ' + str(depthRange['max']) + ']')

		histogram = {}												#  Prepare the histogram--IN METERS
		for i in range(0, 256):
			d = (float(i) / 255.0) * (depthRange['max'] - depthRange['min']) + depthRange['min']
			histogram[d] = 0

		fh = open('FVR_check.log', 'w')

		for enactment in params['enactments']:						#  Make sure depth maps are grayscale/single-channel
			if params['verbose']:
				print('>>> Scanning ' + enactment + ' depth maps')
			for depthmapfilename in sorted(os.listdir(enactment + '/Users/' + params['User'] + '/POV/DepthMapCameraFrames/'), key=lambda x: int(x.split('_')[0])):
				depthmap = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/DepthMapCameraFrames/' + depthmapfilename, cv2.IMREAD_UNCHANGED)
				if len(depthmap.shape) != 2:
					fh.write(enactment + '/Users/' + params['User'] + '/POV/DepthMapCameraFrames/' + depthmapfilename + '\t' + 'is not single-channel' + '\n')
					if params['verbose']:
						print(enactment + '/Users/' + params['User'] + '/POV/DepthMapCameraFrames/' + depthmapfilename + ' is not single-channel')
					if depthmap.shape[3] == 4:						#  If it's RGB + alpha, drop the alpha
						depthmap = cv2.cvtColor(depthmap, cv2.COLOR_BGRA2BGR)
																	#  Convert RGB to gray
					depthmap = cv2.cvtColor(depthmap, cv2.COLOR_BGR2GRAY)
				histg = [int(x) for x in cv2.calcHist([depthmap], [0], None, [256], [0, 256])]
				for i in range(0, 256):
					d = (float(i) / 255.0) * (depthRange['max'] - depthRange['min']) + depthRange['min']
					histogram[d] += histg[i]
		if params['verbose']:
			print('')

		lo = 0														#  Discover least and greatest values used
		while lo < 256 and histogram[ (float(lo) / 255.0) * (depthRange['max'] - depthRange['min']) + depthRange['min'] ] == 0:
			lo += 1
		hi = 255
		while hi >= 0 and histogram[ (float(hi) / 255.0) * (depthRange['max'] - depthRange['min']) + depthRange['min'] ] == 0:
			hi -= 1
		histg = []
		for i in range(0, 256):
			d = (float(i) / 255.0) * (depthRange['max'] - depthRange['min']) + depthRange['min']
			histg.append( histogram[d] )
																	#  Save the histogram
		plt.bar([(float(i) / 255.0) * (depthRange['max'] - depthRange['min']) + depthRange['min'] for i in range(0, 256)], height=histg)
		plt.xlabel('Depth')
		plt.ylabel('Frequency')
		plt.title('Depths Used')
		plt.savefig('histogram.png')

		if params['verbose']:
			print('>>> Least depth encountered = ' + str((float(lo) / 255.0) * (depthRange['max'] - depthRange['min']) + depthRange['min']))
			print('>>> Greatest depth encountered = ' + str((float(hi) / 255.0) * (depthRange['max'] - depthRange['min']) + depthRange['min']))

		for enactment in params['enactments']:						#  Build composite videos so we can spot bogies
			if params['verbose']:
				print('>>> Building composite video for ' + enactment)
			vid = cv2.VideoWriter(enactment + '_composite.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (params['imgw'], params['imgh']))
			for framename in sorted(os.listdir(enactment + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/'), key=lambda x: int(x.split('_')[0])):

				normal_exists = os.path.exists(enactment + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/' + framename)
				segmap_exists = os.path.exists(enactment + '/Users/' + params['User'] + '/POV/ColorMapCameraFrames/' + framename)
				dmap_exists   = os.path.exists(enactment + '/Users/' + params['User'] + '/POV/DepthMapCameraFrames/' + framename)

				if normal_exists and segmap_exists and dmap_exists:

					frame = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/' + framename, cv2.IMREAD_UNCHANGED)
					if frame.shape[2] == 4:							#  Do these guys have alpha channels? I forget.
						frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

					seg_oly = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/ColorMapCameraFrames/' + framename, cv2.IMREAD_UNCHANGED)
					if seg_oly.shape[2] == 4:						#  Some of these guys have an alpha channel; some of these guys don't
						seg_oly = cv2.cvtColor(seg_oly, cv2.COLOR_BGRA2BGR)

					d_oly = cv2.imread(enactment + '/Users/' + params['User'] + '/POV/DepthMapCameraFrames/' + framename, cv2.IMREAD_UNCHANGED)
					if len(d_oly.shape) != 2:
						d_oly = d_oly[:, :, 0]						#  Isolate single channel
					d_oly = cv2.cvtColor(d_oly, cv2.COLOR_GRAY2BGR)	#  Restore color and alpha channels so we can map it to the rest

					d_oly = cv2.addWeighted(d_oly, 1.0, seg_oly, 0.7, 0)

					frame = cv2.addWeighted(frame, 1.0, d_oly, 0.7, 0)

					frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)	#  Flatten alpha

					vid.write(frame)
				else:
					if not normal_exists:
						fh.write(enactment + '/Users/' + params['User'] + '/POV/NormalViewCameraFrames/' + framename + '\tdoes not exist\n')
					if not segmap_exists:
						fh.write(enactment + '/Users/' + params['User'] + '/POV/ColorMapCameraFrames/' + framename + '\tdoes not exist\n')
					if not dmap_exists:
						fh.write(enactment + '/Users/' + params['User'] + '/POV/DepthMapCameraFrames/' + framename + '\tdoes not exist\n')

			vid.release()

		fh.close()
	elif params['verbose']:
		print('*** FVR_check.log already exists. Skipping the checks.')

	return

def getCommandLineParams():
	params = {}
	params['enactments'] = []										#  List of file paths
	params['rules'] = None											#  Rules file
	params['colors'] = None											#  Objects' colors file, if applicable
	params['force'] = False											#  Whether to force recomputing intermediate files
	params['mutex'] = False											#  False: allow overlapping instances;
																	#  True: the smaller object punches a hole in the larger object
	params['verbose'] = False
	params['helpme'] = False

	params['render'] = False										#  Whether to render stuff.

	params['fontsize'] = 1											#  For rendering text to images and videos
	params['imgw'] = 1280											#  It used to be 1920, and I don't want to change a bunch of ints when it changes again
	params['imgh'] = 720											#  It used to be 1080, and I don't want to change a bunch of ints when it changes again
	params['focal'] = None											#  Can be used to override the focal length we otherwise derive from metadata
	params['User'] = 'vr1'											#  It used to be "admin", and I don't want to change a bunch of file paths when it changes again

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-e', '-f', '-r', '-mutex', \
	         '-render', '-colors', \
	         '-v', '-?', '-help', '--help', \
	         '-imgw', '-imgh', '-focal', '-User', '-fontsize']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-v':
				params['verbose'] = True
			elif sys.argv[i] == '-f':
				params['force'] = True
			elif sys.argv[i] == '-mutex':
				params['mutex'] = True
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
				elif argtarget == '-imgw':
					params['imgw'] = int(argval)
				elif argtarget == '-imgh':
					params['imgh'] = int(argval)
				elif argtarget == '-User':
					params['User'] = argval
				elif argtarget == '-fontsize':
					params['fontsize'] = float(argval)
				elif argtarget == '-gpus':
					params['gpus'] = int(argval)
				elif argtarget == '-colors':
					params['colors'] = {}
					fh = open(argval, 'r')
					for line in fh.readlines():
						arr = line.strip().split('\t')
						params['colors'][arr[0]] = [int(arr[1]), int(arr[2]), int(arr[3])]
					fh.close()
				elif argtarget == '-focal':
					params['focal'] = float(argval)

	if params['fontsize'] < 1:
		print('>>> INVALID DATA received for fontsize. Restoring default value.')
		params['fontsize'] = 1

	if params['imgw'] < 1 or params['imgh'] < 1:
		print('>>> INVALID DATA received for image dimensions. Restoring default values.')
		params['imgw'] = 1280
		params['imgh'] = 720

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('You\'ve received new or revised FactualVR enactments.')
	print('Now you want to processes and check them.')
	print('')
	print('This script builds a ground-truth lookup file for each enactment. Each row contains:')
	print('    <timestamp, image-filename, label, object-class-name, score, BBox-x1,BBox-y1;BBox-x2,BBox-y2, mask-filename>')
	print('These are derived from the segmentation maps (ColorMapCameraFrames).')
	print('Frames without detectable objects contain only the first element, image-filename.')
	print('')
	print('Usage:  python3.5 process_enactment.py <parameters, preceded by flags>')
	print(' e.g.:  python3.5 process_enactment.py -r substation.rules -e Enactment11 -v -render -colors colors.txt')
	print('')
	print('Flags:  -e           MUST HAVE AT LEAST ONE: Path to a directory of raw enactment materials: JSONs and color maps.')
	print('        -r           MUST HAVE EXACTLY ONE: Path to a rules file that tells the parser which objects are which.')
	print('        -f           Force re-computing of intermediate files that would otherwise be left as they are.')
	print('        -render      Generate illustrations (videos, 3D representations) for all given enactments.')
	print('        -v           Enable verbosity')
	print('        -?           Display this message')

if __name__ == '__main__':
	main()