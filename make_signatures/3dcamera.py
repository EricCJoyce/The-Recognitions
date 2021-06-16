'''
  Eric C. Joyce, Stevens Institute of Technology, 2020

  Import your object into Blender. Let it sit wherever it wants to appear according to its native coordinate system.
  We will move the camera around the object.
  Position the Blender camera and three lights.
  This initial camera position will form the object's signature AND become the starting point of its orbit around the object.

  This script outputs the following for OBJECT:
    ./depth/OBJECT/depth.N.dat
    ./dimensions/OBJECT/dimensions.N.dat
    ./K/OBJECT/K.N.dat
    ./rgb/OBJECT/rgb.N.png
    ./Rt/OBJECT/Rt.N.dat
  where N is an integer >= 0 to identify different virtual camera positions.

MAKE SURE THE TARGET OBJECT IS SELECTED AND IN OBJECT MODE.
THEN, in the Blender Python console, type:

filename = '/home/eric/Documents/blender/3dcamera.py'
exec(compile(open(filename).read(), filename, 'exec'))

'''

import bpy
from bpy import context
import bpy_extras
import struct
import sys
from array import array
from mathutils import Matrix
import numpy as np
import numpy.linalg

def main():
	targetObject = 'big_switch'										#  <------- CHANGE ME
	renderPath = '/home/eric/Documents/blender/make_signatures/'
	steps_in_degrees = 1.0
	verbose = True

	#################################################################  Now orbit the object and render rgb, K, and Rt files at each step.
	#  REMEMBER THAT THE TARGET OBJECT HAS TO BE SELECTED AND IN OBJECT MODE FOR THIS TO WORK

	centroid = [0.0, 0.0, 0.0]										#  Compute object centroid
	obj = context.active_object
	verts = [vert.co for vert in obj.data.vertices]
	for vert in verts:
		centroid[0] += vert[0]
		centroid[1] += vert[1]
		centroid[2] += vert[2]
	centroid[0] /= float(len(verts))
	centroid[1] /= float(len(verts))
	centroid[2] /= float(len(verts))
	centroid = Vector((centroid[0], centroid[1], centroid[2]))

	lookAt(centroid)

	cam_loc = bpy.data.objects["Camera"].location
	direction = centroid - cam_loc
	up = bpy.data.objects['Camera'].matrix_world.to_quaternion() @ Vector((0.0, 1.0, 0.0))
	radius = np.linalg.norm( np.array([direction[0], direction[1], direction[2]]) )
	direction /= radius

	cross = Vector((direction[1] * up[2] - direction[2] * up[1], \
	                direction[2] * up[0] - direction[0] * up[2], \
	                direction[0] * up[1] - direction[1] * up[0] ))

	i = 0
	while float(i) * steps_in_degrees < 360.0:
																	#  Update camera position
		bpy.data.objects["Camera"].location[0] = centroid[0] + radius * -np.cos( float(i) * steps_in_degrees * np.pi / 180.0 ) * cross[0] \
		                                                     + radius * np.sin( float(i) * steps_in_degrees * np.pi / 180.0 ) * direction[0]
		bpy.data.objects["Camera"].location[1] = centroid[1] + radius * -np.cos( float(i) * steps_in_degrees * np.pi / 180.0 ) * cross[1] \
		                                                     + radius * np.sin( float(i) * steps_in_degrees * np.pi / 180.0 ) * direction[1]
		bpy.data.objects["Camera"].location[2] = centroid[2] + radius * -np.cos( float(i) * steps_in_degrees * np.pi / 180.0 ) * cross[2] \
		                                                     + radius * np.sin( float(i) * steps_in_degrees * np.pi / 180.0 ) * direction[2]

		lookAt(centroid)											#  Look at the centroid again

		depthBuff = bpy.data.images['Viewer Node']					#  Retrieve from node

																	#  Render stuff (step 0 will match signature-image exactly)
		writeDepthBuffer(renderPath + 'depth/' + targetObject + '/depth.' + str(i) + '.dat', depthBuff, verbose)
		writeDimensions(renderPath + 'dimensions/' + targetObject + '/dimensions.' + str(i) + '.dat', depthBuff, verbose)
		writeK(renderPath + 'K/' + targetObject + '/K.' + str(i) + '.dat', verbose)
		writeRGB(renderPath + 'rgb/' + targetObject + '/rgb.' + str(i) + '.png', verbose)
		writeRt(renderPath + 'Rt/' + targetObject + '/Rt.' + str(i) + '.dat', verbose)

		i += 1

	return

def writeRGB(filepath, verbose=False):
	if verbose:
		print('    Writing rendering to file...')
	bpy.context.scene.render.image_settings.file_format = 'PNG'
	bpy.context.scene.render.filepath = filepath
	bpy.ops.render.render(write_still=1)
	if verbose:
		print('      done')
	return

def writeDimensions(filepath, depthBuff, verbose=False):
	if verbose:
		print('    Writing buffer dimensions to file...')
	fh = open(filepath, 'wb')
	if verbose:
		print('      ' + str(depthBuff.size[0]) + ' x ' + str(depthBuff.size[1]))
	fh.write(struct.pack('>I', depthBuff.size[0]))
	fh.write(struct.pack('>I', depthBuff.size[1]))
	fh.close()
	if verbose:
		print('      done')
	return

def writeK(filepath, verbose=False):
	if verbose:
		print('    Writing intrinsic matrix to file...')

	_, K, Rt = get_3x4_P_matrix_from_blender(bpy.context.scene.camera)

	fh = open(filepath, 'wb')
	fh.write(struct.pack('>f', K[0][0]))							#  Write row-major
	fh.write(struct.pack('>f', K[0][1]))
	fh.write(struct.pack('>f', K[0][2]))

	fh.write(struct.pack('>f', K[1][0]))
	#fh.write(struct.pack('>f', K[1][1]))
	fh.write(struct.pack('>f', K[0][0]))
	fh.write(struct.pack('>f', K[1][2]))

	fh.write(struct.pack('>f', K[2][0]))
	fh.write(struct.pack('>f', K[2][1]))
	fh.write(struct.pack('>f', K[2][2]))
	fh.close()
	if verbose:
		print('      [' + str(K[0][0]) + '\t' + str(K[0][1]) + '\t' + str(K[0][2]) + ']')
		#print('      [' + str(K[1][0]) + '\t' + str(K[1][1]) + '\t' + str(K[1][2]) + ']')
		print('      [' + str(K[1][0]) + '\t' + str(K[0][0]) + '\t' + str(K[1][2]) + ']')
		print('      [' + str(K[2][0]) + '\t' + str(K[2][1]) + '\t' + str(K[2][2]) + ']')
		print('      done')

	return

def writeRt(filepath, verbose=False):
	if verbose:
		print('    Writing extrinsic matrix to file...')

	_, K, Rt = get_3x4_P_matrix_from_blender(bpy.context.scene.camera)

	fh = open(filepath, 'wb')
	fh.write(struct.pack('>f', Rt[0][0]))							#  Write row-major
	fh.write(struct.pack('>f', Rt[0][1]))
	fh.write(struct.pack('>f', Rt[0][2]))
	fh.write(struct.pack('>f', Rt[0][3]))

	fh.write(struct.pack('>f', Rt[1][0]))
	fh.write(struct.pack('>f', Rt[1][1]))
	fh.write(struct.pack('>f', Rt[1][2]))
	fh.write(struct.pack('>f', Rt[1][3]))

	fh.write(struct.pack('>f', Rt[2][0]))
	fh.write(struct.pack('>f', Rt[2][1]))
	fh.write(struct.pack('>f', Rt[2][2]))
	fh.write(struct.pack('>f', Rt[2][3]))

	fh.write(struct.pack('>f', 0.0))
	fh.write(struct.pack('>f', 0.0))
	fh.write(struct.pack('>f', 0.0))
	fh.write(struct.pack('>f', 1.0))
	fh.close()
	if verbose:
		print('      [' + str(Rt[0][0]) + '\t' + str(Rt[0][1]) + '\t' + str(Rt[0][2]) + '\t' + str(Rt[0][3]) + ']')
		print('      [' + str(Rt[1][0]) + '\t' + str(Rt[1][1]) + '\t' + str(Rt[1][2]) + '\t' + str(Rt[1][3]) + ']')
		print('      [' + str(Rt[2][0]) + '\t' + str(Rt[2][1]) + '\t' + str(Rt[2][2]) + '\t' + str(Rt[2][3]) + ']')
		print('      [' + str(0.0)      + '\t' + str(0.0)      + '\t' + str(0.0)      + '\t' + str(1.0)      + ']')
		print('      done')
	return

def writeDepthBuffer(filepath, depthBuff, verbose=False):
	if verbose:
		print('    Writing depth buffer to file...')

	fh = open(filepath, 'wb')
	depthBuffSize = int(depthBuff.size[0] * depthBuff.size[1] * 4)
	d = list(depthBuff.pixels)[0:depthBuffSize:4]

	float_array = array('f', d)
	if verbose:
		print('      Depth buffer size:     ' + str(depthBuffSize))
		print('      Length of list:        ' + str(len(d)))
		print('      Length of float array: ' + str(len(float_array)))
	float_array.tofile(fh)

	fh.close()
	if verbose:
		print('      done')
	return

#  Make the camera look at the given point 'pt'
def lookAt(pt):
	cam_loc = bpy.data.objects["Camera"].location
	direction = pt - cam_loc
	rot_quat = direction.to_track_quat('-Z', 'Y')					#  Point the camera's '-Z', use its 'Y' as up
	bpy.data.objects["Camera"].rotation_euler = rot_quat.to_euler()	#  Set camera's Euler rotation
	return

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(camd):
	f_in_mm = camd.lens
	scene = bpy.context.scene
	resolution_x_in_px = scene.render.resolution_x
	resolution_y_in_px = scene.render.resolution_y
	scale = scene.render.resolution_percentage / 100

	sensor_width_in_mm = camd.sensor_width
	sensor_height_in_mm = camd.sensor_height

	pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

	if (camd.sensor_fit == 'VERTICAL'):
		# the sensor height is fixed (sensor fit is horizontal),
		# the sensor width is effectively changed with the pixel aspect ratio
		s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
		s_v = resolution_y_in_px * scale / sensor_height_in_mm

	else: # 'HORIZONTAL' and 'AUTO'
		# the sensor width is fixed (sensor fit is horizontal),
		# the sensor height is effectively changed with the pixel aspect ratio
		s_u = resolution_x_in_px * scale / sensor_width_in_mm
		s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

	# Parameters of intrinsic calibration matrix K

	alpha_u = f_in_mm * s_u
	alpha_v = f_in_mm * s_v

	u_0 = resolution_x_in_px * scale / 2
	v_0 = resolution_y_in_px * scale / 2
	skew = 0 # only use rectangular pixels

	K = Matrix( ((alpha_u, skew,    u_0), \
	             (    0  , alpha_v, v_0), \
	             (    0  , 0,        1 )))
	return K

# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
	# bcam stands for blender camera
	R_bcam2cv = Matrix( ((1, 0,  0), \
                         (0, -1, 0), \
                         (0, 0, -1)))

	# Transpose since the rotation is object rotation,
	# and we want coordinate rotation
	# R_world2bcam = cam.rotation_euler.to_matrix().transposed()
	# T_world2bcam = -1*R_world2bcam * location
	#
	# Use matrix_world instead to account for all constraints
	location, rotation = cam.matrix_world.decompose()[0:2]
	R_world2bcam = rotation.to_matrix().transposed()

	# Convert camera location to translation vector used in coordinate changes
	# T_world2bcam = -1*R_world2bcam*cam.location
	# Use location from matrix_world to account for constraints:
	T_world2bcam = -1 * R_world2bcam @ location

	# Build the coordinate transform matrix from world to computer vision camera
	R_world2cv = R_bcam2cv @ R_world2bcam
	T_world2cv = R_bcam2cv @ T_world2bcam

	# put into 3x4 matrix
	RT = Matrix(( R_world2cv[0][:] + (T_world2cv[0],), \
	              R_world2cv[1][:] + (T_world2cv[1],), \
	              R_world2cv[2][:] + (T_world2cv[2],)  ))
	return RT

def get_3x4_P_matrix_from_blender(cam):
	K = get_calibration_matrix_K_from_blender(cam.data)
	RT = get_3x4_RT_matrix_from_blender(cam)
	return K @ RT, K, RT

if __name__ == '__main__':
	main()
