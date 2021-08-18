'''
  Eric C. Joyce, Stevens Institute of Technology, 2020

  Import your object into Blender. Let it sit wherever it wants to appear according to its native coordinate system.
  We will move the camera around the object.
  Position the Blender camera and three lights.
  This initial camera position will form the object's signature AND become the starting point of its orbit around the object.

  This script outputs the following:
    For the signature:
      ./signature/rgb.png
      ./signature/dimensions.dat
      ./signature/depth.dat
      ./signature/K.dat
      ./signature/Rt.dat
    For the query sequence:
      ./query-seq/rgb.N.png
      ./query-seq/K.N.png
      ./query-seq/Rt.N.png
    where N is an integer >= 0 to identify different virtual camera positions.

MAKE SURE THE TARGET OBJECT IS SELECTED AND IN OBJECT MODE.
THEN, in the Blender Python console, type:

filename = '/home/eric/Documents/blender/make_signatures/snapquery.py'
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
	steps_in_degrees = 1.0
	renderPath = '/home/eric/Documents/Work/Clients/FactualVR/2019-2020/24jun20/signature_creation/'
	signaturePath = 'signature/'
	querySeqPath = 'query-seq/'
	verbose = True

	#################################################################  Now orbit the object and render rgb, K, and Rt files at each step.
	#  REMEMBER THAT THE TARGET OBJECT HAS TO BE SELECTED AND IN OBJECT MODE FOR THIS TO WORK

	cam_loc = bpy.data.objects["Camera"].location

	writeRGB(renderPath + querySeqPath + 'rgb.png', verbose)
	writeRt(renderPath + querySeqPath + 'Rt.dat', verbose)

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
