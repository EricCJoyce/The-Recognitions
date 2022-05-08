import numpy as np

'''
The "cone of attention" that boosts or lowers object signals in the props subvector according to their proximities to gaze and hands.
'''
class Gaussian3D():
	def __init__(self, **kwargs):
		self.mu = (0.0, 0.0, 0.0)									#  Centered on the head
		self.sigma_gaze = (2.0, 1.5, 3.0)							#  In METERS
																	#  No mus for hands: their centers are hand-centroids
		self.sigma_hand = (0.5, 0.5, 0.5)							#  In METERS

		if 'mu' in kwargs:
			assert isinstance(kwargs['mu'], tuple) and len(kwargs['mu']) == 3 and \
			       isinstance(kwargs['mu'][0], float) and isinstance(kwargs['mu'][1], float) and isinstance(kwargs['mu'][2], float), \
			       'Argument \'mu\' passed to Gaussian3D must be a 3-tuple of floats.'
			self.mu = kwargs['mu']

		if 'sigma_gaze' in kwargs:
			assert isinstance(kwargs['sigma_gaze'], tuple) and len(kwargs['sigma_gaze']) == 3 and \
			       isinstance(kwargs['sigma_gaze'][0], float) and isinstance(kwargs['sigma_gaze'][1], float) and isinstance(kwargs['sigma_gaze'][2], float), \
			       'Argument \'sigma_gaze\' passed to Gaussian3D must be a 3-tuple of floats.'
			self.sigma_gaze = kwargs['sigma_gaze']

		if 'sigma_hand' in kwargs:
			assert isinstance(kwargs['sigma_hand'], tuple) and len(kwargs['sigma_hand']) == 3 and \
			       isinstance(kwargs['sigma_hand'][0], float) and isinstance(kwargs['sigma_hand'][1], float) and isinstance(kwargs['sigma_hand'][2], float), \
			       'Argument \'sigma_hand\' passed to Gaussian3D must be a 3-tuple of floats.'
			self.sigma_hand = kwargs['sigma_hand']

	#  'object_centroid' is a Numpy array.
	#  'LH_centroid' is a Numpy array or None if the left hand is not visible.
	#  'RH_centroid' is a Numpy array or None if the right hand is not visible.
	def weigh(self, object_centroid, LH_centroid, RH_centroid):
		x = np.array([object_centroid[0] - self.mu[0], \
		              object_centroid[1] - self.mu[1], \
		              object_centroid[2] - self.mu[2]])
		C = np.array([[self.sigma_gaze[0] * self.sigma_gaze[0], 0.0,                                     0.0                                    ], \
		              [0.0,                                     self.sigma_gaze[1] * self.sigma_gaze[1], 0.0                                    ], \
		              [0.0,                                     0.0,                                     self.sigma_gaze[2] * self.sigma_gaze[2]]])
		C_inv = np.linalg.inv(C)
		f_head = np.exp(-0.5 * np.dot(np.dot(x.T, C_inv), x))

		if LH_centroid is not None:
			x = np.array([object_centroid[0] - LH_centroid[0], \
			              object_centroid[1] - LH_centroid[1], \
			              object_centroid[2] - LH_centroid[2]])
			C = np.array([[self.sigma_hand[0] * self.sigma_hand[0], 0.0,                                     0.0                                    ], \
			              [0.0,                                     self.sigma_hand[1] * self.sigma_hand[1], 0.0                                    ], \
			              [0.0,                                     0.0,                                     self.sigma_hand[2] * self.sigma_hand[2]]])
			C_inv = np.linalg.inv(C)
			f_Lhand = np.exp(-0.5 * np.dot(np.dot(x.T, C_inv), x))
		else:
			f_Lhand = 0.0

		if RH_centroid is not None:
			x = np.array([object_centroid[0] - RH_centroid[0], \
			              object_centroid[1] - RH_centroid[1], \
			              object_centroid[2] - RH_centroid[2]])
			C = np.array([[self.sigma_hand[0] * self.sigma_hand[0], 0.0,                                     0.0                                    ], \
			              [0.0,                                     self.sigma_hand[1] * self.sigma_hand[1], 0.0                                    ], \
			              [0.0,                                     0.0,                                     self.sigma_hand[2] * self.sigma_hand[2]]])
			C_inv = np.linalg.inv(C)
			f_Rhand = np.exp(-0.5 * np.dot(np.dot(x.T, C_inv), x))
		else:
			f_Rhand = 0.0

		return max(f_head, f_Lhand, f_Rhand)

	#  Return a list of this object's parameters.
	def parameter_list(self):
		return list(self.mu) + list(self.sigma_gaze) + list(self.sigma_hand)
