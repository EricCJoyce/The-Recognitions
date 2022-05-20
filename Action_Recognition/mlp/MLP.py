import subprocess

'''
Since the MLP is small, we can run it on the CPU--especially if the GPU is occupied running detection.
'''
class MLP():
	def __init__(self, filename, **kwargs):
		self.mlp_filename = filename

		if 'executable' in kwargs:									#  Were we given a path to the executable?
			assert isinstance(kwargs['executable'], str), \
			       'Argument \'executable\' passed to DistanceProbMLP must be a string.'
			self.executable = kwargs['executable']
		else:
			self.executable = './run_mlp'							#  Default to './run_mlp'

	#  Receive a list of floats.
	#  Return a list of floats.
	def run(self, x_vec):
		args = [self.executable, self.mlp_filename] + [str(x) for x in x_vec]
		comp_proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out = comp_proc.stdout.decode('utf-8')
		err = comp_proc.stderr.decode('utf-8')
		return [float(x) for x in out.split()]
