'''
This is a "loose" sort of data type in that Actions are not really binding until their host Enactment calls apply_actions_to_frames().
Actions contain a label and delineate where things begin and end.
Beginnings are considered inclusive; endings are considered exclusive, mimicking Pythonic slice notation.
'''
class Action():
	def __init__(self, start_time, end_time, **kwargs):
		self.start_time = start_time
		self.end_time = end_time

		self.start_frame = None
		self.end_frame = None
		self.label = None

		if 'start_frame' in kwargs:									#  Were we given a start_frame?
			assert isinstance(kwargs['start_frame'], str), 'Argument \'start_frame\' passed to Action must be a string.'
			self.start_frame = kwargs['start_frame']
		if 'end_frame' in kwargs:									#  Were we given a end_frame?
			assert isinstance(kwargs['end_frame'], str), 'Argument \'end_frame\' passed to Action must be a string.'
			self.end_frame = kwargs['end_frame']
		if 'label' in kwargs:										#  Were we given a label?
			assert isinstance(kwargs['label'], str), 'Argument \'label\' passed to Action must be a string.'
			self.label = kwargs['label']

	def print(self, **kwargs):
		if 'index' in kwargs:										#  Were we given an index?
			assert isinstance(kwargs['index'], int) and kwargs['index'] >= 0, 'Argument \'index\' passed to Action.print() must be an integer >= 0.'
			index = kwargs['index']
		else:
			index = None

		if 'pad_index' in kwargs:									#  Were we given a maximum number of spaces for the index?
			assert isinstance(kwargs['pad_index'], int) and kwargs['pad_index'] >= 0, 'Argument \'pad_index\' passed to Action.print() must be an integer >= 0.'
			pad_index = kwargs['pad_index']
		else:
			pad_index = 0

		if 'pad_label' in kwargs:									#  Were we given a maximum number of spaces for the label?
			assert isinstance(kwargs['pad_label'], int) and kwargs['pad_label'] >= 0, 'Argument \'pad_label\' passed to Action.print() must be an integer >= 0.'
			pad_label = kwargs['pad_label']
		else:
			pad_label = 0

		if 'pad_time' in kwargs:									#  Were we given a maximum number of spaces for time stamps?
			assert isinstance(kwargs['pad_time'], int) and kwargs['pad_time'] >= 0, 'Argument \'pad_time\' passed to Action.print() must be an integer >= 0.'
			pad_time = kwargs['pad_time']
		else:
			pad_time = 0

		if 'pad_path' in kwargs:									#  Were we given a maximum number of spaces for file paths?
			assert isinstance(kwargs['pad_path'], int) and kwargs['pad_path'] >= 0, 'Argument \'pad_path\' passed to Action.print() must be an integer >= 0.'
			pad_path = kwargs['pad_path']
		else:
			pad_path = 0

		if index is not None:
			outstr = '[' + str(index) + ']:' + ' '*(max(1, pad_index - len(str(index)) + 1))
		else:
			outstr = ''
		outstr += self.label + ':' + ' '*(max(1, pad_label - len(self.label) + 1))
		outstr += str(self.start_time) + ' '*(max(1, pad_time - len(str(self.start_time)) + 1))
		if self.start_frame is not None:
			outstr += 'incl.(' + self.start_frame.split('/')[-1] + ')' + ' '*(max(1, pad_path - len(self.start_frame.split('/')[-1]) + 1))
		outstr += '-- '
		outstr += str(self.end_time) + ' '*(max(1, pad_time - len(str(self.end_time)) + 1))
		if self.end_frame is not None:
			outstr += 'excl.(' + self.end_frame.split('/')[-1] + ')'
		print(outstr)
		return
