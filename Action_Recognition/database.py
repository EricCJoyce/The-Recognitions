from classifier import ProcessedEnactment, Classifier
import cv2
import numpy as np
import os
import sys
import time

'''
db = Database(enactments=['BackBreaker1', 'Enactment1', 'Enactment2', 'Enactment3', 'Enactment4', 'Enactment5', 'Enactment6', 'Enactment7', 'Enactment9', 'Enactment10', 'MainFeederBox1', 'Regulator1', 'Regulator2'], verbose=True)
'''
class Database():
	def __init__(self, **kwargs):
		if 'enactments' in kwargs:
			assert isinstance(kwargs['enactments'], list), 'Argument \'enactments\' passed to Database must be a list of strings.'
			self.enactments = kwargs['enactments']
		else:
			self.enactments = []

		if 'snippet_size' in kwargs:
			assert isinstance(kwargs['snippet_size'], int) and kwargs['snippet_size'] > 0, \
			       'Argument \'snippet_size\' passed to Database must be an integer > 0.'
			self.snippet_size = kwargs['snippet_size']
		else:
			self.snippet_size = 10

		if 'stride' in kwargs:
			assert isinstance(kwargs['stride'], int) and kwargs['stride'] > 0, \
			       'Argument \'stride\' passed to Database must be an integer > 0.'
			self.stride = kwargs['stride']
		else:
			self.stride = 2

		if 'verbose' in kwargs:
			assert isinstance(kwargs['verbose'], bool), \
			       'Argument \'verbose\' passed to Database must be a boolean.'
			self.verbose = kwargs['verbose']
		else:
			self.verbose = False									#  Default to False.

		self.recognizable_objects = []								#  Initially empty; this must be determined by the given (processed) enactments.
		self.Xy = {}
		self.original_counts = {}
		self.protected = {}											#  key: (label, index) ==> val: True

	#  Parse the ProcessedEnactments allocated to this database.
	def commit(self):
		recognizable_objects_alignment = {}
		vector_length_alignment = {}
		for enactment in self.enactments:
			e = ProcessedEnactment(enactment, verbose=False)
			recognizable_objects_alignment[ tuple(e.recognizable_objects) ] = True
			vector_length_alignment[e.vector_length] = True
			recognizable_objects = e.recognizable_objects
		if len(recognizable_objects_alignment.keys()) > 1:
			print('ERROR: The given enactments differ in which objects are recognized and are therefore incompatible.')
			return
		if len(vector_length_alignment.keys()) > 1:
			print('ERROR: The given enactments differ in their vector lengths and are therefore incompatible.')
			return
		self.recognizable_objects = [x for x in recognizable_objects_alignment.keys()][0]
		self.vector_length = [x for x in vector_length_alignment.keys()][0]

		self.Xy = {}												#  Putative training set, keyed by labels.
		self.original_counts = {}									#  key:label ==>
		self.protected = {}											#      {key'actions':              ==> val:[ (enactment, start time, start frame,
																	#                                                        end time,   end frame),
																	#                                            (enactment, start time, start frame,
																	#                                                        end time,   end frame),
																	#                                                                ...
																	#                                            (enactment, start time, start frame,
																	#                                                        end time,   end frame) ]
																	#       key:'mean-signal-strength' ==> val:[ strength, strength, ... strength ]
																	#       key:'left-hand-strength'   ==> val:[ strength, strength, ... strength ]
																	#       key:'right-hand-strength'  ==> val:[ strength, strength, ... strength ]
		if self.verbose:											#      }
			print('>>> Computing mean signal strengths of subvectors.')
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.
		num = 0
		for enactment in self.enactments:
			e = ProcessedEnactment(enactment, verbose=False)
			num += len(e.snippets_from_action(self.snippet_size, self.stride))
		prev_ctr = 0
		ctr = 0
		for enactment in self.enactments:
			e = ProcessedEnactment(enactment, verbose=False)
			for action in e.snippets_from_action(self.snippet_size, self.stride):
				label       = action[0]
				start_time  = action[1]
				start_frame = action[2]
				end_time    = action[3]
				end_frame   = action[4]
				if label not in self.Xy:
					self.Xy[label] = {}
					self.Xy[label]['actions'] = []
					self.Xy[label]['mean-signal-strength'] = []
					self.Xy[label]['left-hand-strength'] = []
					self.Xy[label]['right-hand-strength'] = []

				action_tuple = (e.enactment_name, start_time, start_frame, end_time, end_frame)

				self.Xy[label]['actions'].append( action_tuple )
				self.Xy[label]['mean-signal-strength'].append( self.mean_signal_strength(action_tuple)       )
				self.Xy[label]['left-hand-strength'].append(   self.left_hand_signal_strength(action_tuple)  )
				self.Xy[label]['right-hand-strength'].append(  self.right_hand_signal_strength(action_tuple) )

				if self.verbose:
					if int(round(float(ctr) / float(num - 1) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
						prev_ctr = int(round(float(ctr) / float(num - 1) * float(max_ctr)))
						sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(ctr) / float(num - 1) * 100.0))) + '%]')
						sys.stdout.flush()
				ctr += 1

		for label, actions in self.Xy.items():						#  Track the original lengths so we can compute reduction.
			self.original_counts[label] = len(actions['actions'])
			for ctr in range(0, len(actions['actions'])):			#  Initially, set everything to be kept.
				self.protected[ (label, ctr) ] = True

		for label, actions in self.Xy.items():						#  Whether or not we intend to prune them, sort actions/labels by mean signal strength.
			sorted_actions = sorted(list(zip(actions['mean-signal-strength'], \
			                                 actions['left-hand-strength'],   \
			                                 actions['right-hand-strength'],  \
			                                 actions['actions'])), key=lambda x: x[0], reverse=True)
			self.Xy[label]['actions']              = [x[3] for x in sorted_actions]
			self.Xy[label]['mean-signal-strength'] = [x[0] for x in sorted_actions]
			self.Xy[label]['left-hand-strength']   = [x[1] for x in sorted_actions]
			self.Xy[label]['right-hand-strength']  = [x[2] for x in sorted_actions]

		return

	#  Write the contents of this database to file.
	def output(self, db_filename=None):
		if db_filename is None:
			db_filename = str(self.snippet_size) + 'f.db'
		fh = open(db_filename, 'w')
		fh.write('#  Action recognition database made at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		fh.write('#  ENACTMENTS:\n')
		fh.write('#    ' + '\t'.join(self.enactments) + '\n')
		fh.write('#  ACTIONS:\n')
		fh.write('#    ' + '\t'.join(sorted([x for x in self.Xy.keys()])) + '\n')
		fh.write('#  RECOGNIZABLE OBJECTS:\n')
		fh.write('#    ' + '\t'.join(self.recognizable_objects) + '\n')
		fh.write('#  VECTOR LENGTH:\n')
		fh.write('#    ' + str(self.vector_length) + '\n')
		fh.write('#  SNIPPET SIZE:\n')
		fh.write('#    ' + str(self.snippet_size) + '\n')
		fh.write('#  STRIDE:\n')
		fh.write('#    ' + str(self.stride) + '\n')
		fh.write('#  NUMBER OF SOURCE SNIPPETS:\n')
		fh.write('#    ' + str(self.original_size()) + ' snippets\n')
		fh.write('#  DB SIZE:\n')
		fh.write('#    ' + str(self.current_size()) + ' snippets\n')

		for label, actions in sorted(self.Xy.items()):
			ctr = 0
			for action in actions['actions']:
				if (label, ctr) in self.protected:
					fh.write(label + '\t' + actions['actions'][ctr][0]      + '\t' + \
					                        str(actions['actions'][ctr][1]) + '\t' + \
					                        actions['actions'][ctr][2]      + '\t' + \
					                        str(actions['actions'][ctr][3]) + '\t' + \
					                        actions['actions'][ctr][4]      + '\n')
					sequence = self.vectors(action)
					for vector in sequence:
						fh.write('\t' + '\t'.join([str(x) for x in vector]) + '\n')
				ctr += 1
		fh.close()
		return

	#  Load a database from file.
	def load(self, db_filename):
		fh = open(db_filename, 'r')
		reading_enactments = False
		reading_actions = False
		reading_recognizable_objects = False
		reading_vector_length = False
		reading_snippet_size = False
		reading_stride = False
		reading_num_src_snippets = False
		reading_db_size = False

		lines = fh.readlines()
		fh.close()

		if self.verbose:
			print('>>> Loading database "' + db_filename + '".')
		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.
		num = len(lines)
		prev_ctr = 0
		ctr = 0
		for line in lines:
			if line[0] == '#':
				if 'ENACTMENTS:' in line:							#  Prepare to read enactments.
					reading_enactments = True
				elif 'ACTIONS:' in line:							#  Prepare to read actions.
					reading_actions = True
				elif 'RECOGNIZABLE OBJECTS:' in line:				#  Prepare to read recognizable objects.
					reading_recognizable_objects = True
				elif 'VECTOR LENGTH:' in line:						#  Prepare to read the vector length.
					reading_vector_length = True
				elif 'SNIPPET SIZE:' in line:						#  Prepare to read the snippet size.
					reading_snippet_size = True
				elif 'STRIDE:' in line:								#  Prepare to read the stride.
					reading_stride = True
				elif 'NUMBER OF SOURCE SNIPPETS:' in line:			#  Prepare to read the number of source snippets.
					reading_num_src_snippets = True
				elif 'DB SIZE:' in line:							#  Prepare to read the database size.
					reading_db_size = True

				elif reading_enactments:							#  Read the enactments that were used to build this database.
					self.enactments = line[1:].strip().split('\t')
					reading_enactments = False
				elif reading_actions:								#  Read the actions this database can classify.
					self.Xy = {}
					for label in line[1:].strip().split('\t'):
						self.Xy[label] = {}
						self.Xy[label]['actions'] = []
						self.Xy[label]['mean-signal-strength'] = []
						self.Xy[label]['left-hand-strength'] = []
						self.Xy[label]['right-hand-strength'] = []
					reading_actions = False
				elif reading_recognizable_objects:					#  Read the objects this database has been built to recognize.
					self.recognizable_objects = line[1:].strip().split('\t')
					reading_recognizable_objects = False
				elif reading_vector_length:							#  Read the encoded vector length.
					self.vector_length = int(line[1:].strip())
					reading_vector_length = False
				elif reading_snippet_size:							#  Read the snippet length.
					self.snippet_size = int(line[1:].strip())
					reading_snippet_size = False
				elif reading_stride:								#  Read the stride.
					self.stride = int(line[1:].strip())
					reading_stride = False
				elif reading_num_src_snippets:						#  Read the number of source snippets (immaterial *after* the DB has been built.)
					reading_num_src_snippets = False
				elif reading_db_size:								#  Read the number of database snippets (immaterial now, too.)
					reading_db_size = False
			else:
				if line[0] != '\t':									#  Snippet header.
					arr = line.strip().split('\t')
					label = arr[0]
					src_enactment = arr[1]
					start_timestamp = float(arr[2])
					start_frame = arr[3]
					end_timestamp = float(arr[4])
					end_frame = arr[5]

					action_tuple = (src_enactment, start_timestamp, start_frame, end_timestamp, end_frame)

					self.Xy[label]['actions'].append(action_tuple)	#  Append action tuple.
					self.Xy[label]['mean-signal-strength'].append( self.mean_signal_strength(action_tuple)       )
					self.Xy[label]['left-hand-strength'].append(   self.left_hand_signal_strength(action_tuple)  )
					self.Xy[label]['right-hand-strength'].append(  self.right_hand_signal_strength(action_tuple) )

			if self.verbose:
				if int(round(float(ctr) / float(num - 1) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(ctr) / float(num - 1) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(ctr) / float(num - 1) * 100.0))) + '%]')
					sys.stdout.flush()
			ctr += 1

		return

	#  Return the i-th database snippet as an action-tuple: (enactment name, start time, start frame, end time, end frame).
	#  Indexing follows the order in which snippets are written to file--that is, alphabetized by label.
	def index(self, i):
		snippets = []
		for label, actions in sorted(self.Xy.items()):
			snippets += actions['actions']
		if i < len(snippets):
			return snippets[i]
		return None

	#  Print out. If a target_label is passed, then only itemize that label's snippets.
	#  Otherwise, itemize all labels' snippets.
	def itemize(self, target_label=None):
		if target_label is None:
			itemization = [x for x in self.Xy.items()]
		else:
			itemization = [x for x in self.Xy.items() if x[0] == target_label]

		for item in itemization:
			label = item[0]
			actions = item[1]
			print('>>> ' + label + ': u.Sig.\tu.LH\tu.RH\tSnippet')
			for ctr in range(0, len(actions['actions'])):
				if (label, ctr) in self.protected:
					print('    ' + ' '*(len(label) - len(str(ctr)) - 4) + '+ [' + str(ctr) + ']  ' + \
					      "{:.4f}".format(actions['mean-signal-strength'][ctr]) + '\t' + \
					      "{:.4f}".format(actions['left-hand-strength'][ctr]) + '\t' + \
					      "{:.4f}".format(actions['right-hand-strength'][ctr]) + '\t' + \
					      '\t'.join( [actions['actions'][ctr][0], str(actions['actions'][ctr][1]), str(actions['actions'][ctr][3])] ))
				else:
					print('    ' + ' '*(len(label) - len(str(ctr)) - 4) + '- [' + str(ctr) + ']  ' + \
					      "{:.4f}".format(actions['mean-signal-strength'][ctr]) + '\t' + \
					      "{:.4f}".format(actions['left-hand-strength'][ctr]) + '\t' + \
					      "{:.4f}".format(actions['right-hand-strength'][ctr]) + '\t' + \
					      '\t'.join( [actions['actions'][ctr][0], str(actions['actions'][ctr][1]), str(actions['actions'][ctr][3])] ))
				ctr += 1
		return

	#  Just show me the labels, how many are slated to be kept, and how many there were originally.
	def actions(self):
		for label, actions in sorted(self.Xy.items()):
			reduced_set = [x for x in range(0, len(actions['actions'])) if (label, x) in self.protected]
			print(label + ': ' + str(len(reduced_set)) + ' / ' + str(self.original_counts[label]))

	#  Report how far reduced the database is since we called 'commit().'
	#  If a target_label is passed, then only report the reduction achieved for that label's snippets.
	#  Otherwise, report the reduction of the entire database.
	def reduction_achieved(self, target_label=None):
		if target_label is None:
			itemization = [x for x in self.Xy.items()]
		else:
			itemization = [x for x in self.Xy.items() if x[0] == target_label]

		for item in itemization:
			label = item[0]
			actions = item[1]
			reduced_set = [x for x in range(0, len(actions['actions'])) if (label, x) in self.protected]

			print('>>> ' + label + ':')
			print('    Originally ' + str(self.original_counts[label]))
			print('    Currently  ' + str(len(reduced_set)))
			print('    Reduction  ' + "{:.2f}".format(float(self.original_counts[label] - len(reduced_set)) / float(self.original_counts[label]) * 100.0) +  '%')
		return

	#  Mark the index-th snippet under 'label' as one to keep in the final output.
	def keep(self, label, index):
		if isinstance(index, int):
			self.protected[(label, index)] = True
		elif isinstance(index, list):
			for i in index:
				self.protected[(label, i)] = True
		return

	#  If no label is given, then restore everything.
	def keep_all(self, label=None):
		protected = {}
		if label is None:
			for k, v in self.Xy.items():
				for ctr in range(0, len(v['actions'])):
					protected[ (label, ctr) ] = True
		else:
			for k, v in self.Xy.items():
				if k == label:
					for ctr in range(0, len(v['actions'])):
						protected[ (label, ctr) ] = True
		self.protected = protected
		return

	#  Mark the index-th snippet under 'label' as one to omit from the final output.
	def drop(self, label, index):
		if isinstance(index, int):
			del self.protected[(label, index)]
		elif isinstance(index, list):
			for i in index:
				del self.protected[(label, i)]
		return

	#  If no label is given, then drop everything.
	def drop_all(self, label=None):
		protected = {}
		if label is not None:
			for k, v in self.protected.items():
				if k[0] != label:
					protected[k] = v
		self.protected = protected
		return

	#  Drop the lowest 'portion' of actions under 'label'.
	#  'portion' is in [0.0, 1.0].
	def cull(self, label, portion):
		assert isinstance(portion, float) and portion >= 0.0 and portion <= 1.0, \
		       'Argument \'portion\' passed to Database.cull() must be a float in [0.0, 1.0].'
		indices = [x for x in range(0, len(self.Xy[label]['actions']))]
		lim = int(round(float(len(indices)) * portion))
		for ctr in range(lim, len(indices)):
			key = (label, ctr)
			del self.protected[key]
		return

	def relabel(self, old_label, new_label):
		Xy = {}
		original_counts = {}
		protected = {}

		for label, actions in self.Xy.items():
			if label == old_label:
				Xy[new_label] = actions
				original_counts[new_label] = self.original_counts[label]
				for key in [x for x in self.protected.keys() if x[0] == label]:
					protected[ (new_label, key[1]) ] = True
			else:
				Xy[label] = actions
				original_counts[label] = self.original_counts[label]
				for key in [x for x in self.protected.keys() if x[0] == label]:
					protected[ (label, key[1]) ] = True

		self.Xy = Xy
		self.original_counts = original_counts
		self.protected = protected

		return

	#  The given file may have some comment lines, starting with "#",
	#  but apart from that, each line should have the format:
	#  old-label <tab> new-label
	def relabel_from_file(self, relabel_file):
		fh = open(relabel_file, 'r')
		for line in fh.readlines():
			if line[0] != '#':
				arr = line.strip().split('\t')
				old_label = arr[0]
				new_label = arr[1]

				self.relabel(old_label, new_label)
		fh.close()
		return

	#  Fetch the vectors of the given snippet.
	def vectors(self, snippet):
		pe = ProcessedEnactment(snippet[0])
		sequence = []
		for timestamp_frame in pe.get_frames():						#  pe.frames is a dictionary of dictionaries: file, ground-truth-label, vector.
			timestamp = timestamp_frame[0]
			frame = timestamp_frame[1]
			if timestamp >= snippet[1] and timestamp < snippet[3]:
				sequence.append(frame['vector'])
		return sequence

	def snippets(self, label):
		return self.Xy[label]['actions']

	#  Given a list of sequences all under a single label, and a lambda function 'condition',
	#  return a list or indices into 'sequences' that uphold 'condition'.
	#  Be careful if you plan to use this to identify snippets containing a state-transition:
	#  given the layout of the classroom, state-transitions are really more instances of new state appearances.
	#  Meaning, what you actually want to look for are sequences were a particular signal goes from 0.0 to something > 0.0.
	#  One door may open, but the signal for closed doors will not go away because adjacent doors in view remain closed.
	#  Example:
	#    db.lambda_identify( db.snippets('Close Disconnect (MFB)'),
	#                       (lambda seq: db.contains_downturn(seq, db.recognizable_objects.index('Disconnect_Open') + 12) or
	#                                    db.contains_downturn(seq, db.recognizable_objects.index('Disconnect_Unknown') + 12)) ))
	def lambda_identify(self, snippets, condition):
		passing_indices = []
		index = 0
		for snippet in snippets:									#  Each 'snippet' is (enactment, start time, start frame, end time, end frame).
			sequence = self.vectors(snippet)
			if condition(sequence):									#  Apply the test.
				passing_indices.append(index)
			index += 1
		return passing_indices

	#  Given a sequence and an index into the representation vector (say for the MainFeederBox_Open)
	#  determine whether the sequence starts with a zero signal for 'index' and flips to non-zero ONCE.
	#  No other pattern will do: we cannot never flip from non-zero to zero, and we cannot flip and flip back.
	def contains_upturn(self, sequence, index):
		bool_vec = []
		for vector in sequence:										#  For each frame, is the target signal greater than zero?
			bool_vec.append(vector[index] > 0.0)

		current = bool_vec[0]										#  Now "flatten" the list of Booleans.
		compressed = [current]
		for b in bool_vec:
			if b != current:
				compressed.append(b)
				current = b

		if len(compressed) == 2 and compressed[0] == False:			#  If, after flattening, we have a state not > 0.0
			return True												#  followed by a state > 0.0, and no more. This is what we want.

		return False												#  No other pattern will do.

	#  Given a sequence and an index into the representation vector (say for the MainFeederBox_Open)
	#  determine whether the sequence starts with a non-zero signal for 'index' and flips to zero ONCE.
	#  No other pattern will do: we cannot never flip from zero to non-zero, and we cannot flip and flip back.
	def contains_downturn(self, sequence, index):
		bool_vec = []
		for vector in sequence:										#  For each frame, is the target signal greater than zero?
			bool_vec.append(vector[index] > 0.0)

		current = bool_vec[0]										#  Now "flatten" the list of Booleans.
		compressed = [current]
		for b in bool_vec:
			if b != current:
				compressed.append(b)
				current = b

		if len(compressed) == 2 and compressed[0] == True:			#  If, after flattening, we have a state > 0.0
			return True												#  followed by a state not > 0.0, and no more. This is what we want.

		return False												#  No other pattern will do.

	#  For each vector in the given 'snippet', compute its magnitude, excluding the one-hot-encoded elements for left and right hands.
	#  Return the average of all magnitudes.
	#  Vector encoding is [0]LH_x [1]LH_y [2]LH_z [3]LH_0 [4]LH_1 [5]LH_2 [6]RH_x [7]RH_y [8]RH_z [9]RH_0 [10]RH_1 [11]RH_2 [12]Prop_1 ...
	#  'snippet' is (enactment, start time, start frame, end time, end frame).
	def mean_signal_strength(self, snippet, hands_coeff=1.0, props_coeff=1.0):
		sequence = self.vectors(snippet)
		magnitudes = []
		for vector in sequence:
			vec = [hands_coeff * x for x in vector[:3]]  + \
			      [hands_coeff * x for x in vector[6:9]] + \
			      [props_coeff * x for x in vector[12:]]
			magnitudes.append( np.linalg.norm(vec) )
		return np.mean(magnitudes)

	#  For each vector in the given 'snippet', compute the magnitude of the left-hand subvector only, excluding the one-hot-encoding.
	#  Return the average of all magnitudes.
	#  Vector encoding is [0]LH_x [1]LH_y [2]LH_z [3]LH_0 [4]LH_1 [5]LH_2 [6]RH_x [7]RH_y [8]RH_z [9]RH_0 [10]RH_1 [11]RH_2 [12]Prop_1 ...
	#  'snippet' is (enactment, start time, start frame, end time, end frame).
	def left_hand_signal_strength(self, snippet, hands_coeff=1.0):
		sequence = self.vectors(snippet)
		magnitudes = []
		for vector in sequence:
			vec = [hands_coeff * x for x in vector[:3]]
			magnitudes.append( np.linalg.norm(vec) )
		return np.mean(magnitudes)

	#  For each vector in the given 'snippet', compute the magnitude of the right-hand subvector only, excluding the one-hot-encoding.
	#  Return the average of all magnitudes.
	#  Vector encoding is [0]LH_x [1]LH_y [2]LH_z [3]LH_0 [4]LH_1 [5]LH_2 [6]RH_x [7]RH_y [8]RH_z [9]RH_0 [10]RH_1 [11]RH_2 [12]Prop_1 ...
	#  'snippet' is (enactment, start time, start frame, end time, end frame).
	def right_hand_signal_strength(self, snippet, hands_coeff=1.0):
		sequence = self.vectors(snippet)
		magnitudes = []
		for vector in sequence:
			vec = [hands_coeff * x for x in vector[6:9]]
			magnitudes.append( np.linalg.norm(vec) )
		return np.mean(magnitudes)

	#  For each vector in the given 'snippet', compute the magnitude of the props subvector only.
	#  Return the average of all magnitudes.
	#  Vector encoding is [0]LH_x [1]LH_y [2]LH_z [3]LH_0 [4]LH_1 [5]LH_2 [6]RH_x [7]RH_y [8]RH_z [9]RH_0 [10]RH_1 [11]RH_2 [12]Prop_1 ...
	#  'snippet' is (enactment, start time, start frame, end time, end frame).
	def prop_signal_strength(self, snippet, props_coeff=1.0):
		sequence = self.vectors(snippet)
		magnitudes = []
		for vector in sequence:
			vec = [props_coeff * x for x in vector[12:]]
			magnitudes.append( np.linalg.norm(vec) )
		return np.mean(magnitudes)

	def	original_size(self):
		return sum([x for x in self.original_counts.values()])
		self.Xy = {}
		self.original_counts = {}

	def current_size(self):
		s = []
		for label, actions in self.Xy.items():
			reduced_set = [x for x in range(0, len(actions['actions'])) if (label, x) in self.protected]
			s.append(len(reduced_set))
		return sum(s)

	#  Show me.
	#  The given 'snippet' is an action-tuple: (enactment name, start time, start frame path, end time, end frame path)
	def render_snippet(self, snippet, video_name=None, color_map=None):
		colors = {}													#  key:recognizable-object ==> val: (r, g, b)
		if color_map is None:
			for recog_object in self.recognizable_objects:
				colors[recog_object] = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
		else:
			fh = open(color_map, 'r')
			for line in fh.readlines():
				if line[0] != '#':
					arr = line.strip().split('\t')
					colors[arr[0]] = (int(arr[1]), int(arr[2]), int(arr[3]))
			fh.close()

		pe = ProcessedEnactment(snippet[0])
		animation_frames = []
		for timestamp_frame in pe.get_frames():						#  pe.frames is a dictionary of dictionaries: file, ground-truth-label, vector.
			timestamp = timestamp_frame[0]
			frame = timestamp_frame[1]
			if timestamp >= snippet[1] and timestamp < snippet[3]:
				animation_frames.append(frame['file'])

		if video_name is None:
			video_name = str(snippet[1]) + '-' + str(snippet[3])
		vid = cv2.VideoWriter( snippet[0] + '_' + video_name + '.avi', \
		                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
		                       pe.fps, \
		                      (pe.width, pe.height) )
		for frame in animation_frames:
			img = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
			mask_oly = np.zeros((pe.height, pe.width, 3), dtype='uint8')

			fh = open(snippet[0] + '_props.txt', 'r')
			for line in fh.readlines():
				if line[0] != '#':
					arr = line.strip().split('\t')
					img_filename = arr[1]
					obj_type = arr[3]
					mask_path = arr[7]
					if img_filename == frame and obj_type not in ['LeftHand', 'RightHand']:
						mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
						mask[mask > 1] = 1							#  All values greater than 1 become 1
																	#  Extrude to three channels
						mask = mask[:, :, None] * np.ones(3, dtype='uint8')[None, None, :]
						mask[:, :, 0] *= colors[ obj_type ][2]		#  Convert this to a graphical overlay
						mask[:, :, 1] *= colors[ obj_type ][1]
						mask[:, :, 2] *= colors[ obj_type ][0]

						mask_oly += mask							#  Add mask to mask accumulator
						mask_oly[mask_oly > 255] = 255				#  Clip accumulator to 255
			fh.close()

			img = cv2.addWeighted(img, 1.0, mask_oly, 0.7, 0)		#  Add mask accumulator to source frame
			img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)				#  Flatten alpha

			vid.write(img)
		vid.release()
		return
