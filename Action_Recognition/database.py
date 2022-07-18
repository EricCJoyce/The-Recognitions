from enactment import ProcessedEnactment
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

		if 'accept_small' in kwargs:
			assert isinstance(kwargs['accept_small'], bool), \
			       'Argument \'accept_small\' passed to Database must be a boolean.'
			self.accept_small = kwargs['accept_small']
		else:
			self.accept_small = False

		if 'verbose' in kwargs:
			assert isinstance(kwargs['verbose'], bool), \
			       'Argument \'verbose\' passed to Database must be a boolean.'
			self.verbose = kwargs['verbose']
		else:
			self.verbose = False									#  Default to False.

		self.recognizable_objects = []								#  Initially empty; this must be determined by the given (processed) enactments.
		self.Xy = {}
		self.original_counts = {}
		self.dropped_actions = {}									#  For actions. key: (enactment, index) ==> val: True
		self.protected = {}											#  For snippets. key: (label, index) ==> val: True
		self.protected_vector = {}									#  For elements of the props subvector. key: object-label ==> val: True

	#  Parse the ProcessedEnactments allocated to this database.
	def commit(self):
		assert self.align_recognizable_objects(), 'The given enactments must recognize the same objects and have the same vector length.'

		self.Xy = {}												#  Putative training set, keyed by labels.
		self.original_counts = {}									#  key:label ==>                           length N for N snippets
		self.protected = {}											#      {key'actions':              ==> val:[ (enactment, start time, start frame,
																	#                                                        end time,   end frame),
																	#                                            (enactment, start time, start frame,
																	#                                                        end time,   end frame),
																	#                                                                ...
																	#                                            (enactment, start time, start frame,
																	#                                                        end time,   end frame) ]
																	#                                          length N for N snippets
																	#       key:'mean-signal-strength' ==> val:[ strength, strength, ... strength ]
																	#                                          length N for N snippets
																	#       key:'left-hand-strength'   ==> val:[ strength, strength, ... strength ]
																	#                                          length N for N snippets
																	#       key:'right-hand-strength'  ==> val:[ strength, strength, ... strength ]
																	#                                          length N for N snippets
																	#       key:'prop-strength'        ==> val:[ strength, strength, ... strength ]
																	#                                          length N for N snippets
																	#       key:'sample-conf-weight'   ==> val:[ weight,   weight,   ... weight   ]
		if self.verbose:											#      }
			print('>>> Parsing source enactments into snippets.')

		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.
		num = 0
		for enactment in self.enactments:
			e = ProcessedEnactment(enactment, verbose=False)
			for i in range(0, e.num_actions()):
																	#  Only admit actions of size sufficient to make at least one snippet.
				if e.action_len(i) >= self.snippet_size or self.accept_small:
					num += 1
		prev_ctr = 0
		ctr = 0
		for enactment in self.enactments:
			e = ProcessedEnactment(enactment, verbose=False)
			action_ctr = 0
			for i in range(0, e.num_actions()):
																	#  Only admit actions of size sufficient to make at least one snippet.
				if e.action_len(i) >= self.snippet_size or self.accept_small:
					action = e.actions[i]							#  Action = (label, start time, start frame, end time, end frame).
					label       = action[0]
					start_time  = action[1]
					start_frame = action[2]
					end_time    = action[3]
					end_frame   = action[4]
																	#  If we are not wholesale omitting this action (several snippets), this enactment.
					if (enactment, action_ctr) not in self.dropped_actions:
						for snippet in e.snippets_from_action(self.snippet_size, self.stride, action_ctr):
							snippet_label       = snippet[0]
							snippet_start_time  = snippet[1]
							snippet_start_frame = snippet[2]
							snippet_end_time    = snippet[3]
							snippet_end_frame   = snippet[4]
							snippet_tuple = (e.enactment_name, snippet_start_time, snippet_start_frame, snippet_end_time, snippet_end_frame)

							if snippet_label not in self.Xy:
								self.Xy[snippet_label] = {}
								self.Xy[snippet_label]['actions'] = []
								self.Xy[snippet_label]['mean-signal-strength'] = []
								self.Xy[snippet_label]['left-hand-strength'] = []
								self.Xy[snippet_label]['right-hand-strength'] = []
								self.Xy[snippet_label]['prop-strength'] = []
								self.Xy[snippet_label]['sample-conf-weight'] = []

							self.Xy[snippet_label]['actions'].append( snippet_tuple )
																	#  Computing signal strengths takes time!
																	#  Only do it if requested.
							self.Xy[snippet_label]['mean-signal-strength'].append(0.0)
							self.Xy[snippet_label]['left-hand-strength'].append(0.0)
							self.Xy[snippet_label]['right-hand-strength'].append(0.0)
							self.Xy[snippet_label]['prop-strength'].append(0.0)
																	#  Initially, all samples carry full weight.
							self.Xy[snippet_label]['sample-conf-weight'].append(1.0)

					action_ctr += 1

					if self.verbose:
						if int(round(float(ctr) / float(num - 1) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
							prev_ctr = int(round(float(ctr) / float(num - 1) * float(max_ctr)))
							sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(ctr) / float(num - 1) * 100.0))) + '%]')
							sys.stdout.flush()

					ctr += 1

		self.keep_all()												#  Initially, set everything to be kept.
		self.count()												#  (Re)count.

		return

	#  Write the contents of this database to a database-formatted file.
	def output_db(self, db_filename=None):
		if db_filename is None:
			db_filename = str(self.snippet_size) + 'f.db'

		if self.verbose:
			print('>>> Writing "' + db_filename + '".')

		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.
		num = 0
		for label, actions in sorted(self.Xy.items()):
			for ctr in range(0, len(actions['actions'])):
				if (label, ctr) in self.protected:
					num += 1
		prev_ctr = 0
		complete_ctr = 0

		fh = open(db_filename, 'w')
		fh.write('#  Action recognition database made at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		fh.write('#  ENACTMENTS:\n')
		fh.write('#    ' + '\t'.join(self.enactments) + '\n')
		fh.write('#  ACTIONS:\n')
		fh.write('#    ' + '\t'.join(sorted([x for x in self.Xy.keys() if self.current_size(x) > 0])) + '\n')
		fh.write('#  RECOGNIZABLE OBJECTS:\n')
		fh.write('#    ' + '\t'.join( [x for x in self.recognizable_objects if self.protected_vector[x]] ) + '\n')
		fh.write('#  VECTOR LENGTH:\n')
		fh.write('#    ' + str(12 + len([x for x in self.recognizable_objects if self.protected_vector[x] == True])) + '\n')
		fh.write('#  ITEMIZATION:\n')
		longest_action_str = 0
		for key in sorted([x for x in self.Xy.keys() if self.current_size(x) > 0]):
			if len(key) > longest_action_str:
				longest_action_str = len(key)
		for key in sorted([x for x in self.Xy.keys() if self.current_size(x) > 0]):
			fh.write('#    ' + key + ' '*(longest_action_str - len(key)) + ' ' + str(self.current_size(key)) + ' snippets\n')
		fh.write('#  SNIPPET SIZE:\n')
		fh.write('#    ' + str(self.snippet_size) + '\n')
		fh.write('#  STRIDE:\n')
		fh.write('#    ' + str(self.stride) + '\n')
		if self.accept_small:
			fh.write('#  SMALLER SNIPPETS ACCEPTED\n')
		fh.write('#  NUMBER OF SOURCE SNIPPETS:\n')
		fh.write('#    ' + str(self.original_size()) + ' snippets\n')
		fh.write('#  DB SIZE:\n')
		fh.write('#    ' + str(self.current_size()) + ' snippets\n')

		db_ctr = 0
		for label, actions in sorted(self.Xy.items()):
			ctr = 0
			for action in actions['actions']:
				if (label, ctr) in self.protected:					#  If this is a sample we are keeping,
																	#  then write down the label, its DB index, its weight, and its source information.
					fh.write(label + '\t' + '[' + str(db_ctr) + ']'                 + '\t' + \
					                        str(actions['sample-conf-weight'][ctr]) + '\t' + \
					                        actions['actions'][ctr][0]              + '\t' + \
					                        str(actions['actions'][ctr][1])         + '\t' + \
					                        actions['actions'][ctr][2]              + '\t' + \
					                        str(actions['actions'][ctr][3])         + '\t' + \
					                        actions['actions'][ctr][4]              + '\n')
																	#  Write down the snippet vectors.
					sequence = self.vectors(action)
					for vector in sequence:
						vec = list(vector[:12])						#  Always include the hands.
						props_subvec = list(vector[12:])			#  Include props' signals if they're marked for inclusion.
						for i in range(0, len(self.recognizable_objects)):
							if self.protected_vector[ self.recognizable_objects[i] ] == True:
								vec.append( props_subvec[i] )
						fh.write('\t' + '\t'.join([str(x) for x in vec]) + '\n')

					if self.verbose:
						if int(round(float(complete_ctr) / float(num - 1) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
							prev_ctr = int(round(float(complete_ctr) / float(num - 1) * float(max_ctr)))
							sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(complete_ctr) / float(num - 1) * 100.0))) + '%]')
							sys.stdout.flush()

					complete_ctr += 1
					db_ctr += 1										#  Increment only after writing to file.
				ctr += 1
		fh.close()
		return

	#  Write the contents of this database to file as a training set.
	def output_Xy(self, filename=None):
		if filename is None:
			filename = str(self.snippet_size) + 'f.txt'

		if self.verbose:
			print('>>> Writing "' + filename + '".')

		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.
		num = 0
		for label, actions in sorted(self.Xy.items()):
			for ctr in range(0, len(actions['actions'])):
				if (label, ctr) in self.protected:
					num += 1
		prev_ctr = 0
		complete_ctr = 0

		fh = open(filename, 'w')
		fh.write('#  Action recognition dataset made at ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		fh.write('#  ENACTMENTS:\n')
		fh.write('#    ' + '\t'.join(self.enactments) + '\n')
		fh.write('#  RECOGNIZABLE ACTIONS:\n')
		fh.write('#    ' + '\t'.join(sorted([x for x in self.Xy.keys() if self.current_size(x) > 0])) + '\n')
		fh.write('#  RECOGNIZABLE OBJECTS:\n')
		fh.write('#    ' + '\t'.join( [x for x in self.recognizable_objects if self.protected_vector[x]] ) + '\n')
		fh.write('#  VECTOR LENGTH:\n')
		fh.write('#    ' + str(12 + len([x for x in self.recognizable_objects if self.protected_vector[x] == True])) + '\n')
		fh.write('#  ITEMIZATION:\n')
		longest_action_str = 0
		for key in sorted([x for x in self.Xy.keys() if self.current_size(x) > 0]):
			if len(key) > longest_action_str:
				longest_action_str = len(key)
		for key in sorted([x for x in self.Xy.keys() if self.current_size(x) > 0]):
			fh.write('#    ' + key + ' '*(longest_action_str - len(key)) + ' ' + str(self.current_size(key)) + ' snippets\n')
		fh.write('#  SNIPPET SIZE:\n')
		fh.write('#    ' + str(self.snippet_size) + '\n')
		fh.write('#  STRIDE:\n')
		fh.write('#    ' + str(self.stride) + '\n')
		if self.accept_small:
			fh.write('#  SMALLER SNIPPETS ACCEPTED\n')
		fh.write('#  NUMBER OF SOURCE SNIPPETS:\n')
		fh.write('#    ' + str(self.original_size()) + ' snippets\n')
		fh.write('#  TRAINING SET SIZE:\n')
		fh.write('#    ' + str(self.current_size()) + ' snippets\n')
		fh.write('#  FORMAT:\n')
		for i in range(0, self.snippet_size):
			fh.write('#    <Descriptor vector, elements tab-separated>\n')
		fh.write('#    <Label>\n')

		for label, actions in sorted(self.Xy.items()):
			ctr = 0
			for action in actions['actions']:
				if (label, ctr) in self.protected:
					sequence = self.vectors(action)
					for vector in sequence:
						vec = list(vector[:12])						#  Always include the hands.
						props_subvec = list(vector[12:])			#  Include props' signals if they're marked for inclusion.
						for i in range(0, len(self.recognizable_objects)):
							if self.protected_vector[ self.recognizable_objects[i] ] == True:
								vec.append( props_subvec[i] )
						fh.write('\t'.join([str(x) for x in vec]) + '\n')
					fh.write(label + '\n')

					if self.verbose:
						if int(round(float(complete_ctr) / float(num - 1) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
							prev_ctr = int(round(float(complete_ctr) / float(num - 1) * float(max_ctr)))
							sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(complete_ctr) / float(num - 1) * 100.0))) + '%]')
							sys.stdout.flush()

					complete_ctr += 1
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

		self.recognizable_objects = []								#  Reset everything.
		self.Xy = {}
		self.original_counts = {}
		self.protected = {}
		self.protected_vector = {}

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
				elif 'SMALLER SNIPPETS ACCEPTED' in line:			#  Accept snippets smaller than the snippet size.
					self.accept_small = True
				elif 'NUMBER OF SOURCE SNIPPETS:' in line:			#  Prepare to read the number of source snippets.
					reading_num_src_snippets = True
				elif 'DB SIZE:' in line:							#  Prepare to read the database size.
					reading_db_size = True

				elif reading_enactments:							#  Read the enactments that were used to build this database.
					self.enactments = line[1:].strip().split('\t')
					assert self.align_recognizable_objects(), 'The given enactments must recognize the same objects and have the same vector length.'
					reading_enactments = False
				elif reading_actions:								#  Read the actions this database can classify.
					self.Xy = {}
					for label in line[1:].strip().split('\t'):
						self.Xy[label] = {}
						self.Xy[label]['actions'] = []
						self.Xy[label]['mean-signal-strength'] = []
						self.Xy[label]['left-hand-strength'] = []
						self.Xy[label]['right-hand-strength'] = []
						self.Xy[label]['prop-strength'] = []
						self.Xy[label]['sample-conf-weight'] = []
					reading_actions = False
				elif reading_recognizable_objects:					#  Read the objects this database has been built to recognize.
					self.recognizable_objects = line[1:].strip().split('\t')
																	#  Initially mark all recognizable objects as included.
					for recognizable_object in self.recognizable_objects:
						self.protected_vector[recognizable_object] = True
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
					db_index = int(arr[1][1:-1])					#  Snip off brackets.
					sample_conf_weight = float(arr[2])
					src_enactment = arr[3]
					start_timestamp = float(arr[4])
					start_frame = arr[5]
					end_timestamp = float(arr[6])
					end_frame = arr[7]

					action_tuple = (src_enactment, start_timestamp, start_frame, end_timestamp, end_frame)

					self.Xy[label]['actions'].append(action_tuple)	#  Append action tuple.
					self.Xy[label]['mean-signal-strength'].append( self.mean_signal_strength(action_tuple)       )
					self.Xy[label]['left-hand-strength'].append(   self.left_hand_signal_strength(action_tuple)  )
					self.Xy[label]['right-hand-strength'].append(  self.right_hand_signal_strength(action_tuple) )
					self.Xy[label]['prop-strength'].append(        self.prop_signal_strength(action_tuple)       )
					self.Xy[label]['sample-conf-weight'].append(   sample_conf_weight                            )
			if self.verbose:
				if int(round(float(ctr) / float(num - 1) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
					prev_ctr = int(round(float(ctr) / float(num - 1) * float(max_ctr)))
					sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(ctr) / float(num - 1) * 100.0))) + '%]')
					sys.stdout.flush()
			ctr += 1

		self.keep_all()												#  Initially, set everything to be kept.

		return

	#  Return data about the i-th database snippet:
	#    Return the label.
	#    Return the index under that label.
	#    Return an action-tuple: (enactment name, start time, start frame, end time, end frame).
	#  Indexing follows the order in which snippets are written to file--that is, alphabetized by label.
	def index(self, i):
		snippets = []
		for label, actions in sorted(self.Xy.items()):
			for i in range(0, len(actions['actions'])):
				snippets.append( (label, i, actions['actions'][i]) )
		if i < len(snippets):
			return snippets[i][0], snippets[i][1], snippets[i][2]
		return None, None, None

	#  Print out. If a target_label is passed, then only itemize that label's snippets.
	#  Otherwise, itemize all labels' snippets.
	def itemize(self, target_label=None):
		if target_label is None:
			itemization = [x for x in sorted(self.Xy.items())]
		else:
			itemization = [x for x in self.Xy.items() if x[0] == target_label]

		for item in itemization:
			label = item[0]
			actions = item[1]
			print('>>> ' + label + ': u.Sig.\tu.LH\tu.RH\tu.P\tWght\tSnippet')
			for ctr in range(0, len(actions['actions'])):
				if (label, ctr) in self.protected:
					print('    ' + ' '*(len(label) - len(str(ctr)) - 4) + '+ [' + str(ctr) + ']  ' + \
					      "{:.4f}".format(actions['mean-signal-strength'][ctr]) + '\t' + \
					      "{:.4f}".format(actions['left-hand-strength'][ctr]) + '\t' + \
					      "{:.4f}".format(actions['right-hand-strength'][ctr]) + '\t' + \
					      "{:.4f}".format(actions['prop-strength'][ctr]) + '\t' + \
					      "{:.4f}".format(actions['sample-conf-weight'][ctr]) + '\t' + \
					      '\t'.join( [actions['actions'][ctr][0], str(actions['actions'][ctr][1]), str(actions['actions'][ctr][3])] ))
				else:
					print('    ' + ' '*(len(label) - len(str(ctr)) - 4) + '- [' + str(ctr) + ']  ' + \
					      "{:.4f}".format(actions['mean-signal-strength'][ctr]) + '\t' + \
					      "{:.4f}".format(actions['left-hand-strength'][ctr]) + '\t' + \
					      "{:.4f}".format(actions['right-hand-strength'][ctr]) + '\t' + \
					      "{:.4f}".format(actions['prop-strength'][ctr]) + '\t' + \
					      "{:.4f}".format(actions['sample-conf-weight'][ctr]) + '\t' + \
					      '\t'.join( [actions['actions'][ctr][0], str(actions['actions'][ctr][1]), str(actions['actions'][ctr][3])] ))
		return

	def labels(self):
		return sorted([x for x in self.Xy.keys()])

	#  Just show me the labels, how many are slated to be kept, and how many there were originally.
	def actions(self):
		for label, actions in sorted(self.Xy.items()):
			reduced_set = [x for x in range(0, len(actions['actions'])) if (label, x) in self.protected]
			print(label + ': ' + str(len(reduced_set)) + ' / ' + str(self.original_counts[label]))

	#  Track the original lengths so we can compute reduction.
	def count(self):
		self.original_counts = {}
		for label, actions in self.Xy.items():
			self.original_counts[label] = len(actions['actions'])
		return

	#  Sort self.Xy[*]['actions'] by mean signal strength.
	def sort(self):
		for label, actions in self.Xy.items():						#  Whether or not we intend to prune them, sort actions/labels by mean signal strength.
			sorted_actions = sorted(list(zip(actions['mean-signal-strength'], \
			                                 actions['left-hand-strength'],   \
			                                 actions['right-hand-strength'],  \
			                                 actions['prop-strength'], \
			                                 actions['sample-conf-weight'],  \
			                                 actions['actions'])), key=lambda x: x[0], reverse=True)
			self.Xy[label]['actions']              = [x[5] for x in sorted_actions]
			self.Xy[label]['mean-signal-strength'] = [x[0] for x in sorted_actions]
			self.Xy[label]['left-hand-strength']   = [x[1] for x in sorted_actions]
			self.Xy[label]['right-hand-strength']  = [x[2] for x in sorted_actions]
			self.Xy[label]['prop-strength']        = [x[3] for x in sorted_actions]
			self.Xy[label]['sample-conf-weight']   = [x[4] for x in sorted_actions]
		return

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

	def compute_signal_strength(self, include_dropped=False):
		if self.verbose:
			print('>>> Computing mean signal strengths of subvectors.')

		max_ctr = os.get_terminal_size().columns - 7				#  Leave enough space for the brackets, space, and percentage.
		num = 0
		for label, action_data in self.Xy.items():
			num += len(action_data['actions'])
		prev_ctr = 0
		ctr = 0

		for label, action_data in self.Xy.items():
			for i in range(0, len(action_data['actions'])):
				if (label, i) in self.protected or include_dropped:	#  Ordinarily, don't bother about dropped snippets.
					self.Xy[label]['mean-signal-strength'][i] = self.mean_signal_strength( action_data['actions'][i] )
					self.Xy[label]['left-hand-strength'][i]   = self.left_hand_signal_strength( action_data['actions'][i] )
					self.Xy[label]['right-hand-strength'][i]  = self.right_hand_signal_strength( action_data['actions'][i] )
					self.Xy[label]['prop-strength'][i]        = self.prop_signal_strength( action_data['actions'][i] )

				if self.verbose:
					if int(round(float(ctr) / float(num - 1) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
						prev_ctr = int(round(float(ctr) / float(num - 1) * float(max_ctr)))
						sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(ctr) / float(num - 1) * 100.0))) + '%]')
						sys.stdout.flush()
				ctr += 1

		self.sort()													#  Sort by mean signal strength.
		return

	#################################################################
	#  Mark actions for wholesale inclusion or omission.            #
	#  One action likely yields several snippets.                   #
	#################################################################
	#  The AtemporalClassifier class can load and export train-test splits.
	def load_action_split(self, split_file_name, split_set):
		fh = open(split_file_name, 'r')
		for line in fh.readlines():
			if line[0] != '#':
				arr = line.strip().split('\t')
				enactment_name = arr[0]
				action_index = int(arr[1])
				action_label = arr[2]
				set_name = arr[3]

				if set_name != split_set:
					self.drop_enactment_action(enactment_name, action_index)
		fh.close()
		return

	#  Prevent the index-th action in 'enactment_name' from generating any snippets.
	#  (Some actions are just no good, from head to tail.)
	def drop_enactment_action(self, enactment_name, index):
		self.dropped_actions[ (enactment_name, index) ] = True
		return

	#  "No! Wait! I changed my mind!"
	def keep_enactment_action(self, enactment_name, index):
		if (enactment_name, index) in self.dropped_actions:
			del self.dropped_actions[ (enactment_name, index) ]
		return

	def itemize_actions(self, elist=None):
		if elist is None:
			elist = self.enactments[:]

		for enactment in elist:
			print('\n' + enactment + ':')
			print('='*(len(enactment) + 1))

			maxlen_label = 0
			maxlen_timestamp = 0
			maxlen_filepath = 0

			e = ProcessedEnactment(enactment, verbose=False)
			maxlen_index = len(str(len(e.actions) - 1))
			for action in e.actions:
				maxlen_label = max(maxlen_label, len(action[0]))
				maxlen_timestamp = max(maxlen_timestamp, len(str(action[1])), len(str(action[3])))
				maxlen_filepath = max(maxlen_filepath, len(action[2].split('/')[-1]), len(action[4].split('/')[-1]))

			i = 0
			for action in e.actions:								#  Print all nice and tidy like.

				if (enactment, i) in self.dropped_actions:			#  If we are wholesale omitting this action (several snippets), this enactment.
					print('- [' + str(i) + ' '*(maxlen_index - len(str(i))) + ']: ' + \
					      action[0] + ' '*(maxlen_label - len(action[0])) + ': incl. ' + \
					      str(action[1]) + ' '*(maxlen_timestamp - len(str(action[1]))) + ' ' + \
					      action[2].split('/')[-1] + ' '*(maxlen_filepath - len(action[2].split('/')[-1])) + ' --> excl. ' + \
					      str(action[3]) + ' '*(maxlen_timestamp - len(str(action[3]))) + ' ' + \
					      action[4].split('/')[-1] + ' '*(maxlen_filepath - len(action[4].split('/')[-1])) )
				else:												#  We are NOT wholesale omitting this action, this enactment.
					print('+ [' + str(i) + ' '*(maxlen_index - len(str(i))) + ']: ' + \
					      action[0] + ' '*(maxlen_label - len(action[0])) + ': incl. ' + \
					      str(action[1]) + ' '*(maxlen_timestamp - len(str(action[1]))) + ' ' + \
					      action[2].split('/')[-1] + ' '*(maxlen_filepath - len(action[2].split('/')[-1])) + ' --> excl. ' + \
					      str(action[3]) + ' '*(maxlen_timestamp - len(str(action[3]))) + ' ' + \
					      action[4].split('/')[-1] + ' '*(maxlen_filepath - len(action[4].split('/')[-1])) )
				i += 1

		return

	def which_enactment_has_action(self, query_action):
		elist = []
		for enactment in self.enactments:
			e = ProcessedEnactment(enactment, verbose=False)
			for action in e.actions:
				if action[0] == query_action:
					if enactment not in elist:
						elist.append(enactment)
		return elist

	#################################################################
	#  Mark snippets for inclusion or omission.                     #
	#################################################################
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
					protected[ (k, ctr) ] = True
		else:
			for k, v in self.Xy.items():
				if k == label:
					for ctr in range(0, len(v['actions'])):
						protected[ (k, ctr) ] = True
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
			if key in self.protected:
				del self.protected[key]
		return

	#  Keep the highest 'portion' of actions under 'label'.
	#  'portion' is in [0.0, 1.0].
	#  (Idunno, what would you call the opposite of "cull()"?)
	def endorse(self, label, portion):
		assert isinstance(portion, float) and portion >= 0.0 and portion <= 1.0, \
		       'Argument \'portion\' passed to Database.endorse() must be a float in [0.0, 1.0].'
		indices = [x for x in range(0, len(self.Xy[label]['actions']))]
		lim = int(round(float(len(indices)) * portion))
		for ctr in range(0, lim):
			key = (label, ctr)
			self.protected[key] = True
		return

	#################################################################
	#  Editing.                                                     #
	#################################################################
	#  Really just marks the given 'object_label' for omission from outputs and vector means.
	def remove_column(self, object_label):
		if object_label in self.recognizable_objects:
			self.protected_vector[object_label] = False
		return

	#  Find all instances of 'old_label' and rename them under 'new_label'.
	#  Relabeling can also merge labels:
	#  e.g. Both "OpenDoor("TargetBackBreaker")" and "OpenDoor("Target2BackBreaker")" become "Open (BB)".
	def relabel(self, old_label, new_label):
		Xy = {}
		protected = {}

		if old_label in self.Xy:									#  Don't bother if it's not here.

			if new_label in self.Xy:								#  Merge 'old_label' into existing 'new_label'.
																	#  Copy over everything EXCEPT the label to be relabeled.
				for label, action_data in [x for x in self.Xy.items() if x[0] != old_label]:
					Xy[label] = action_data
					for key in [x for x in self.protected.keys() if x[0] == label]:
						protected[ key ] = True

				offset = len(self.Xy[new_label]['actions'])

				Xy[new_label]['actions']              += self.Xy[old_label]['actions']
				Xy[new_label]['mean-signal-strength'] += self.Xy[old_label]['mean-signal-strength']
				Xy[new_label]['left-hand-strength']   += self.Xy[old_label]['left-hand-strength']
				Xy[new_label]['right-hand-strength']  += self.Xy[old_label]['right-hand-strength']
				Xy[new_label]['prop-strength']        += self.Xy[old_label]['prop-strength']
				Xy[new_label]['sample-conf-weight']   += self.Xy[old_label]['sample-conf-weight']

				for key in [x for x in self.protected.keys() if x[0] == old_label]:
					protected[ (new_label, offset + key[1]) ] = True
			else:													#  Simply relabel 'old_label' as 'new_label'.
				for label, action_data in self.Xy.items():
					if label == old_label:
						Xy[new_label] = action_data
						for key in [x for x in self.protected.keys() if x[0] == label]:
							protected[ (new_label, key[1]) ] = True
					else:
						Xy[label] = action_data
						for key in [x for x in self.protected.keys() if x[0] == label]:
							protected[ key ] = True

			self.Xy = Xy											#  Replace attributes.
			self.protected = protected

			self.count()
			self.sort()

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

	#################################################################
	#  Utilities.                                                   #
	#################################################################
	#  Make sure that all the enactments we were given identify the same objects.
	def align_recognizable_objects(self):
		recognizable_objects_alignment = {}
		vector_length_alignment = {}
		for enactment in self.enactments:
			e = ProcessedEnactment(enactment, verbose=False)
			recognizable_objects_alignment[ tuple(e.recognizable_objects) ] = True
			vector_length_alignment[e.vector_length] = True
			recognizable_objects = e.recognizable_objects
		if len(recognizable_objects_alignment.keys()) > 1:
			print('ERROR: The given enactments differ in which objects are recognized and are therefore incompatible.')
			return False
		if len(vector_length_alignment.keys()) > 1:
			print('ERROR: The given enactments differ in their vector lengths and are therefore incompatible.')
			return False
		self.recognizable_objects = [x for x in recognizable_objects_alignment.keys()][0]
		for recognizable_object in self.recognizable_objects:		#  Initially mark all recognizable objects as included.
			self.protected_vector[recognizable_object] = True
		self.vector_length = [x for x in vector_length_alignment.keys()][0]
		return True

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
	#  given the layout of the "classroom," state-transitions are really more instances of new state appearances.
	#  Meaning, what you actually want to look for are sequences were a particular signal goes from 0.0 to something > 0.0.
	#  One door may open, but the signal for closed doors will not go away because adjacent doors in view remain closed.
	#
	#  Example: identify all "Close Disconnect (MFB)" snippets that contain a downturn for Disconnect_Open or Disconnect_Unknown.
	#    keepers = db.lambda_identify( db.snippets('Close Disconnect (MFB)'),
	#                                  (lambda seq: db.contains_downturn(seq, db.recognizable_objects.index('Disconnect_Open') + 12) or
	#                                               db.contains_downturn(seq, db.recognizable_objects.index('Disconnect_Unknown') + 12)) )
	#    db.drop_all('Close Disconnect (MFB)')
	#    db.keep('Close Disconnect (MFB)', keepers)
	#
	#  Example: identify all "Release (Meter)" snippets that contain a change in either hand's status from 1=grabbing
	#                                                                 to either 0=open or 2=pointing (or disappeared).
	#    keepers = db.lambda_identify( db.snippets('Release (Meter)'),
	#                                  lambda seq: db.contains_hand_status_change_from(seq, 1) )
	#    db.drop_all('Release (Meter)')
	#    db.keep('Release (Meter)', keepers)
	#
	#  Example: identify all "Grab (Meter)" snippets that contain a change in either hand's status from either 0=open or 2=pointing (or disappeared)
	#                                                                                              to 1=grabbing.
	#    keepers = db.lambda_identify( db.snippets('Grab (Meter)'),
	#                                  lambda seq: db.contains_hand_status_change_to(seq, 1) )
	#    db.drop_all('Grab (Meter)')
	#    db.keep('Grab (Meter)', keepers)
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

	#  Given a sequence and a hand-status, determine whether the sequence starts with either hand's one-hot
	#  subvector IN the state and then changes OUT of that state exactly ONCE.
	#  No other pattern will do: we cannot never leave the status, and we cannot reach leave and return to the status.
	#
	#  Recall that there are three values for hand status: {0: open; 1: grasping; 2: pointing}.
	#  When a hand is not visible on screen, its subvector is the zero vector.
	#
	#  Also notice that *sometimes*, all we care about is that a hand is NOT in some state.
	#  This becomes relevant when you recall that the hands are controlled by paddle buttons, and that a user may
	#  accidentally point when merely wishing to cease clutching an object.
	def contains_hand_status_change_from(self, sequence, status):
		bool_vec_lh = []
		bool_vec_rh = []

		if isinstance(status, int):									#  Accept ints, but wrap them in lists.
			status = [status]

		for vector in sequence:										#  For each frame, is the target signal greater than zero?
			lh = [x for x in range(0, 3) if vector[3:6][x] > 0.0]	#  Left Hand One-hot = [3:5] = [3:6)
			rh = [x for x in range(0, 3) if vector[9:12][x] > 0.0]	#  Right Hand One-hot = [9:11] = [9:12)

			if len(lh) > 0:											#  Is left hand even visible? (Exists any element > 0?)
				bool_vec_lh.append( len(list(set(lh).intersection(status))) > 0 )
			else:
				bool_vec_lh.append(False)							#  Not in any state.
			if len(rh) > 0:											#  Is right hand even visible? (Exists any element > 0?)
				bool_vec_rh.append( len(list(set(rh).intersection(status))) > 0 )
			else:
				bool_vec_rh.append(False)							#  Not in any state.

		current_lh = bool_vec_lh[0]									#  Now "flatten" the lists of Booleans.
		current_rh = bool_vec_rh[0]

		compressed_lh = [current_lh]
		compressed_rh = [current_rh]

		for b in bool_vec_lh:
			if b != current_lh:
				compressed_lh.append(b)
				current_lh = b

		for b in bool_vec_rh:
			if b != current_rh:
				compressed_rh.append(b)
				current_rh = b
																	#  If, after flattening, we have for either hand, a True state
																	#  followed by a False state, and no more. This is what we want.
		if (len(compressed_lh) == 2 and compressed_lh[0] == True) or \
		   (len(compressed_rh) == 2 and compressed_rh[0] == True):
			return True

		return False												#  No other pattern will do.

	#  Similar to above, but reversed: return true if the sequence starts OUT of 'status' and enters 'status'.
	def contains_hand_status_change_to(self, sequence, status):
		bool_vec_lh = []
		bool_vec_rh = []

		if isinstance(status, int):									#  Accept ints, but wrap them in lists.
			status = [status]

		for vector in sequence:										#  For each frame, is the target signal greater than zero?
			lh = [x for x in range(0, 3) if vector[3:6][x] > 0.0]	#  Left Hand One-hot = [3:5] = [3:6)
			rh = [x for x in range(0, 3) if vector[9:12][x] > 0.0]	#  Right Hand One-hot = [9:11] = [9:12)

			if len(lh) > 0:											#  Is left hand even visible? (Exists any element > 0?)
				bool_vec_lh.append( len(list(set(lh).intersection(status))) == 0 )
			else:
				bool_vec_lh.append(True)							#  Not in any state means not in the target state.
			if len(rh) > 0:											#  Is right hand even visible? (Exists any element > 0?)
				bool_vec_rh.append( len(list(set(rh).intersection(status))) == 0 )
			else:
				bool_vec_rh.append(True)							#  Not in any state means not in the target state.

		current_lh = bool_vec_lh[0]									#  Now "flatten" the lists of Booleans.
		current_rh = bool_vec_rh[0]

		compressed_lh = [current_lh]
		compressed_rh = [current_rh]

		for b in bool_vec_lh:
			if b != current_lh:
				compressed_lh.append(b)
				current_lh = b

		for b in bool_vec_rh:
			if b != current_rh:
				compressed_rh.append(b)
				current_rh = b
																	#  If, after flattening, we have for either hand, a True state
																	#  followed by a False state, and no more. This is what we want.
		if (len(compressed_lh) == 2 and compressed_lh[0] == True) or \
		   (len(compressed_rh) == 2 and compressed_rh[0] == True):
			return True

		return False												#  No other pattern will do.

	#  For each vector in the given 'snippet', compute its magnitude, excluding the one-hot-encoded elements for left and right hands.
	#  Return the average of all magnitudes.
	#  Vector encoding is [0]LH_x [1]LH_y [2]LH_z [3]LH_0 [4]LH_1 [5]LH_2 [6]RH_x [7]RH_y [8]RH_z [9]RH_0 [10]RH_1 [11]RH_2 [12]Prop_1 ...
	#  'snippet' is (enactment, start time, start frame, end time, end frame).
	def mean_signal_strength(self, snippet, hands_coeff=1.0, one_hot_coeff=1.0, props_coeff=1.0):
		sequence = self.vectors(snippet)
		magnitudes = []
		for vector in sequence:
			props_vec = [props_coeff * vector[12:][i] for i in range(0, len(self.recognizable_objects)) \
			                                           if self.protected_vector[ self.recognizable_objects[i] ] == True]
			vec = [hands_coeff * x for x in vector[:3]]  + \
			      [one_hot_coeff * x for x in vector[3:6]] + \
			      [hands_coeff * x for x in vector[6:9]] + \
			      [one_hot_coeff * x for x in vector[9:12]] + \
			      props_vec

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
	#  Remember that the user may have dropped some elements from the props subvector--this is not allowed for the hands subvectors.
	def prop_signal_strength(self, snippet, props_coeff=1.0):
		sequence = self.vectors(snippet)
		magnitudes = []
		for vector in sequence:
			vec = [props_coeff * vector[12:][i] for i in range(0, len(self.recognizable_objects)) \
			                                     if self.protected_vector[ self.recognizable_objects[i] ] == True]
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
			vec = [props_coeff * vector[12:][i] for i in range(0, len(self.recognizable_objects)) \
			                                     if self.protected_vector[ self.recognizable_objects[i] ] == True]
			#vec = [props_coeff * x for x in vector[12:]]
			magnitudes.append( np.linalg.norm(vec) )
		return np.mean(magnitudes)

	def	original_size(self, label_size=None):
		if label_size is None:
			return sum([x for x in self.original_counts.values()])
		return self.original_counts[label_size]

	def current_size(self, label_size=None):
		s = []

		if label_size is None:
			for label, actions in self.Xy.items():
				reduced_set = [x for x in range(0, len(actions['actions'])) if (label, x) in self.protected]
				s.append(len(reduced_set))
		else:
			s.append( len( [x for x in range(0, len(self.Xy[label_size]['actions'])) if (label_size, x) in self.protected] ) )

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
					if img_filename == frame and obj_type not in ['LeftHand', 'RightHand'] and self.protected_vector[ obj_type ] == True:
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