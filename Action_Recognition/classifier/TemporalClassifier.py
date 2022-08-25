import cv2
import numpy as np
import os
import sys
import time

from classifier.Classifier import Classifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'							#  Suppress TensorFlow barf.
import tensorflow as tf
from object_detection.utils import label_map_util					#  Works with TensorFlow Object-Model Zoo's model library.

sys.path.append('../enactment')
from enactment import Enactment, Gaussian3D, ProcessedEnactment

'''
Give this classifier a database file and enactment files.
The database becomes the training set, self.X_train, and the enactments will be marched through in simulated real time.
Buffers-full of vectors from the enactments are given to the classification engine.
This constitutes "temporal" classification because sequence boundaries are NOT known a priori.

In the interpreter:
temporal = TemporalClassifier(rolling_buffer_length=10, rolling_buffer_stride=2, db_file='10f.db', relabel='relabels.txt', inputs=['Enactment11', 'Enactment12'], verbose=True)
'''
class TemporalClassifier(Classifier):
	def __init__(self, **kwargs):
		super(TemporalClassifier, self).__init__(**kwargs)

		if 'rolling_buffer_length' in kwargs:						#  Were we given a rolling buffer length?
			assert isinstance(kwargs['rolling_buffer_length'], int) and kwargs['rolling_buffer_length'] > 0, \
			       'Argument \'rolling_buffer_length\' passed to Classifier must be an integer > 0.'
			self.rolling_buffer_length = kwargs['rolling_buffer_length']
		else:
			self.rolling_buffer_length = 10							#  Default to 10.

		if 'rolling_buffer_stride' in kwargs:						#  Were we given a rolling buffer stride?
			assert isinstance(kwargs['rolling_buffer_stride'], int) and kwargs['rolling_buffer_stride'] > 0, \
			       'Argument \'rolling_buffer_stride\' passed to Classifier must be an integer > 0.'
			self.rolling_buffer_stride = kwargs['rolling_buffer_stride']
		else:
			self.rolling_buffer_stride = 2							#  Default to 2.

		if 'temporal_buffer_length' in kwargs:						#  Were we given a temporal buffer length?
			assert isinstance(kwargs['temporal_buffer_length'], int) and kwargs['temporal_buffer_length'] > 0, \
			       'Argument \'temporal_buffer_length\' passed to Classifier must be an integer > 0.'
			self.temporal_buffer_length = kwargs['temporal_buffer_length']
		else:
			self.temporal_buffer_length = 3							#  Default to 3.

		if 'temporal_buffer_stride' in kwargs:						#  Were we given a temporal buffer stride?
			assert isinstance(kwargs['temporal_buffer_stride'], int) and kwargs['temporal_buffer_stride'] > 0, \
			       'Argument \'temporal_buffer_stride\' passed to Classifier must be an integer > 0.'
			self.temporal_buffer_stride = kwargs['temporal_buffer_stride']
		else:
			self.temporal_buffer_stride = 1							#  Default to 1.

		if 'db_file' in kwargs:										#  Were we given a database file?
			assert isinstance(kwargs['db_file'], str), 'Argument \'db_file\' passed to TemporalClassifier must be a string.'
			self.database_file = kwargs['db_file']
		else:
			self.database_file = None

		if 'inputs' in kwargs:										#  Were we given a list of enactments to use in simulated real time?
			assert isinstance(kwargs['inputs'], list), 'Argument \'inputs\' passed to TemporalClassifier must be a list of enactment name strings.'
			self.enactment_inputs = kwargs['inputs']
		else:
			self.enactment_inputs = []

		if 'relabel' in kwargs and kwargs['relabel'] is not None:	#  Were we given a relableing file?
			assert isinstance(kwargs['relabel'], str), 'Argument \'relabel\' passed to TemporalClassifier must be a string: the filepath for a relabeling file.'
			if kwargs['relabel'] is not None:
				self.relabelings = {}
				fh = open(kwargs['relabel'], 'r')
				for line in fh.readlines():
					if line[0] != '#':
						arr = line.strip().split('\t')
						self.relabelings[arr[0]] = arr[1]
				fh.close()
		else:
			self.relabelings = {}									#  key: old label ==> val: new label

		if 'gaussian' in kwargs:									#  Were we given a Gaussian3D object or a 3-tuple of 3-tuples?
			assert isinstance(kwargs['gaussian'], Gaussian3D) or (isinstance(kwargs['gaussian'], tuple) and len(kwargs['gaussian']) == 3), \
			  'Argument \'gaussian\' passed to TemporalClassifier must be either a Gaussian3D object or a 3-tuple of 3-tuples of floats: (mu for the gaze, sigma for the gaze, sigma for the hands).'
			if isinstance(kwargs['gaussian'], Gaussian3D):
				self.gaussian = kwargs['gaussian']
			else:
				self.gaussian = Gaussian3D(mu=kwargs['gaussian'][0], \
				                           sigma_gaze=kwargs['gaussian'][1], \
				                           sigma_hand=kwargs['gaussian'][2])
		else:
			self.gaussian = Gaussian3D(mu=(0.0, 0.0, 0.0), sigma_gaze=(4.0, 3.5, 6.5), sigma_hand=(2.0, 2.0, 2.0))

		if 'min_bbox' in kwargs:									#  Were we given a minimum bounding box area for recognized objects?
			assert isinstance(kwargs['min_bbox'], int) and kwargs['min_bbox'] > 0, \
			  'Argument \'min_bbox\' passed to TemporalClassifier must be an integer greater than 0.'
			self.minimum_bbox_area = kwargs['min_bbox']
		else:
			self.minimum_bbox_area = 1

		if 'detection_confidence' in kwargs:						#  Were we given a detection confidence threshold for recognized objects?
			assert isinstance(kwargs['detection_confidence'], float) and kwargs['detection_confidence'] >= 0.0 and kwargs['detection_confidence'] <= 1.0, \
			  'Argument \'detection_confidence\' passed to TemporalClassifier must be an integer greater than 0.'
			self.detection_confidence = kwargs['detection_confidence']
		else:
			self.detection_confidence = 0.0

		if 'use_detection_source' in kwargs:						#  Were we given a detection source?
			assert isinstance(kwargs['use_detection_source'], str), 'Argument \'use_detection_source\' passed to TemporalClassifier must be a string.'
			self.use_detection_source = kwargs['use_detection_source']
		else:
			self.use_detection_source = None

		if 'render_modes' in kwargs:								#  Were we given render modes?
			assert isinstance(kwargs['render_modes'], list), \
			       'Argument \'render_modes\' passed to TemporalClassifier must be a list of zero or more strings in {rolling-buffer, confidence, probabilities, smoothed}.'
			self.render_modes = kwargs['render_modes']
			self.render = True
		else:
			self.render_modes = []

		#############################################################
		#  Main attributes for this class.                          #
		#############################################################
		self.X_train = []											#  Contains lists of tuples (sequences of vectors).
		self.y_train = []											#  Contains strings (labels).
																	#  Becomes 'query'.
		self.rolling_buffer = [None for i in range(0, self.rolling_buffer_length)]
		self.rolling_buffer_filling = True							#  Buffer has yet to reach capacity
																	#  Holds probability distributions.
		self.temporal_buffer = [None for i in range(0, self.temporal_buffer_length)]
		self.temporal_buffer_filling = True							#  Buffer has yet to reach capacity
																	#  Holds buffer-fulls of ground-truth labels.
																	#  Used to determine whether a evaluation is "fair."
		self.buffer_labels = [None for i in range(0, self.rolling_buffer_length)]

		self.train_sample_lookup = {}								#  key: index into X_train ==> val: (source enactment, start time, start frame,
		self.vector_length = None									#                                                      end time,   end frame)
		self.width = None
		self.height = None

		#############################################################
		#  Attributes used for rendering and display.               #
		#############################################################
		self.seismograph_length = 10								#  Cache length.
		self.max_seismograph_linewidth = 15

		self.seismograph_enactment_name_super = {}					#  Where to locate and how to type the source of an annotated frame.
		self.seismograph_enactment_name_super['x'] = 10
		self.seismograph_enactment_name_super['y'] = 50
		self.seismograph_enactment_name_super['fontsize'] = 1.0

		self.seismograph_ground_truth_label_super = {}				#  Where to locate and how to type the label of an annotated frame.
		self.seismograph_ground_truth_label_super['x'] = 10
		self.seismograph_ground_truth_label_super['y'] = 90
		self.seismograph_ground_truth_label_super['fontsize'] = 1.0

		self.seismograph_prediction_super = {}						#  Where to locate and how to type the predicted label in an annotated frame.
		self.seismograph_prediction_super['x'] = 10
		self.seismograph_prediction_super['y'] = 130
		self.seismograph_prediction_super['fontsize'] = 1.0

		if self.database_file is not None:
			self.load_db(self.database_file)

	#################################################################
	#  Labels.                                                      #
	#################################################################

	#  Return a list of unique action labels as evidences by the snippet labels in 'y_train' and/or the test enactments.
	#  It is possible that the training set and test enactments contain different action labels.
	def labels(self, sets='both'):
		if sets == 'train':
			return list(np.unique(self.y_train))
		if sets == 'test':
			labels = {}												#  Used for its set-like properties.
			for enactment_inputs in self.enactment_inputs:			#  Include all (potentially unknown) labels from the input.
				pe = ProcessedEnactment(enactment_inputs, verbose=False)
				for x in pe.labels():
					if x in self.relabelings:						#  Allow that we may have asked to relabel some or all of these.
						labels[ self.relabelings[x] ] = True
					else:
						labels[x] = True
			return sorted([x for x in labels.keys()])				#  Return a sorted list of unique labels.

		labels = {}													#  Used for its set-like properties.
		for enactment_inputs in self.enactment_inputs:				#  Include all (potentially unknown) labels from the input.
			pe = ProcessedEnactment(enactment_inputs, verbose=False)
			for x in pe.labels():
				if x in self.relabelings:							#  Allow that we may have asked to relabel some or all of these.
					labels[ self.relabelings[x] ] = True
				else:
					labels[x] = True
		for label in list(np.unique(self.y_train)):					#  Also include the training set labels.
			labels[label] = True
		return sorted([x for x in labels.keys()])					#  Return a sorted list of unique labels.

	#  Return a list of unique action labels as evidences by the snippet labels in 'y_train' and/or the test enactments.
	#  It is possible that the training set and test enactments contain different action labels.
	def num_labels(self, sets='both'):
		labels = self.labels(sets)
		return len(labels)

	#################################################################
	#  Classify.                                                    #
	#################################################################

	#  March through each vector/frame in input enactments as if they were received in real time.
	#  Note that this is not our best approximation of a deployed system, since it avoids all the work
	#  of image processing/detection by network, centroid location, etc. As such, this method is best for
	#  testing the accuracy of the matching mechanism.
	#  And, yes, this is "simulated real time," but if we know we won't save the results of a test, then why perform that test?
	#  This is what the 'skip_unfair' parameter is for: check the accumulated ground-truth labels first BEFORE running DTW.
	#  Save time; run more tests.
	#  A noteworthy caveat of this method is that IF YOUR TEMPORAL BUFFER LENGTH IS > 1, THEN SKIPPING UNFAIR TESTS WILL
	#  CREATE DISCONTINUITIES IN THE TEMPORAL BUFFER: there may be gaps between the last outgoing fair probability distribution
	#  and the next incoming probability distribution. USE THIS METHOD WITH CAUTION!
	def simulated_classify(self, skip_unfair=True):
		assert isinstance(skip_unfair, bool), 'Argument \'skip_unfair\' passed to TemporalClassifier.classify() must be a Boolean.'

		n_labels = self.num_labels('train')							#  Store the number of training-set labels.
		classification_stats = self.initialize_stats()				#  Init.

		self.initialize_timers()									#  (Re)set.

		t0_start = time.process_time()								#  Start timer.
		for enactment_input in self.enactment_inputs:				#  Treat each input enactment as a separate slice of time.
																	#  (Re)initialize the rolling buffer.
			self.rolling_buffer = [None for i in range(0, self.rolling_buffer_length)]
			self.rolling_buffer_filling = True						#  Buffer has yet to reach capacity
																	#  (Re)initialize the temporal buffer.
			self.temporal_buffer = [None for i in range(0, self.temporal_buffer_length)]
			self.temporal_buffer_filling = True						#  Buffer has yet to reach capacity
																	#  (Re)initialize the ground-truth labels buffer.
			self.buffer_labels = [None for i in range(0, self.rolling_buffer_length)]

			if self.render:											#  If we're rendering, we may need to cache some things
				if 'confidence' in self.render_modes:				#  we do not need when simply making predictions.
					confidence_store = []
				if 'probabilities' in self.render_modes:
					probability_store = []
				if 'smooth' in self.render_modes:
					smoothed_probability_store = []

			vector_buffer = []										#  Takes the place of X_test or an input stream.
			ground_truth_buffer = []								#  Can include "*" or "nothing" labels.
			time_stamp_buffer = []
			frame_path_buffer = []

			t1_start = time.process_time()							#  Start timer.
			fh = open(enactment_input + '.enactment', 'r')			#  Read in the input-enactment.
			lines = fh.readlines()
			fh.close()
			for line in lines:
				if line[0] != '#':
					arr = line.strip().split('\t')
					timestamp = float(arr[0])						#  Save the time stamp.
					frame_filename = arr[1]							#  Save the frame file path.
					ground_truth_label = arr[2]						#  Save the true label (these include the nothing-labels.)
					vector = [float(x) for x in arr[3:]]			#  This is already Gaussian-weighed and maxed.

					if self.hand_schema == 'strong-hand':			#  Apply hand-schema (if applicable.)
						vector = self.strong_hand_encode(vector)

					vec = list(vector[:12])							#  Apply vector drop-outs (where applicable.)
					for i in range(0, len(self.vector_drop_map)):
						if self.vector_drop_map[i] == True:
							vec.append(vector[i + 12])
					vector = tuple(vec)
																	#  Apply coefficients (if applicable.)
					vector_buffer.append( self.apply_vector_coefficients(vector) )

					if ground_truth_label in self.relabelings:
						ground_truth_buffer.append( self.relabelings[ground_truth_label] )
					else:
						ground_truth_buffer.append( ground_truth_label )

					time_stamp_buffer.append( timestamp )
					frame_path_buffer.append( frame_filename )
			t1_stop = time.process_time()							#  Stop timer.
			self.timing['load-enactment'].append(t1_stop - t1_start)

			num = len(vector_buffer)
			prev_ctr = 0
			max_ctr = os.get_terminal_size().columns - 7			#  Leave enough space for the brackets, space, and percentage.
			if self.verbose:
				if skip_unfair:
					print('>>> Classifying from "' + enactment_input + '" in simulated real time (skipping unfair tests.)')
				else:
					print('>>> Classifying from "' + enactment_input + '" in simulated real time.')
				print('    Minimum detection area:       ' + str(self.minimum_bbox_area))
				print('    Minimum detection confidence: ' + str(self.detection_confidence))
				print('    Prediction threshold:         ' + str(self.threshold))

			nearest_neighbor_label = None							#  Initially nothing.
			prediction = None
			metadata = None											#  Initially nothing (applies when rendering).

			if self.render:											#  Rendering? Create the video file now and add to it through the loop.
				pe = ProcessedEnactment(enactment_input, verbose=False)
				self.width = pe.width
				self.height = pe.height
				if 'rolling-buffer' in self.render_modes:
					vid_rolling_buffer = cv2.VideoWriter(enactment_input + '_seismograph.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
					                                     pe.fps, (pe.width * 2, pe.height) )
				if 'confidence' in self.render_modes:
					vid_confidence = cv2.VideoWriter(enactment_input + '_confidence_seismograph.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
					                                 pe.fps, (pe.width * 2, pe.height) )
				if 'probabilities' in self.render_modes:
					vid_probabilities = cv2.VideoWriter(enactment_input + '_probabilities_seismograph.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
					                                    pe.fps, (pe.width * 2, pe.height) )
				if 'smooth' in self.render_modes:
					vid_smoothed_probabilities = cv2.VideoWriter(enactment_input + '_smoothed-probabilities_seismograph.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), \
					                                             pe.fps, (pe.width * 2, pe.height) )

			#########################################################
			#  Pseudo-boot-up is over. March through the buffer.    #
			#########################################################

			for frame_ctr in range(0, num):							#  March through vector_buffer.
				self.push_rolling( vector_buffer[frame_ctr] )		#  Push vector to rolling buffer.
																	#  Push G.T. label to rolling G.T. buffer.
				self.push_ground_truth_buffer( ground_truth_buffer[frame_ctr] )
																	#  Are the contents of the ground-truth buffer "fair"?
				#fair = self.ground_truth_buffer_full() and self.ground_truth_buffer_uniform() and self.ground_truth_buffer_familiar()
				fair = self.ground_truth_buffer_full() and (self.ground_truth_buffer_newest_frame_label() in self.labels('train') or self.ground_truth_buffer_newest_frame_label() == '*')

				#####################################################
				#  If the rolling buffer is full, then classify.    #
				#  This will always produce a tentative prediction  #
				#  based on lowest matching cost. Actual            #
				#  predictions are subject to threshold and         #
				#  smoothed probabilities.                          #
				#####################################################
				if self.is_rolling_buffer_full() and (fair or not skip_unfair):
																	#  Identify ground-truth.
					ground_truth_label = self.ground_truth_buffer_newest_frame_label()

					prediction = None								#  (Re)set.

					t1_start = time.process_time()					#  Start timer.
																	#  Call the parent class's core matching engine.
					nearest_neighbor_label, matching_costs, confidences, probabilities, metadata = super(TemporalClassifier, self).classify(self.rolling_buffer)
					t1_stop = time.process_time()					#  Stop timer.
					self.timing['dtw-classification'].append(t1_stop - t1_start)

					#################################################
					#  nearest_neighbor_label:        label or None #
					#  matching_costs: key: label ==> val: cost     #
					#                  in R^N                       #
					#  confidences:    key: label ==> val: score    #
					#                  in R^N                       #
					#  probabilities:  key: label ==> val: prob.    #
					#                  in R^N or R^{N+1}            #
					#  metadata:       key: label ==>               #
					#                    val: {query-indices,       #
					#                          template-indices,    #
					#                          db-index}            #
					#################################################

					#################################################
					#  When probabilities is in R^N the nothing-    #
					#  label is predicted only if all other scores  #
					#  are below a threshold.                       #
					#################################################
					if len(probabilities) == n_labels:				#  Probabilities is in R^N.
																	#  Among other tasks, this method pushes to the temporal buffer.
						sorted_confidences, sorted_probabilities = self.process_dtw_results(matching_costs, confidences, probabilities)

																	#  Save all costs for all labels.
						classification_stats['_costs'].append( tuple([time_stamp_buffer[ frame_ctr ]] + \
						                                             [time_stamp_buffer[ max(0, frame_ctr - (self.rolling_buffer_length - 1)) ]] + \
						                                             [enactment_input] + \
						                                             [matching_costs[label] for label in self.labels('train')] + \
						                                             [ground_truth_label]) )
																	#  Save confidence scores for all labels, regardless of what the system picks.
						for label in self.labels('train'):			#  We use these values downstream in the pipeline for isotonic regression.
							classification_stats['_conf'].append( (confidences[label], label, ground_truth_label, \
							                                       enactment_input, \
							                                       time_stamp_buffer[ max(0, frame_ctr - (self.rolling_buffer_length - 1)) ], \
							                                       time_stamp_buffer[ frame_ctr ]) )

						for label in self.labels('train'):			#  Save probabilities for all labels, regardless of what the system picks.
							classification_stats['_prob'].append( (probabilities[label], label, ground_truth_label, \
							                                       enactment_input, \
							                                       time_stamp_buffer[ max(0, frame_ctr - (self.rolling_buffer_length - 1)) ], \
							                                       time_stamp_buffer[ frame_ctr ]) )

																	#  If we are rendering confidences, then add to the confidences buffer.
						if self.render and 'confidence' in self.render_modes:
							confidence_store = self.push_buffer(sorted_confidences, confidence_store)
																	#  If we are rendering probabilities, then add to the probabilities buffer.
						if self.render and 'probabilities' in self.render_modes:
							probability_store = self.push_buffer(sorted_probabilities, probability_store)

						#############################################
						#  Smooth the contents of the temporal      #
						#  buffer and make a prediction (or abstain #
						#  from predicting).                        #
						#############################################

						t1_start = time.process_time()				#  Start timer.
						smoothed_probabilities = np.mean(np.array([x for x in self.temporal_buffer if x is not None]), axis=0)
						smoothed_norm = sum(smoothed_probabilities)
						if smoothed_norm > 0.0:
							smoothed_probabilities /= smoothed_norm
						smoothed_probabilities = list(smoothed_probabilities)
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['temporal-smoothing'].append(t1_stop - t1_start)

						if self.render and 'smooth' in self.render_modes:
							smoothed_probability_store = self.push_buffer(smoothed_probabilities, smoothed_probability_store)

						#############################################
						#  Predict... possibly an IMPLICIT nothing. #
						#############################################
						t1_start = time.process_time()				#  Start timer.
						prediction = None
						if nearest_neighbor_label is not None:		#  (It was not first impossible to select any nearest neighbor at all.)
							tentative_probability = smoothed_probabilities[ self.labels('train').index(nearest_neighbor_label) ]
							if tentative_probability >= self.threshold:
								prediction = nearest_neighbor_label
								if prediction in self.hidden_labels:#  Is this a hidden label? Then dummy up.
									prediction = None
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['make-temporally-smooth-decision'].append(t1_stop - t1_start)

						classification_stats = self.update_stats(prediction, ground_truth_label, (fair or not skip_unfair), classification_stats)
																	#  Whether or not it's skipped, put it on record.
						if nearest_neighbor_label is not None:
							classification_stats['_tests'].append( (prediction, ground_truth_label, \
							                                        confidences[nearest_neighbor_label], tentative_probability, \
							                                        enactment_input, time_stamp_buffer[frame_ctr], \
							                                        metadata[nearest_neighbor_label]['db-index'], \
							                                        fair) )
						else:										#  The tentative prediction is None if applied conditions make ALL possibilities impossible.
							classification_stats['_tests'].append( (prediction, ground_truth_label, \
							                                        0.0, 0.0, \
							                                        enactment_input, time_stamp_buffer[frame_ctr], \
							                                        -1, \
							                                        fair) )

																	#  First-Timestamp    Final-Timestamp    Source-Enactment    Confs...    Ground-Truth-Label    {fair,unfair}
						classification_stats['_test-conf'].append( tuple([time_stamp_buffer[max(0, frame_ctr - (self.rolling_buffer_length - 1))], \
						                                                  time_stamp_buffer[frame_ctr], enactment_input] + \
						                                                 [confidences[x] for x in self.labels('train')] + \
						                                                 [ground_truth_label, fair]) )
																	#  First-Timestamp    Final-Timestamp    Source-Enactment    Probs...    Ground-Truth-Label    {fair,unfair}
						classification_stats['_test-prob'].append( tuple([time_stamp_buffer[max(0, frame_ctr - (self.rolling_buffer_length - 1))], \
						                                                  time_stamp_buffer[frame_ctr], enactment_input] + \
						                                                 [probabilities[x] for x in self.labels('train')] + \
						                                                 [ground_truth_label, fair]) )
																	#  First-Timestamp    Final-Timestamp    Source-Enactment    Smoothed-Probs...    Ground-Truth-Label    {fair,unfair}
						classification_stats['_test-smooth-prob'].append( tuple([time_stamp_buffer[max(0, frame_ctr - (self.rolling_buffer_length - 1))], \
						                                                         time_stamp_buffer[frame_ctr], enactment_input] + \
						                                                        [smoothed_probabilities[x] for x in range(0, n_labels)] + \
						                                                        [ground_truth_label, fair]) )
					#################################################
					#  When probabilities is in R^{N+1} the nothing-#
					#  label has an explicit prediction.            #
					#  This becomes the system's prediction if this #
					#  probability is the highest and high enough,  #
					#  OR if no other label is probable enough.     #
					#################################################
					else:											#  Probabilities is in R^{N+1}. Meaning the nothing label has been explicitly predicted.
						labels_and_nothing = self.labels('train') + ['*']

						sorted_probabilities = [probabilities[label] for label in labels_and_nothing]
						t1_start = time.process_time()				#  Start timer.

						self.push_temporal( sorted_probabilities )	#  Add this probability distribution to the temporal buffer.
																	#  The temporal buffer always receives distributions sorted
																	#  according to the labels('train') method.
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['push-temporal-buffer'].append(t1_stop - t1_start)

																	#  Save all costs for all labels.
																	#  (There is no matching cost for the explicitly predicted nothing-label.)
						classification_stats['_costs'].append( tuple([time_stamp_buffer[ frame_ctr ]] + \
						                                             [time_stamp_buffer[ max(0, frame_ctr - (self.rolling_buffer_length - 1)) ]] + \
						                                             [enactment_input] + \
						                                             [matching_costs[label] for label in labels_and_nothing[:-1]] + \
						                                             [ground_truth_label]) )
																	#  Save confidence scores for all labels, regardless of what the system picks.
						for label in self.labels('train'):			#  We use these values downstream in the pipeline for isotonic regression.
							classification_stats['_conf'].append( (confidences[label], label, ground_truth_label, \
							                                       enactment_input, \
							                                       time_stamp_buffer[ max(0, frame_ctr - (self.rolling_buffer_length - 1)) ], \
							                                       time_stamp_buffer[ frame_ctr ]) )

						for label in labels_and_nothing:			#  Save probabilities for all labels, regardless of what the system picks.
							classification_stats['_prob'].append( (probabilities[label], label, ground_truth_label, \
							                                       enactment_input, \
							                                       time_stamp_buffer[ max(0, frame_ctr - (self.rolling_buffer_length - 1)) ], \
							                                       time_stamp_buffer[ frame_ctr ]) )

						#############################################
						#  Smooth the contents of the temporal      #
						#  buffer and make a prediction (or abstain #
						#  from predicting).                        #
						#############################################

						t1_start = time.process_time()				#  Start timer.
						smoothed_probabilities = np.mean(np.array([x for x in self.temporal_buffer if x is not None]), axis=0)
						smoothed_norm = sum(smoothed_probabilities)
						if smoothed_norm > 0.0:
							smoothed_probabilities /= smoothed_norm
						smoothed_probabilities = list(smoothed_probabilities)
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['temporal-smoothing'].append(t1_stop - t1_start)

						if self.render and 'smooth' in self.render_modes:
							smoothed_probability_store = self.push_buffer(smoothed_probabilities, smoothed_probability_store)

						#############################################
						#  Predict... possibly an EXPLICIT nothing. #
						#############################################
						t1_start = time.process_time()				#  Start timer.
						prediction = None
						if nearest_neighbor_label is not None:		#  (It was not first impossible to select any nearest neighbor at all.)
							tentative_probability = smoothed_probabilities[ labels_and_nothing.index(nearest_neighbor_label) ]
							nothing_probability = smoothed_probabilities[ -1 ]

							if tentative_probability > nothing_probability and tentative_probability >= self.threshold:
								prediction = nearest_neighbor_label
								if prediction in self.hidden_labels:#  Is this a hidden label? Then dummy up.
									prediction = None
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['make-temporally-smooth-decision'].append(t1_stop - t1_start)

						classification_stats = self.update_stats(prediction, ground_truth_label, (fair or not skip_unfair), classification_stats)
																	#  Whether or not it's skipped, put it on record.
						test_tuple = [prediction, ground_truth_label]
						test_tuple.append(confidences[nearest_neighbor_label])
						test_tuple.append(tentative_probability)
						test_tuple.append(enactment_input)
						test_tuple.append(time_stamp_buffer[frame_ctr])
						if 'db-index' in metadata[nearest_neighbor_label]:
							test_tuple.append(metadata[nearest_neighbor_label]['db-index'])
						else:
							test_tuple.append(None)
						test_tuple.append(fair)
						classification_stats['_tests'].append( tuple(test_tuple) )

																	#  First-Timestamp    Final-Timestamp    Source-Enactment    Confs...    Ground-Truth-Label    {fair,unfair}
						classification_stats['_test-conf'].append( tuple([time_stamp_buffer[max(0, frame_ctr - (self.rolling_buffer_length - 1))], \
						                                                  time_stamp_buffer[frame_ctr], enactment_input] + \
						                                                 [confidences[x] for x in self.labels('train')] + \
						                                                 [ground_truth_label, fair]) )
																	#  First-Timestamp    Final-Timestamp    Source-Enactment    Probs...    Ground-Truth-Label    {fair,unfair}
						classification_stats['_test-prob'].append( tuple([time_stamp_buffer[max(0, frame_ctr - (self.rolling_buffer_length - 1))], \
						                                                  time_stamp_buffer[frame_ctr], enactment_input] + \
						                                                 [probabilities[x] for x in self.labels('train')] + \
						                                                 [ground_truth_label, fair]) )
																	#  First-Timestamp    Final-Timestamp    Source-Enactment    Smoothed-Probs...    Ground-Truth-Label    {fair,unfair}
						classification_stats['_test-smooth-prob'].append( tuple([time_stamp_buffer[max(0, frame_ctr - (self.rolling_buffer_length - 1))], \
						                                                         time_stamp_buffer[frame_ctr], enactment_input] + \
						                                                        [smoothed_probabilities[x] for x in range(0, n_labels)] + \
						                                                        [ground_truth_label, fair]) )

				#####################################################
				#  Rendering?                                       #
				#####################################################
				if self.render:
																	#  Whatever you are rendering, you will need the reference source frame.
					t1_start = time.process_time()					#  Start timer.
					if prediction is not None:
						annotated_prediction = prediction
					else:
						annotated_prediction = ''
					annotated_source_frame = self.render_annotated_source_frame(enactment_input, frame_path_buffer[frame_ctr],     \
					                                                            time_stamp=time_stamp_buffer[frame_ctr],           \
					                                                            ground_truth_label=ground_truth_buffer[frame_ctr], \
					                                                            prediction=annotated_prediction)
					t1_stop = time.process_time()					#  Stop timer.
					self.timing['render-annotated-source'].append(t1_stop - t1_start)

					if 'rolling-buffer' in self.render_modes:
						t1_start = time.process_time()				#  Stop timer.
						graph = self.render_rolling_buffer_seismograph()
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['render-rolling-buffer'].append(t1_stop - t1_start)
						concat_frame = np.zeros((self.height, self.width * 2, 3), dtype='uint8')
						concat_frame[:, :self.width, :] = annotated_source_frame[:, :, :]
						concat_frame[:, self.width:, :] = graph[:, :, :]
						vid_rolling_buffer.write(concat_frame)

					if 'confidence' in self.render_modes:
						t1_start = time.process_time()				#  Stop timer.
						if metadata is not None:
							label_picks = dict( [(x[0], x[1]['db-index']) for x in metadata.items() if x[0] in self.labels('train')] )
						else:
							label_picks = None
						graph = self.render_confidence_seismograph(confidence_store, label_picks)
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['render-confidence'].append(t1_stop - t1_start)
						concat_frame = np.zeros((self.height, self.width * 2, 3), dtype='uint8')
						concat_frame[:, :self.width, :] = annotated_source_frame[:, :, :]
						concat_frame[:, self.width:, :] = graph[:, :, :]
						vid_confidence.write(concat_frame)

					if 'probabilities' in self.render_modes:
						t1_start = time.process_time()				#  Stop timer.
						if metadata is not None:
							label_picks = dict( [(x[0], x[1]['db-index']) for x in metadata.items() if x[0] in self.labels('train')] )
						else:
							label_picks = None
						graph = self.render_probabilities_seismograph(probability_store, label_picks)
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['render-probabilities'].append(t1_stop - t1_start)
						concat_frame = np.zeros((self.height, self.width * 2, 3), dtype='uint8')
						concat_frame[:, :self.width, :] = annotated_source_frame[:, :, :]
						concat_frame[:, self.width:, :] = graph[:, :, :]
						vid_probabilities.write(concat_frame)

					if 'smooth' in self.render_modes:
						t1_start = time.process_time()				#  Stop timer.
						if metadata is not None:
							label_picks = dict( [(x[0], x[1]['db-index']) for x in metadata.items() if x[0] in self.labels('train')] )
						else:
							label_picks = None
						graph = self.render_smoothed_probabilities_seismograph(smoothed_probability_store, label_picks)
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['render-smoothed-probabilities'].append(t1_stop - t1_start)
						concat_frame = np.zeros((self.height, self.width * 2, 3), dtype='uint8')
						concat_frame[:, :self.width, :] = annotated_source_frame[:, :, :]
						concat_frame[:, self.width:, :] = graph[:, :, :]
						vid_smoothed_probabilities.write(concat_frame)

				if self.verbose:									#  Progress bar.
					if int(round(float(frame_ctr) / float(num - 1) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
						prev_ctr = int(round(float(frame_ctr) / float(num - 1) * float(max_ctr)))
						sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(frame_ctr) / float(num - 1) * 100.0))) + '%]')
						sys.stdout.flush()

			if self.render:											#  Rendering? Close and save the newly created video.
				if 'rolling-buffer' in self.render_modes:
					vid_rolling_buffer.release()
				if 'confidence' in self.render_modes:
					vid_confidence.release()
				if 'probabilities' in self.render_modes:
					vid_probabilities.release()
				if 'smooth' in self.render_modes:
					vid_smoothed_probabilities.release()
			if self.verbose:
				print('')

		t0_stop = time.process_time()
		self.timing['total'] = t0_stop - t0_start
		return classification_stats

	#  THIS is the really real real-time method.
	#  We cannot temporally "skip over" unfair snippets, but if this parameter is left to 'True', then unfair snippets will not
	#  negatively impact classifier accuracy.
	def classify(self, model=None, skip_unfair=True):
		assert isinstance(skip_unfair, bool), 'Argument \'skip_unfair\' passed to TemporalClassifier.classify() must be a Boolean.'

		gpus = tf.config.experimental.list_physical_devices('GPU')	#  List all GPUs on this system.
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)		#  For each GPU, limit memory use.

		n_labels = self.num_labels('train')							#  Store the number of training-set labels.
		flip = np.array([[-1.0,  0.0, 0.0], \
		                 [ 0.0, -1.0, 0.0], \
		                 [ 0.0,  0.0, 1.0]], dtype=np.float32)		#  Build the flip matrix

		classification_stats = self.initialize_stats()				#  Init.

		self.initialize_timers()									#  (Re)set.

		if self.verbose:
			if model is None:
				if self.use_detection_source is None:
					print('>>> Bootup: no object-detection model to load; using GROUND-TRUTH (*_props.txt) for detection')
				else:
					print('>>> Bootup: no object-detection model to load; using DETECTION SOURCE (*_' + self.use_detection_source + '_detections.txt) for detection')
			else:
				print('>>> Bootup: loading object-detection model "' + model + '"')

		if model is not None:										#  Load saved model and build detection function.
			t1_start = time.process_time()							#  Start timer.
			detect_function = tf.saved_model.load(model + '/saved_model')
			label_path = '/'.join(model.split('/')[:-2] + ['annotations', 'label_map.pbtxt'])
			recognizable_objects = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)
			t1_stop = time.process_time()							#  Stop timer.
			self.timing['load-model'] = t1_stop - t1_start

		if self.verbose:
			print('    Done')

		t0_start = time.process_time()								#  Start timer.
		for enactment_input in self.enactment_inputs:				#  Treat each input enactment as a separate slice of time.

			e = Enactment(enactment_input, enactment_file=enactment_input + '.enactment')
			if model is None:										#  Using ground-truth? Load all the detections.
				if self.use_detection_source is not None:			#  We may be reading object detections from file, but was a detections file specified?
					e.load_parsed_objects(enactment_input + '_' + self.use_detection_source + '_detections.txt')
				else:												#  Nothing specified: expect to use GROUND-TRUTH detections.
					e.load_parsed_objects()							#  Default to "*_props.txt".
				recognizable_objects = e.recognizable_objects[:]

			metadata = e.load_metadata()
			min_depth = metadata['depthImageRange']['x']			#  Save the current enactment's minimum and maximum depths.
			max_depth = metadata['depthImageRange']['y']

			K_inv = np.linalg.inv(e.K())							#  Build inverse K-matrix
																	#  (Re)initialize the rolling buffer.
			self.rolling_buffer = [None for i in range(0, self.rolling_buffer_length)]
			self.rolling_buffer_filling = True						#  Buffer has yet to reach capacity
																	#  (Re)initialize the temporal buffer.
			self.temporal_buffer = [None for i in range(0, self.temporal_buffer_length)]
			self.temporal_buffer_filling = True						#  Buffer has yet to reach capacity
																	#  (Re)initialize the ground-truth labels buffer.
			self.buffer_labels = [None for i in range(0, self.rolling_buffer_length)]

			hand_vector_buffer = []									#  Only store the hand poses:
																	#  these would be independently computed in a deployed system (HoloLens).
			ground_truth_buffer = []								#  Can include "*" or "nothing" labels.
			time_stamp_buffer = []									#  Store time stamps for reference.
			frame_path_buffer = []									#  Store file paths (NormalViewCameraFrames) for reference.
			depth_map_path_buffer = []								#  Store file paths (DepthMapCameraFrames) for reference.

			if self.verbose:
				print('>>> Bootup: caching hand subvectors for "' + enactment_input + '"')

			t1_start = time.process_time()							#  Start timer.
			fh = open(enactment_input + '.enactment', 'r')			#  Read in the input-enactment.
			lines = fh.readlines()
			fh.close()
			for line in lines:										#  All we want are: time stamps; file paths; hand poses.
				if line[0] != '#':
					arr = line.strip().split('\t')
					timestamp = float(arr[0])						#  Save the time stamp.
					frame_filename = arr[1]							#  Save the frame file path.
																	#  Construct a depth map file path.
					depth_map_filename = frame_filename.replace('NormalViewCameraFrames', 'DepthMapCameraFrames')
					ground_truth_label = arr[2]						#  Save the true label (these include the nothing-labels.)
					vector = [float(x) for x in arr[3:]]
																	#  But we will not actually use the entire enactment vector here: only the
					hand_vector_buffer.append(vector[:12])			#  hands are assumed to be given to us; object detection must happen in real time.

					if ground_truth_label in self.relabelings:		#  Is this ground-truth label to be relabeled?
						ground_truth_buffer.append( self.relabelings[ground_truth_label] )
					else:
						ground_truth_buffer.append( ground_truth_label )

					time_stamp_buffer.append( timestamp )
					frame_path_buffer.append( frame_filename )
					depth_map_path_buffer.append( depth_map_filename )

			t1_stop = time.process_time()							#  Stop timer.
			self.timing['load-enactment'].append(t1_stop - t1_start)

			if self.verbose:
				print('    Done')

			num = len(time_stamp_buffer)
			prev_ctr = 0
			max_ctr = os.get_terminal_size().columns - 7			#  Leave enough space for the brackets, space, and percentage.
			if self.verbose:
				if skip_unfair:
					print('>>> Classifying from "' + enactment_input + '" in real time (unfair tests will not be skipped but will not be counted.)')
				else:
					print('>>> Classifying from "' + enactment_input + '" in real time.')
				print('    Minimum detection area:       ' + str(self.minimum_bbox_area))
				print('    Minimum detection confidence: ' + str(self.detection_confidence))
				print('    Prediction threshold:         ' + str(self.threshold))

			prediction = None										#  Initially nothing.

			#########################################################
			#  Pseudo-boot-up is over. March through the enactment. #
			#########################################################
			for frame_ctr in range(0, num):							#  March through vector_buffer.
				t1_start = time.process_time()						#  Start timer.
				t2_start = t1_start									#  Duplicate timer.
																	#  Openg frame image.
				frame_img = cv2.imread(frame_path_buffer[frame_ctr], cv2.IMREAD_COLOR)
				frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
				imgt = tf.convert_to_tensor(frame_img)				#  Convert image to tensor.
				input_tensor = imgt[tf.newaxis, ...]
																	#  Open depth map.
				depth_map = cv2.imread(depth_map_path_buffer[frame_ctr], cv2.IMREAD_UNCHANGED)

				t1_stop = time.process_time()						#  Stop timer.
				self.timing['image-open'].append(t1_stop - t1_start)

				if model is None:
					t1_start = time.process_time()					#  Start timer.

					detections = e.frames[ time_stamp_buffer[frame_ctr] ].detections
					num_detections = len(detections)

					t1_stop = time.process_time()					#  Stop timer.
					self.timing['object-detection'].append(t1_stop - t1_start)
				else:
					t1_start = time.process_time()					#  Start timer.
					detections = detect_function(input_tensor)		#  DETECT!
					t1_stop = time.process_time()					#  Stop timer.
					self.timing['object-detection'].append(t1_stop - t1_start)

					num_detections = int(detections.pop('num_detections'))
					detections = {key: val[0, :num_detections].numpy() for key, val in detections.items()}
					detections['num_detections'] = num_detections
					detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

				hands_subvector = hand_vector_buffer[frame_ctr]		#  We already have the hands subvector.

																	#  If hands are not present, then they cannot influence object weights.
				if hands_subvector[0] > 0.0 or hands_subvector[1] > 0.0 or hands_subvector[2] > 0.0:
					lh = np.array(hands_subvector[:3])				#  Left hand influences Gaussian3D weight.
				else:
					lh = None										#  No left-hand influence on Gaussian3D weight.
				if hands_subvector[6] > 0.0 or hands_subvector[7] > 0.0 or hands_subvector[8] > 0.0:
					rh = np.array(hands_subvector[6:9])				#  Right hand influences Gaussian3D weight.
				else:
					rh = None										#  No right-hand influence on Gaussian3D weight.
																	#  Initialize the props subvector to zero.
				props_subvector = [0.0 for i in self.recognizable_objects]
																	#  For every detection...
				if model is None:
					for i in range(0, num_detections):				#  GROUND-TRUTH:
																	#    object_name, bounding_box, confidence
						if detections[i].object_name in recognizable_objects:
							detection_class = detections[i].object_name
							bbox            = detections[i].bounding_box
							detection_score = detections[i].confidence
							bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

							if detection_score >= self.detection_confidence and bbox_area >= self.minimum_bbox_area:
								object_index = self.recognizable_objects.index(detection_class)

								t1_start = time.process_time()		#  Start timer.
								bbox_center = ( int(round(float(bbox[0] + bbox[2]) * 0.5)), \
								                int(round(float(bbox[1] + bbox[3]) * 0.5)) )
																	#  In meters.
								d = min_depth + (float(depth_map[min(bbox_center[1], e.height - 1), min(bbox_center[0], e.width - 1)]) / 255.0) * (max_depth - min_depth)
								centroid = np.dot(K_inv, np.array([bbox_center[0], bbox_center[1], 1.0]))
								centroid *= d						#  Scale by known depth (meters from head).
								centroid_3d = np.dot(flip, centroid)#  Flip point.
								t1_stop = time.process_time()		#  Stop timer.
								self.timing['centroid-computation'].append(t1_stop - t1_start)

								g = self.gaussian.weigh(centroid_3d, lh, rh)
																	#  Each recognizable object's slot receives the maximum signal for that prop.
								props_subvector[object_index] = max(g, props_subvector[object_index])
				else:
					for i in range(0, num_detections):				#  SSD MOBILE-NET:
																	#    detection_classes, detection_multiclass_scores, detection_anchor_indices,
																	#    detection_boxes, raw_detection_boxes,
																	#    detection_scores, raw_detection_scores,
																	#    num_detections
						detection_class = recognizable_objects[ detections['detection_classes'][i] ]['name']
						detection_box   = detections['detection_boxes'][i]
						detection_score = float(detections['detection_scores'][i])

						bbox = ( int(round(detection_box[1] * e.width)), int(round(detection_box[0] * e.height)), \
						         int(round(detection_box[3] * e.width)), int(round(detection_box[2] * e.height)) )
						bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

						if detection_score >= self.detection_confidence and bbox_area >= self.minimum_bbox_area:
							object_index = self.recognizable_objects.index(detection_class)

							t1_start = time.process_time()			#  Start timer.
							bbox_center = ( int(round(float(bbox[0] + bbox[2]) * 0.5)), \
							                int(round(float(bbox[1] + bbox[3]) * 0.5)) )
																	#  In meters.
							d = min_depth + (float(depth_map[min(bbox_center[1], e.height - 1), min(bbox_center[0], e.width - 1)]) / 255.0) * (max_depth - min_depth)
							centroid = np.dot(K_inv, np.array([bbox_center[0], bbox_center[1], 1.0]))
							centroid *= d							#  Scale by known depth (meters from head).
							centroid_3d = np.dot(flip, centroid)	#  Flip point.
							t1_stop = time.process_time()			#  Stop timer.
							self.timing['centroid-computation'].append(t1_stop - t1_start)

							g = self.gaussian.weigh(centroid_3d, lh, rh)
																	#  Each recognizable object's slot receives the maximum signal for that prop.
							props_subvector[object_index] = max(g, props_subvector[object_index])

				del frame_img										#  Free the memory!
				del depth_map

				vector = hands_subvector + props_subvector			#  Assemble the vector.
				if self.hand_schema == 'strong-hand':				#  Apply hand-schema (if applicable.)
					vector = self.strong_hand_encode(vector)

				vec = list(vector[:12])								#  Apply vector drop-outs (where applicable.)
				for i in range(0, len(self.vector_drop_map)):
					if self.vector_drop_map[i] == True:
						vec.append(vector[i + 12])
				vector = tuple(vec)

				vector = self.apply_vector_coefficients(vector)		#  Apply coefficients (if applicable.)

				self.push_rolling( vector )							#  Push vector to rolling buffer.
																	#  Push G.T. label to rolling G.T. buffer.
				self.push_ground_truth_buffer( ground_truth_buffer[frame_ctr] )
																	#  Are the contents of the ground-truth buffer "fair"?
				#fair = self.ground_truth_buffer_full() and self.ground_truth_buffer_uniform() and self.ground_truth_buffer_familiar()
				fair = self.ground_truth_buffer_full() and (self.ground_truth_buffer_newest_frame_label() in self.labels('train') or self.ground_truth_buffer_newest_frame_label() == '*')

				#####################################################
				#  If the rolling buffer is full, then classify.    #
				#  This will always produce a tentative prediction  #
				#  based on lowest matching cost. Actual            #
				#  predictions are subject to threshold and         #
				#  smoothed probabilities.                          #
				#####################################################
				if self.is_rolling_buffer_full():
																	#  Identify ground-truth.
					ground_truth_label = self.ground_truth_buffer_newest_frame_label()

					prediction = None								#  (Re)set.

					t1_start = time.process_time()					#  Start timer.
																	#  Call the parent class's core matching engine.
					nearest_neighbor_label, matching_costs, confidences, probabilities, metadata = super(TemporalClassifier, self).classify(self.rolling_buffer)
					t1_stop = time.process_time()					#  Stop timer.
					self.timing['dtw-classification'].append(t1_stop - t1_start)

					#################################################
					#  nearest_neighbor_label:        label or None #
					#  matching_costs: key: label ==> val: cost     #
					#                  in R^N                       #
					#  confidences:    key: label ==> val: score    #
					#                  in R^N                       #
					#  probabilities:  key: label ==> val: prob.    #
					#                  in R^N or R^{N+1}            #
					#  metadata:       key: label ==>               #
					#                    val: {query-indices,       #
					#                          template-indices,    #
					#                          db-index}            #
					#################################################

					#################################################
					#  When probabilities is in R^N the nothing-    #
					#  label is predicted only if all other scores  #
					#  are below a threshold.                       #
					#################################################
					if len(probabilities) == n_labels:				#  Probabilities is in R^N.
																	#  Among other tasks, this method pushes to the temporal buffer.
						sorted_confidences, sorted_probabilities = self.process_dtw_results(matching_costs, confidences, probabilities)
																	#  Save all costs for all labels.
						classification_stats['_costs'].append( tuple([time_stamp_buffer[ frame_ctr ]] + \
						                                             [time_stamp_buffer[ max(0, frame_ctr - (self.rolling_buffer_length - 1)) ]] + \
						                                             [enactment_input] + \
						                                             [matching_costs[label] for label in self.labels('train')] + \
						                                             [ground_truth_label]) )
																	#  Save confidence scores for all labels, regardless of what the system picks.
						for label in self.labels('train'):			#  We use these values downstream in the pipeline for isotonic regression.
							classification_stats['_conf'].append( (confidences[label], label, ground_truth_label, \
							                                       enactment_input, \
							                                       time_stamp_buffer[ max(0, frame_ctr - (self.rolling_buffer_length - 1)) ], \
							                                       time_stamp_buffer[ frame_ctr ]) )

						for label in self.labels('train'):			#  Save probabilities for all labels, regardless of what the system picks.
							classification_stats['_prob'].append( (probabilities[label], label, ground_truth_label, \
							                                       enactment_input, \
							                                       time_stamp_buffer[ max(0, frame_ctr - (self.rolling_buffer_length - 1)) ], \
							                                       time_stamp_buffer[ frame_ctr ]) )

						#############################################
						#  Smooth the contents of the temporal      #
						#  buffer and make a prediction (or abstain #
						#  from predicting).                        #
						#############################################

						t1_start = time.process_time()				#  Start timer.
						smoothed_probabilities = np.mean(np.array([x for x in self.temporal_buffer if x is not None]), axis=0)
						smoothed_norm = sum(smoothed_probabilities)
						if smoothed_norm > 0.0:
							smoothed_probabilities /= smoothed_norm
						smoothed_probabilities = list(smoothed_probabilities)
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['temporal-smoothing'].append(t1_stop - t1_start)

						#############################################
						#  Predict... possibly an IMPLICIT nothing. #
						#############################################
						t1_start = time.process_time()				#  Start timer.
						prediction = None
						if nearest_neighbor_label is not None:		#  (It was not first impossible to select any nearest neighbor at all.)
							tentative_probability = smoothed_probabilities[ self.labels('train').index(nearest_neighbor_label) ]
							if tentative_probability >= self.threshold:
								prediction = nearest_neighbor_label
								if prediction in self.hidden_labels:#  Is this a hidden label? Then dummy up.
									prediction = None
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['make-temporally-smooth-decision'].append(t1_stop - t1_start)

						classification_stats = self.update_stats(prediction, ground_truth_label, (fair or not skip_unfair), classification_stats)
																	#  Whether or not it's skipped, put it on record.
						if nearest_neighbor_label is not None:
							classification_stats['_tests'].append( (prediction, ground_truth_label, \
							                                        confidences[nearest_neighbor_label], tentative_probability, \
							                                        enactment_input, time_stamp_buffer[frame_ctr], \
							                                        metadata[nearest_neighbor_label]['db-index'], \
							                                        fair) )
						else:										#  The tentative prediction is None if applied conditions make ALL possibilities impossible.
							classification_stats['_tests'].append( (prediction, ground_truth_label, \
							                                        0.0, 0.0, \
							                                        enactment_input, time_stamp_buffer[frame_ctr], \
							                                        -1, \
							                                        fair) )

																	#  First-Timestamp    Final-Timestamp    Source-Enactment    Confs...    Ground-Truth-Label    {fair,unfair}
						classification_stats['_test-conf'].append( tuple([time_stamp_buffer[max(0, frame_ctr - (self.rolling_buffer_length - 1))], \
						                                                  time_stamp_buffer[frame_ctr], enactment_input] + \
						                                                 [confidences[x] for x in self.labels('train')] + \
						                                                 [ground_truth_label, fair]) )
																	#  First-Timestamp    Final-Timestamp    Source-Enactment    Probs...    Ground-Truth-Label    {fair,unfair}
						classification_stats['_test-prob'].append( tuple([time_stamp_buffer[max(0, frame_ctr - (self.rolling_buffer_length - 1))], \
						                                                  time_stamp_buffer[frame_ctr], enactment_input] + \
						                                                 [probabilities[x] for x in self.labels('train')] + \
						                                                 [ground_truth_label, fair]) )
																	#  First-Timestamp    Final-Timestamp    Source-Enactment    Smoothed-Probs...    Ground-Truth-Label    {fair,unfair}
						classification_stats['_test-smooth-prob'].append( tuple([time_stamp_buffer[max(0, frame_ctr - (self.rolling_buffer_length - 1))], \
						                                                         time_stamp_buffer[frame_ctr], enactment_input] + \
						                                                        [smoothed_probabilities[x] for x in range(0, n_labels)] + \
						                                                        [ground_truth_label, fair]) )
					#################################################
					#  When probabilities is in R^{N+1} the nothing-#
					#  label has an explicit prediction.            #
					#  This becomes the system's prediction if this #
					#  probability is the highest and high enough,  #
					#  OR if no other label is probable enough.     #
					#################################################
					else:											#  Probabilities is in R^{N+1}. Meaning the nothing label has been explicitly predicted.
						labels_and_nothing = self.labels('train') + ['*']

						sorted_probabilities = [probabilities[label] for label in labels_and_nothing]
						t1_start = time.process_time()				#  Start timer.
																	#  Add this probability distribution to the temporal buffer.
						self.push_temporal( sorted_probabilities )
																	#  The temporal buffer always receives distributions sorted
																	#  according to the labels('train') method.
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['push-temporal-buffer'].append(t1_stop - t1_start)

																	#  Save all costs for all labels.
																	#  (There is no matching cost for the explicitly predicted nothing-label.)
						classification_stats['_costs'].append( tuple([time_stamp_buffer[ frame_ctr ]] + \
						                                             [time_stamp_buffer[ max(0, frame_ctr - (self.rolling_buffer_length - 1)) ]] + \
						                                             [enactment_input] + \
						                                             [matching_costs[label] for label in labels_and_nothing[:-1]] + \
						                                             [ground_truth_label]) )
																	#  Save confidence scores for all labels, regardless of what the system picks.
						for label in self.labels('train'):			#  We use these values downstream in the pipeline for isotonic regression.
							classification_stats['_conf'].append( (confidences[label], label, ground_truth_label, \
							                                       enactment_input, \
							                                       time_stamp_buffer[ max(0, frame_ctr - (self.rolling_buffer_length - 1)) ], \
							                                       time_stamp_buffer[ frame_ctr ]) )

						for label in labels_and_nothing:			#  Save probabilities for all labels, regardless of what the system picks.
							classification_stats['_prob'].append( (probabilities[label], label, ground_truth_label, \
							                                       enactment_input, \
							                                       time_stamp_buffer[ max(0, frame_ctr - (self.rolling_buffer_length - 1)) ], \
							                                       time_stamp_buffer[ frame_ctr ]) )

						#############################################
						#  Smooth the contents of the temporal      #
						#  buffer and make a prediction (or abstain #
						#  from predicting).                        #
						#############################################

						t1_start = time.process_time()				#  Start timer.
						smoothed_probabilities = np.mean(np.array([x for x in self.temporal_buffer if x is not None]), axis=0)
						smoothed_norm = sum(smoothed_probabilities)
						if smoothed_norm > 0.0:
							smoothed_probabilities /= smoothed_norm
						smoothed_probabilities = list(smoothed_probabilities)
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['temporal-smoothing'].append(t1_stop - t1_start)

						#############################################
						#  Predict... possibly an EXPLICIT nothing. #
						#############################################
						t1_start = time.process_time()				#  Start timer.
						prediction = None
						if nearest_neighbor_label is not None:		#  (It was not first impossible to select any nearest neighbor at all.)
							tentative_probability = smoothed_probabilities[ labels_and_nothing.index(nearest_neighbor_label) ]
							nothing_probability = smoothed_probabilities[ -1 ]

							if tentative_probability > nothing_probability and tentative_probability >= self.threshold:
								prediction = nearest_neighbor_label
								if prediction in self.hidden_labels:#  Is this a hidden label? Then dummy up.
									prediction = None
						t1_stop = time.process_time()				#  Stop timer.
						self.timing['make-temporally-smooth-decision'].append(t1_stop - t1_start)

						classification_stats = self.update_stats(prediction, ground_truth_label, (fair or not skip_unfair), classification_stats)

																	#  Whether or not it's skipped, put it on record.
						test_tuple = [prediction, ground_truth_label]
						if nearest_neighbor_label in confidences:
							test_tuple.append(confidences[nearest_neighbor_label])
						else:
							test_tuple.append(None)
						test_tuple.append(tentative_probability)
						test_tuple.append(enactment_input)
						test_tuple.append(time_stamp_buffer[frame_ctr])
						if nearest_neighbor_label in metadata:
							test_tuple.append(metadata[nearest_neighbor_label]['db-index'])
						else:
							test_tuple.append(None)
						test_tuple.append(fair)
						classification_stats['_tests'].append( tuple(test_tuple) )
																	#  First-Timestamp    Final-Timestamp    Source-Enactment    Confs...    Ground-Truth-Label    {fair,unfair}
						classification_stats['_test-conf'].append( tuple([time_stamp_buffer[max(0, frame_ctr - (self.rolling_buffer_length - 1))], \
						                                                  time_stamp_buffer[frame_ctr], enactment_input] + \
						                                                 [confidences[x] for x in self.labels('train')] + \
						                                                 [ground_truth_label, fair]) )
																	#  First-Timestamp    Final-Timestamp    Source-Enactment    Probs...    Ground-Truth-Label    {fair,unfair}
						classification_stats['_test-prob'].append( tuple([time_stamp_buffer[max(0, frame_ctr - (self.rolling_buffer_length - 1))], \
						                                                  time_stamp_buffer[frame_ctr], enactment_input] + \
						                                                 [probabilities[x] for x in labels_and_nothing] + \
						                                                 [ground_truth_label, fair]) )
																	#  First-Timestamp    Final-Timestamp    Source-Enactment    Smoothed-Probs...    Ground-Truth-Label    {fair,unfair}
						classification_stats['_test-smooth-prob'].append( tuple([time_stamp_buffer[max(0, frame_ctr - (self.rolling_buffer_length - 1))], \
						                                                         time_stamp_buffer[frame_ctr], enactment_input] + \
						                                                        [smoothed_probabilities[x] for x in range(0, n_labels)] + \
						                                                        [ground_truth_label, fair]) )

				if self.verbose:									#  Progress bar.
					if int(round(float(frame_ctr) / float(num - 1) * float(max_ctr))) > prev_ctr or prev_ctr == 0:
						prev_ctr = int(round(float(frame_ctr) / float(num - 1) * float(max_ctr)))
						sys.stdout.write('\r[' + '='*prev_ctr + ' ' + str(int(round(float(frame_ctr) / float(num - 1) * 100.0))) + '%]')
						sys.stdout.flush()

				t2_stop = time.process_time()						#  Stop per-frame timer.
				self.timing['per-frame'].append(t2_stop - t2_start)	#  Save per-frame time.

		t0_stop = time.process_time()
		self.timing['total'] = t0_stop - t0_start

		return classification_stats

	#################################################################
	#  Edit the frame-vectors.                                      #
	#################################################################

	def drop_vector_element(self, index):
		super(TemporalClassifier, self).drop_vector_element(index)	#  The parent class's method treats the training set.
																	#  This child class treats the test set itself.
		assert isinstance(index, int) or isinstance(index, list), \
		  'Argument \'index\' passed to TemporalClassifier.drop_vector_element() must be either a single integer or a list of integers.'

		if isinstance(index, int):									#  Cut a single index from everything in self.X_train.
			assert index < self.vector_length, \
			  'Argument \'index\' passed to TemporalClassifier.drop_vector_element() must be an integer less than the number of recognizable objects.'
			self.vector_length -= 1									#  Decrement the vector length.
			self.vector_drop_map[index] = False						#  Mark the index-th element for omission in the test set, too!

		elif isinstance(index, list):								#  Cut all given indices from everything in self.X_train.
																	#  Accept all or nothing.
			assert len([x for x in index if x < self.vector_length]) == len(index), \
			  'Argument \'index\' passed to TemporalClassifier.drop_vector_element() must be a list of integers, all less than the number of recognizable objects.'
			self.vector_length -= len(index)						#  Shorten the vector length.
			for i in index:
				self.vector_drop_map[i] = False						#  Mark all indices for omission in the test set, too!

		return

	#################################################################
	#  Buffer update.                                               #
	#################################################################

	#  Used in both the simulated and in the real-time case, it made sense to have a single method they can share.
	#  This method receives the matching_costs, confidences, and probabilities returned by DTW.
	#  The received 'matching_costs' is a dictionary, key: label ==> val: cost.
	#  The received 'confidences'    is a dictionary, key: label ==> val: score.
	#  The received 'probabilities'  is a dictionary, key: label ==> val: prob.
	#
	#  This method pushes probabilities sorted according to the labels('train') method to the temporal buffer.
	#
	#  This method returns:
	#    'sorted_confidences',   which is all labels' confidence scores, sorted according to the labels('train') method;
	#    'sorted_probabilities', which is all labels' probabilities, sorted according to the labels('train') method.
	def process_dtw_results(self, matching_costs, confidences, probabilities):
		t1_start = time.process_time()								#  Start timer.
		sorted_confidences = []
		for label in self.labels('train'):							#  Maintain the order of label scores.
			sorted_confidences.append( confidences[label] )
		t1_stop = time.process_time()								#  Stop timer.
		self.timing['sort-confidences'].append(t1_stop - t1_start)

		t1_start = time.process_time()								#  Start timer.
		sorted_probabilities = []
		for label in self.labels('train'):							#  Maintain the order of label scores.
			sorted_probabilities.append( probabilities[label] )
		t1_stop = time.process_time()								#  Stop timer.
		self.timing['sort-probabilities'].append(t1_stop - t1_start)

		t1_start = time.process_time()								#  Start timer.
		self.push_temporal( sorted_probabilities )					#  Add this probability distribution to the temporal buffer.
																	#  The temporal buffer always receives distributions sorted
																	#  according to the labels('train') method.
		t1_stop = time.process_time()								#  Stop timer.
		self.timing['push-temporal-buffer'].append(t1_stop - t1_start)

		return sorted_confidences, sorted_probabilities

	#  Add the given vector to the rolling buffer, kicking out old vectors if necessary.
	def push_rolling(self, vector):
		self.rolling_buffer_filling = not self.is_rolling_buffer_full()

		if self.rolling_buffer_filling:
			i = 0
			while i < self.rolling_buffer_length and self.rolling_buffer[i] is not None:
				i += 1
			self.rolling_buffer[i] = vector[:]
		else:
			if None not in self.rolling_buffer:
				self.rolling_buffer = self.rolling_buffer[self.rolling_buffer_stride:] + [None for i in range(0, self.rolling_buffer_stride)]
			self.rolling_buffer[ self.rolling_buffer.index(None) ] = vector[:]
		return

	#  Add the given ground-truth label to the ground-truth buffer, kicking out old labels if necessary.
	def push_ground_truth_buffer(self, label):
		if self.rolling_buffer_filling:
			i = 0
			while i < self.rolling_buffer_length and self.buffer_labels[i] is not None:
				i += 1
			self.buffer_labels[i] = label[:]
		else:
			if None not in self.buffer_labels:
				self.buffer_labels = self.buffer_labels[self.rolling_buffer_stride:] + [None for i in range(0, self.rolling_buffer_stride)]
			self.buffer_labels[ self.buffer_labels.index(None) ] = label[:]
		return

	#  Add the given probability distribution to the temporal buffer, kicking out old distributions if necessary.
	def push_temporal(self, distribution):
		self.temporal_buffer_filling = not self.is_temporal_buffer_full()

		if self.temporal_buffer_filling:
			i = 0
			while i < self.temporal_buffer_length and self.temporal_buffer[i] is not None:
				i += 1
			self.temporal_buffer[i] = distribution[:]
		else:
			if None not in self.temporal_buffer:
				self.temporal_buffer = self.temporal_buffer[self.temporal_buffer_stride:] + [None for i in range(0, self.temporal_buffer_stride)]
			self.temporal_buffer[ self.temporal_buffer.index(None) ] = distribution[:]
		return

	def is_rolling_buffer_full(self):
		return None not in self.rolling_buffer

	def is_temporal_buffer_full(self):
		return None not in self.temporal_buffer

	def is_temporal_buffer_not_empty(self):
		return len([x for x in self.temporal_buffer if x is not None]) > 0

	#  A more generic function used by the caches for rendering.
	#  (This assumes the rendering caches use a stride of 1.)
	#  (Also, unlike the above push operations, this one has to return an updated buffer.)
	def push_buffer(self, vector, buffer):
		if len(buffer) == self.seismograph_length:					#  If the buffer is at its maximum length, then shift 'stride' elements out.
			buffer = buffer[1:]
		buffer.append( vector )										#  Append the latest.
		return buffer

	#################################################################
	#  Ground-Truth buffer testing.                                 #
	#################################################################

	#  Is self.buffer_labels currently full?
	def ground_truth_buffer_full(self):
		return None not in self.buffer_labels

	#  Does self.buffer_labels contain the same label?
	def ground_truth_buffer_uniform(self):
		return self.buffer_labels.count(self.buffer_labels[0]) == self.rolling_buffer_length

	#  Is self.buffer_labels free of any labels that are not in the database?
	def ground_truth_buffer_familiar(self):
		valid_labels = self.labels('train')
		return all([x in valid_labels for x in self.buffer_labels])

	#  Get the most recently added label (this is capable of returning None).
	def ground_truth_buffer_newest_frame_label(self):
		i = 0
		label = self.buffer_labels[i]

		while i < len(self.buffer_labels) and self.buffer_labels[i] is not None:
			label = self.buffer_labels[i]
			i += 1

		return label

	#################################################################
	#  Profiling.                                                   #
	#################################################################

	#  How many snippets will we encounter as we march through time with the current rolling buffer size and stride?
	def itemize(self, skip_unfair=True):
		maxindexlen = 0
		maxlabellen = 0
		total_snippets = 0											#  Across all enactments.
		labels = self.labels('train')

		for enactment_input in self.enactment_inputs:
			pe = ProcessedEnactment(enactment_input, verbose=False)
			snippets = pe.snippets_from_frames(self.rolling_buffer_length, self.rolling_buffer_stride)
			for snippet in snippets:
				label = snippet[0]
				if label in self.relabelings:						#  Look up.
					label = self.relabelings[label]
																	#  Just because something has been relabeled does not mean that the DB knows it.
				if (skip_unfair and label in labels) or not skip_unfair:
					total_snippets += 1
					if len(label) > maxlabellen:
						maxlabellen = len(label)
		maxindexlen = len(str(total_snippets))

		for enactment_input in self.enactment_inputs:
			pe = ProcessedEnactment(enactment_input, verbose=False)
			print('>>> "' + enactment_input + '" in simulated real-time:')
			ctr = 0
			for snippet in pe.snippets_from_frames(self.rolling_buffer_length, self.rolling_buffer_stride):
				label = snippet[0]
				if label in self.relabelings:
					label = self.relabelings[label]

				if (skip_unfair and label in labels) or not skip_unfair:
					print_str = '    [' + str(ctr) + ']:' + ' '*(maxindexlen - len(str(ctr))) + \
					                 label + ' '*(maxlabellen - len(label)) + '\t' + \
					                 str(snippet[1]) + '\t-->\t' + str(snippet[3])
					print(print_str)
					ctr += 1;
			print('')

		if skip_unfair:
			print('>>> Total (fair) snippets in test set: ' + str(total_snippets))
		else:
			print('>>> Total snippets in test set: ' + str(total_snippets))
		return

	def itemize_snippets(self):
		maxlabellen = 0
		maxcountlen = 0
		training_counts = {}
		testing_counts = {}

		X_test = []													#  These are perishable, local variables.
		y_test = []													#  We hold snippets and labels here *as if* they did not exist in time.

		for label in list(np.unique(self.y_train)):
			if len(label) > maxlabellen:
				maxlabellen = len(label)
			training_counts[label] = len([x for x in self.y_train if x == label])
			if len(str(training_counts[label])) > maxcountlen:
				maxcountlen = len(str(training_counts[label]))

		for enactment in self.enactment_inputs:
			pe = ProcessedEnactment(enactment, verbose=False)
			snippets = pe.snippets_from_frames(self.rolling_buffer_length, self.rolling_buffer_stride)
			for snippet in snippets:
				label = snippet[0]
				if label in self.relabelings:
					label = self.relabelings[label]
				X_test.append( (snippet[1], snippet[2], snippet[3], snippet[4]) )
				y_test.append( label )

				if label not in testing_counts:
					testing_counts[label] = 0
				testing_counts[label] += 1
				if len(label) > maxlabellen:
					maxlabellen = len(label)
				if len(str(testing_counts[label])) > maxcountlen:
					maxcountlen = len(str(testing_counts[label]))

		num_fair_labels = len([x for x in testing_counts.keys() if x in training_counts])
		num_fair_snippets = len([x for x in y_test if x in training_counts])

		if len(self.X_train) > 0:
			print('    Training set: ' + str(len(self.labels('train'))) + ' unique labels, ' + str(len(self.X_train)) + ' total ' + str(self.rolling_buffer_length) + '-frame snippets: ' + \
			           "{:.2f}".format(float(len(self.X_train)) / float(len(self.X_train) + len(X_test)) * 100.0) + ' %.')

			for label in self.labels('train'):
				print('        ' + label + ': ' + ' '*(maxlabellen - len(label)) + \
				                   str(training_counts[label]) + ' '*(maxcountlen - len(str(training_counts[label]))) + \
				   ' snippets, ' + "{:.3f}".format(float(training_counts[label]) / float(len(self.y_train)) * 100.0) + ' % of train.')

		if len(X_test) > 0:
			print('    Test set:     ' + str(len(self.labels('test'))) + ' unique labels, ' + str(len(X_test))  + ' total ' + str(self.rolling_buffer_length) + '-frame snippets: ' + \
			           "{:.2f}".format(float(len(X_test)) / float(len(self.X_train) + len(X_test)) * 100.0) + ' %.')
			print('                  ' + str(num_fair_labels) + ' unique fair labels, ' + str(num_fair_snippets) + ' total fair ' + str(self.rolling_buffer_length) + '-frame snippets: ' + \
			           "{:.2f}".format(float(num_fair_snippets) / float(len(self.X_train) + len(X_test)) * 100.0) + ' %.')

			for label in sorted(list(np.unique(y_test))):
				if label not in list(np.unique(self.y_train)):			#  Let users know that a label in the test set is not in the training set.
					print('     !! ' + label + ': ' + ' '*(maxlabellen - len(label)) + \
					                   str(testing_counts[label]) + ' '*(maxcountlen - len(str(testing_counts[label]))) + \
					   ' snippets, ' + "{:.3f}".format(float(testing_counts[label]) / float(len(y_test)) * 100.0) + ' % of test.')
				else:
					print('        ' + label + ': ' + ' '*(maxlabellen - len(label)) + \
					                   str(testing_counts[label]) + ' '*(maxcountlen - len(str(testing_counts[label]))) + \
					   ' snippets, ' + "{:.3f}".format(float(testing_counts[label]) / float(len(y_test)) * 100.0) + ' % of test.')
		return

	#  Print out every input-enactment's timestamps and labels, just to check what is currently on hand.
	def preview(self):
		for enactment_input in self.enactment_inputs:
			fh = open(enactment_input + '.enactment', 'r')			#  Read in the input-enactment.
			lines = fh.readlines()
			fh.close()

			for line in lines:
				if line[0] != '#':
					arr = line.strip().split('\t')
					timestamp = float(arr[0])						#  Save the time stamp.
					ground_truth_label = arr[2]						#  Save the true label (these include the nothing-labels.)
					if ground_truth_label in self.relabelings:
						ground_truth_label = self.relabelings[ground_truth_label]

					print(enactment_input + ':\t' + str(timestamp) + '\t' + ground_truth_label)
		return

	#################################################################
	#  Timing.                                                      #
	#################################################################

	#  Set up 'self.timing' to start collecting measurements.
	def initialize_timers(self):
		self.timing['total'] = 0									#  Prepare to capture time taken by the entire classification run.
		self.timing['load-enactment'] = []							#  Prepare to capture enactment loading times.
		self.timing['image-open'] = []								#  Prepare to capture frame opening times.
		self.timing['object-detection'] = []						#  Prepare to capture object detection times.
		self.timing['centroid-computation'] = []					#  Prepare to capture centroid computation times.
		self.timing['dtw-classification'] = []						#  This is a coarser grain: time each classification process
																	#  (directly affected by the size of the database).
		self.timing['sort-confidences'] = []						#  Prepare to collect confidence-score sorting times.
		self.timing['sort-probabilities'] = []						#  Prepare to collect probability sorting times.
		self.timing['push-temporal-buffer'] = []					#  Prepare to collect temporal-buffer update times.
		self.timing['temporal-smoothing'] = []						#  Prepare to collect temporal-smoothing runtimes.
		self.timing['make-temporally-smooth-decision'] = []			#  Prepare to collect final decision-making runtimes.
		self.timing['per-frame'] = []								#  Prepare to collect per-frame times.
		if self.render:
			self.timing['render-annotated-source'] = []				#  Prepare to collect rendering times.
			self.timing['render-rolling-buffer'] = []
			self.timing['render-confidence'] = []
			self.timing['render-probabilities'] = []
			self.timing['render-smoothed-probabilities'] = []
		return

	#################################################################
	#  Rendering.                                                   #
	#################################################################

	#  Take the given frame name, superimpose object masks, centroids, other data.
	def render_annotated_source_frame(self, enactment, file_name, **kwargs):
		vid_frame = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
		masks = self.get_masks_for_frame(enactment, file_name)
		maskcanvas = np.zeros((self.height, self.width, 3), dtype='uint8')
		for mask in masks:											#  Each is (mask file name, recog-object, bounding-box).
			if mask[1] not in ['LeftHand', 'RightHand']:
				mask_img = cv2.imread(mask[0], cv2.IMREAD_UNCHANGED)
				mask_img[mask_img > 1] = 1							#  All values greater than 1 become 1.
																	#  Extrude to three channels.
				mask_img = mask_img[:, :, None] * np.ones(3, dtype='uint8')[None, None, :]
																	#  Convert this to a graphical overlay:
				mask_img[:, :, 0] *= self.robject_colors[ mask[1] ][2]
				mask_img[:, :, 1] *= self.robject_colors[ mask[1] ][1]
				mask_img[:, :, 2] *= self.robject_colors[ mask[1] ][0]

				maskcanvas += mask_img								#  Add mask to mask accumulator.
				maskcanvas[maskcanvas > 255] = 255					#  Clip accumulator to 255.
																	#  Add mask accumulator to source frame.
		vid_frame = cv2.addWeighted(vid_frame, 1.0, maskcanvas, 0.7, 0)
																	#  Flatten alpha.
		vid_frame = cv2.cvtColor(vid_frame, cv2.COLOR_RGBA2RGB)

		for mask in masks:											#  Second pass: draw bounding boxes and 2D centroids.
			if mask[1] not in ['LeftHand', 'RightHand']:
				bbox = mask[2]
				cv2.rectangle(vid_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (self.robject_colors[ mask[1] ][2], \
				                                                                  self.robject_colors[ mask[1] ][1], \
				                                                                  self.robject_colors[ mask[1] ][0]), 1)
				cv2.circle(vid_frame, (int(round(np.mean([bbox[0], bbox[2]]))), int(round(np.mean([bbox[1], bbox[3]])))), \
				           5, (self.robject_colors[ mask[1] ][2], \
				               self.robject_colors[ mask[1] ][1], \
				               self.robject_colors[ mask[1] ][0]), 3)

		if 'time_stamp' in kwargs:
			assert isinstance(kwargs['time_stamp'], float), \
			       'Argument \'time_stamp\' passed to TemporalClassifier.render_annotated_source_frame() must be a float.'
			time_stamp_str = str(kwargs['time_stamp'])
		else:
			time_stamp_str = ''

		cv2.putText(vid_frame, enactment + ', ' + time_stamp_str, (self.seismograph_enactment_name_super['x'], self.seismograph_enactment_name_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.seismograph_enactment_name_super['fontsize'], (255, 255, 255, 255), 3)

		if 'ground_truth_label' in kwargs:
			assert isinstance(kwargs['ground_truth_label'], str), \
			       'Argument \'ground_truth_label\' passed to TemporalClassifier.render_annotated_source_frame() must be a string.'
																	#  Bright green for truth!
			cv2.putText(vid_frame, kwargs['ground_truth_label'], (self.seismograph_ground_truth_label_super['x'], self.seismograph_ground_truth_label_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.seismograph_ground_truth_label_super['fontsize'], (0, 255, 0, 255), 3)

		if 'prediction' in kwargs:
			assert isinstance(kwargs['prediction'], str), \
			       'Argument \'prediction\' passed to TemporalClassifier.render_annotated_source_frame() must be a string.'
																	#  Use bright green for the prediction, too, when it matches ground-truth.
			if 'ground_truth_label' in kwargs and isinstance(kwargs['ground_truth_label'], str) and kwargs['prediction'] == kwargs['ground_truth_label']:
				cv2.putText(vid_frame, kwargs['prediction'], (self.seismograph_prediction_super['x'], self.seismograph_prediction_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.seismograph_prediction_super['fontsize'], (0, 255, 0, 255), 3)
			else:
				cv2.putText(vid_frame, kwargs['prediction'], (self.seismograph_prediction_super['x'], self.seismograph_prediction_super['y']), cv2.FONT_HERSHEY_SIMPLEX, self.seismograph_prediction_super['fontsize'], (255, 255, 255, 255), 3)

		return vid_frame

	#  Create a seismograph-like plot of the rolling buffer's components.
	def render_rolling_buffer_seismograph(self):
																	#  All-white RGB
		graph = np.ones((self.height, self.width, 3), dtype='uint8') * 255

		buffer_len = self.rolling_buffer_length
		vector_len = self.vector_length
		img_center = (int(round(float(self.width) * 0.5)), int(round(float(self.height) * 0.5)))

																	#  Build a (buffer_len) X (vector_len) matrix:
																	#  [LHx, LHy, LHz, LH0, LH1, LH2, RHx, ..., num_objects] in frame 0
																	#  [LHx, LHy, LHz, LH0, LH1, LH2, RHx, ..., num_objects] in frame 1
		B = []														#  [LHx, LHy, LHz, LH0, LH1, LH2, RHx, ..., num_objects] in frame 2
		for i in range(0, self.rolling_buffer_length):
			if self.rolling_buffer[i] is not None:
				B.append( [x for x in self.undo_vector_coefficients(self.rolling_buffer[i])] )
			else:
				B.append( [0.0 for x in range(0, vector_len)] )
		B = np.array(B)

		x_intervals = int(np.floor(float(self.width) / float(vector_len)))
		y_intervals = int(np.floor(float(self.height) / float(buffer_len)))
		nudge_right = 10

		for y in range(0, buffer_len):
			color_ctr = 2

			for x in range(0, vector_len):
				point_a = (x * x_intervals + nudge_right,  y      * y_intervals)
				point_b = (x * x_intervals + nudge_right, (y + 1) * y_intervals)

				if x >= 0 and x < 6:								#  Always green for left hand/strong hand
					color = (0, 255, 0)
				elif x >= 6 and x < 12:								#  Always red for right hand/weak hand
					color = (0, 0, 255)
				else:
					if color_ctr == 0:
						color = (0, 0, 255)
					elif color_ctr == 1:
						color = (0, 255, 0)
					else:
						color = (255, 0, 0)
				dark_color = (color[0] // 4, color[1] // 4, color[2] // 4)

				graph = cv2.line(graph, point_a, point_b, color, max(1, int(round(float(self.max_seismograph_linewidth) * B[y, x]))) )

				if x >= 12:
					tmp = np.zeros((self.height, self.width, 3), dtype='uint8')
					pt = (x * x_intervals, self.height - 10)
					cv2.putText(tmp, self.recognizable_objects[x - 12], img_center, cv2.FONT_HERSHEY_SIMPLEX, 0.4, dark_color, 1)
					tmp = self.rotate_text(tmp, 90.0, (x * x_intervals, self.height - 10))
					indices = np.where(tmp > 0)
					for i in range(0, len(indices[0])):
						graph[indices[0][i], indices[1][i]] = tmp[indices[0][i], indices[1][i]]

					color_ctr += 1
					if color_ctr == 3:
						color_ctr = 0
		return graph

	#  Create a seismograph-like plot of confidence scores.
	#  Receives 'confidence_accumulator', a buffer of 'self.seismograph_length' confidence vectors for all recognizable actions.
	def render_confidence_seismograph(self, confidence_accumulator, label_picks=None):
																	#  All-white RGB
		graph = np.ones((self.height, self.width, 3), dtype='uint8') * 255

		labels = self.labels('train')
		buffer_len = self.seismograph_length
		vector_len = len(labels)
		img_center = (int(round(float(self.width) * 0.5)), int(round(float(self.height) * 0.5)))
																	#  Build a (buffer_len) X (length of labels) matrix:
																	#  [class0, class1, class2, ..., classn] in frame 0
																	#  [class0, class1, class2, ..., classn] in frame 1
		B = []														#  [class0, class1, class2, ..., classn] in frame 2
		if len(confidence_accumulator) < buffer_len:
			for i in range(0, buffer_len - len(confidence_accumulator)):
				B.append( [0.0 for x in range(0, vector_len)] )
		for i in range(0, len(confidence_accumulator)):
			B.append( [x for x in confidence_accumulator[i]] )
		B = np.array(B)

		x_intervals = int(np.floor(float(self.width) / float(vector_len)))
		y_intervals = int(np.floor(float(self.height) / float(buffer_len)))
		nudge_right = 30
		label_margin = 5

		for y in range(0, buffer_len):
			color_ctr = 0

			for x in range(0, vector_len):
				point_a = (x * x_intervals + nudge_right,  y      * y_intervals)
				point_b = (x * x_intervals + nudge_right, (y + 1) * y_intervals)

				if color_ctr == 0:
					color = (0, 0, 255)
				elif color_ctr == 1:
					color = (0, 255, 0)
				else:
					color = (255, 0, 0)
				dark_color = (color[0] // 4, color[1] // 4, color[2] // 4)

				graph = cv2.line(graph, point_a, point_b, color, max(1, int(round(float(self.max_seismograph_linewidth) * B[y, x]))) )

				tmp = np.zeros((self.height, self.width, 3), dtype='uint8')
				pt = (x * x_intervals, self.height - 10)
				if label_picks is not None:
					cv2.putText(tmp, labels[x] + ' [' + str(label_picks[labels[x]]) + ']', img_center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, dark_color, 1)
				else:
					cv2.putText(tmp, labels[x], img_center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, dark_color, 1)
				tmp = self.rotate_text(tmp, 90.0, (x * x_intervals + nudge_right - label_margin, self.height - 10))
				indices = np.where(tmp > 0)
				for i in range(0, len(indices[0])):
					graph[indices[0][i], indices[1][i]] = tmp[indices[0][i], indices[1][i]]

				color_ctr += 1
				if color_ctr == 3:
					color_ctr = 0

		return graph

	#  Create a seismograph-like plot of probabilities.
	#  Receives 'probability_accumulator', a buffer of 'self.seismograph_length' probability distributions for all recognizable actions.
	def render_probabilities_seismograph(self, probability_accumulator, label_picks=None):
																	#  All-white RGB
		graph = np.ones((self.height, self.width, 3), dtype='uint8') * 255

		labels = self.labels('train')
		buffer_len = self.seismograph_length
		vector_len = len(labels)
		img_center = (int(round(float(self.width) * 0.5)), int(round(float(self.height) * 0.5)))
																	#  Build a (buffer_len) X (length of labels) matrix:
																	#  [class0, class1, class2, ..., classn] in frame 0
																	#  [class0, class1, class2, ..., classn] in frame 1
		B = []														#  [class0, class1, class2, ..., classn] in frame 2
		if len(probability_accumulator) < buffer_len:
			for i in range(0, buffer_len - len(probability_accumulator)):
				B.append( [0.0 for x in range(0, vector_len)] )
		for i in range(0, len(probability_accumulator)):
			B.append( [x for x in probability_accumulator[i]] )
		B = np.array(B)

		x_intervals = int(np.floor(float(self.width) / float(vector_len)))
		y_intervals = int(np.floor(float(self.height) / float(buffer_len)))
		nudge_right = 30
		label_margin = 5

		for y in range(0, buffer_len):
			color_ctr = 0

			for x in range(0, vector_len):
				point_a = (x * x_intervals + nudge_right,  y      * y_intervals)
				point_b = (x * x_intervals + nudge_right, (y + 1) * y_intervals)

				if color_ctr == 0:
					color = (0, 0, 255)
				elif color_ctr == 1:
					color = (0, 255, 0)
				else:
					color = (255, 0, 0)
				dark_color = (color[0] // 4, color[1] // 4, color[2] // 4)

				graph = cv2.line(graph, point_a, point_b, color, max(1, int(round(float(self.max_seismograph_linewidth) * B[y, x]))) )

				tmp = np.zeros((self.height, self.width, 3), dtype='uint8')
				pt = (x * x_intervals, self.height - 10)
				if label_picks is not None:
					cv2.putText(tmp, labels[x] + ' [' + str(label_picks[labels[x]]) + ']', img_center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, dark_color, 1)
				else:
					cv2.putText(tmp, labels[x], img_center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, dark_color, 1)
				tmp = self.rotate_text(tmp, 90.0, (x * x_intervals + nudge_right - label_margin, self.height - 10))
				indices = np.where(tmp > 0)
				for i in range(0, len(indices[0])):
					graph[indices[0][i], indices[1][i]] = tmp[indices[0][i], indices[1][i]]

				color_ctr += 1
				if color_ctr == 3:
					color_ctr = 0

		return graph

	#  Create a seismograph-like plot of smoothed probabilities.
	#  Receives 'smoothed_probability_accumulator', a buffer of 'self.seismograph_length' probability distributions for all recognizable actions.
	def render_smoothed_probabilities_seismograph(self, smoothed_probability_accumulator, label_picks=None):
																	#  All-white RGB
		graph = np.ones((self.height, self.width, 3), dtype='uint8') * 255

		labels = self.labels('train')
		buffer_len = self.seismograph_length
		vector_len = len(labels)
		img_center = (int(round(float(self.width) * 0.5)), int(round(float(self.height) * 0.5)))
																	#  Build a (buffer_len) X (length of labels) matrix:
																	#  [class0, class1, class2, ..., classn] in frame 0
																	#  [class0, class1, class2, ..., classn] in frame 1
		B = []														#  [class0, class1, class2, ..., classn] in frame 2
		if len(smoothed_probability_accumulator) < buffer_len:
			for i in range(0, buffer_len - len(smoothed_probability_accumulator)):
				B.append( [0.0 for x in range(0, vector_len)] )
		for i in range(0, len(smoothed_probability_accumulator)):
			B.append( [x for x in smoothed_probability_accumulator[i]] )
		B = np.array(B)

		x_intervals = int(np.floor(float(self.width) / float(vector_len)))
		y_intervals = int(np.floor(float(self.height) / float(buffer_len)))
		nudge_right = 30
		label_margin = 5

		for y in range(0, buffer_len):
			color_ctr = 0

			for x in range(0, vector_len):
				point_a = (x * x_intervals + nudge_right,  y      * y_intervals)
				point_b = (x * x_intervals + nudge_right, (y + 1) * y_intervals)

				if color_ctr == 0:
					color = (0, 0, 255)
				elif color_ctr == 1:
					color = (0, 255, 0)
				else:
					color = (255, 0, 0)
				dark_color = (color[0] // 4, color[1] // 4, color[2] // 4)

				graph = cv2.line(graph, point_a, point_b, color, max(1, int(round(float(self.max_seismograph_linewidth) * B[y, x]))) )

				tmp = np.zeros((self.height, self.width, 3), dtype='uint8')
				pt = (x * x_intervals, self.height - 10)
				if label_picks is not None:
					cv2.putText(tmp, labels[x] + ' [' + str(label_picks[labels[x]]) + ']', img_center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, dark_color, 1)
				else:
					cv2.putText(tmp, labels[x], img_center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, dark_color, 1)
				tmp = self.rotate_text(tmp, 90.0, (x * x_intervals + nudge_right - label_margin, self.height - 10))
				indices = np.where(tmp > 0)
				for i in range(0, len(indices[0])):
					graph[indices[0][i], indices[1][i]] = tmp[indices[0][i], indices[1][i]]

				color_ctr += 1
				if color_ctr == 3:
					color_ctr = 0

		return graph

	def rotate_text(self, src, angle, true_point):
		center = (int(round(float(self.width) * 0.5)), int(round(float(self.height) * 0.5)))
		img = src[:, :, :]

		rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
		img = cv2.warpAffine(img, rotMat, (self.width, self.height))

		transMat = np.array([[1.0, 0.0, true_point[0] - center[0]], \
		                     [0.0, 1.0, true_point[1] - center[1]]])
		img = cv2.warpAffine(img, transMat, (self.width, self.height))

		return img

	#  Return a list of tuples: [ (mask file name, recog-object, bounding-box), (mask file name, recog-object, bounding-box), ... ].
	#  One tuple for each recognizable object in the given enactment, at the given frame.
	def get_masks_for_frame(self, enactment, frame_path):
		masks = []
		if self.use_detection_source is None:
			fh = open(enactment + '_props.txt', 'r')
		else:
			fh = open(enactment + '_' + self.use_detection_source + '_detections.txt')
		for line in fh.readlines():
			if line[0] != '#':
				arr = line.strip().split('\t')
				file_path = arr[1]
				r_object = arr[3]
				bbox_str = arr[6]
				mask_path = arr[7]
				if file_path == frame_path:
					bbox_arr = bbox_str.split(';')
					bbox = tuple([int(x) for x in bbox_arr[0].split(',')] + [int(x) for x in bbox_arr[1].split(',')])
					masks.append( (mask_path, r_object, bbox) )
		fh.close()
		return masks
