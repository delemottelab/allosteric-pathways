import os
import sys
from joblib import Parallel, delayed
import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

import postprocessing as pp

def unwrap_current_flow_multiple_sinks(arg, **kwarg):
	return CurrentFlow._current_flow_loop(*arg, **kwarg)


class CurrentFlow():

	def __init__(self, similarity_map_filename, contact_map_filenames, sources_filename, sinks_filename, similarity_map_diagonal_filename='',
				 out_directory='', file_label='', n_chains=1, n_cores=1, compute_edge_CF=False,
				 cheap_write=False, start_frame=0, normalize_similarity_map=False, auxiliary_protein_indices='',
				 additional_interactor_protein_contacts='', compute_current_flow_closeness=False):

		self.file_end_name_ = file_label
		self.out_directory_ = out_directory

		self.n_chains_ = n_chains
		self.n_protein_residues_ = None
		self.n_cores_ = n_cores
		self.compute_edge_CF_ = compute_edge_CF
		self.cheap_write_ = cheap_write
		self.start_frame_ = start_frame

		self.similarity_map_filename_ = similarity_map_filename
		self.similarity_map_diagonal_filename_ = similarity_map_diagonal_filename
		self.norm_similarity_map_ = normalize_similarity_map
		self.contact_map_filenames_ = contact_map_filenames

		self.auxiliary_protein_indices_ = auxiliary_protein_indices
		self.additional_interactor_protein_contacts_ = additional_interactor_protein_contacts

		self.sources_filename_ = sources_filename
		self.sinks_filename_ = sinks_filename

		self.source_nodes_ = []
		self.sink_nodes_ = []

		self.similarity_map_ = None
		self.adjacency_matrix_ = []
		self.Laplacian_ = []
		self.L1_inv_ = []

		self.current_sink_ind_ = []
		self.node_distributions_ = []
		self.all_net_flow_edges_ = []

		self.calc_current_flow_betweenness_ = not(compute_current_flow_closeness)

		print('*----------Current flow analysis of protein residue-residue networks----------*')
		print('')
		print('   similarity_map = ' + str(self.similarity_map_filename_))
		print('   similarity_map_diagonal = ' + str(self.similarity_map_diagonal_filename_))
		print('   normalize_similarity_map = '+str(self.norm_similarity_map_))
		print('   distance_mask = ' + str(self.contact_map_filenames_))
		print('   auxiliary_protein_index = ' + str(self.auxiliary_protein_indices_))
		if self.additional_interactor_protein_contacts_ != '':
			print('   additional_interactor_protein_contacts = ' + str(self.additional_interactor_protein_contacts_))
		print()
		print('   source_nodes = ' + str(self.sources_filename_))
		print('   sink_nodes = ' + str(self.sinks_filename_))
		print('   n_chains = ' + str(self.n_chains_))
		print('   n_cores = ' + str(self.n_cores_))
		print('   Computing edge current flows = ' +str(self.compute_edge_CF_))
		print('   Condense output = ' +str(self.cheap_write_))
		print('   out_directory = ' + str(self.out_directory_))
		print('   file_end_name = ' + str(self.file_end_name_))
		print('')
		if compute_current_flow_closeness:
			print('Computing current flow closeness centrality.')
		else:
			print('Computing current flow betweenness centrality.')
		print('')
		print('*-----------------------------------------------------------------------------*')

		return

	def run(self):
		"""
		Running calculations as specified by input parameters.
		:return:
		"""
		counter = self.start_frame_
		original_file_end_name = self.file_end_name_
		if self.contact_map_filenames_ is None:
			all_contact_maps = ['']
		else:
			all_contact_maps = self.contact_map_filenames_

		for contact_map_file in all_contact_maps:
			if len(all_contact_maps) > 1:
				print('Frame '+str(counter)+'/'+str(self.start_frame_+len(all_contact_maps)-1))

			if len(all_contact_maps) > 1:
				self.file_end_name_ = 'frame_'+str(counter)+'_'+original_file_end_name
			else:
				self.file_end_name_ = original_file_end_name

			if self.calc_current_flow_betweenness_:
				# Compute current flow betweenness
				all_chain_current_flows, all_chain_current_flows_edges = self._compute_current_flow_betweenness(*self._read_data(contact_map_file))

				if self.n_chains_ > 1:
					pp.postprocess_CF_data(self, all_chain_current_flows, n_protein_residues=self.n_protein_residues_)
					if self.compute_edge_CF_:
						print('Postprocessing edge current flows.')
						pp.postprocess_CF_data(self, all_chain_current_flows_edges, is_edges=True)
			else:
				# Compute current flow closeness centrality
				self._compute_current_flow_closeness(*self._read_data(contact_map_file))

			counter += 1
			print()
		return

	def _read_data(self, contact_map_file=None):
		"""
		Reading the contact map, similartiy matrix, and the sources and sinks used in the current flow calculations.
		:param contact_map_file:
		:return:
		"""
		print('Reading data...')

		if contact_map_file is None:
			if self.contact_map_filenames_ is None:
				contact_map_file = None
			else:
				contact_map_file = self.contact_map_filenames_[0]

		if self.similarity_map_ is None:
			if self.similarity_map_filename_[-4::] == '.npy':
				similarity_map = np.load(self.similarity_map_filename_, allow_pickle=True)
			else:
				similarity_map = np.loadtxt(self.similarity_map_filename_)
			similarity_map = np.abs(similarity_map)

			if len(similarity_map.shape) == 1:
				similarity_map = squareform(similarity_map)

			# Read diagonal values of similarity_map (in compressed format, the similarity_map has diagonal values 0)
			if self.similarity_map_diagonal_filename_ == '':
				print('No similarity map diagonal supplied in input.')
			else:
				print('Setting diagonal of similarity map.')
				if self.similarity_map_diagonal_filename_[-4::] == '.npy':
					diagonal = np.load(self.similarity_map_diagonal_filename_)
				else:
					diagonal = np.loadtxt(self.similarity_map_diagonal_filename_)

				for i in range(diagonal.shape[0]):
					similarity_map[i, i] = diagonal[i]
					if similarity_map[i, i]==0:
						print('Warning: Zero encountered in diagonal: '+str(i))

			similarity_map = np.abs(similarity_map)

			if self.norm_similarity_map_:
				similarity_map = self._normalize_similarity_map(similarity_map)
		else:
			similarity_map = np.copy(self.similarity_map_)

		# Set number of protein residues tentatively as the number of residues in the similarity map.
		# This will be overwritten if a contact map is supplied.
		self.n_protein_residues_ = similarity_map.shape[0]

		if self.contact_map_filenames_ is not None:
			if len(self.contact_map_filenames_) > 1:
				self.similarity_map_ = np.copy(similarity_map)

			contact_map = self._set_contact_map(contact_map_file)

			if contact_map.shape[0] != contact_map.shape[0] or contact_map.shape[0] != similarity_map.shape[0]:
				print('Error: Wrong dimensions on contact map and/or similarity matrix.')
				print('Contact map shape: '+str(contact_map.shape))
				print('Similarity matrix shape:'+str(similarity_map.shape))
				sys.exit(0)
		else:
			print('No distance mask supplied! Using full similarity map data.')
			contact_map = np.ones(similarity_map.shape)

		source_nodes = np.sort(np.ravel(np.loadtxt(self.sources_filename_)).astype(int))
		sink_nodes = np.sort(np.ravel(np.loadtxt(self.sinks_filename_)).astype(int))

		return similarity_map, contact_map, source_nodes, sink_nodes

	def _set_contact_map(self, contact_map_file):
		"""
		Load the protein contact map and add potential interactor contact map
 		:return:
		"""

		print('Setting protein contact map.')

		if contact_map_file[-4::] == '.npy':
			contact_map = np.load(contact_map_file)
			if len(contact_map.shape) == 1:
				contact_map = squareform(contact_map)
				print('Distance mask size: ' + str(contact_map.shape[0]))
		else:
			contact_map = np.loadtxt(contact_map_file)
			if len(contact_map.shape) == 1:
				contact_map = squareform(contact_map)
				print('Distance mask size: ' + str(contact_map.shape[0]))

		# The number of protein residues is set to the number of residues in the protein contact map.
		self.n_protein_residues_ = contact_map.shape[0]

		if self.additional_interactor_protein_contacts_ != '':
			contact_map = self._add_interactor_cmap(self.additional_interactor_protein_contacts_, contact_map)

		return contact_map

	def _add_interactor_cmap(self, additional_interactor_protein_contacts, contact_map):
		"""
		Adding additional interactor sontact map to the protein contact map.
		:param additional_interactor_protein_contacts:
		:param contact_map:
		:return:
		"""
		print('Adding interactor contact map.')
		int_prot_cmap = np.load(additional_interactor_protein_contacts)
		new_cmap = np.zeros((int_prot_cmap.shape[1],int_prot_cmap.shape[1]))

		# Add protein contacts
		new_cmap[0:contact_map.shape[0], 0:contact_map.shape[1]] = np.copy(contact_map)
		# Add interactor contacts
		new_cmap[contact_map.shape[0]::, :] = np.copy(int_prot_cmap)
		new_cmap[:, contact_map.shape[1]::] = np.copy(int_prot_cmap.T)

		# Overwrite the contact map to include both protein and interactor interactions
		contact_map = np.copy(new_cmap)
		return contact_map

	def _normalize_similarity_map(self, similarity_map):
		"""
		Normalize similarity map with symmetric uncertainty.
		:param similarity_map:
		:return:
		"""
		print('Normalizing similarity map.')
		normalized_similarity_map = np.copy(similarity_map)
		n_residues = similarity_map.shape[0]
		# Normalize the mutual information
		for i in range(0, n_residues):
			for j in range(i, n_residues):
				normalized_similarity_map[i, j] = 2.0 * similarity_map[i, j] / (similarity_map[i, i] + similarity_map[j, j])
				normalized_similarity_map[j, i] = normalized_similarity_map[i, j]
		return normalized_similarity_map

	def _set_graph_Laplacian(self):
		"""
		Setting the (normalized) graph Laplacian.
		"""
		D = np.sum(self.adjacency_matrix_, axis=1)
		self.Laplacian_ = np.diag(D) - self.adjacency_matrix_
		return

	def _set_reduced_inverse_Laplacian(self, sink_node_inds):
		"""
		Computing the reduced system inverse Laplacian of multiple absorbing states.
		This enables computing properties from absorbing markov chains, such as current flow.
		"""
		tmp_L1_inv = np.zeros(self.Laplacian_.shape)

		n_sinks = sink_node_inds.shape[0]

		# used in current flow calculations
		self.current_sink_ind_ = np.zeros(sink_node_inds.shape[0], int)

		# Compute inverse of reduced Laplacian
		reduced_L = np.delete(self.Laplacian_, sink_node_inds, axis=0)
		reduced_L = np.delete(reduced_L, sink_node_inds, axis=1)

		L_red_inv = np.linalg.inv(reduced_L)

		# Create a matrix with zero rows and cols at current sink node, and reduced inverse Laplacian elsewhere
		prev_sink_ind_row = 0
		for i in range(n_sinks):

			self.current_sink_ind_[i] = sink_node_inds[i]

			prev_sink_ind_col = 0
			next_sink_ind_row = sink_node_inds[i]

			for j in range(n_sinks):
				next_sink_ind_col = sink_node_inds[j]

				tmp_L1_inv[prev_sink_ind_row:next_sink_ind_row, prev_sink_ind_col:next_sink_ind_col] = L_red_inv[
																									   prev_sink_ind_row - i:next_sink_ind_row - i,
																									   prev_sink_ind_col - j:next_sink_ind_col - j]
				prev_sink_ind_col = next_sink_ind_col + 1

			# Set last columns at the current rows
			tmp_L1_inv[prev_sink_ind_row:next_sink_ind_row, prev_sink_ind_col::] = L_red_inv[(prev_sink_ind_row - i):(
						next_sink_ind_row - i), prev_sink_ind_col - n_sinks::]

			prev_sink_ind_row = next_sink_ind_row + 1

		# Set lowest rows
		prev_sink_ind_col = 0
		for j in range(n_sinks):
			next_sink_ind_col = sink_node_inds[j]
			tmp_L1_inv[prev_sink_ind_row::, prev_sink_ind_col:next_sink_ind_col] = L_red_inv[
																				   prev_sink_ind_row - n_sinks::,
																				   prev_sink_ind_col - j:next_sink_ind_col - j]
			prev_sink_ind_col = next_sink_ind_col + 1

		# Set final square
		tmp_L1_inv[prev_sink_ind_row::, prev_sink_ind_col::] = L_red_inv[prev_sink_ind_row - n_sinks::,
															   prev_sink_ind_col - n_sinks::]

		self.L1_inv_ = np.copy(tmp_L1_inv)
		return

	def _current_flow_loop(self, j):
		"""
		One iteration in loop: Calculating current flow calculation of one source node to all sinks.
		:param j: current loop index
		"""
		tmp_source_ind = self.source_nodes_[j]

		b = np.zeros(self.adjacency_matrix_.shape[0])
		b[self.current_sink_ind_] = -1.0 / float(self.current_sink_ind_.shape[0])
		b[tmp_source_ind] = 1.0

		# Compute throughput of each edge and node
		potentials = np.dot(self.L1_inv_, b)

		if np.sum(np.abs(potentials-self.L1_inv_[:,tmp_source_ind])) != 0:
			print('Warning! potentials vs. source column in Linv: '+str(np.sum(np.abs(potentials-self.L1_inv_[:,tmp_source_ind]))))

		[p_x, p_y] = np.meshgrid(potentials, potentials)

		currents = np.multiply((p_y - p_x), self.adjacency_matrix_)

		net_current_flow = np.abs(currents)

		if self.compute_edge_CF_:
			self.all_net_flow_edges_[j] = net_current_flow
		self.node_distributions_[j] = 0.5 * (np.sum(net_current_flow, axis=1))

		# Remove sources and sinks from analysis
		self.node_distributions_[j][tmp_source_ind] = 0
		self.node_distributions_[j][self.current_sink_ind_] = 0
		return

	def _compute_current_flow_betweenness(self, similarity_map, distance_mask, source_nodes, sink_nodes):
		"""
		Computing the current flow betweenness on nodes and edges.
		"""
		print('Running calculations.')

		A = np.multiply(similarity_map, distance_mask)

		self.adjacency_matrix_ = np.copy(A)

		n_sources_per_chain = int(source_nodes.shape[0] / self.n_chains_)
		all_chain_CF = [None]*self.n_chains_	
		all_chain_CF_edges = [None]*self.n_chains_

		for i_chain in range(self.n_chains_):
			self._set_graph_Laplacian()

			n_nodes = A.shape[0]

			# Compute current flow with absorbing markov chains
			edge_distribution = np.zeros(A.shape)
			node_distribution = np.zeros(n_nodes)
			self.source_nodes_ = np.copy(source_nodes[i_chain * n_sources_per_chain:(i_chain + 1) * n_sources_per_chain])

			# Calculating the average visiting times between all nodes
			self._set_reduced_inverse_Laplacian(sink_nodes)

			self.node_distributions_ = [None] * n_sources_per_chain
			if self.compute_edge_CF_:
				self.all_net_flow_edges_ = [None] * n_sources_per_chain

			if self.n_cores_ == 1:
				for j in range(self.source_nodes_.shape[0]):
					self._current_flow_loop(j)
					node_distribution += self.node_distributions_[j]
					self.node_distributions_[j] = None

					if self.compute_edge_CF_:
						edge_distribution += self.all_net_flow_edges_[j]
						self.all_net_flow_edges_[j] = None
			else:
				# Compute current flow from all sources
				Parallel(n_jobs=self.n_cores_, backend="threading")(
					delayed(unwrap_current_flow_multiple_sinks)(j) for j in
					zip([self] * self.source_nodes_.shape[0], range(self.source_nodes_.shape[0])))

				# Add node and edge distributions together
				for j in range(n_sources_per_chain):
					node_distribution += self.node_distributions_[j]
					if self.compute_edge_CF_:
						edge_distribution += self.all_net_flow_edges_[j]

			# Average distributions
			node_distribution /= float(n_sources_per_chain)
			if self.compute_edge_CF_:
				edge_distribution /= float(n_sources_per_chain)

			all_chain_CF[i_chain] = np.copy(node_distribution)
			if self.compute_edge_CF_:
				all_chain_CF_edges[i_chain] = np.copy(edge_distribution)

			if self.n_chains_ > 1:
				print('Chain ' + str(i_chain + 1))
				if not(self.cheap_write_):
					np.save(self.out_directory_ + 'current_flow_betweenness_chain_averaged_' + self.file_end_name_ + '_chain_' + str(
						i_chain + 1) + '.npy', node_distribution)
					print('Saved current flow betweenness centrality.')
					if self.compute_edge_CF_:
						np.save(self.out_directory_ + 'edge_current_flow_betweenness_chain_averaged_'+
								self.file_end_name_+'_chain_'+str(i_chain+1)+'.npy', edge_distribution)
						print('Saved edge current flow betweenness centrality.')

		if self.n_chains_ == 1:
			if not(self.cheap_write_):
				np.save(self.out_directory_ + 'current_flow_betweenness_' + self.file_end_name_ + '.npy', node_distribution)
				print('Saved current flow betweenness centrality.')

				if self.compute_edge_CF_:
					np.save(self.out_directory_ + 'edge_current_flow_betweenness__'+self.file_end_name_+'.npy', edge_distribution)
					print('Saved current flow betweenness centrality.')

		return all_chain_CF, all_chain_CF_edges

	def _compute_current_flow_closeness(self, similarity_map, distance_mask, source_nodes, sink_nodes):
		"""
		Computing the current flow closeness centrality between each source and the sinks.
		"""
		print('Running calculations.')

		A = np.multiply(similarity_map, distance_mask)

		self.adjacency_matrix_ = np.copy(A)

		n_sources_per_chain = int(source_nodes.shape[0] / self.n_chains_)

		CFCs = np.zeros((n_sources_per_chain, self.n_chains_))

		for i_chain in range(self.n_chains_):
			self._set_graph_Laplacian()
			self._set_reduced_inverse_Laplacian(sink_nodes)

			self.source_nodes_ = np.copy(
				source_nodes[i_chain * n_sources_per_chain:(i_chain + 1) * n_sources_per_chain])

			for i_source in range(n_sources_per_chain):
				tmp_source_ind = self.source_nodes_[i_source]

				b = np.zeros(self.adjacency_matrix_.shape[0])
				b[sink_nodes] = -1.0 / float(sink_nodes.shape[0])
				b[tmp_source_ind] = 1.0

				# Compute throughput of each edge and node
				potentials = np.dot(self.L1_inv_, b)

				[p_x, p_y] = np.meshgrid(potentials, potentials)

				if np.sum(np.abs(potentials[sink_nodes])) > 0:
					print('WARNING! Non-zero sink potentials.')
					print('sink potentials:')
					print(potentials[sink_nodes])
					print('source potentials:')
					print(potentials[tmp_source_ind])

				potential_differences = np.abs(p_y - p_x)
				CFCs[i_source, i_chain] = 1.0 / np.mean(potential_differences[tmp_source_ind, sink_nodes])

		if self.n_chains_ > 1 and not(self.cheap_write_):
			np.save(self.out_directory_ + 'all_chain_current_flow_closeness_' + self.file_end_name_ + '.npy', CFCs)

		# Average over chains
		CFCs_average = np.mean(CFCs, axis=1)
		CFCs_std = np.std(CFCs, axis=1)
		np.save(self.out_directory_ + 'average_current_flow_closeness_' + self.file_end_name_ + '.npy', CFCs_average)
		np.save(self.out_directory_ + 'std_current_flow_closeness_' + self.file_end_name_ + '.npy', CFCs_std)
		print('Saved current flow closeness centrality.')
		return