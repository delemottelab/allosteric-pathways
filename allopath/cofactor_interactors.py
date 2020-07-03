import os
import sys
import numpy as np
import mdtraj as md
import lap # https://github.com/gatagat/lap
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed

current_dir = os.path.dirname(__file__)

sys.path.append(current_dir)

import MD_init
from cmap_utils import *


class CofactorInteractors():

	def __init__(self, topology_file, trajectory_files='', cofactor_domain_selection='', cofactor_interactor_inds='',
				cofactor_interactor_coords='', compute_interactor_node_fluctuations=False, cofactor_interactor_atom_inds='',
				use_internal_coordinates=False, dt=1, out_directory='', file_label='', multiple_trajectories=False,
				trajectory_file_directory='', cutoff=0.45, std_dev=0.138):


		traj_init_kwargs = {
			'topology_file': topology_file,
			'trajectory_files': trajectory_files,
			'trajectory_file_directory': trajectory_file_directory,
			'dt': dt,
			'multiple_trajectories': multiple_trajectories,
			'file_label': file_label,
			'out_directory': out_directory,
		}

		init = MD_init.MD_initializer()
		self.traj_ = init.initialize_trajectory(**traj_init_kwargs)

		self.file_end_name_ = file_label
		self.out_directory_ = init.out_directory_
		self.n_cores_ = 1

		self.cutoff_ = cutoff
		self.std_dev_ = std_dev

		self.protein_traj_ = self.traj_.atom_slice(self.traj_.topology.select('protein'))

		self.n_residues_ = int(self.protein_traj_.n_residues)
		self.n_interactor_residues_ = int(self.traj_.n_residues) - self.n_residues_


		self.n_frames_ = int(self.traj_.n_frames)

		self.n_pairs_ = int(self.n_residues_*(self.n_residues_-1)/2)

		self.cofactor_interactor_indices_ = None
		self.cofactor_interactor_inds_file_ = cofactor_interactor_inds

		self.cofactor_interactor_atom_indices_ = []
		self.cofactor_interactor_atom_inds_file_ = cofactor_interactor_atom_inds

		self.compute_interactor_centroids_ = compute_interactor_node_fluctuations
		self.use_cartesian_coords_ = not(use_internal_coordinates)

		if self.use_cartesian_coords_:
			print('Using Cartesian coordinates.')
		else:
			print('Using protein-internal coordinates.')

		self.domain_decomposition_names_ = self._set_domain_decomposition(cofactor_domain_selection)

		self.cofactor_interactor_coords_ = None
		self.cofactor_interactor_coords_file_ = cofactor_interactor_coords

		self.cofactor_interactor_protein_cmap_ = None

		return


	def _set_domain_decomposition(self,cofactor_domain_selection):
		"""
		Read the user-supplied file with cofactor domain selections
		:return:
		"""
		if cofactor_domain_selection != '':
			domain_selections = []
			fID1 = open(cofactor_domain_selection, 'r')
			for line in fID1:
				domain_selections.append(' and ' + line.replace('\n', ''))
		else:
			domain_selections = ['']

		print()
		print('cofactor interactor domain selections: ')
		for isel, sel in enumerate(domain_selections):
			print('not protein'+str(sel))
		print()
		return domain_selections


	def _get_resid_indices(self, n_residues, query='protein and !(type H)', domain_decomposition=None):
		print('Extracting residue indices with query = '+query)
		if domain_decomposition is not None:
			print('Using supplied domain decomposition to split cofactors into '+str(len(domain_decomposition))+' domains...')

		residue_inds = []
		for i_resid in range(n_residues):
			if domain_decomposition is None:
				ind = self.traj_.topology.select(query + ' and resid ' + str(i_resid))
				if len(ind) > 0:
					residue_inds.append(ind)
			else:
				for atom_sel in domain_decomposition:
					ind = self.traj_.topology.select(query + ' and resid ' + str(i_resid)+atom_sel)
					if len(ind) > 0:
						residue_inds.append(ind)

		print('# Residues extracted: '+str(len(residue_inds)))
		return residue_inds

	def _compute_cofactor_interactor_coordinates(self):
		"""
		Protein Cartesian coordinates: Compute the Cartesian centroid of each interactor in each frame.
		Protein internal coordinates: Compute distances between cofactor Ps and and protein C-alphas. Then, use these distances
		to determine cofactor interactor indices.
		:return:
		"""
		n_domains = len(self.domain_decomposition_names_)

		cofactor_inds = self._get_resid_indices(self.n_residues_+self.n_interactor_residues_, query='not protein',
											   domain_decomposition=self.domain_decomposition_names_)

		self.cofactor_interactor_atom_indices_ = cofactor_inds
		self.n_interactor_residues_ = int(len(cofactor_inds))


		if not(self.use_cartesian_coords_):
			protein_CA_inds = self._get_resid_indices(self.n_residues_+self.n_interactor_residues_, query='protein and name CA')
			all_atom_pairs = []
			for cofactor_domain_inds in cofactor_inds:
				all_atom_pairs.append(get_atom_pairs(cofactor_domain_inds, protein_CA_inds))

		print('Calculating interactor coordinates. ')

		if self.use_cartesian_coords_:
			# Compute distances
			self.cofactor_interactor_coords_ = np.zeros((self.n_frames_, self.n_interactor_residues_, 3))

			# Superpose system on the protein C-alphas
			if len(self.traj_.topology.select('protein and name CA')) > 0:
				self.traj_.superpose(self.traj_, frame=0, atom_indices=self.traj_.topology.select('protein and name CA'))
			else:
				self.traj_.superpose(self.traj_, frame=0, atom_indices=self.traj_.topology.select('protein'))

			all_points = self.traj_.xyz

			# Interactor centroid in protein-internal coordinate system
			for i_interactor, interactor_inds in enumerate(cofactor_inds):
				self.cofactor_interactor_coords_[:, i_interactor, :] = np.mean(all_points[:,interactor_inds,:], axis=1)
				sys.stdout.write(
					"\r Interactor index: " + str(i_interactor+1) + '/' + str(self.n_interactor_residues_) + ' ')
				sys.stdout.flush()
		else:
			# Compute distances
			self.cofactor_interactor_coords_ = np.zeros((self.n_frames_,self.n_interactor_residues_,self.n_residues_))

			# Interactor centroid in protein-internal coordinate system
			i_interactor = 0
			for atom_pairs in all_atom_pairs:
				tmp_dists = md.compute_distances(self.traj_, atom_pairs, periodic=True)
				tmp_dists= np.reshape(tmp_dists, (self.n_frames_, int(len(atom_pairs)/self.n_residues_), self.n_residues_))
				self.cofactor_interactor_coords_[:,int(i_interactor),:] = np.mean(tmp_dists,axis=1)
				i_interactor += 1
				sys.stdout.write("\r Interactor index: "+str(i_interactor)+'/'+str(self.n_interactor_residues_)+' ')
				sys.stdout.flush()
		print()
		np.save(self.out_directory_+'cofactor_interactor_coords_'+self.file_end_name_+'.npy',self.cofactor_interactor_coords_)
		np.save(self.out_directory_+'cofactor_interactor_atom_indices_'+self.file_end_name_+'.npy',cofactor_inds)

		print('Files written:')
		print(self.out_directory_+'cofactor_interactor_coords_'+self.file_end_name_+'.npy')
		print(self.out_directory_ + 'cofactor_interactor_atom_indices_' + self.file_end_name_ + '.npy')
		return

	def _compute_distances_PBC(self, coords1, coords2, unitcell_size):
		"""
		Compute minimum distances between two particles with PBC.
		OBS: Assumes an orthorhombic box.

		:param coords1: Coordinates of the first particle in the original simulation box.
		:param coords2: Coordinates of the second particle in the original simulation box.
		:param unitcell_size: The size of the simulation box corresponding to coords1 (unitcell lengths).
		:return: The minimum distances across periodic boundary condition
		"""

		n_images=unitcell_size.shape[0]**3

		image_dists = np.zeros((coords1.shape[0], coords2.shape[0], n_images))
		dX = np.zeros(3)

		counter = 0
		for dx in np.arange(-1, 2):
			dX[0] = dx*unitcell_size[0]
			for dy in np.arange(-1, 2):
				dX[1] = dy*unitcell_size[1]
				for dz in np.arange(-1, 2):
					dX[2] = dz*unitcell_size[2]
					# Compute distances between the two particles for image translated by dX
					image_dists[:, :, counter] = cdist(coords1 + dX, coords2)
					counter+=1

		# Return the minimum distances across periodic boundary condition
		min_distance = image_dists.min(axis=2)

		is_wrong=False
		if np.max(min_distance) > np.min(unitcell_size):
			print('Error: Min distance > box size: ')
			print('Image distances: '+str(image_dists))
			print('Unitcell size: '+str(unitcell_size))
			print('coords1:' +str(coords1))
			print('coords2: '+str(coords2))
			is_wrong=True
		return min_distance, is_wrong

	def _compute_mean_PBC(self, coords, unitcell_sizes):
		"""
		Compute the centroid of an interactor along the trajectory.
		OBS: Assumes an orthorhombic box.

		:param coords: Coordinates of the interactor in the original simulation box.
		:param unitcell_size: The size of the simulation box (unitcell lengths).
		:return: The minimum distances across periodic boundary condition
		"""
		n_images = unitcell_sizes.shape[1]**3
		n_frames = coords.shape[0]
		n_dims = coords.shape[1]
		image_coords = np.zeros((n_frames * n_images, n_dims))
		dX = np.zeros((n_frames, n_dims))

		ref_ind = 0
		counter = 0
		for dx in np.arange(-1, 2):
			dX[:, 0] = dx * unitcell_sizes[:, 0]
			for dy in np.arange(-1, 2):
				dX[:, 1] = dy * unitcell_sizes[:, 1]
				for dz in np.arange(-1, 2):
					dX[:, 2] = dz * unitcell_sizes[:, 2]

					# Translate the particle to image translated by dX
					image_coords[counter * n_frames:(counter + 1) * n_frames, :] = np.copy(coords + dX)
					if np.sum(np.abs(dX)) == 0:
						ref_ind = counter * n_frames 	# Index of first coordinate in the original box
					counter += 1

		# Clustering based on cutoff: distance between two points should not be larger
		# than a half the shortest box side
		cutoff = np.min(unitcell_sizes)/2.0
		point_distances = cdist(image_coords[ref_ind, np.newaxis, :], image_coords)
		component_indices = np.zeros(n_frames * n_images)
		component_indices[point_distances[0, :] < cutoff] = 1

		# Get component centroid
		centroid = np.mean(image_coords[component_indices == 1, :], axis=0)

		# Return the centroid
		return centroid


	def _compute_interactor_indices_per_frame(self):
		# Set cofactor interactor indices
		all_cofactors = np.arange(self.n_interactor_residues_)
		print('Computing cofactor interactor indices.')
		ref_frame_coords = np.copy(self.cofactor_interactor_coords_[0])
		self.cofactor_interactor_indices_ = np.zeros((self.n_frames_, self.n_interactor_residues_))
		self.cofactor_interactor_indices_[0] = np.arange(self.n_interactor_residues_)
		for i_frame in range(self.n_frames_):
			sys.stdout.write("\r Frame: "+str(i_frame+1)+'/'+str(self.n_frames_)+' ')
			sys.stdout.flush()

			# Get cofactor coordinates of current frame
			current_frame_coords = self.cofactor_interactor_coords_[i_frame]

			# Compute distances to the coordinates in frame 0
			if not(self.use_cartesian_coords_):	# When using internal coordinates, periodic boundary conditions are taken into account in the coordinates
				distances = cdist(current_frame_coords, ref_frame_coords)
			else:
				distances, _ = self._compute_distances_PBC(current_frame_coords, ref_frame_coords, self.traj_.unitcell_lengths[i_frame])

			# Solve the linear sum assignment problem
			_, node_inds, _ = lap.lapjv(distances)

			# Get cofactor interactor index in each frame by finding the index of the closest cofactor in the reference frame
			self.cofactor_interactor_indices_[i_frame, all_cofactors] = node_inds

		print()
		# Save data
		np.save(self.out_directory_+'cofactor_interactor_indices_'+self.file_end_name_+'.npy',self.cofactor_interactor_indices_)

		print('cofactor interactor coordinates and indices written to files: ')
		print(self.out_directory_+'cofactor_interactor_indices_'+self.file_end_name_+'.npy')

		return

	def _get_resid_number(self, atom_ind):
		"""
		Get the resID of the residue containing the atom given by atom_ind.
		:param atom_ind:
		:return:
		"""
		return self.traj_.topology.atom(atom_ind).residue.index

	def _compute_interactor_centroid_fluctuations(self):
		"""
		Compute the interactor centroids in each frame and write results to file.
		:return:
		"""

		if not (self.use_cartesian_coords_):
			print('Computing interactor centroid fluctuations in the protein-internal coordinate system.')
		else:
			print('Computing interactor centroid fluctuations in Cartesian coordinate system.')

		self.n_interactor_residues_ = self.cofactor_interactor_indices_.shape[1]

		if self.cofactor_interactor_coords_ is None:
			self._compute_cofactor_interactor_coordinates()

		print('Permuting interactor coordinates according to interactor indices.')
		tmp_cofactor_interactor_coords = np.copy(self.cofactor_interactor_coords_)
		for i_frame in range(self.n_frames_):
			sys.stdout.write("\r Interactor index: " + str(i_frame+1) + '/' + str(
				self.n_frames_))
			sys.stdout.flush()

			# Permute the coordinates
			self.cofactor_interactor_coords_[i_frame, self.cofactor_interactor_indices_[i_frame].astype(int), :] = np.copy(tmp_cofactor_interactor_coords[i_frame])

		del self.cofactor_interactor_indices_
		del tmp_cofactor_interactor_coords
		print()

		print('Computing interactor node fluctuations.')
		interactor_centroid_fluctuations = np.zeros((self.n_frames_, self.n_interactor_residues_))
		n_coords = self.cofactor_interactor_coords_.shape[2]
		# Compute the interactor centroid fluctuations in each frame
		for i_resid in range(self.n_interactor_residues_):
			sys.stdout.write("\r Interactor index: " + str(i_resid+1) + '/' + str(
				self.n_interactor_residues_))
			sys.stdout.flush()
			ave_centroid = self._compute_mean_PBC(self.cofactor_interactor_coords_[:, i_resid, :], self.traj_.unitcell_lengths)
			ave_centroid = ave_centroid[np.newaxis, :]
			for i_frame in range(self.n_frames_):
				interactor_coord = np.copy(self.cofactor_interactor_coords_[np.newaxis, i_frame, i_resid, :])

				if not(self.use_cartesian_coords_): # When using internal coordinates, periodic boundary conditions are taken into account in the coordinates
					# Normalize the fluctuations by 3.0/n_coords to be in comparable size to 3D fluctuations (residue fluctuations).
					interactor_centroid_fluctuations[i_frame,i_resid] = 3.0/n_coords*cdist(interactor_coord, ave_centroid)
				else:
					interactor_centroid_fluctuations[i_frame,i_resid],_ = self._compute_distances_PBC(interactor_coord,
																									ave_centroid,
																									self.traj_.unitcell_lengths[i_frame])
		print()

		# Write data to file
		np.save(self.out_directory_+'interactor_centroid_fluctuations_'+self.file_end_name_+'.npy', interactor_centroid_fluctuations)
		print('Centroid fluctuations written to file: '+self.out_directory_+'interactor_centroid_fluctuations_'+self.file_end_name_+'.npy')
		return


	def _cofactor_interactor_cmap_loop(self,i_resid):
		"""
		One loop iteration of contact map construction.
		:return:
		"""
		sys.stdout.write("\r " +str(i_resid+1-self.n_residues_)+'/'+str(self.n_interactor_residues_))
		sys.stdout.flush()

		i_interactor_inds = self.cofactor_interactor_indices_[:, i_resid - self.n_residues_].astype(int)

		for j_resid in range(i_resid):

			atom_pairs = get_atom_pairs(self.all_resid_inds_[i_resid],self.all_resid_inds_[j_resid])
			distances = np.min(md.compute_distances(self.traj_, atom_pairs, periodic=True),axis=1)
			distances = semi_Gaussian_kernel(distances, cutoff=self.cutoff_, std_dev=self.std_dev_)

			average_contact = np.copy(distances/float(self.n_frames_))

			# Assign contacts to their corresponding network nodes.
			if j_resid >= self.n_residues_:
				j_interactor_inds = self.cofactor_interactor_indices_[:,j_resid-self.n_residues_]
				for i_frame in range(self.n_frames_):
					j_int_ind = int(j_interactor_inds[i_frame])
					i_int_ind = int(i_interactor_inds[i_frame])

					if i_int_ind >= 0 and j_int_ind >= 0:
						self.cofactor_interactor_protein_cmap_[i_int_ind, j_int_ind+self.n_residues_] += average_contact[i_frame]
						self.cofactor_interactor_protein_cmap_[j_int_ind, i_int_ind+self.n_residues_] += average_contact[i_frame]
			else:
				for i_frame in range(self.n_frames_):
					i_int_ind = int(i_interactor_inds[i_frame])
					self.cofactor_interactor_protein_cmap_[i_int_ind, j_resid] += average_contact[i_frame]
		return

	def _compute_cofactor_interactor_cmaps(self):
		"""
		Compute the semi-binary contact map of cofactor-cofactor and cofactor-protein interactions using the
		cofactor interactor indices. The final contact map has elements of the interactors, not the specific cofactor residues.
		:return:
		"""
		self.all_resid_inds_= self._get_resid_indices(self.n_residues_,query='!(type H)')

		for ind in self.cofactor_interactor_atom_indices_:
			self.all_resid_inds_.append(ind)

		self.n_interactor_residues_ = len(self.cofactor_interactor_atom_indices_)

		print('Computing interactor node contact maps.')
		self.cofactor_interactor_protein_cmap_ = np.zeros((self.n_interactor_residues_,self.n_residues_+self.n_interactor_residues_))

		for i_resid in range(self.n_residues_,self.n_residues_+self.n_interactor_residues_):
			self._cofactor_interactor_cmap_loop(i_resid)
		print()

		np.save(self.out_directory_+'cofactor_protein_residue_semi_binary_cmap_'+self.file_end_name_+'.npy',self.cofactor_interactor_protein_cmap_)

		print('cofactor-protein interactor contact map written to file: ')
		print(self.out_directory_+'cofactor_protein_residue_semi_binary_cmap_'+self.file_end_name_+'.npy')
		return

	def run(self):

		if (self.cofactor_interactor_inds_file_ == '' or self.cofactor_interactor_atom_inds_file_== '') and self.cofactor_interactor_coords_file_ == '':
			self._compute_cofactor_interactor_coordinates()
			self._compute_interactor_indices_per_frame()

		elif self.cofactor_interactor_inds_file_ != '' and self.cofactor_interactor_atom_inds_file_ != '':

			print('Reading interactor atom indices and permutation indices.')
			self.cofactor_interactor_indices_ = np.load(self.cofactor_interactor_inds_file_)
			self.cofactor_interactor_atom_indices_ = np.load(self.cofactor_interactor_atom_inds_file_, allow_pickle=True)
			print('Done reading data.')

		elif self.cofactor_interactor_coords_file_ != '' and not(self.compute_interactor_centroids_):
			self._compute_interactor_indices_per_frame()

		if self.compute_interactor_centroids_:
			if self.cofactor_interactor_coords_file_ != '' and self.cofactor_interactor_inds_file_ != '':

				self.cofactor_interactor_coords_ = np.load(self.cofactor_interactor_coords_file_)
				self.cofactor_interactor_indices_ = np.load(self.cofactor_interactor_inds_file_)

			self._compute_interactor_centroid_fluctuations()
		else:
			self._compute_cofactor_interactor_cmaps()

		print('Data written to files.')
		return
