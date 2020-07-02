import sys
import os
import time

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(__file__))

sys.path.append(current_dir)

from joblib import Parallel, delayed
import mdtraj as md
import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt

import MD_init

def unwrap_atom_select(arg,**kwarg):
	return MutualInformation._atom_select_loop(*arg,**kwarg)

def unwrap_MI_loop(arg,**kwarg):
	return MutualInformation._MI_loop(*arg,**kwarg)

def unwrap_MI_diagonal_loop(arg,**kwarg):
	return MutualInformation._MI_diagonal_loop(*arg,**kwarg)

class MutualInformation():
	
	def __init__(self, topology_file, trajectory_files='', contact_map_file='', i_block=0,
				 n_matrix_block_cols=1, n_split_sets=0, dt=1, out_directory='', file_label='', n_cores=-1,
				 additional_interactor_protein_contacts='', additional_interactor_fluctuations='',
				 multiple_trajectories=False, trajectory_file_directory='', do_diagonal=False, do_local_alignment=False,
				 alignment_residue_range=2, lag_time=0, n_components_range=[1,4], query='protein and !(type H)'):

		self.traj_init_kwargs_ = {
			'topology_file': topology_file,
			'trajectory_files': trajectory_files,
			'trajectory_file_directory': trajectory_file_directory,
			'dt': dt,
			'multiple_trajectories': multiple_trajectories,
			'file_label': file_label,
			'out_directory': out_directory,
		}

		self.file_end_name_ = file_label
		self.out_directory_ = out_directory
		self.n_cores_ = n_cores

		self.points_ = 0
		self.protein_points_ = 0
		self.alignment_residue_range_ = 2
		self.n_residues_ = 0
		self.query_ = query

		# Create the blocks and keep the current block part (index)
		self.lag_time_ = int(lag_time)
		self.n_cols_ = int(n_matrix_block_cols)
		if self.lag_time_ == 0:
			self.n_blocks_ = self.n_cols_*(self.n_cols_-1)/2+self.n_cols_
		else:
			self.n_blocks_ = self.n_cols_*self.n_cols_

		self.n_split_sets_ = int(n_split_sets)

		self.contact_map_file_ = contact_map_file
		self.cmap_ = []
		
		self.MIs_ = []
		self.all_MIs_ = []
		self.fluctuations_ = None

		self.i_block_ = int(i_block-1)

		self.do_diagonal_ = do_diagonal
		self.do_local_alignment_ = do_local_alignment
		self.alignment_residue_range_ = alignment_residue_range

		self.topology_ = None
		self.n_component_lim_ = n_components_range

		print('Estimating density with #components in range: '+str(self.n_component_lim_))

		self.additional_interactor_protein_contacts_ = additional_interactor_protein_contacts
		self.additional_interactor_fluctuations_ = additional_interactor_fluctuations

		return


	def _compute_split_PDFs(self, GMM, nPoints):
		"""
		Compute the entropy using the probability density model of MD points and bootstrapped points.
		Is used for computing obtaining a more accurate MI estimate.
		:param GMM:
		:param nPoints:
		:return:
		"""
		PDFs = []
		entropies = np.zeros(self.n_split_sets_ + 1)
		all_sampled_points = []

		for i_set in range(self.n_split_sets_):
			sampled_points, _ = GMM.sample(nPoints)

			# Compute the density of sampled points in the model
			pdf_x = self._GMM_density(GMM,sampled_points)
			entropy = self._Monte_Carlo_entropy(pdf_x)

			PDFs.append(pdf_x)
			entropies[i_set] = entropy
			all_sampled_points.append(sampled_points)

		return all_sampled_points, PDFs, entropies

	def _Monte_Carlo_entropy(self,pdf_x):
		"""
		Estimate the entropy of pdf_x using Monte-Carlo integration.
		"""
		entropy = np.mean(-np.log(pdf_x+1e-16))
		return entropy

	def _estimate_GMM(self, x, n_component_lim):
		"""
        Find the GMM that best fit the data in x using Bayesian information criterion.
        """
		min_comp = n_component_lim[0]
		max_comp = n_component_lim[1]

		lowest_BIC = np.inf
		best_GMM_ind = 0
		all_GMMs = []
		counter = 0
		for i_comp in range(min_comp, max_comp + 1):
			all_GMMs.append(GaussianMixture(i_comp))
			all_GMMs[-1].fit(x)
			BIC = all_GMMs[-1].bic(x)
			if BIC < lowest_BIC:
				lowest_BIC = BIC
				best_GMM_ind = counter
			counter += 1

		best_GMM = all_GMMs[best_GMM_ind]
		return best_GMM

	def _GMM_density(self, GMM, x):
		n_dims = x.shape[1]
		density = 0.0
		for i_component in range(GMM.weights_.shape[0]):
			density += GMM.weights_[i_component] * multivariate_normal.pdf(x, mean=GMM.means_[i_component],
																		   cov=GMM.covariances_[
																				   i_component] + 1e-15 * np.eye(n_dims))
		return density

	def _Monte_Carlo_MI(self, x, y, n_component_lim, entropy_x=[], GMM_x=None):
		"""
		Estimate MI between x and y.
		Use _Monte_Carlo_entropy to estimate entropies.
		"""
		entropies_y = np.zeros(self.n_split_sets_ + 1)
		entropies_x = np.zeros(self.n_split_sets_ + 1)
		MI_all = np.zeros(self.n_split_sets_ + 1)

		nPoints = x.shape[0]
		xy = np.zeros((nPoints, 2))
		xy[:, 0] = x[:, 0]
		xy[:, 1] = y[:, 0]
		
		# Compute full 2D density
		GMM = self._estimate_GMM(xy, n_component_lim)
		density_xy = self._GMM_density(GMM,xy)
		entropy_xy = self._Monte_Carlo_entropy(density_xy)

		all_sampled_points, PDFs_xy, entropies_xy = self._compute_split_PDFs(GMM, nPoints)

		# Compute the x densities and entropies of sampled points
		if GMM_x is None:
			GMM_x = self._estimate_GMM(x, n_component_lim)

		if entropy_x == []:
			density_x = self._GMM_density(GMM_x,x)
			entropy_x = self._Monte_Carlo_entropy(density_x)

		PDFs_x = []
		for i_set in range(self.n_split_sets_):
			sampled_points = all_sampled_points[i_set][:, 0]
			pdf_x = self._GMM_density(GMM_x,sampled_points[:, np.newaxis])
			PDFs_x.append(pdf_x)
			entropies_x[i_set] = self._Monte_Carlo_entropy(pdf_x)

		# Compute the y densities and entropies of sampled points
		GMM_y = self._estimate_GMM(y, n_component_lim)
		density_y = self._GMM_density(GMM_y,y)
		entropy_y = self._Monte_Carlo_entropy(density_y)

		PDFs_y = []
		for i_set in range(self.n_split_sets_):
			sampled_points = all_sampled_points[i_set][:, 1]
			pdf_x = self._GMM_density(GMM_y,sampled_points[:, np.newaxis])
			PDFs_y.append(pdf_x)
			entropies_y[i_set] = self._Monte_Carlo_entropy(pdf_x)

		entropies_xy[self.n_split_sets_] = entropy_xy
		entropies_x[self.n_split_sets_] = entropy_x
		entropies_y[self.n_split_sets_] = entropy_y

		for i_set in range(self.n_split_sets_ + 1):
			MI_all[i_set] = entropies_x[i_set] + entropies_y[i_set] - entropies_xy[i_set]

		MI = np.mean(MI_all)
		return MI, MI_all

	def _MI_loop(self,i):
		use_cmap = False
		calc_MI = True
		if self.cmap_ != []:
			use_cmap = True
		
		print(str(i+1)+'/'+str(self.n_resid_vec_[self.i_block_,0]+self.n_resids_i))
		
		if self.lag_time_ > 0:
			x = self.fluctuations_[0:-self.lag_time_:1,i,np.newaxis]
		else:
			x = self.fluctuations_[:,i,np.newaxis]

		GMM_x = self._estimate_GMM(x, self.n_component_lim_)
		density_x = self._GMM_density(GMM_x,x)
		entropy_x = self._Monte_Carlo_entropy(density_x)
		
		for j in range(self.n_resid_vec_[self.i_block_,1],self.n_resid_vec_[self.i_block_,1]+self.n_resids_j):
			
			#if np.mod(j,200)==0:
			#	print('('+str(i+1)+','+str(j+1)+')'+'/'+'('+str(self.n_resid_vec_[self.i_block_,0]+self.n_resids_i)+','+str(self.n_resid_vec_[self.i_block_,1]+self.n_resids_j)+')')
			
			if use_cmap:
				calc_MI = False
				if self.cmap_[i,j] > 1e-7:
					calc_MI=True
			
			if calc_MI:
				if self.lag_time_ > 0:
					y = self.fluctuations_[self.lag_time_::,j,np.newaxis]
				else:
					y = self.fluctuations_[:,j,np.newaxis]

				tmpMI, tmp_all_MIs = self._Monte_Carlo_MI(x, y, self.n_component_lim_, entropy_x, GMM_x)
				
				self.MIs_[i-self.n_resid_vec_[self.i_block_,0],j-self.n_resid_vec_[self.i_block_,1]] = tmpMI
				for i_set in range(self.n_split_sets_):
					self.all_MIs_[i_set][i-self.n_resid_vec_[self.i_block_,0],j-self.n_resid_vec_[self.i_block_,1]] = tmp_all_MIs[i_set]
			
		return
	
	def _local_alignment_CAs(self,traj):
		nFrames = int(traj.n_frames)
		self.topology_ = traj.topology
		# Store residue centroid points
		print('Store protein residue (locally aligned) centroid points.')
		self.points_ = np.zeros((nFrames,self.n_cols_*self.n_residues_,3))
		
		# Store residue C_alphas
		for i in range(0, self.n_cols_*self.n_residues_):
			sys.stdout.write("\r"+str(i+1)+'/'+str(self.n_cols_*self.n_residues_)+' ')			
			sys.stdout.flush()	
			
			tmp_resid_traj_ind = traj.topology.select('protein and resid ' + str(i) + ' and !(type H)')
			
			min_ind = i-self.alignment_residue_range_
			max_ind = i+self.alignment_residue_range_
			if min_ind < 0:
				min_ind = 0
			
			if max_ind > self.n_cols_*self.n_residues_-1:
				max_ind = self.n_cols_*self.n_residues_-1
			
			# Extract C-alphas of +-2 residues around the current residue
			tmp_CA_traj_ind = traj.topology.select('protein and resid ' + str(min_ind) +' to '+str(max_ind)+ ' and name CA')
			
			# Align tracjectory to C-alphas around current residue
			traj.superpose(traj,frame=0,atom_indices=tmp_CA_traj_ind)
			
			# Extract current residue
			tmpPoints = traj.atom_slice(tmp_resid_traj_ind).xyz
			self.points_[::,i,::] = np.mean(tmpPoints,axis=1)
		return
	
	def _global_alignment_CAs(self,traj):
		nFrames = int(traj.n_frames)
		n_residues = int(traj.n_residues)

		# Structurally align the frames on C-alphas (global alignment)	
		tmp_CA_traj_ind = traj.topology.select('protein and name CA')
		if len(tmp_CA_traj_ind)>0:
			traj.superpose(traj, frame=0, atom_indices=tmp_CA_traj_ind)
		else:
			print('Superposing on protein heavy atoms because no C-alphas are found.')
			traj.superpose(traj, frame=0, atom_indices=traj.topology.select('protein and !(type H)'))
		
		self.topology_ = traj.topology
		self.protein_points_ = traj.xyz
		self.points_ = np.zeros((nFrames, n_residues, 3))
		
		# Store residue centroid points
		print('Store residue (globally aligned) centroid points. ')
		for i in range(0, n_residues):
			if np.mod(i, 100) == 0:
				print(str(i+1)+'/'+str(n_residues))
			inds = self.topology_.select('resid ' + str(i))

			tmp_points = self.protein_points_[:, inds.astype(int), :]
			self.points_[:, i, :] = np.mean(tmp_points, axis=1)
		return
		
	def _get_fluctuations(self):
		n_frames = int(self.points_.shape[0])
		n_residues = self.points_.shape[1]
		self.fluctuations_ = np.zeros((n_frames, n_residues))
		for i in range(0, n_residues):
			x1 = self.points_[:,i,:]
			x1 = x1-np.mean(x1,axis=0)	# fluctuation
			self.fluctuations_[:,i] = np.linalg.norm(x1,axis=1)	# distance from equilibrium position
		return

	def _add_interactors_to_cmap(self, additional_interactor_protein_contacts):
		int_prot_cmap = np.load(additional_interactor_protein_contacts)
		new_cmap = np.zeros((int_prot_cmap.shape[1],int_prot_cmap.shape[1]))
		n_protein_residues = self.cmap_.shape[0]

		# Add protein contacts
		new_cmap[0:n_protein_residues,0:n_protein_residues] = np.copy(self.cmap_)
		# Add interactor contacts
		new_cmap[n_protein_residues::,:] = np.copy(int_prot_cmap)
		new_cmap[:,n_protein_residues::] = np.copy(int_prot_cmap.T)

		# Overwrite global contact matrix
		self.cmap_ = np.copy(new_cmap)
		return

	def _add_interactors(self, additional_interactor_fluctuations, additional_interactor_protein_contacts):
		print('Adding interactors to MI analysis.')

		interactor_flucts = np.load(additional_interactor_fluctuations)
		n_protein_residues = self.fluctuations_.shape[1]
		n_interactors = interactor_flucts.shape[1]
		new_flucts = np.zeros((int(self.points_.shape[0]),n_protein_residues + n_interactors))

		# Set protein fluctuations
		new_flucts[:,0:n_protein_residues] = np.copy(self.fluctuations_)
		# Set interactor fluctuations
		new_flucts[:,n_protein_residues::] = np.copy(interactor_flucts)

		# Overwrite global fluctuation matrix
		self.fluctuations_ = np.copy(new_flucts)

		# Update contact map
		if self.cmap_ != []:
			if additional_interactor_protein_contacts == '':
				self.cmap_ = []
				print('Warning: No interactor-protein contact map supplied => estimating MI between all residue pairs.')
			else:
				self._add_interactors_to_cmap(additional_interactor_protein_contacts)
				if self.cmap_.shape[0] != self.cmap_.shape[1] or self.cmap_.shape[0] != self.fluctuations_.shape[1]:
					print('Error: Contact map and fluctuation matrix contains different amount of residues.')
					print('Contact map shape: '+str(self.cmap_.shape))
					print('Fluctuation matrix shape: '+str(self.fluctuations_.shape))
					sys.exit(0)

		# Update #residues per block
		self.n_residues_ = int((n_protein_residues + n_interactors)/self.n_cols_)
		return

	def _set_block_start_end_resids(self):
		"""
		Set the start and end residues of each matrix block.
		:return:
		"""

		# Set start and end residues of each block
		self.n_resid_vec_ = np.zeros((int(self.n_blocks_), 2), int)
		counter = 0
		iNResids = 0
		if self.lag_time_ == 0:
			for i in range(self.n_cols_):
				jNResids = iNResids
				for j in range(i, self.n_cols_):
					self.n_resid_vec_[counter, 0] = iNResids
					self.n_resid_vec_[counter, 1] = jNResids
					jNResids += self.n_residues_
					counter += 1
				iNResids += self.n_residues_
		else:
			for i in range(self.n_cols_):
				jNResids = 0
				for j in range(self.n_cols_):
					self.n_resid_vec_[counter, 0] = iNResids
					self.n_resid_vec_[counter, 1] = jNResids
					jNResids += self.n_residues_
					counter += 1
				iNResids += self.n_residues_

		return

	def _residue_2_residue_MI(self):
		""" 
		Compute the MI correlation map of one protein. Residue vs. residue correlations.
		The MI is computed on fluctuations around the mean position of the side-chain centroids.
		"""

		self.n_resids_i = self.n_residues_
		self.n_resids_j = self.n_residues_

		# Adjust self.n_resids_i for uneven number of residues
		if self.n_resid_vec_[self.i_block_,0] == np.max(self.n_resid_vec_[:,0]):
			self.n_resids_i = self.fluctuations_.shape[1] - self.n_resid_vec_[self.i_block_,0]
			print('New #vertical residues in block: '+str(self.n_resids_j))

		if self.n_resid_vec_[self.i_block_,1] == np.max(self.n_resid_vec_[:,1]):
			self.n_resids_j = self.fluctuations_.shape[1] - self.n_resid_vec_[self.i_block_,1]
			print('New #horizontal residues in block: '+str(self.n_resids_j))

		self.MIs_ = np.zeros((self.n_resids_i, self.n_resids_j))
		# Store all MIs so that proper standard error can be computed on everything
		self.all_MIs_ = []
		for i_set in range(self.n_split_sets_):
			self.all_MIs_.append(np.zeros((self.n_resids_i, self.n_resids_j)))
		
		# Compute mutual informations
		print("Constructing MI correlation map")
		start = time.time()

		# Parallel version of for-loop
		Parallel(n_jobs=self.n_cores_, backend="threading")(delayed(unwrap_MI_loop)(i) for i in zip([self]*self.n_resids_i,range(self.n_resid_vec_[self.i_block_,0],self.n_resid_vec_[self.i_block_,0]+self.n_resids_i)))
		
		t1 = time.time()-start
		
		print('time python:'+ str(t1))
		print('Writing results to files.')

		if not(self.do_diagonal_):
			np.save(self.out_directory_ + 'res_res_MI_part_'+ str(self.i_block_)+ self.file_end_name_ + '.npy',self.MIs_)
			for i_set in range(self.n_split_sets_):
				np.save(self.out_directory_ + 'res_res_MI_' + str(self.i_block_)+ 'bootstrap_' + str(i_set) + '_' +  self.file_end_name_ + '.npy',self.all_MIs_[i_set])
		else:
			np.save(self.out_directory_ + 'diagonal_MI_' + self.file_end_name_ + '.npy', np.diag(self.MIs_))
			print('Diagonal extracted and written to file: ' + self.out_directory_ + 'diagonal_MI_' + self.file_end_name_ + '.npy')
			for i_set in range(self.n_split_sets_):
				np.save(self.out_directory_ + 'diagonal_MI_bootstrap_' + str(i_set) + '_' + self.file_end_name_ + '.npy', np.diag(self.all_MIs_[i_set]))

		print()
		print('Data written to files.')
		return


	def run(self):
		
		init = MD_init.MD_initializer()
		
		# Initialize trajectory
		traj = init.initialize_trajectory(**self.traj_init_kwargs_)
		self.out_directory_ = init.out_directory_

		# Keep only query (default: protein (heavy) ) atoms
		protein_traj_inds = traj.topology.select(self.query_)
		traj = traj.atom_slice(protein_traj_inds)
		print(traj)
		
		# Number of residues in the column of one block. The block has size [nResidues x nResidues] 
		self.n_residues_ = int(traj.n_residues/self.n_cols_)

		if self.contact_map_file_ != '':
			self.cmap_ = np.loadtxt(self.contact_map_file_)
			# Normalize the contact map in case it is not binary
			self.cmap_ = self.cmap_/np.max(self.cmap_)
			if len(self.cmap_.shape) == 1:
				self.cmap_ = squareform(self.cmap_)


		# Align the protein residues
		if self.do_local_alignment_:
			self._local_alignment_CAs(traj)
		else:
			self._global_alignment_CAs(traj)

		# Get fluctuations of protein residues
		self._get_fluctuations()

		# Add potential additional interactors (input files should include already computed fluctuations and contact map to protein and interactors)
		if self.additional_interactor_fluctuations_ != '':
			self._add_interactors(self.additional_interactor_fluctuations_, self.additional_interactor_protein_contacts_)

		if self.do_diagonal_:
			self.cmap_ = np.eye(self.n_residues_*self.n_cols_)

		# Set start and end residues of each block
		self._set_block_start_end_resids()

		# Construct the MI-matrix of current block
		self._residue_2_residue_MI()
