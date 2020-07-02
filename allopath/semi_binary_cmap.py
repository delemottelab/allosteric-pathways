import sys
import os
import time

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(__file__))

sys.path.append(current_dir)

import mdtraj as md
import numpy as np
from scipy import stats
from scipy.spatial.distance import squareform, pdist, cdist
from joblib import Parallel, delayed
import MD_init
from cmap_utils import *

def unwrap_cmap_semi_bin_loop(arg,**kwarg):
	return ContactMap.distance_matrix_semi_bin_loop(*arg,**kwarg)

def unwrap_cmap_semi_bin_all_frames_loop(arg,**kwarg):
	return ContactMap.distance_matrix_semi_bin_all_frames_loop(*arg,**kwarg)

class ContactMap():
	
	def __init__(self, topology_file,trajectory_files='',dt=1, n_cores=4, out_directory='', file_label='',
				 multiple_trajectories=False, trajectory_file_directory='', cutoff=0.45, query='protein and !(type H)',
				 start_frame=0,end_frame=-1, ref_cmap_file='', per_frame=False, std_dev=0.138):

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

		# Initialize trajectory
		self.traj_ = init.initialize_trajectory(**traj_init_kwargs)

		self.out_directory_ = init.out_directory_
		self.file_end_name_ = file_label
		self.n_residues_ = int(self.traj_.n_residues)
		self.n_cores_ = int(n_cores)

		# Set reference cmap
		self.ref_cmap_ = None
		if len(ref_cmap_file) > 0:
			print('Setting average contact map.')
			self.ref_cmap_ = squareform(np.loadtxt(ref_cmap_file))

		self.start_frame_ = int(start_frame)
		self.end_frame_ = int(end_frame)

		self.per_frame_ = per_frame
		self.cutoff_ = cutoff
		self.std_dev_ = std_dev
		self.query_ = query

		if self.per_frame_:
			print('Start frame: ' + str(start_frame))
			print('End frame: ' + str(end_frame))

		self.cmap = []
		return

	def distance_matrix_semi_bin_loop(self, i):
		# Compute a semi-binary contact map. Residue pair within the cutoff (4.5 angstrom) is a contact. Outside, the "degree" of contact decreases with a gaussian (for smoothness).
		sys.stdout.write('\rResid: '+str(i + 1) + '/' + str(self.n_residues_))
		sys.stdout.flush()
		for j in range(i + 1, self.n_residues_):

			atom_ind1 = self.atom_inds_[i]
			atom_ind2 = self.atom_inds_[j]

			atom_pairs = get_atom_pairs(atom_ind1, atom_ind2)

			distances = md.compute_distances(self.traj_, atom_pairs, periodic=False)
			if len(distances) == 0:
				print('The chosen residue does not exist!')
				continue

			min_distances = np.min(distances, axis=1)
			semi_gaussians = semi_Gaussian_kernel(min_distances, cutoff=self.cutoff_, std_dev=self.std_dev_)

			# The distance between residues is the kernel of min distance between all heavy atoms. Take mean over all frames.
			self.cmap_[i, j] = np.mean(semi_gaussians, axis=0)
			self.cmap_[j, i] = np.mean(semi_gaussians, axis=0)

		return

	def distance_matrix_semi_bin_all_frames_loop(self,i):
		# Compute a semi-binary contact map per frame. Residue pair within the cutoff (4.5 angstrom) is a contact.
		# Outside, the "degree" of contact decreases with a gaussian (for smoothness).
		n_frames = self.traj_.n_frames
		sys.stdout.write('\r Frame: '+str(self.current_frame_+1)+'/'+str(n_frames)+', Resid: '+str(i+1)+'/'+str(self.n_residues_)+'      ')
		sys.stdout.flush()

		tmp_traj = self.traj_[self.current_frame_]

		for j in range(i+1,self.n_residues_):
			
			do_pair = True
			if self.ref_cmap_ is not None:
				do_pair = self.ref_cmap_[i,j] > 1e-8

			if do_pair:

				atom_ind1 = self.atom_inds_[i]
				atom_ind2 = self.atom_inds_[j]
			
				atom_pairs = get_atom_pairs(atom_ind1, atom_ind2)
			
				distances = md.compute_distances(tmp_traj, atom_pairs, periodic=False)
				if len(distances) == 0:
					print('The chosen residue does not exist!')
					continue
			
				min_distances =	np.min(distances,axis=1)		
				semi_gaussians = semi_Gaussian_kernel(min_distances, cutoff=self.cutoff_, std_dev=self.std_dev_)

				# The distance between residues is the kernel of min distance between
				# all heavy atoms. Take mean over all frames.
				self.cmap_[i,j] = semi_gaussians
				self.cmap_[j,i] = semi_gaussians
		
		return 
	

	def compute_semi_bin_cmap(self, start_frame, end_frame):

		n_frames = int(self.traj_.n_frames)
		self.atom_inds_ = []

		# Do atom selections, save list with all heavy atoms.
		for i in range(0,self.n_residues_):
			query = "resid " + str(i)
			tmpInd = self.traj_.topology.select(query)
			self.atom_inds_.append(tmpInd)

		self.n_residues_ = int(len(self.atom_inds_))	

		if self.per_frame_:
			print('Compute per-frame contact map.')
			if start_frame == -1:
				start_frame = 0

			if end_frame == -1:
				end_frame = n_frames

			for i_frame in range(start_frame, end_frame+1):
				self.cmap_ = np.zeros((self.n_residues_,self.n_residues_))
				self.current_frame_ = i_frame

				Parallel(n_jobs=self.n_cores_, backend="threading")(delayed(unwrap_cmap_semi_bin_all_frames_loop)(i) for i in zip([self]*self.n_residues_,range(self.n_residues_)))

				# Save distance matrices to .npy file
				np.save(self.out_directory_ + 'distance_matrices_semi_bin_frame_'+str(i_frame)+'_'+self.file_end_name_+'.npy', squareform(self.cmap_))
		else:
			self.cmap_ = np.zeros((self.n_residues_, self.n_residues_))
			Parallel(n_jobs=28, backend="threading")(
				delayed(unwrap_cmap_semi_bin_loop)(i) for i in zip([self] * self.n_residues_, range(self.n_residues_)))

			# Save distance matrix to file
			np.savetxt(self.out_directory_ + 'distance_matrix_semi_bin_' + self.file_end_name_ + '.txt',
					   squareform(self.cmap_))
		print()
		print('Data saved to file!')
		return
	
	
	def run(self):
		# Construct the average distance matrix (residue-residue) with the distance between residues defined as the
		# minimum distance between all heavy atoms of the two residues.
		atom_indices = self.traj_.topology.select(self.query_)
		self.traj_.atom_slice(atom_indices, inplace=True)
		print('Computing average semi binary sidechain contact map with cutoff = '+str(self.cutoff_)+' and std_dev = '+str(self.std_dev_))
		print('Atom query: ' + self.query_)
		print(self.traj_)

		self.n_residues_ = self.traj_.n_residues
		self.compute_semi_bin_cmap(self.start_frame_, self.end_frame_)
		return
