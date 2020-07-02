import mdtraj as md
import numpy as np
import os

class MD_initializer():
	
	def __init__(self):
		return
	
	def initialize_trajectory(self, topology_file='', trajectory_files='', dt=1, trajectory_file_directory='', out_directory='', file_label='',
							  multiple_trajectories=False):

		# Get command line input parameters
		self.out_directory_ = out_directory
		self.file_end_name__ = file_label
		
		# Put / at end of out directory if not present. Check so that folder extsts, otherwise construct it.
		if out_directory !='':
			if out_directory[-1] != '/':
				out_directory += '/'
		
			if not os.path.exists(out_directory):
				os.makedirs(out_directory)

		print('Output directory: ' + out_directory)
		self.out_directory_ = out_directory

		if trajectory_files=='' and trajectory_file_directory=='':
			print('You need to supply a trajectory or a directory with trajectories:')
			print('trajectory_files/trajectory_file_directory')

		if not(multiple_trajectories) or not(trajectory_file_directory==''):
			# Get the main trajectory
			traj = self._get_single_trajectory(topology_file[0], trajectory_files, dt)
		else:
			# Get list of trajectories
			traj = self._get_multiple_trajectories(topology_file, trajectory_files, trajectory_file_directory, dt)

		print('# Frames after stride: '+str(traj.n_frames))
		print('File end name: ' + self.file_end_name__)
		print('-----------------------------------------')		
		print()

		return traj


	def _get_single_trajectory(self, topology_file, trajectory_files, dt):
		trajectory_string = "Trajectory files: "
		topology_string = "Topology files: " + topology_file
	
		# Print file names in string
		for i in range(0,len(trajectory_files)):
			trajectory_string += trajectory_files[i] + " "
		
		print(topology_string)
		print(trajectory_string)
		print('Stride: '+str(int(dt)))

		if len(trajectory_files)==1:
			traj = md.load(trajectory_files[0], top = topology_file, stride=int(dt))
		elif len(trajectory_files)==0:
			traj = md.load(topology_file)		
		else:
			# Stack trajectories
			traj = md.load(trajectory_files[0], top = topology_file, stride=1)
		
			for i in range(1,len(trajectory_files)):
				print("Stacking extra trajectory: " + str(i))
			
				trajectory = md.load(trajectory_files[i], top = topology_file, stride=1)
				traj = traj.join(trajectory)
				print("Number of frames: " + str(traj.n_frames))
		
			# Keep every dt:th frame (dt=1 by default)
			traj = traj[0::int(dt)]
		return traj

	def _get_multiple_trajectories(self, topology_files, trajectory_files, trajectory_file_directory, dt):
		# Load multiple trajectories
		trajs = []
		do_separate_topologies = (len(topology_files)>1)

		# Create "trajs" with only .pdb files (topology)
		if len(trajectory_files)==0 and do_separate_topologies:
			for i in range(len(topology_files)):
				trajs.append(self._get_single_trajectory(topology_files[i], [],1))

		# Create "trajs" with both topology and trajectory
		if trajectory_file_directory == '':
			# Loop over pre-specified trajectories
			for i in range(len(trajectory_files)):
				if do_separate_topologies:
					trajs.append(self._get_single_trajectory(topology_files[i], [trajectory_files[i]], dt))
				else:
					trajs.append(self._get_single_trajectory(topology_files[0], [trajectory_files[i]], dt))

		else:
			# Loop over all trajectory files within the specified directory
			counter = 0
			for trajectory_file in os.listdir(trajectory_file_directory):
				if trajectory_file.endswith(".dcd") or trajectory_file.endswith(".xtc"): 
					trajs.append(self._get_single_trajectory(topology_files[0], [trajectory_file_directory+trajectory_file], dt))
		
		return trajs
