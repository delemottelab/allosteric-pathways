import numpy as np
import mdtraj as md

def get_atom_pairs(inds1, inds2):
	# Construct array with all pairs
	atom_pairs = np.zeros((len(inds1)*len(inds2),2))
	counter = 0
	for k in range(0,len(inds1)):
		for l in range(0,len(inds2)):
			atom_pairs[counter,0] = inds1[k]
			atom_pairs[counter,1] = inds2[l]
			counter += 1
	atom_pairs = atom_pairs[0:counter,::]
	
	return atom_pairs


def semi_Gaussian_kernel(distances, std_dev = 0.138, cutoff = 0.45):
	# Transform distances to semi-binary contacts through a semi-Gaussian kernel.
	# 1.38 angstrom standard deviation => gives 1e-5 weight at 0.8 nm and c = 0.45 nm cutoff..
	# within 4.5 angstrom, the weight is 1.

	# Compute normalizing factor
	cutoff_value = np.exp(-cutoff**2/(2*std_dev**2))

	gaussians = np.exp(-distances**2/(2*std_dev**2))/cutoff_value
	gaussians[distances < cutoff] = 1.0
	return gaussians

