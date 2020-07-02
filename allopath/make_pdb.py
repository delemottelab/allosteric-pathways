import os
import sys
import argparse
import numpy as np
from biopandas.pdb import PandasPdb

def project_current_flow(pdb_file, current_flow_file,  out_directory='', file_label='', max_min_normalize=False,
						 interactor_atom_inds_file=None, data_name='current_flow', remove_uncertain=False, std_CF_file=''):
	"""
	Write current flow into beta column of pdb file. Assumes that the PDB has the same interactor order
	as the reference frame when assigning interaction indices.
	"""

	if not os.path.exists(out_directory):
			os.makedirs(out_directory)

	out_file = out_directory + data_name +'_'+file_label+'.pdb'
	
	interactor_atom_indices=None
	if interactor_atom_inds_file is not None:
		interactor_atom_indices = np.load(interactor_atom_inds_file, allow_pickle=True)

	current_flow = np.load(current_flow_file)

	if remove_uncertain and len(std_CF_file) > 0:
		std_current_flow = np.load(std_CF_file)
		min_uncertainty = current_flow-std_current_flow
		current_flow[min_uncertainty<0] = 0

	if max_min_normalize:
		current_flow -= current_flow.min()
		current_flow /= current_flow.max()
	else:
		current_flow *= 10

	pdb = PandasPdb()
	pdb.read_pdb(pdb_file)

	n_interactors = 0
	if interactor_atom_indices is not None:
		n_interactors = len(interactor_atom_indices)

	n_protein_resids = current_flow.shape[0]-n_interactors

	atoms = pdb.df['ATOM']
	resid_counter = -1
	current_resid = -1
	print('Writing current flow to beta column - protein atoms.')
	for i_atom, line in atoms.iterrows():
		res_seq = int(line['residue_number'])

		if resid_counter < n_protein_resids-1:
			if res_seq != current_resid:
				resid_counter += 1
				current_resid = res_seq
		else:
			break
		# Set protein current flows
		atoms.at[i_atom, 'b_factor'] = current_flow[resid_counter]
	
	if n_interactors > 0:
		print('Writing current flow to beta column - interactor atoms.')
	
	# Set interactor current flows
	for i_int in range(n_interactors):
		sys.stdout.write(
			"\r -> Interactor index: " + str(i_int+1) + '/' + str(n_interactors) + ' ')
		sys.stdout.flush()
		for i_atom in interactor_atom_indices[i_int]:
			atoms.at[i_atom, 'b_factor'] = current_flow[i_int+n_protein_resids]
	
	print()
	print('Writing to file.')
	pdb.to_pdb(path=out_file, records=None, gz=False, append_newline=True)
	print('File written to: '+out_file)
	return

if __name__ == "__main__":
	parser = argparse.ArgumentParser(epilog='Writing current flow with protein and potential other interactors '
											'to beta column in pdb file.')
	parser.add_argument('-top','--PDB_file',help='PDB file with protein (and potential interactors).')
	parser.add_argument('-od','--out_directory',help='Directory where output is written.',default='')
	parser.add_argument('-f','--current_flow',help='File with current flow data.',default='current_flow_.npy')
	parser.add_argument('-f_std','--std_current_flow',help='File with current flow data.',default='std_current_flow_.npy')
	parser.add_argument('-norm','--max_min_normalize',help='Normalize data between zero and one.',action='store_true')
	parser.add_argument('-rm_uncer','--remove_uncertain',help='Set uncertain data to zero (standard deviation crossing zero).',action='store_true')
	parser.add_argument('-fe','--file_label',help='File end name (file label).',default='')
	parser.add_argument('-int_atom_inds', '--interactor_atom_inds',
						help='File with interactor atom indices.')

	args = parser.parse_args()
	if not os.path.exists(args.out_directory):
			os.makedirs(args.out_directory)
	out_file = args.out_directory + 'current_flow_'+args.file_label+'.pdb'
	
	interactor_atom_inds = None
	if args.interactor_atom_inds is not None:
		interactor_atom_inds = np.load(args.interactor_atom_inds, allow_pickle=True)

	project_current_flow(args.PDB_file, args.current_flow,  args.out_directory, args.file_label, args.max_min_normalize,
						 args.interactor_atom_inds, remove_uncertain=args.remove_uncertain, std_CF_file=args.std_current_flow)
