# This file contains all the necessary steps for current flow analysis of a protein-cofactor 
# trajectory from command-line arguments.
#
# For more information about the steps, functions and inputs, see the jupyter notebook tutorial.
# 
# A. Decide what to do. 
# 	The variables below indicate whether or not a calculation/step should be done.
#	If set to 1, the calculation will be carried out, otherwise not.
#	The steps follow a logical order (more details are found in the jupyter notebook): 
#	1.a Compute the protein contact map.
#	1.b Compute the cofactor contribution to the contact map.
#	2.a Compute cofactor fluctuations.
#	2.b Compute the mutual information (MI) matrix (using the contact map as input).
#	3. Compute the current flow using the contact map and MI matrix as input.
#	4. Plotting the current flow on the beta column in the .PDB (using current flow as input).
#
compute_prot_cmap=1
compute_cofactor_prot_cmap=1
compute_interactor_node_fluctuations=1
compute_MI=1
current_flow=1
project_on_structure=1

# B. Set general input parameters
top_file=input_data/my_system.pdb
trajs=(input_data/system_traj1.dcd input_data/system_traj2.dcd)

n_cores=4
n_bootstraps=10

out_dir=Results_data/
out_dir_MI=${out_dir}MI_data/
file_end_name=my_system
dt=1
n_chains=4 # number of chains in a homomeric protein.

cofactor_domain_sel=input_data/cofactor_domain_selection.txt # Input file with selections for 
														     # cofactor domains (defining interactors 
														     # - see juputer notebook for more info)

source_inds=input_data/inds_sources.txt	# Residue indices of sources used in the current flow
sink_inds=input_data/inds_sinks.txt		# Residue indices of sinks used in the current flow

aux_inds=input_data/auxiliary_prot_inds.txt	# Residue indices of auxiliary subunits. 
											# Used when symmeterizing current flow over 
											# homomeric subunits (chains). 
											# This assumes one auxiliary protein per chain, ie.
											# n_chains identical auxiliary proteins.
											# If there is no auxiliary subunit, just remove 
											# the "-iaux ${aux_inds}" 
											# input to the current flow script.


# C. Call the python scripts
if [ $compute_prot_cmap -eq 1 ] 
then
	# 1.a Compute the protein contact map.
	python run_contact_map.py -top ${top_file} -trj ${trajs[@]} -fe ${file_end_name} -od ${out_dir} -dt ${dt} 
fi


if [ ${compute_cofactor_prot_cmap} -eq 1 ]
then
	# 1.b Compute the cofactor contribution to the contact map.
	python run_cofactor_interactors.py -top ${top_file} -trj ${trajs[@]} -fe ${file_end_name} \
		-od ${out_dir} -dt ${dt} -cofactor_domain_sel ${cofactor_domain_sel} -dt ${dt}

fi


if [ ${compute_interactor_node_fluctuations} -eq 1 ]
then
	# 2.a Compute cofactor fluctuations.
	#
	# By adding the flag "-fluctuations" we will compute the node fluctuations.
	# We also supply the cofactor coordinates and permutation indices computed in the previous step
	# to avoid redoing this.

	python run_cofactor_interactors.py -top ${top_file} -trj ${trajs[@]} \
		-fe ${file_end_name} -od ${out_dir} -dt ${dt} \
		-cofactor_coords ${out_dir}cofactor_interactor_coords_${file_end_name}.npy \
		-cofactor_inds ${out_dir}cofactor_interactor_indices_${file_end_name}.npy \
		-fluctuations
fi


if [ $compute_MI -eq 1 ]
then
	# 2.b Compute the mutual information (MI) matrix (using the contact map as input).
	#
	# Computing MI with 4 blocks per matrix column => 4*(4-1)/2 + 4 = 10 blocks in total.
	#
	# The density is by default estimated with 1 to 5 number of Gaussian components.
	# To change the range of allowed components, use the flag "-GMM_range ${min_comp} ${max_comp}",
	# e.g. "-GMM_range 1 5" corresponds to the default setting.
	
	for((i_block=1; i_block<=10; i_block++))
	do
		echo $i_block
		python run_mutual_information.py -top ${top_file} -trj ${trajs[@]} -fe ${file_end_name} \
			-od $out_dir_MI -dt ${dt} -n ${n_cores} -i_block $i_block -n_blocks_col 4 \
			-cmap ${out_dir}distance_matrix_semi_bin_${file_end_name}.txt -n_splits \
			-aif ${out_dir}interactor_centroid_fluctuations_${file_end_name}.npy \
            -aipc ${out_dir}cofactor_protein_residue_semi_binary_cmap_${file_end_name}.npy \
			$n_bootstraps 
	done
	
	# When the contact map is used as input, we might need to compute the MI the matrix diagonal. 
	# This should only be done if the MI on the diagonal is used to normalize the MIs prior to the current flow analysis.
	python run_mutual_information.py -top ${top_file} -trj ${trajs[@]} -od ${out_dir_MI} \
		-fe ${file_end_name} -n ${n_cores} -i_block 0 -n_blocks_col 1 \
		-aif ${out_dir}interactor_centroid_fluctuations_${file_end_name}.npy \
        -aipc ${out_dir}cofactor_protein_residue_semi_binary_cmap_${file_end_name}.npy \
		-n_splits ${n_bootstraps} -MI_map_diag -dt ${dt}
	
	# Build a complete MI matrix from the blocks (written to compressed format)
	python allopath/from_matrix_blocks.py -f ${out_dir_MI}res_res_MI_part_ \
		-fe ${file_end_name} -od ${out_dir_MI} -n_blocks 10 

fi

if [ $current_flow -eq 1 ]
then
	# 3. Compute the current flow using the contact map and MI matrix as input.
	# 
	# Current flow betweenness is computed by default. To compute current flow closeness,
	# we need to add the flag "-CFC" 
	#
	# Current flow analysis is averaged over subunits (ie. chains) if n_chains > 1
	#
	# Note: Adding the flag "-norm" and input "-f_map_diag ${out_dir_MI}diagonal_MI_${file_end_name}.npy" 
	# would result in symmetric uncertainty normalization of the MI (Witter & Frank, 2005).
	python run_current_flow_analysis.py -f_map ${out_dir_MI}res_res_MI_compressed_${file_end_name}.npy \
	-cmap ${out_dir}distance_matrix_semi_bin_${file_end_name}.txt  \
	-od ${out_dir} -fe ${file_end_name} \
	-f_source ${source_inds} \
	-f_sink ${sink_inds} \
	-iaux ${aux_inds} \
	-aipc ${out_dir}cofactor_protein_residue_semi_binary_cmap_${file_end_name}.npy \
	-n_cores ${n_cores} -n_chains ${n_chains} 
fi

if [ $project_on_structure -eq 1 ]
then
	# 4. Plotting the current flow on the beta column in the .PDB (using current flow as input).
	#
	# Plotting the current flow on the beta-column in the PDB file. Note: It has to be .pdb	
	# Note: Adding the flag "-norm" would scale the current flow values between 0 and 1 in 
	# the generated .pdb.
	python allopath/make_pdb.py -top ${top_file} -f ${out_dir}average_current_flow_${file_end_name}.npy \
		-od ${out_dir}PDBs/ -fe ${file_end_name} \
		-int_atom_inds ${out_dir}cofactor_interactor_atom_indices_${file_end_name}.npy \

fi


