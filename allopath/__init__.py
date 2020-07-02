from .semi_binary_cmap import ContactMap
from .mutual_information import MutualInformation
from .current_flow import CurrentFlow
from .cofactor_interactors import CofactorInteractors
import from_matrix_blocks
import make_pdb

def set_traj_init_parser(parser):
	parser.add_argument('-top','--topology_file',help='Input 1 topology file (.gro, .pdb, etc)',type=str,default='',nargs='+')
	parser.add_argument('-trj','--trajectory_files',help='Input trajectory files (.xtc, .dcd, etc)',nargs='+',default='')
	parser.add_argument('-trjdir','--trajectory_file_directory',help='Input directory with trajectory files '
																	 '(.xtc, .dcd, etc.). This will load all '
																	 'trajectory files in the specified directory.',default='')

	parser.add_argument('-multitraj','--multiple_trajectories',help='Flag for reading multiple trajectories. '
																	'Either put as many arguments in -top as in -trj, '
																	'or use one -top for multiple -trj, or use multiple '
																	'-top and leave -trj empty.',action='store_true')

	parser.add_argument('-fe','--file_label',type=str,help='Output file end name (optional)', default='')
	parser.add_argument('-od','--out_directory',type=str,help='The directory where data should be saved (optional)',default='')
	
	parser.add_argument('-dt','--dt',help='Keep every dt frame (stride).',default=1, type=int)
	return parser


def set_cmap_parser(parser):
	parser.add_argument('-q','--query',help='Query for analyzing trajectory, e.g. -protein and !(type H)-',default='protein and !(type H)')
	
	parser.add_argument('-sf','--start_frame',help='Defines which frame to start calculations from.',default=0)
	parser.add_argument('-ef','--end_frame',help='Defines which frame to end calculations at.',default=-1)
	
	parser.add_argument('-n','--n_cores',help='Number of cores. ',default=4,type=int)
	parser.add_argument('-ref','--ref_cmap',help='File with reference cmap (e.g. average over all frames). Is used to make computations sparse/speed up calculations when computing multiple frame contact maps.',default='')
	parser.add_argument('-per_frame', '--per_frame', help='Computing contact map per frame instead of averaging.', action='store_true')
	parser.add_argument('-cutoff', '--cutoff', help='Cutoff value for binary residue-cofactor contacts.', default=0.45, type=float)
	parser.add_argument('-std_dev', '--std_dev', help='Standard deviation value for binary residue-cofactor contacts.', default=0.138, type=float)
	return parser


def set_MI_parser(parser):
	parser.add_argument('-MI_map_diag', '--do_diagonal',
						help='Flag for computing diagonal of residue-residue mutual information (optional)',
						action='store_true')

	parser.add_argument('-q','--query',help='Query for analyzing trajectory, e.g. -protein and !(type H)-',default='protein and !(type H)')

	parser.add_argument('-cmap', '--contact_map_file', help='File with binary contact map. Used to speed up calculations by only computing MI where there are contacts.', default='')
	parser.add_argument('-n', '--n_cores', help='Number of threads', default=-1, type=int)

	# Arguments controlling alignment and time-lag options
	parser.add_argument('-local', '--do_local_alignment', help='Flag for performing local alignment before calculating the fluctuations.',
						action='store_true')
	parser.add_argument('-lag', '--lag_time', help='Lag time in time-lagged MI.', default=0, type=int)

	parser.add_argument('-align_resid_range', '--alignment_residue_range',
						help='The number of residues above and below the current to align on when using local alignment. Ex: n=2 => alignment residues = resid i-2 to i+2.',
						default=2, type=int)

	# Arguments for splitting the calculations to various splits
	parser.add_argument('-i_block', '--i_block', help='The part of the matrix that should be calculated',
						default=0, type=int)
	parser.add_argument('-n_blocks_col', '--n_matrix_block_cols',
						help='Number of blocks of the column of the MI matrix. Example: 4 blocks => 10 parts (upper triangle + diagonal)',
						default=1, type=int)

	# Arguments for estimated densities
	parser.add_argument('-GMM_range','--n_components_range', help='The minimum and maximum number of components to estimate GMM densities (default=[1,5].', nargs='+' ,default=[1,4], type=int)
	parser.add_argument('-n_splits', '--n_split_sets',
						help='Number of sampled sets with the same size as the original data set to use for estimating entropy and errors.',
						default=0, type=int)


	# Extra arguments for including the environment
	parser.add_argument('-aif','--additional_interactor_fluctuations',help='.npy file containing the fluctuations of additional interactors.',default='')
	parser.add_argument('-aipc','--additional_interactor_protein_contacts',help='.npy file containing the additional interactor-(protein+interactor) contacts. '
																				'Elements should be ordered with protein first, interactors second. '
																				'Used together with supplied protein contact map to sparsify the MI matrices.',default='')

	return parser

def set_CF_parser(parser):
	parser.add_argument('-f_map', '--similarity_map_file',
						help='File containing the compressed (squareform) similarity map (text file with e.g. MI or (absolute) Pearson correlation between residues movements)', default='')

	parser.add_argument('-cmap', '--contact_map_file',
						help='Binary (or semi-binary weight) protein-residue contact map (compressed, squareform) used to sparsify the similarity map (.txt, .npy). '
							 'If multiple contact maps are supplied, one current flow profile per cmap will be calculated.',
						nargs='+')

	parser.add_argument('-f_sources', '--file_sources', help='File containing the source indices.',
						default='')
	parser.add_argument('-f_sinks', '--file_sinks', help='File containing the sink indices.', default='')

	parser.add_argument('-f_map_diag', '--similarity_map_diagonal_filename',
						help='File containing the diagonal of the similarity (text file)',
						default='')

	parser.add_argument('-norm', '--normalize_similarity_map',
						help='Flag to normalize similarity map with symmetric uncertainty.',
						action='store_true')

	parser.add_argument('-iaux', '--auxiliary_protein_indices',
						help='File with auxiliary protein indices (used for treating such parts of the '
							 'network separately in postprocessing chain averaging).', default='')
	parser.add_argument('-aipc', '--additional_interactor_protein_contacts',
						help='File with contact map between additional interactors (e.g. cofactors) and the protein. '
							 '(not supported for the per-frame current flow).', default='')

	parser.add_argument('-n_chains', '--number_of_chains',
						help='Number of chains. Used to obtain averaged current flow.',
						default='1', type=int)

	parser.add_argument('-sf', '--start_frame',
						help='Start frame in the list of supplied contact maps (used only for numbering the output when supplying multiple contact maps).',
						default=0, type=int)

	parser.add_argument('-n_cores', '--n_cores', help='Number of jobs to run in parallel (default=1)', default=1,
						type=int)
	parser.add_argument('-edges', '--edge_CF', help='Flag to compute edge current flows.', action='store_true')

	parser.add_argument('-CFC', '--compute_current_flow_closeness', help='Flag to compute current flow closeness instead of betweenness.', action='store_true')

	parser.add_argument('-od', '--out_directory', help='Folder to save data in.', default='')
	parser.add_argument('-fe', '--file_label', help='End of file name (can be used for labeling different datasets).', default='')
	parser.add_argument('-cheap_write', '--cheap_write',
						help='Flag for not writing files in intermediate steps, and condensing to one subunit each (in case of multi-chain proteins).',
						action='store_true')

	return parser

def set_CI_parser(parser):

	parser.add_argument('-cofactor_domain_sel', '--cofactor_domain_selection',
						help='File with cofactor domain selections (used for extracting atom indices of each cofactor '
							 'domain which will be the interactors).',default='')

	parser.add_argument('-cofactor_inds', '--cofactor_interactor_inds', help='File with cofactor interactor indices (determined with '
																	   'the protein-internal coordinate system).',default='')
	parser.add_argument('-cofactor_atom_inds', '--cofactor_interactor_atom_inds', help='File with atom indices of each '
																				 'cofactor interactor (each interactor '
																				 'is a quasi-rigid domain).',default='')
	parser.add_argument('-cofactor_coords', '--cofactor_interactor_coords', help='File with coordinates of each '
																				 'cofactor interactor.', default='')
	parser.add_argument('-fluctuations','--compute_interactor_node_fluctuations',help='Flag for computing the centroids '
																				 'of each interactor node in all '
																				 'frames (will be used to estimate '
																				 'MI matrix).',action='store_true')

	parser.add_argument('-internal','--internal_coordinates',help='Flag for using internal instead of Cartesian coordinates when computing '
																	'centroids and/or fluctuations. Note that this leads to high ' 		
																	'dimensionality data.',action='store_true')
	parser.add_argument('-cutoff', '--cutoff', help='Cutoff value for binary residue-cofactor contacts.', default=0.45, type=float)
	parser.add_argument('-std_dev', '--std_dev', help='Standard deviation value for binary residue-cofactor contacts.', default=0.138, type=float)
	return parser


def set_cmap_input_args(parser):
	args = parser.parse_args()
	
	in_args=[args.topology_file]
	
	kwargs={	
	'trajectory_files': args.trajectory_files,
	'trajectory_file_directory': args.trajectory_file_directory,
	'file_label': args.file_label,
	'out_directory': args.out_directory,
	'dt': args.dt,
	'multiple_trajectories': args.multiple_trajectories,
	'query': args.query,
	'n_cores': args.n_cores,
	'start_frame': args.start_frame,
	'end_frame': args.end_frame,
	'ref_cmap_file': args.ref_cmap,
	'cutoff': args.cutoff,
	'std_dev': args.std_dev,
	'per_frame': args.per_frame
	}
	
	return in_args, kwargs


def set_MI_input_args(parser):
	args = parser.parse_args()
	
	in_args=[args.topology_file]
	
	kwargs={	
	'trajectory_files': args.trajectory_files,
	'trajectory_file_directory': args.trajectory_file_directory,
	'dt': args.dt,
	'query': args.query,
	'multiple_trajectories': args.multiple_trajectories,
	'contact_map_file': args.contact_map_file,
	'i_block': args.i_block,
	'n_matrix_block_cols': args.n_matrix_block_cols,
	'n_split_sets': args.n_split_sets,
	'file_label': args.file_label,
	'out_directory': args.out_directory,
	'do_diagonal': args.do_diagonal,
	'do_local_alignment': args.do_local_alignment,
	'alignment_residue_range': args.alignment_residue_range,
	'lag_time': args.lag_time,
	'n_cores': args.n_cores,
	'n_components_range': args.n_components_range,
	'additional_interactor_protein_contacts': args.additional_interactor_protein_contacts,
	'additional_interactor_fluctuations': args.additional_interactor_fluctuations
	}
	
	return in_args, kwargs

def set_CF_args(parser):
	args = parser.parse_args()

	in_args=[args.similarity_map_file, args.contact_map_file, args.file_sources, args.file_sinks]
	
	kwargs={
	'file_label': args.file_label,
	'out_directory': args.out_directory,
	'n_chains': args.number_of_chains,
	'n_cores': args.n_cores,
	'compute_edge_CF': args.edge_CF,
	'cheap_write': args.cheap_write,
	'similarity_map_diagonal_filename': args.similarity_map_diagonal_filename,
	'normalize_similarity_map': args.normalize_similarity_map,
	'auxiliary_protein_indices': args.auxiliary_protein_indices,
	'additional_interactor_protein_contacts': args.additional_interactor_protein_contacts,
	'start_frame': args.start_frame,
	'compute_current_flow_closeness': args.compute_current_flow_closeness,
	}
	
	return in_args, kwargs

def set_CI_args(parser):
	args = parser.parse_args()
	
	in_args=[args.topology_file]
	
	kwargs={	
	'trajectory_files': args.trajectory_files,
	'trajectory_file_directory': args.trajectory_file_directory,
	'dt': args.dt,
	'multiple_trajectories': args.multiple_trajectories,
	'cofactor_domain_selection': args.cofactor_domain_selection,
	'cofactor_interactor_inds': args.cofactor_interactor_inds,
	'cofactor_interactor_atom_inds': args.cofactor_interactor_atom_inds,
	'cofactor_interactor_coords': args.cofactor_interactor_coords,
	'compute_interactor_node_fluctuations': args.compute_interactor_node_fluctuations,
	'use_internal_coordinates': args.internal_coordinates,
	'file_label': args.file_label,
	'out_directory': args.out_directory,
	'cutoff': args.cutoff,
	'std_dev': args.std_dev
	}
	
	return in_args, kwargs
