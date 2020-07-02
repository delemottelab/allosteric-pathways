import numpy as np
import argparse
from scipy.spatial.distance import squareform
"""
Post-processing if the MI matrix was computed in blocks. 
These blocks are joined to the full MI matrix and saved to compressed format.
"""

def _read_data(base_file_name, n_blocks, file_label=''):

	filename = base_file_name

	# Read all blocks in a list
	print('Reading all blocks')
	n_cols_per_block = 0
	n_cols_last_block = 0
	all_blocks = []

	for i in range(int(n_blocks)):
		try:
			tmp_block = np.loadtxt(filename + str(i) + file_label+'.txt')
		except:
			tmp_block = np.load(filename + str(i) + file_label+'.npy')
	
		all_blocks.append(tmp_block)
		if i == 0:
			n_cols_per_block = tmp_block.shape[1]
		# Get the last block size, if last block is not rectangular
		if i == n_blocks-1:
			n_cols_last_block = tmp_block.shape[1]
	
	return all_blocks, n_cols_per_block, n_cols_last_block

def _remove_diagonal(full_matrix):
	total_cols = full_matrix.shape[0]
	for i in range(total_cols):
		full_matrix[i,i] = 0
	return full_matrix

def _save_full_MI_matrix(full_matrix, save_folder, file_label):
	print('Writing to files.')
	# Save data
	np.save(save_folder + 'res_res_MI_compressed_'+file_label+'.npy',squareform(full_matrix))
	print('MI matrix written to file: '+save_folder + 'res_res_MI_compressed_'+file_label+'.npy')
	return

def _get_full_matrix(all_blocks, total_cols, n_blocks_row, n_blocks_col, n_cols_per_block, n_cols_last_block,
					 using_lag_time=False):
	print('Building matrix.')
	full_matrix = np.zeros((total_cols, total_cols))

	if using_lag_time:
		i_row_start = 0
		counter = 0
		for i_block in range(n_blocks_row):
			i_col_start = 0
			for j_block in range(n_blocks_col):
				tmp_block = all_blocks[counter]

				full_matrix[i_row_start:i_row_start + n_cols_per_block,
				i_col_start:i_col_start + n_cols_per_block] = tmp_block
				i_col_start += n_cols_per_block
				counter += 1
			i_row_start += n_cols_per_block

		# Symmeterize (enforce detailed balance)
		full_matrix = 0.5 * (full_matrix + full_matrix.T)

	else:
		i_row_start = 0
		counter = 0
		for i_block in range(n_blocks_row):
			i_col_start = i_row_start
			for j_block in range(i_block, n_blocks_col):
				tmp_block = all_blocks[counter]

				# Get #columns in current block
				if i_block == n_blocks_row - 1:
					n_cols_i = n_cols_last_block
				else:
					n_cols_i = n_cols_per_block

				if j_block == n_blocks_row - 1:
					n_cols_j = n_cols_last_block
				else:
					n_cols_j = n_cols_per_block

				if i_block != j_block:
					full_matrix[i_row_start:i_row_start + n_cols_i, i_col_start:i_col_start + n_cols_j] = tmp_block
					full_matrix[i_col_start:i_col_start + n_cols_j, i_row_start:i_row_start + n_cols_i] = np.transpose(
						tmp_block)
				else:
					for i in range(n_cols_i):
						for j in range(i, n_cols_j):
							tmp_element1 = tmp_block[i, j]
							tmp_element2 = tmp_block[j, i]
							full_matrix[i_row_start + i, i_col_start + j] = (tmp_element1 + tmp_element2) / 2.0
							full_matrix[i_col_start + j, i_row_start + i] = (tmp_element1 + tmp_element2) / 2.0
				i_col_start += n_cols_per_block
				counter += 1
			i_row_start += n_cols_per_block

	return full_matrix

def build_matrix(base_file_name, n_blocks, using_lag_time=False, file_label='', out_directory=''):
	"""
	Build and write the full matrix to file.
	:param base_file_name:
	:param n_blocks:
	:param using_lag_time:
	:param file_label:
	:param out_directory:
	:return:
	"""
	all_blocks, n_cols_per_block, n_cols_last_block = _read_data(base_file_name, n_blocks, file_label=file_label)

	if using_lag_time:
		n_blocks_col = int(np.sqrt(n_blocks))
	else:
		n_blocks_col = int(np.sqrt(2 * n_blocks + 0.25) - 0.5)

	n_blocks_row = n_blocks_col
	total_cols = (n_blocks_col - 1) * n_cols_per_block + n_cols_last_block

	full_matrix = _get_full_matrix(all_blocks, total_cols, n_blocks_row, n_blocks_col, n_cols_per_block,
								   n_cols_last_block, using_lag_time=using_lag_time)

	print('Number of blocks: ' + str(int(n_blocks)))
	print('Number of blocks per column: ' + str(n_blocks_col))
	print('Total number of cols: ' + str(total_cols))

	full_matrix = _remove_diagonal(full_matrix)
	_save_full_MI_matrix(full_matrix, out_directory, file_label)
	return


def main(parser):
	print('Building full matrix from blocks.')
	args = parser.parse_args()
	print(args)

	file_end_name = args.file_end_name
	n_blocks = float(args.n_blocks)
	out_directory = args.out_directory

	base_file_name = args.base_file_name

	build_matrix(base_file_name, n_blocks, file_label=file_end_name, out_directory=out_directory, using_lag_time=args.using_time_lag)
	return

if __name__=='__main__':
	parser = argparse.ArgumentParser(epilog='Build complete matrix (e.g. MI matrix) from blocks. Annie Westerlund.')
	parser.add_argument('-f','--base_file_name',help='Base file name of matrices.',default='')
	parser.add_argument('-fe','--file_end_name',help='End of file name (file label) of matrices.',default='')
	parser.add_argument('-n_blocks','--n_blocks',help='Number of blocks that constitutes the upper triangle (including diagonal).',default=2)
	parser.add_argument('-od','--out_directory',help='The folder in which the new data should be saved.',default='')
	parser.add_argument('-lag','--using_time_lag',help='Flag for denoting when the MI matrix is time-lagged (ie. assymmetric).',action='store_true')
	main(parser)




