import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

def order_protein_domains(data, domain_composition):
    """    Change the domain composition of the protein
    so that it goes together
    with the symmetrization assumption (for a protein with chains A, B, C, D):
         0
     3      1
        2

    Inputs:
    - data: Numpy array with data that will
    be shuffled around.
    - domain_composition: string with current domain composition (clockwise),
    e.g.the domain composition below would be '0 2 3 1':
         0
     1      2
        3
    """

    n_dims = len(data.shape)
    n_domains = len(domain_composition)
    n_resids = data.shape[0]
    print('# Residues per domain: '+str(n_resids/n_domains))
    n_resids_per_domain = int(n_resids/n_domains)

    swap_inds = np.zeros(n_domains)
    shuffled_data = np.zeros(data.shape)

    # Find the whereabouts of the domains so
    # that we can put the domain data at correct position in the matrix.
    for i_domain in range(n_domains):
        swap_inds[i_domain] = np.where(domain_composition == i_domain)[0]

    # Flip the matrix rows
    for i_domain in range(n_domains):
        tmp_domain_data = data[i_domain*n_resids_per_domain:(i_domain+1)*n_resids_per_domain]
        domain_ind = int(swap_inds[i_domain])

        shuffled_data[domain_ind*n_resids_per_domain:(domain_ind+1)*n_resids_per_domain] = np.copy(tmp_domain_data)

    if n_dims > 1:
        data = np.copy(shuffled_data)
        # Flip the matrix columns
        for i_domain in range(n_domains):
            tmp_domain_data = data[:, i_domain*n_resids_per_domain:(i_domain+1)*n_resids_per_domain]
            domain_ind = int(swap_inds[i_domain])

            shuffled_data[:, domain_ind*n_resids_per_domain:(domain_ind+1)*n_resids_per_domain] = np.copy(tmp_domain_data)

    return shuffled_data


def get_data(args):
    print('Reading data...')

    try:
        data = np.load(args.file_name)
    except:
        data = np.loadtxt(args.file_name)

    if args.is_squareform:
        data = squareform(data)

    if args.protein_contact_map is not None:
        cmap = squareform(np.loadtxt(args.protein_contact_map))
        n_prot_resids = cmap.shape[0]
    else:
        n_prot_resids = data.shape[0]

    if args.auxiliary_protein_indices is not None:
        aux_inds = np.loadtxt(args.auxiliary_protein_indices).astype(int)
    else:
        aux_inds = None

    shuffle_aux = args.shuffle_auxiliary_proteins
    domain_composition = np.asarray(args.protein_domain_composition)

    file_name_out = args.file_name[0:-4]+'_shuffled.npy'

    return data, domain_composition, file_name_out, n_prot_resids, args.is_squareform, aux_inds, shuffle_aux


def remove_auxiliary_from_main(auxiliary_inds, data):

    # Remove auxiliary protein indices from main protein to enable simple averaging and error estimation
    if auxiliary_inds is None:
        auxiliary_inds = np.zeros((0),int)

    mask = np.ones(data.shape[0], dtype=bool)
    invert_mask = np.zeros(data.shape[0], dtype=bool)

    mask[auxiliary_inds] = False
    invert_mask[auxiliary_inds] = True

    if len(data.shape) == 2:
        tmp_c1 = data[mask,:]
        tmp_c1 = tmp_c1[:,mask]
        tmp_c2 = data[invert_mask,:]
        tmp_c2 = tmp_c2[:,invert_mask]

        main_data = np.copy(tmp_c1)
        auxiliary_data = np.copy(tmp_c2)
    else:
        main_data = data[mask]
        auxiliary_data = data[invert_mask]

    return main_data, auxiliary_data, mask, invert_mask


def set_shuffled_data(data, shuffled_data, mask):

    resid_indices = np.arange(mask.shape[0])
    resid_indices = resid_indices[mask]

    n_dims = len(data.shape)
    if n_dims > 1:
        for counter, mask_ind in enumerate(resid_indices):
            data[mask_ind, mask] = shuffled_data[counter]
    else:
        data[mask] = shuffled_data

    return data


def main(data, domain_composition, file_name, n_prot_resids, is_squareform=True, aux_inds=None, shuffle_aux=False):

    n_dims = len(data.shape)

    prot_data = np.copy(data[0:n_prot_resids])
    if n_dims > 1:
        prot_data = prot_data[:, 0:n_prot_resids]

    print('Shuffle domains.')
    main_data, auxiliary_data, main_mask, aux_mask = remove_auxiliary_from_main(aux_inds, prot_data)

    shuffled_data = np.copy(prot_data)
    if shuffle_aux:
        auxiliary_shuffled_data = order_protein_domains(auxiliary_data, domain_composition)
        prot_data = set_shuffled_data(prot_data, auxiliary_shuffled_data, aux_mask)
    else:
        main_shuffled_data = order_protein_domains(main_data, domain_composition)
        prot_data = set_shuffled_data(prot_data, main_shuffled_data, main_mask)

    if n_dims > 1:
        for i in range(n_prot_resids):
            for j in range(n_prot_resids):
                data[i, j] = prot_data[i, j]
    else:
        data[0:n_prot_resids] = prot_data

    if is_squareform:
        data = squareform(data)

    np.save(file_name, data)
    print('Shuffled data saved to: '+file_name)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(epilog='Order the domains of the protein according to clockwise symmetry.')
    parser.add_argument('-f','--file_name',help='Data file to shuffle data in.')
    parser.add_argument('-cmap','--protein_contact_map',help='Protein contact map, used to set number of residues (optional).')
    parser.add_argument('-sq','--is_squareform',help='Note whether data is in squareform or not.',action='store_true')
    parser.add_argument('-iaux','--auxiliary_protein_indices',help='Residue indices of auxiliary protein.')
    parser.add_argument('-saux','--shuffle_auxiliary_proteins',help='Flag to shuffle auxiliary proteins.',action='store_true')
    parser.add_argument('-pdc','--protein_domain_composition',help='The order of chains in the composition. First chain is called 0, second 1 etc.',nargs='+',type=int)

    args = parser.parse_args()

    print('---------------------------------------------')
    print('Shuffle protein domains for symmetrization.')
    print()
    print('File name: '+args.file_name)
    print('---------------------------------------------')

    main(*get_data(args))

