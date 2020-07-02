import numpy as np


def rearrange_chains(data, n_chains, is_edges=False):
    # 1D rearrangment of chain data with i_chain set as the dominant chain.
    new_data = []
    n_residues_per_chain = int(data[0].shape[0] / n_chains)

    for i_chain in range(n_chains):
        if is_edges:
            # Select second part of the adjacency matrix (starting from the first
            # residue of chain i_chain)
            tmp_chain_dist1 = data[i_chain][i_chain * n_residues_per_chain::]

            # Select first part of the adjacency matrix (starting from the first
            # residue of the protein to the first residue of chain i) => wrapping
            tmp_chain_dist2 = data[i_chain][0:i_chain * n_residues_per_chain]

            tmp_chain_dist = np.concatenate((tmp_chain_dist1, tmp_chain_dist2),axis=0)

            # Flip the matrix columns
            tmp_chain_dist1 = tmp_chain_dist[:,i_chain * n_residues_per_chain::]
            tmp_chain_dist2 = tmp_chain_dist[:,0:i_chain * n_residues_per_chain]

            tmp_chain_dist = np.concatenate((tmp_chain_dist1, tmp_chain_dist2),axis=1)

            # Append rearranged matrix
            new_data.append(tmp_chain_dist)
        else:
            # Select second part of the adjacency matrix (starting from the first
            # residue of chain i)
            tmp_chain_dist1 = data[i_chain][i_chain * n_residues_per_chain::]

            # Select first part of the adjacency matrix (starting from the first
            # residue of the protein to the first residue of chain i) => wrapping
            tmp_chain_dist2 = data[i_chain][0:i_chain * n_residues_per_chain]

            tmp_chain_dist = np.concatenate((tmp_chain_dist1, tmp_chain_dist2))
            new_data.append(tmp_chain_dist)

    return new_data


def replicate_chain(data, n_chains, is_edges=False):
    # Replicate the data to different chains.
    new_data = []
    n_residues_per_chain = int(data.shape[0] / n_chains)

    for i_chain in range(n_chains):
        if is_edges:
            # Select second part of the adjacency matrix (starting from the first
            # residue of chain i_chain)
            tmp_chain_dist1 = data[i_chain * n_residues_per_chain::]

            # Select first part of the adjacency matrix (starting from the first
            # residue of the protein to the first residue of chain i) => wrapping
            tmp_chain_dist2 = data[0:i_chain * n_residues_per_chain]

            tmp_chain_dist = np.concatenate((tmp_chain_dist1, tmp_chain_dist2), axis=0)

            # Flip the matrix columns
            tmp_chain_dist1 = tmp_chain_dist[:, i_chain * n_residues_per_chain::]
            tmp_chain_dist2 = tmp_chain_dist[:, 0:i_chain * n_residues_per_chain]

            tmp_chain_dist = np.concatenate((tmp_chain_dist1, tmp_chain_dist2), axis=1)
        else:
            # Select first part of the adjacency matrix (starting from the first
            # residue of chain i_chain)
            tmp_chain_dist1 = data[i_chain * n_residues_per_chain::]

            # Select first part of the adjacency matrix (starting from the first
            # residue of the protein to the first residue of chain i_chain) => wrapping
            tmp_chain_dist2 = data[0:i_chain * n_residues_per_chain]

            # Put the two parts together
            tmp_chain_dist = np.concatenate((tmp_chain_dist1, tmp_chain_dist2))
        new_data.append(tmp_chain_dist)

    return new_data

def get_average_node_current_flow(data, n_chains):
    # Average current flow considering all chains. This is centered around the first chain.
    # Rearrange chains
    data = rearrange_chains(data, n_chains)

    # Convert to numpy-array
    all_current_flows = np.zeros((n_chains, data[0].shape[0]))
    all_wrapped_current_flows = np.zeros((n_chains, data[0].shape[0]))
    for i in range(n_chains):
        all_current_flows[i, :] = data[i]
        # wrap (symmeterize current flow of each chain)
        tmp_wrapped_CF = replicate_chain(all_current_flows[i,:],n_chains)
        for i_chain in range(n_chains):
            all_wrapped_current_flows[i,:] += tmp_wrapped_CF[i_chain]/n_chains

    # Compute mean and standard deviation
    average_current_flow = np.mean(all_wrapped_current_flows, axis=0)
    std_current_flow = np.std(all_wrapped_current_flows, axis=0)

    return average_current_flow, std_current_flow, all_wrapped_current_flows

def get_average_current_flow(data, n_chains,is_edges=False):
    # Average current flow considering all chains. This is centered around the first chain.
    # Rearrange chains
    data = rearrange_chains(data, n_chains, is_edges=is_edges)

    # Convert data to numpy-arrays
    all_current_flows = np.zeros((n_chains, *data[0].shape))
    all_wrapped_current_flows = np.zeros((n_chains, *data[0].shape))
    for i in range(n_chains):
        all_current_flows[i] = data[i]
        # wrap (symmeterize current flow of each chain)
        tmp_wrapped_CF = replicate_chain(all_current_flows[i],n_chains, is_edges=is_edges)
        for i_chain in range(n_chains):
            all_wrapped_current_flows[i] += tmp_wrapped_CF[i_chain]/n_chains

    # Compute mean and standard deviation
    average_current_flow = np.mean(all_wrapped_current_flows, axis=0)
    std_current_flow = np.std(all_wrapped_current_flows, axis=0)

    return average_current_flow, std_current_flow, all_wrapped_current_flows

def remove_auxiliary_from_main(auxiliary_indices, all_current_flows, is_edges=False):
    # Remove auxiliary protein indices from main protein to enable simple averaging and error estimation
    if auxiliary_indices != '':
        auxiliary_inds = np.loadtxt(auxiliary_indices)
        auxiliary_inds = auxiliary_inds.astype(int)
    else:
        auxiliary_inds = np.zeros((0),int)

    mask = np.ones(all_current_flows[0].shape[0], dtype=bool)
    invert_mask = np.zeros(all_current_flows[0].shape[0], dtype=bool)

    mask[auxiliary_inds] = False
    invert_mask[auxiliary_inds] = True

    main_current_flows = []
    auxiliary_current_flows = []

    if is_edges:
        for i in range(len(all_current_flows)):
            tmp_c1 = all_current_flows[i][mask,:]
            tmp_c1 = tmp_c1[:,mask]
            tmp_c2 = all_current_flows[i][invert_mask,:]
            tmp_c2 = tmp_c2[:,invert_mask]

            main_current_flows.append(tmp_c1)
            auxiliary_current_flows.append(tmp_c2)
    else:
        for i in range(len(all_current_flows)):
            main_current_flows.append(all_current_flows[i][mask])
            auxiliary_current_flows.append(all_current_flows[i][invert_mask])

    return main_current_flows, auxiliary_current_flows, mask, invert_mask

def extract_protein_and_interactor_current_flows(all_current_flows,n_protein_residues):
    """
    Extract protein current flow from other interactors
    :param all_current_flows:
    :param n_protein_residues:
    :return:
    """
    n_chains = len(all_current_flows)
    all_current_flows_protein = []
    average_flows_interactors = np.zeros(all_current_flows[0].shape[0]-n_protein_residues) # average interactor current flows imediately

    for CF in all_current_flows:
        all_current_flows_protein.append(CF[0:n_protein_residues])
        average_flows_interactors += CF[n_protein_residues::]/n_chains

    return all_current_flows_protein, average_flows_interactors

def postprocess_CF_data(CF, all_current_flows, is_edges=False, n_protein_residues=None, write_to_file=True):

    print('Computing average and standard deviation current flows across chains.')
    n_chains = len(all_current_flows)
    cheap_write=CF.cheap_write_
    file_label = CF.file_end_name_

    # Put protein current flows in all_current_flows, and potential interactor current flows in interactor_current_flow
    n_residues_total = all_current_flows[0].shape[0]
    if n_protein_residues < n_residues_total:
        all_current_flows, interactor_average_current_flows = extract_protein_and_interactor_current_flows(all_current_flows,n_protein_residues)
        np.save(CF.out_directory_ + 'average_interactor_current_flow_' + file_label + '.npy', interactor_average_current_flows)

    # Remove the auxiliary protein from the main (easier for averaging)
    main_current_flows, auxiliary_current_flows, main_mask, auxiliary_mask = \
        remove_auxiliary_from_main(CF.auxiliary_protein_indices_, all_current_flows, is_edges=is_edges)

    # Compute average current flow through each residue
    average_main_CF, std_main_CF, all_main_CFs = get_average_current_flow(main_current_flows, n_chains, is_edges=is_edges)
    average_auxiliary_CF, std_auxiliary_CF, all_auxiliary_CFs = get_average_current_flow(auxiliary_current_flows, n_chains, is_edges=is_edges)

    # Put data back to original main-auxiliary-protein form
    if is_edges:
        average_current_flow = np.zeros(n_residues_total)
        std_current_flow = np.zeros(n_residues_total)
        all_chain_current_flows = np.zeros((n_chains, *all_current_flows[0].shape))

        main_indices = np.arange(main_mask.shape[0])
        aux_indices = main_indices[auxiliary_mask]
        main_indices = main_indices[main_mask]

        for counter, mask_ind in enumerate(main_indices):
            average_current_flow[mask_ind,main_mask] = average_main_CF[counter]
            std_current_flow[mask_ind,main_mask] = std_main_CF[counter]

            all_chain_current_flows[:, mask_ind, main_mask] = all_main_CFs[:,counter]

        for counter,aux_mask_ind in enumerate(aux_indices):
            average_current_flow[aux_mask_ind, auxiliary_mask] = average_auxiliary_CF[counter]
            std_current_flow[aux_mask_ind,auxiliary_mask] = std_auxiliary_CF[counter]
            all_chain_current_flows[:, aux_mask_ind, auxiliary_mask] = all_auxiliary_CFs[:,counter]

        # Save average edge current flow to file
        if write_to_file:
            if not(cheap_write):
                np.save(CF.out_directory_ + 'all_chain_current_flows_edges_main_protein_' + file_label + '.npy', all_main_CFs)

            np.save(CF.out_directory_ + 'all_chain_current_flows_edges_' + file_label + '.npy', all_chain_current_flows)
            return
        else:
            return average_current_flow, std_current_flow

    else:
        average_current_flow = np.zeros(n_residues_total)
        std_current_flow = np.zeros(n_residues_total)
        all_chain_current_flows = np.zeros((n_chains,n_residues_total))

        prot_resids = np.arange(n_protein_residues,dtype=int)

        average_current_flow[prot_resids[main_mask]] = average_main_CF
        average_current_flow[prot_resids[auxiliary_mask]] = average_auxiliary_CF

        if n_protein_residues < n_residues_total:
            average_current_flow[n_protein_residues::] = interactor_average_current_flows

        std_current_flow[prot_resids[main_mask]] = std_main_CF
        std_current_flow[prot_resids[auxiliary_mask]] = std_auxiliary_CF

        all_chain_current_flows[:,prot_resids[main_mask]] = all_main_CFs
        all_chain_current_flows[:,prot_resids[auxiliary_mask]] = all_auxiliary_CFs

        if write_to_file:
            # Save average current flow to file
            if not(cheap_write):
                np.save(CF.out_directory_ + 'all_chain_current_flows_main_protein_' + file_label + '.npy', all_main_CFs)
                np.save(CF.out_directory_ + 'all_chain_current_flows_' + file_label + '.npy', all_chain_current_flows)
                np.save(CF.out_directory_ + 'average_current_flow_' + file_label + '.npy', average_current_flow)
                np.savetxt(CF.out_directory_ + 'average_current_flow_' + file_label + '.txt', average_current_flow)
                np.savetxt(CF.out_directory_ + 'std_current_flow_' + file_label + '.txt', std_current_flow)
                np.save(CF.out_directory_ + 'std_current_flow_' + file_label + '.npy', std_current_flow)

                print('Chain average and standard deviation current flows written to files.')

            else:
                n_resids_per_chain = int(all_main_CFs.shape[1]/n_chains)
                # Remove redundant subunits (all is averaged and replicated)
                all_main_CFs = all_main_CFs[:,0:n_resids_per_chain]
                all_auxiliary_CFs = all_auxiliary_CFs[:,0:n_resids_per_chain]

                np.save(CF.out_directory_ + 'all_chain_current_flows_cheap_write_main_protein_' + file_label + '.npy', all_main_CFs)
                np.save(CF.out_directory_ + 'all_chain_current_flows_cheap_write_aux_protein_' + file_label + '.npy', all_auxiliary_CFs)

                print('Chain current flows per subunit written to files.')