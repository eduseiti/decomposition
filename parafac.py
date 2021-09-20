import numpy as np
import tensorly as tl

import pickle

import pandas as pd

from tensorly.decomposition import parafac

import seaborn as sn

import matplotlib.pyplot as plt

import torch

import sys

from scipy.io import savemat


EGF_TIMES = [0., 5.5, 7., 9., 13., 17., 23., 30., 40., 60.]

INHIBITORS_TIMES = [ 0.,  7.,  9., 13., 17., 40., 60.]

CELL_LINES_MORE_1K = [
    '184A1', 'BT20', 'BT474', 'BT549', 'CAL148', 'CAL51', 'CAL851',
    'EFM192A', 'EVSAT', 'HBL100', 'HCC1395',
    'HCC1419', 'HCC1500', 'HCC1937', 'HCC1954',
    'HCC2157', 'HCC2185', 'HCC3153', 'HCC38', 'HCC70', 'HDQP1',
    'JIMT1', 'MCF10A', 'MCF7', 'MDAMB134VI', 'MDAMB157',
    'MDAMB175VII', 'MDAMB361', 'MDAMB415', 'MDAMB453', 'MDAkb2',
    'MFM223', 'MPE600', 'MX1', 'OCUBM', 'T47D', 'UACC812', 'UACC893',
    'ZR7530'
]

COMPLETE_CELL_LINES_MORE_1K = [
    '184A1', 'BT474', 'BT549', 'CAL51', 'CAL851', 'EFM192A', 'HBL100',
    'HCC1419', 'HCC1500', 'HCC1937', 'HCC1954', 'HCC2185', 'HCC3153',
    'HCC38', 'HDQP1', 'JIMT1', 'MCF7', 'MDAMB134VI', 'MDAMB175VII',
    'MDAMB361', 'MDAMB453', 'MFM223', 'MPE600', 'OCUBM', 'T47D',
    'UACC812', 'UACC893', 'ZR7530'
]

COMPLETE_CELL_LINES_ALL_MARKERS_MORE_1K = [
    '184A1', 'BT474', 'CAL51', 'CAL851', 'EFM192A', 'HBL100',
    'HCC2185', 'HCC3153', 'HDQP1', 'JIMT1', 'MCF7', 'MDAMB175VII',
    'MFM223', 'MPE600', 'OCUBM', 'UACC812', 'ZR7530'
]

NO_S_PHASE_CELL_LINES_ALL_MARKERS_MORE_1K = [
    '184A1', 'BT474', 'CAL51', 'CAL851', 'EFM192A', 'HBL100',
    'HCC2185', 'HCC3153', 'HDQP1', 'JIMT1', 
    'MFM223', 'MPE600', 'OCUBM', 'ZR7530'
]



if len(sys.argv) > 1:
    PARAFAC_FACTORS = int(sys.argv[1])
else:
    PARAFAC_FACTORS=260

if len(sys.argv) > 2:
    CELL_LINES = sys.argv[2].split(" ")
else:
    CELL_LINES = [CELL_LINES_MORE_1K[0]]

if len(sys.argv) > 3:
    TREATMENTS = sys.argv[3].split(" ")
else:
    TREATMENTS = ['iEGFR']


if len(sys.argv) > 4:
    CELL_COUNTS = int(sys.argv[4])
else:
    CELL_COUNTS = 1000



#
# Read computed data
#

with open("./data/complete_cell_lines_df.pkl", "rb") as inputFile:
    cell_lines_df = pickle.load(inputFile)
    
ALL_MARKERS = cell_lines_df.columns[5:]


with open("./data/kmeans_lda_split_class_0_s_phase.pkl", "rb") as inputFile:
    class_0_s_phase = pickle.load(inputFile)

with open("./data/kmeans_lda_split_class_1_non_s_phase.pkl", "rb") as inputFile:
    class_1_non_s_phase = pickle.load(inputFile)




non_s_phase_cells_df = cell_lines_df.iloc[class_1_non_s_phase]


#
# Initialize tensorly backend
#

tl.set_backend('pytorch')


### Helper functions

def single_cell_line_multiple_treatments(which_cell_lines, which_treatments, times_list, cells_count, all_data_df):
    
    cell_lines_filters = all_data_df['cell_line'] == which_cell_lines[0]
    
    if len(which_cell_lines) > 1:
        for cell_line in which_cell_lines[1:]:
            cell_lines_filters |= all_data_df['cell_line'] == cell_line
    
    
    treatments_filters = all_data_df['treatment'] == which_treatments[0]
    
    if len(which_treatments) > 1:
        for treatment in which_treatments[1:]:
            treatments_filters |= all_data_df['treatment'] == treatment
            
    
    filtered_data_df = all_data_df.loc[(cell_lines_filters & treatments_filters)]
    
    all_cell_lines = {}

    for cell_line_group_name, cell_line_group_df in filtered_data_df.groupby('cell_line'):

        all_treatments = {}

        for treatment_group_name, treatment_group_df in cell_line_group_df.groupby('treatment'):

            all_times = []

            for time in times_list:
                all_cells_markers = treatment_group_df.loc[(treatment_group_df['time'] == time)][ALL_MARKERS].to_numpy()

                cells_sample = np.random.choice(list(range(all_cells_markers.shape[0])), cells_count, replace=False)

                all_times.append(all_cells_markers[cells_sample])

            all_treatments[treatment_group_name] = np.stack(all_times)

        all_cell_lines[cell_line_group_name] = np.stack([all_treatments[key] for key in which_treatments])
        
    return np.squeeze(np.stack([all_cell_lines[key] for key in which_cell_lines]))




def plot_parafac(parafac_results, which_components_indexes, dimensions_labels):
    
    how_many_components = parafac_results.weights.shape[0]
    
    components_labels = ['comp_{}'.format(component) for component in list(range(how_many_components))]
    
    for i, component in enumerate(which_components_indexes):
     
        if len(dimensions_labels[i]) > 10:
            y_size = 15
        else:
            y_size = 4
            
        if len(components_labels) > 20:
            x_size = 30
        else:
            x_size = 6
            
        plt.figure(figsize = (x_size, y_size))
            
        sn.heatmap(parafac_results.factors[component], annot=True, xticklabels = components_labels, yticklabels = dimensions_labels[i])
        
        plt.show()




def select_data_and_factorize(which_cell_lines, 
                              which_treatments, 
                              times_list, 
                              cells_count, 
                              all_data_df, 
                              parafac_components, 
                              normalize, 
                              plot=False,
                              result_filename=None,
                              export_matlab=False):

    print("\nCalculating PARAFAC decomposition, {} factors:\n- cell line(s): {}\n- treatment(s): {}\n- time(s): {}\n- cells per condition: {}\n".format(
        parafac_components, which_cell_lines, which_treatments, times_list, cells_count))


    selected_data = single_cell_line_multiple_treatments(which_cell_lines, which_treatments, times_list, cells_count, all_data_df)
    
    if export_matlab:

        which_filename = result_filename.split(".pkl")[0].format("_".join(which_cell_lines), "_".join(which_treatments), parafac_components)

        print("Saving the {} MATLAB file".format(which_filename))

        savemat(which_filename, {'phospho': selected_data, 'label': "single-cell_phosphorylation"})

    all_cells_tensor = tl.tensor(selected_data)
    
    if len(which_cell_lines) > 1:
        first_dimension = which_cell_lines
    else:
        first_dimension = which_treatments
        
    second_dimension = times_list
    last_dimension = ALL_MARKERS

    components_to_plot = [0, 1, len(all_cells_tensor.shape) - 1]
    dimensions_labels = [first_dimension, second_dimension, last_dimension]
    
    results = parafac(all_cells_tensor, parafac_components, n_iter_max=200, 
                      tol=1e-8, linesearch=False, verbose=1, normalize_factors=normalize, cvg_criterion='abs_rec_error', init='random')
    
    if plot:
        plot_parafac(results, components_to_plot, dimensions_labels)
    
    processed_data = {
        'which_cell_lines': which_cell_lines,
        'which_treatments': which_treatments,
        'times_list': times_list,
        'cells_count': cells_count,
        'data':selected_data, 
        'parafac': results, 
        'components': components_to_plot, 
        'dimensions': dimensions_labels
    }

    if result_filename is not None:
        with open(result_filename.format("_".join(which_cell_lines), "_".join(which_treatments), parafac_components), "wb") as outputFile:
            pickle.dump(processed_data, outputFile, pickle.HIGHEST_PROTOCOL)

    return processed_data




def explore_parafac(decomposed_data, factors_to_combine=None, min_cluster_size=5, samples_labels=None):
    
    if factors_to_combine is not None:
        decomposed_data_combined = torch.matmul(decomposed_data['parafac'].factors[factors_to_combine[0]], torch.transpose(decomposed_data['parafac'].factors[factors_to_combine[1]], 0, 1))
    else:
        decomposed_data_combined = decomposed_data
        
    
    plt.figure(figsize = (30, 15))

    if decomposed_data_combined.shape[0] > 50:
        samples = np.random.choice(list(range(decomposed_data_combined.shape[0])), 20, replace=False)

        sn.heatmap(decomposed_data_combined[samples], annot=True)
    else:
        if samples_labels is not None:
            sn.heatmap(decomposed_data_combined, annot=True, yticklabels=samples_labels)
        else:
            sn.heatmap(decomposed_data_combined, annot=True)
    
    
    distances = sklearn.metrics.pairwise_distances(decomposed_data_combined, metric='cosine')
    
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True, metric='precomputed')
    
    clusterer.fit(distances.astype(np.double))
    
    print("Not noisy points: {}".format(sum(clusterer.labels_ > -1)))
    
    print("\nhdbscan results: \n")
    print(np.unique(clusterer.labels_, return_counts=True))
    
    
    pca = PCA()
    
    decomposed_data_combined_pca = pca.fit_transform(decomposed_data_combined)
    
    print("\npca explained variance ratio: \n")
    print(pca.explained_variance_ratio_)
    
    return ({"decomposed_data_combined": decomposed_data_combined,
             "clusterer": clusterer,
             "pca": pca})



#
# Perform some tests
#

decomposition_results = select_data_and_factorize(CELL_LINES, 
                                                  TREATMENTS, 
                                                  INHIBITORS_TIMES, 
                                                  CELL_COUNTS, 
                                                  non_s_phase_cells_df, PARAFAC_FACTORS, False,
                                                  result_filename="non_s_phase_{}_treatments_{}_parafac_decomposition_{}.pkl",
                                                  export_matlab=True)