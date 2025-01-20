import numpy as np
import scanpy as sc
import pandas as pd
import click
from scipy import stats
from scipy.stats import binomtest
from scipy.stats import combine_pvalues


def calculate_metrics(adata):
    results = []
    
    mean_prob = np.mean(adata.obs['probability'])
    std_prob = np.std(adata.obs['probability'])
    
    adata.obs['z_score'] = (adata.obs['probability'] - mean_prob) / std_prob
    

    adata.obs['cell_pvalue'] = 1 * (1 - stats.norm.cdf(abs(adata.obs['z_score'])))

    for patient in adata.obs.patient.unique():
        patient_data = adata.obs[adata.obs.patient == patient]
        
        n_cells = len(patient_data)
        
        significant_cells = sum(patient_data['cell_pvalue'] < 0.05)
        

        # binom_test = binomtest(significant_cells, n_cells, p=0.05)
        statistic, combined_pvalue = combine_pvalues(patient_data['cell_pvalue'], method='fisher')
        
        results.append({
            'patient': patient,
            'total_cells': n_cells,
            'significant_cells': significant_cells,
            # 'patient_pvalue': binom_test.pvalue
            'patient_pvalue': combined_pvalue
            
        })
    
    return pd.DataFrame(results)


    
    


@click.command()
@click.option(
    "-c", "--checkpoint", help="Step to evaluate (e.g. 100000)", required=True
)
def evaluate(checkpoint):
    adata = sc.read_h5ad('../dataset_full/temp_test_ad.h5ad')
    prob_df = pd.read_csv('results_'+checkpoint+'.csv', header=0, index_col=0)
    # Get the intersection of indices
    common_cells = list(set(adata.obs_names).intersection(set(prob_df.index)))

    # Print matching statistics
    print(f"Total cells in adata: {adata.n_obs}")
    print(f"Matched cells: {len(common_cells)}")
    print(f"Unmatched cells: {adata.n_obs - len(common_cells)}")

    # Keep only matched cells in adata
    adata = adata[common_cells].copy()

    probability_values = prob_df.loc[adata.obs_names, '1']
    adata.obs['probability'] = probability_values

    results_df = calculate_metrics(adata)

    print("\nPatient-level results:")
    print(results_df)

    for idx, row in results_df.iterrows():
        patient = row['patient']
        adata.obs.loc[adata.obs.patient == patient, 'patient_pvalue'] = row['patient_pvalue']
    


if __name__ == "__main__":
    evaluate()

