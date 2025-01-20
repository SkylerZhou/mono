import pandas as pd
import scanpy as sc
import os
import click
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@click.command()
@click.option(
    "-csv", "--csv_path", help="Path for read-in csv file", required=True
)
@click.option(
    "-adata", "--adata_path", help="Path for read-in adata file", required=True
)


def _load_fn(csv_path, adata_path):
    csv_file = csv_path
    df = pd.read_csv(csv_file, header=0, names=['cell_name', 'logit_1', 'logit_2'])

    df['predicted_category'] = df.apply(lambda row: 0 if row[1] > row[2] else 1, axis=1)
    # print(df.head())
    # print(sum(df['predicted_category']))


    adata = sc.read_h5ad(adata_path)
    patient_cells = adata.obs[['patient']]


    patient_categories = {}
    original = {}
    ori_cells = []
    pred_cells = []
    pred_1 = {}
    pred_0 = {}
    cell_num = {}


    for patient in patient_cells['patient'].unique():
        cells = patient_cells[patient_cells['patient'] == patient].index
        patient_data = adata.obs[adata.obs['patient'] == patient]
        
        cell_categories = df[df['cell_name'].isin(cells)]['predicted_category']

        
        if not cell_categories.empty:
            cell_num[patient] = len(cells)
            # most_common_category = Counter(cell_categories).most_common(1)[0][0]
            if sum(cell_categories) > 0.5*len(cell_categories):
                most_common_category = 1
            else:
                most_common_category = 0
            patient_categories[patient] = most_common_category
            if (patient_data['perturbation'] == 0).any():
                original_val =  0
            else:
                original_val =  1
            original[patient] = original_val
            pred_1[patient] = sum(cell_categories)
            pred_0[patient] = len(cell_categories) - sum(cell_categories)
            ori_cells.extend([original_val] * len(cell_categories))
            pred_cells.extend(list(cell_categories))

            

    print('----------------------Patient Level----------------------------------')
    for patient, category in patient_categories.items():
        print(f'Patient: {patient}| Num of cells: {pred_0[patient]+pred_1[patient]},[{pred_0[patient]}, {pred_1[patient]}]| Predicted Cat: {category}| Original Cat: {original[patient]}')

    accuracy = accuracy_score(ori_cells, pred_cells)
    precision = precision_score(ori_cells, pred_cells)
    recall = recall_score(ori_cells, pred_cells)
    f1 = f1_score(ori_cells, pred_cells)
    print('-------------------------Cell Level----------------------------------')
    print(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}')

    
    

if __name__ == "__main__":
    _load_fn()
