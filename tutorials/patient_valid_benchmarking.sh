#!/bin/bash

# Read arguments from the command line
adata_file_path=$1
num=$2

# Repeat the Python processing 5 times
# rm -rf ../dataset_full/finetune/model_temp
# folder_path = ../dataset_full/$test
# if [ -d "$folder_path" ]; then
#     rm -rf "$folder_path"/*
# else
#     mkdir -p "$folder_path"
# fi

for i in {1..5}
do

    echo "Starting iteration $i"
    python - <<END
import anndata as ad
import numpy as np   
import random
import time
import numpy as np
import scanpy as sc
import scipy
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import time


# Load the adata file
adata = ad.read_h5ad("$adata_file_path")

# Function to check perturbation ratio
def check_perturbation_ratio(data):
    perturbation_counts = data.obs['perturbation'].value_counts(normalize=True)
    return 0.3 <= perturbation_counts.get(0, 0) <= 0.7 and 0.3 <= perturbation_counts.get(1, 0) <= 0.7

def check_perturbation_ratio_2(data):
    perturbation_counts = data.obs['perturbation'].value_counts(normalize=True)
    return 0 < perturbation_counts.get(0, 0) < 1

# Split the data
while True:
    patients = list(adata.obs['patient'].unique())
    random.seed(time.time())
    selected_patients = random.sample(patients, $num)
    # selected_patients = list([1,2,22,21,18])
    
    # test_ad = adata[adata.obs['patient'].isin(selected_patients)]
    # train_ad = adata[~adata.obs['patient'].isin(selected_patients)]

    # test_perturbation_counts = test_ad.obs['perturbation'].value_counts()
    # train_perturbation_counts = train_ad.obs['perturbation'].value_counts()

    # print(f"Test set - Perturbation 0: {test_perturbation_counts.get(0, 0)}, Perturbation 1: {test_perturbation_counts.get(1, 0)}")
    # print(f"Train set - Perturbation 0: {train_perturbation_counts.get(0, 0)}, Perturbation 1: {train_perturbation_counts.get(1, 0)}")
    
    # if check_perturbation_ratio(train_ad):
    #     break

    test_ad = adata[adata.obs['patient'].isin(selected_patients)]
    train_ad = adata[~adata.obs['patient'].isin(selected_patients)]

    value_counts = train_ad.obs['perturbation'].value_counts()

    n = value_counts.min()

    balanced_indices = []
    for perturbation in value_counts.index:
        mask = train_ad.obs['perturbation'] == perturbation
        indices = train_ad.obs[mask].index
        sampled_indices = pd.Series(indices).sample(n=n, random_state=42)
        balanced_indices.extend(sampled_indices)


    train_ad = train_ad[balanced_indices].copy()


    test_perturbation_counts = test_ad.obs['perturbation'].value_counts()
    train_perturbation_counts = train_ad.obs['perturbation'].value_counts()

    print(f"Test set - {test_perturbation_counts}")
    print(f"Train set - {train_perturbation_counts}")
    break
    

    # test_ad = adata[adata.obs['patient'].isin(selected_patients)]
    # train_ad = adata[~adata.obs['patient'].isin(selected_patients)]

    # idx_0 = train_ad.obs['perturbation'] == 0
    # idx_1 = train_ad.obs['perturbation'] == 1
    # n_0 = sum(idx_0)
    # n_1 = sum(idx_1)
    # n_keep = min(n_0, n_1)

    # if n_0 > n_1:
    #     idx_0_selected = np.random.choice(np.where(idx_0)[0], size=n_keep, replace=False)
    #     idx_1_selected = np.where(idx_1)[0]
    # else:
    #     idx_0_selected = np.where(idx_0)[0]
    #     idx_1_selected = np.random.choice(np.where(idx_1)[0], size=n_keep, replace=False)

    # selected_idx = np.concatenate([idx_0_selected, idx_1_selected])

    # train_ad = train_ad[selected_idx].copy()

    # test_perturbation_counts = test_ad.obs['perturbation'].value_counts()
    # train_perturbation_counts = train_ad.obs['perturbation'].value_counts()

    # print(f"Test set - Perturbation 0: {test_perturbation_counts.get(0, 0)}, Perturbation 1: {test_perturbation_counts.get(1, 0)}")
    # print(f"Train set - Perturbation 0: {train_perturbation_counts.get(0, 0)}, Perturbation 1: {train_perturbation_counts.get(1, 0)}")
    
    # if check_perturbation_ratio(train_ad) and check_perturbation_ratio_2(test_ad):
    # break

# Save the split datasets with iteration index
print(selected_patients)
print(patients)
# train_ad.write('../dataset_full/temp_train_'+$test_name+'.h5ad')
# test_ad.write('../dataset_full/temp_test_'+$test_name+'.h5ad')


np.random.seed(42)

# Calculate the fraction to keep (20% = 0.2)
fraction = 0.25

# Subsample the AnnData object
sc.pp.subsample(train_ad, fraction=fraction, random_state=42)
sc.pp.subsample(test_ad, fraction=fraction*4, random_state=42)

rawcount = train_ad.X
if not isinstance(rawcount, np.ndarray):
    rawcount = rawcount.toarray()
rawcount = np.squeeze(rawcount.astype(int))
X_train = rawcount
y_train = list(train_ad.obs['perturbation'])

rawcount = test_ad.X
if not isinstance(rawcount, np.ndarray):
    rawcount = rawcount.toarray()
rawcount = np.squeeze(rawcount.astype(int))
X_test = rawcount
y_test = list(test_ad.obs['perturbation'])

print(train_ad)

models = {
    'SVM': SVC(kernel='rbf', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'MLP': MLPClassifier(max_iter=1000, random_state=42),
    'LDA': LinearDiscriminantAnalysis()
}

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

start_time = time.time()


for name, model in models.items():

    model.fit(X_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    

    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Macro Score: {f1_macro:.4f}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Running time: {elapsed_time} seconds")
    start_time = time.time()

END

    echo "Completed iteration $i"
done