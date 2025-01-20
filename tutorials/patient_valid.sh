#!/bin/bash

# Read arguments from the command line
adata_file_path=$1
num=$2
test_name=$3

# Repeat the Python processing 5 times
rm -rf ../dataset_full/finetune/model_temp
for i in {1}
do

    echo "Starting iteration $i"
    python - <<END
import anndata as ad
import numpy as np
import random
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


m = 0
# selected_list = ['108', '13413', '16', '37', '220', '8017', '8113']

# Split the data
while True:
    patients = list(adata.obs['patient'].unique())
    random.seed(time.time())
    # selected_patients = random.sample(patients, $num)
    selected_patients = list(['220'])
    
    test_ad = adata[adata.obs['patient'].isin(selected_patients)]
    train_ad = adata[~adata.obs['patient'].isin(selected_patients)]

    idx_0 = train_ad.obs['perturbation'] == 0
    idx_1 = train_ad.obs['perturbation'] == 1
    n_0 = sum(idx_0)
    n_1 = sum(idx_1)
    n_keep = min(n_0, n_1)

    if n_0 > n_1:
        idx_0_selected = np.random.choice(np.where(idx_0)[0], size=n_keep, replace=False)
        idx_1_selected = np.where(idx_1)[0]
    else:
        idx_0_selected = np.where(idx_0)[0]
        idx_1_selected = np.random.choice(np.where(idx_1)[0], size=n_keep, replace=False)

    selected_idx = np.concatenate([idx_0_selected, idx_1_selected])

    train_ad = train_ad[selected_idx].copy()

    test_perturbation_counts = test_ad.obs['perturbation'].value_counts()
    train_perturbation_counts = train_ad.obs['perturbation'].value_counts()

    print(f"Test set - Perturbation 0: {test_perturbation_counts.get(0, 0)}, Perturbation 1: {test_perturbation_counts.get(1, 0)}")
    print(f"Train set - Perturbation 0: {train_perturbation_counts.get(0, 0)}, Perturbation 1: {train_perturbation_counts.get(1, 0)}")
    
    # if check_perturbation_ratio(train_ad) and check_perturbation_ratio_2(test_ad):
    if check_perturbation_ratio(train_ad):
        break

# Save the split datasets with iteration index
print(selected_patients)
print(patients)
train_ad.write('../dataset_full/temp_train_ad.h5ad')
test_ad.write('../dataset_full/temp_test_ad.h5ad')
END
    

    # Run subsequent scripts once after all iterations
    bash launch_train.sh
    #!/bin/bash

    parameters=(150)

    for param in "${parameters[@]}"
    do
        echo "Running steps $param"
        python load_results.py -c $param
        python cell2patient.py -csv "results_${param}.csv" -adata "../dataset_full/temp_test_ad.h5ad"
        # python cell2patient_binom.py -c ${param}
    done
    rm -rf ../dataset_full/finetune/model_temp

    echo "--------------------------Completed iteration $i------------------------------------"
done