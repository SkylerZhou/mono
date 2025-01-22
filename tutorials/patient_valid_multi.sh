#!/bin/bash

# Read arguments from the command line
adata_file_path=$1
num=$2
test=$3

# Repeat the Python processing 5 times

folder_path="../results/$test"

file_name="../common/config.json"

echo "{
    \"saved_dir\": \"$folder_path\"
}" > $file_name

# # Confirm the file was created
# if [ -f "$file_name" ]; then
#     echo "JSON file '$file_name' created successfully."
# else
#     echo "Failed to create JSON file."
# fi

for i in {1}
do
    if [ -d "$folder_path" ]; then
        rm -rf "$folder_path"/*
    else
        mkdir -p "$folder_path"
    fi

    echo "Starting iteration $i"
    python - <<END
import anndata as ad
import numpy as np
import random
import time
import pandas as pd


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
    # selected_patients = list(['VC09_CD19', 'VC29_CD19', 'VC22_CD19', 'VC01_CD19', 'DC45_CD19'])
    
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
    
    # if check_perturbation_ratio(train_ad) and check_perturbation_ratio_2(test_ad):
    #     break

# Save the split datasets with iteration index
print(selected_patients)
print(patients)
train_ad.write('$folder_path/temp_train_ad.h5ad')
test_ad.write('$folder_path/temp_test_ad.h5ad')
END
    

    # Run subsequent scripts once after all iterations
    bash launch_train.sh $test
    #!/bin/bash

    parameters=(100 500 1000 1500)

    for param in "${parameters[@]}"
    do
        echo "Running steps $param"
        python load_results.py -c $param -p $folder_path
        python cell2patient.py -csv "$folder_path/results_${param}.csv" -adata ".$folder_path/temp_test_ad.h5ad"
        # python cell2patient_binom.py -c ${param}
    done

    echo "--------------------------Completed iteration $i------------------------------------"
done