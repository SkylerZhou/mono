# Get overall patient distribution
import numpy as np
import scanpy as sc

adata = sc.read_h5ad('/home/pany3/pany3/mono/dataset_full/lymphoma_5k.h5ad')
patient_counts = adata.obs.patient.value_counts()
print("\nPatient Distribution:")
print(patient_counts)

# Get perturbation distribution for each patient
print("\nPerturbation Distribution within each Patient:")
for patient in adata.obs.patient.unique():
    patient_data = adata.obs[adata.obs.patient == patient]
    perturb_counts = patient_data.perturbation.value_counts()
    perturb_props = patient_data.perturbation.value_counts(normalize=True)
    
    print(f"\nPatient {patient}:")
    print(f"Total cells: {len(patient_data)}")
    for perturb_value in sorted(perturb_counts.index):
        count = perturb_counts[perturb_value]
        prop = perturb_props[perturb_value]
        print(f"Perturbation {perturb_value}: {count} cells ({prop:.2%})")


