from dataloader import  *
from torch.utils.data import DataLoader

batch_size = 3

default_uncertain_label="LSR-Ones"
default_unknown_label=0
default_annotation_percent=100
default_normalization="imagenet"

chexpert_data_dir="/data/jliang12/mhossei2/Dataset"
chexpert_train_list="dataset/CheXpert_train.csv"

chexpert_dataset = CheXpertDataset(
    images_path=chexpert_data_dir, 
    file_path=chexpert_train_list,
    augment=build_transform_classification(normalize=default_normalization, mode="train"), 
    uncertain_label=default_uncertain_label, 
    unknown_label=default_unknown_label, 
    annotation_percent=default_annotation_percent
    )

chexpert_dataloader = DataLoader(chexpert_dataset, batch_size=batch_size, shuffle=True)
chexpert_examples = enumerate(chexpert_dataloader)
batch_idx, (chexpert_sources, chexpert_targets) = next(chexpert_examples) 

diseases=['No Finding', 'Atelectasis','Cardiomegaly','Consolidation','Edema',
    'Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
    'Pleural Effusion','Pneumonia','Pneumothorax','Pleural Other',
    'Support Devices']

mimic_data_dir="/data/jliang12/jpang12/dataset/MIMIC_jpeg"

mimic_dataset = MIMIC_CXR(
        dataset_directory=mimic_data_dir, 
        partition="train",
        augment=build_transform_classification(normalize=default_normalization, mode="train"),
        diseases=diseases,
        from_modality="image", 
        to_modality="chexpert",
        annotation_percent=default_annotation_percent, 
        uncertain_label=default_uncertain_label, 
        unknown_label=default_unknown_label
    )

mimic_dataloader = DataLoader(mimic_dataset, batch_size=batch_size, shuffle=True)
mimic_examples = enumerate(mimic_dataloader)
batch_idx, (mimic_sources, mimic_targets) = next(mimic_examples)

print("chexpert_sources", chexpert_sources.shape)
print("chexpert_targets", chexpert_targets.shape)
print("mimic_source", mimic_sources.shape)
print("mimic_targets", mimic_targets.shape)