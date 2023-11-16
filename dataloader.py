import os
import torch
import random
import copy
import csv
import gzip
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import pydicom as dicom
import cv2
from skimage import transform, io, img_as_float, exposure
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomBrightnessContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)



def build_transform_classification(normalize, crop_size=224, resize=256, mode="train", test_augment=True):
    transformations_list = []

    if normalize.lower() == "imagenet":
      normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "none":
      normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)
    if mode == "train":
      transformations_list.append(transforms.RandomResizedCrop(crop_size))
      transformations_list.append(transforms.RandomHorizontalFlip())
      transformations_list.append(transforms.RandomRotation(7))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "valid":
      transformations_list.append(transforms.Resize((resize, resize)))
      transformations_list.append(transforms.CenterCrop(crop_size))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "test":
      if test_augment:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.TenCrop(crop_size))
        transformations_list.append(
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
          transformations_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
      else:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.CenterCrop(crop_size))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
          transformations_list.append(normalize)
    transformSequence = transforms.Compose(transformations_list)

    return transformSequence

def build_transform_segmentation():
  AUGMENTATIONS_TRAIN = Compose([
    # HorizontalFlip(p=0.5),
    OneOf([
        RandomBrightnessContrast(),
        RandomGamma(),
         ], p=0.3),
    OneOf([
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
    RandomSizedCrop(min_max_height=(156, 224), height=224, width=224,p=0.25),
    ToFloat(max_value=1)
    ],p=1)

  return AUGMENTATIONS_TRAIN




class ChestXray14Dataset(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()

        if line:
          lineItems = line.split()

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)


# ---------------------------------------------Downstream CheXpert------------------------------------------
class CheXpertDataset(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14,
               uncertain_label="LSR-Ones", unknown_label=0, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
        label = line[5:]
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              label[i] = -1
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        imageLabel = [int(i) for i in label]
        self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    label = []
    for l in self.img_label[index]:
      if l == -1:
        if self.uncertain_label == "Ones":
          label.append(1)
        elif self.uncertain_label == "Zeros":
          label.append(0)
        elif self.uncertain_label == "LSR-Ones":
          label.append(random.uniform(0.55, 0.85))
        elif self.uncertain_label == "LSR-Zeros":
          label.append(random.uniform(0, 0.3))
      else:
        label.append(l)
    imageLabel = torch.FloatTensor(label)

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------Downstream Shenzhen------------------------------------------
class ShenzhenCXR(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=1, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split(',')

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------Downstream VinDrCXR------------------------------------------
class VinDrCXR(Dataset):
    def __init__(self, images_path, file_path, augment, num_class=6, annotation_percent=100):
        self.img_list = []
        self.img_label = []
        self.augment = augment

        with open(file_path, "r") as fr:
            line = fr.readline().strip()
            while line:
                lineItems = line.split()
                imagePath = os.path.join(images_path, lineItems[0]+".jpeg")
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                self.img_list.append(imagePath)
                self.img_label.append(imageLabel)
                line = fr.readline()

        if annotation_percent < 100:
            indexes = np.arange(len(self.img_list))
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * annotation_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []

            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])

    def __getitem__(self, index):

        imagePath = self.img_list[index]
        imageLabel = torch.FloatTensor(self.img_label[index])
        imageData = Image.open(imagePath).convert('RGB')
        if self.augment != None: imageData = self.augment(imageData)
        return imageData, imageLabel
    def __len__(self):

        return len(self.img_list)

# ---------------------------------------------Downstream RSNA Pneumonia------------------------------------------
class RSNAPneumonia(Dataset):

  def __init__(self, images_path, file_path, augment, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.strip().split(' ')
          imagePath = os.path.join(images_path, lineItems[0])


          self.img_list.append(imagePath)
          self.img_label.append(int(lineItems[-1]))

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = np.zeros(3)
    imageLabel[self.img_label[index]] = 1
    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------MIMIC Dataset------------------------------------------
class MIMIC_CXR(Dataset):

    def label_reorganize(self, goal_meta_list, current_meta_list, label_list):
        labels = []
        for i, disease in enumerate(goal_meta_list):
            index = current_meta_list.index(disease)
            label = label_list[index]
            ##Convert Datatype
            if label and '.' in label:
                label = int(float(label))
            elif label == '': label=0
            else: label = int(label)

            if label > 0: labels.append(1)
            else: labels.append(0)

        return labels

    def __init__(self, dataset_directory, partition, augment, diseases, from_modality="report", to_modality="chexpert", uncertain_label="LSR-Ones", unknown_label=0, annotation_percent=100):
        
        self.from_modality = from_modality
        self.to_modality = to_modality
        self.augment = augment
        self.uncertain_label = uncertain_label
    
        metadata_directory = os.path.join(dataset_directory, "physionet.org", "files", "mimic-cxr-jpg", "2.0.0")
        filepaths_file_path = os.path.join(metadata_directory, "cxr_paths.csv")
        chexpert_labels_file_path = os.path.join(metadata_directory, "mimic-cxr-2.0.0-chexpert.csv.gz")
        negbio_labels_file_path = os.path.join(metadata_directory, "mimic-cxr-2.0.0-negbio.csv.gz")
        partitions_file_path = os.path.join(metadata_directory, "mimic-cxr-2.0.0-split.csv.gz") 

        diseases=diseases

        MIMIC_CXR_home_directory = "/data/jliang12/jpang12/dataset/MIMIC"
        MIMIC_CXR_data_directory = os.path.join(MIMIC_CXR_home_directory, "physionet.org", "files", "mimic-cxr", "2.0.0", "files")

        studies = []

        with open(filepaths_file_path, 'r') as studies_file:
            csv_reader=csv.reader(studies_file)
            next(csv_reader) #skip header
            for file_path in csv_reader:
                #Populate list of studies, each study is a list: [patientId, studyId, imageId, image path]

                ##study starts with last three file directory/file references: ../patientId/studyid/imageid.jpg
                study = file_path[0].split("/")[-3:]
                ##strip .jpg off imageid's
                study[2] = study[2].split(".")[0]
                ##add full file_path to study image
                study.append(file_path[0])
                ##add full report_path to study report
                report_path = os.path.join(MIMIC_CXR_data_directory, *file_path[0].split("/")[-4:-1]) + ".txt"
                study.append(report_path)
                studies.append(study)

        study_keys = ['subject_id', 'study_id', 'image_id', 'image_path', 'study_path']
        studies = [dict(zip(study_keys, study)) for study in studies]
        #this preps the data for joining later by nominating the image_id as the dictionary key
        studies_dict = {study['image_id']: study for study in studies}

        partitions = []

        with gzip.open(partitions_file_path, 'rb') as partitions_file:
            decoded_partitions_file = partitions_file.read().decode('utf-8')
            csv_reader=csv.reader(decoded_partitions_file.splitlines(), delimiter=',')
            next(csv_reader) #skip header
            for data in csv_reader:
                partitions.append(data)

        partition_keys=['image_id', 'study_id', 'subject_id', 'split']
        partitions = [dict(zip(partition_keys, partition)) for partition in partitions]

        chexpert_labels = []

        with gzip.open(chexpert_labels_file_path, 'rb') as chexpert_labels_file:
            decoded_chexpert_labels_file = chexpert_labels_file.read().decode('utf-8')
            csv_reader=csv.reader(decoded_chexpert_labels_file.splitlines(), delimiter=',')
            chexpert_header = next(csv_reader)
            for data in csv_reader:
                label_data = data[0:2]
                labels = data[2:]
                labels = self.label_reorganize(diseases, chexpert_header[2:], labels)
                label_data.append(labels)
                chexpert_labels.append(label_data)

        chexpert_keys = ['subject_id', 'study_id', 'label_chexpert']
        chexpert_labels = [dict(zip(chexpert_keys, chexpert_label)) for chexpert_label in chexpert_labels]
        #this preps the data for joining later by nominating (subject_id, study_id) as the dictionary key
        chexpert_labels_dict = {(chexpert_label['subject_id'], chexpert_label['study_id']): chexpert_label for chexpert_label in chexpert_labels}

        negbio_labels = []

        with gzip.open(negbio_labels_file_path, 'rb') as negbio_labels_file:
            decoded_negbio_labels_file = negbio_labels_file.read().decode('utf-8')
            csv_reader=csv.reader(decoded_negbio_labels_file.splitlines(), delimiter=',')
            negbio_header = next(csv_reader)
            for data in csv_reader:
                label_data = data[0:2]
                labels = data[2:]
                labels = self.label_reorganize(diseases, negbio_header[2:], labels)
                label_data.append(labels)
                negbio_labels.append(label_data)

        negbio_keys = ['subject_id', 'study_id', 'label_negbio']
        negbio_labels = [dict(zip(negbio_keys, negbio_label)) for negbio_label in negbio_labels]
        negbio_labels_dict = {(negbio_label['subject_id'], negbio_label['study_id']): negbio_label for negbio_label in negbio_labels}

        self.mimic_data = []
        #Combine ALL data
        for partition_data in partitions:
            if partition_data["split"] == partition:
                image_data = {}

                combined_data = [partition_data]

                #get data from all other dicts
                study_data = studies_dict.get(partition_data["image_id"])
                if study_data: combined_data.append(study_data)

                chexpert_label_data = chexpert_labels_dict.get((partition_data["subject_id"], partition_data["study_id"]))
                if chexpert_label_data: combined_data.append(chexpert_label_data)

                negbio_label_data = negbio_labels_dict.get((partition_data["subject_id"], partition_data["study_id"]))
                if negbio_label_data: combined_data.append(negbio_label_data)

                for sub_dict in combined_data:
                    for key, value in sub_dict.items():
                        if key not in image_data:
                            image_data[key] = value

                #For now only Chexpert labels are double checked for existing
                if 'label_chexpert' in image_data:        
                    self.mimic_data.append(image_data)
                    
        indexes = np.arange(len(self.mimic_data))
        if annotation_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * annotation_percent / 100.0)
            indexes = indexes[:num_data]

            _mimic_data = copy.deepcopy(self.mimic_data)
            self.mimic_data = []

            for i in indexes:
                self.img_list.append(_mimic_data[i])
    
    def __getitem__(self, index):
        mimic_data = self.mimic_data[index]
        
        #From Modality Options: Image, Report, Both
        if self.from_modality in ['image', 'both']:
            image_data = Image.open(mimic_data['image_path']).convert('RGB')            
            if self.augment != None: image_data = self.augment(image_data)
            
        # if self.from_modality in ['report', 'both']:
        #     source['report'] = transform_report(mimic_data['study_path'])
        
        #To Modality Options: negbio, chexpert, both
        
        if self.to_modality in ['negbio', 'both']:
            raw_label = torch.FloatTensor(mimic_data['label_negbio'])
        if self.to_modality in ['chexpert', 'both']:
            raw_label = torch.FloatTensor(mimic_data['label_chexpert'])
            
        label = []
        for l in raw_label:
            if l == -1:
                if self.uncertain_label == "Ones":
                    label.append(1)
                elif self.uncertain_label == "Zeros":
                    label.append(0)
                elif self.uncertain_label == "LSR-Ones":
                    label.append(random.uniform(0.55, 0.85))
                elif self.uncertain_label == "LSR-Zeros":
                    label.append(random.uniform(0, 0.3))
            else:
                label.append(l)

        imageLabel = torch.FloatTensor(label)

        return image_data, imageLabel
    
    def __len__(self):
        return len(self.mimic_data)