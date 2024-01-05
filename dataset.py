import os
import torch
import numpy as np
import pickle
import random
from collections import defaultdict
import torchvision.transforms as transforms
import pandas as pd

import monai
from monai.apps import download_and_extract
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)
import torch

data_dir = '.'
fold_divisions = [.8, .9, 1]

class MRIDataset(torch.utils.data.Dataset):
  "Dataset for Pytorch"
  # sources is a list of strings which each string represents a source dataset.
  def __init__(self, list_IDs, labels, task, data_dir=data_dir):
        'Initialization'
        self.labels = labels # maps id to label
        self.list_IDs = list_IDs # Maps index to id
        self.task = task
        self.data_dir = data_dir

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        fileID = self.list_IDs[index]
        X = np.load(os.path.join(self.data_dir, fileID +  '.npy'))
        if X.shape[0] == 1:
            X = X.squeeze(0)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])  
        X = transform(X).unsqueeze(0).float()

        y = self.labels[fileID][self.task]
        #y = self.labels[fileID]

        tensors = []
        labels = []
        tensors.append(X)
        labels.append(y)

        return {'tensor':tensors, 'label': labels}

# Returns fold corresponding to i.
# total is total samples
def get_fold(i, total, folds):
    fold = folds[0]
    for fold_num,f in enumerate(folds[1:]):
        if i >= total * fold_divisions[fold_num] and i < total * fold_divisions[fold_num+1]:
            return fold_num + 1, f
    return 0, fold

"""This function packages the stored dataset into the MRIDataset torch dataset class implemented above. We have 
included this code as reference – due to dataset access limitations with ABCD and NCANDA, we are 
unable to share the preprocessed dataset and labels publically.
To easily use our provided training loop, we would recommend implementing your own get_dataset 
function for the dataset you'd like to use, which returns MRIDataset objects.
Returns:
    5 MRIDataset objects, which are respectively the train, augmented train, validation and 
    test sets, as well as source sets. The source sets are the dataset specific validation and 
    test sets – since we combined two datasets (ABCD and NCANDA) in our study, we extracted the 
    dataset-specific val and test sets for metric computation as well. 
"""

def get_dataset_JB(data_dir, augment=True):

    # Loading MRI filenames and labels into lists
    labels_pd = pd.read_excel(
        os.path.join(data_dir, 'IXI.xls')).drop_duplicates(subset='IXI_ID').set_index('IXI_ID')['SEX_ID (1=m, 2=f)']

    images = []
    labels = []
    valid_IDs = []
    for MRI_filename in os.listdir(os.path.join(data_dir, 'IXI-T1')):
        print(int(MRI_filename[3:6]))
        MRI_ID = int(MRI_filename[3:6])
        if MRI_ID in labels_pd.keys():
            images.append(MRI_filename)
            valid_IDs.append(MRI_ID)
            labels.append(labels_pd[MRI_ID])
    labels_pd = labels_pd[labels_pd.index.isin(valid_IDs)]
    # Define transforms
    if augment:
        train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((64, 64, 64)), RandRotate90()])
        val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((64, 64, 64))])

    else:
        train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((64, 64, 64))])
        val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((64, 64, 64))])

    # Define nifti dataset, data loader
    check_ds = ImageDataset(image_files=images.copy(), labels=labels, transform=train_transforms)
    check_ds.list_IDs = {}
    check_ds.data_dir = data_dir
    check_ds.labels_pd = labels_pd
    for i in range(len(check_ds.image_files)):
        check_ds.list_IDs[i] = int(check_ds.image_files[i][3:6])
        check_ds.image_files[i] = os.path.join(data_dir, 'IXI-T1', check_ds.image_files[i])
        
    # check_loader = DataLoader(check_ds, batch_size=3, num_workers=2, pin_memory=pin_memory)

    # im, label = monai.utils.misc.first(check_loader)
    # print(type(im), im.shape, label, label.shape)

    # create a training data loader
    train_ds = ImageDataset(image_files=images[:10].copy(), labels=labels[:10], transform=train_transforms)
    train_ds.list_IDs = {}
    train_ds.data_dir = data_dir
    for i in range(len(train_ds.image_files)):
        train_ds.list_IDs[i] = int(train_ds.image_files[i][3:6])
        train_ds.image_files[i] = os.path.join(data_dir, 'IXI-T1', train_ds.image_files[i])
    # train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=pin_memory)

    # create a validation data loader
    val_ds = ImageDataset(image_files=images[-10:].copy(), labels=labels[-10:], transform=val_transforms)
    val_ds.list_IDs = {}
    val_ds.data_dir = data_dir
    for i in range(len(val_ds.image_files)):
        val_ds.list_IDs[i] = int(val_ds.image_files[i][3:6])
        val_ds.image_files[i] = os.path.join(data_dir, 'IXI-T1', val_ds.image_files[i])
    # val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=pin_memory)

    return check_ds, train_ds, val_ds

    






def get_dataset(sources, task, data_dir, augment=False):
    task = task[0]

    with open('./parsing/parsed_labels/patient_labels.pkl', 'rb') as f:
        patient_labels = pickle.load(f)
    with open('./parsing/parsed_labels/file_labels.pkl', 'rb') as f:
        file_labels = pickle.load(f)
    
    train_labels, val_labels, test_labels = {}, {}, {}
    train_IDs, val_IDs, test_IDs = {}, {}, {}
    # {abcd: [{}, {}], ncanda: [{}, {}]} dict from
    source_splits = [s + '_val' for s in sources] + [s + '_test' for s in sources] 
    source_files = dict(map(lambda s: (s, [{}, {}]), source_splits))
    
    # Find subject IDs that correspond to datasets in the list sources
    # and contain labels that correspond to tasks.
    # this is mapping from patient id to list of fileIDs corresponding patient.
    patient_files = defaultdict(list)
    for fileID in file_labels:
        file_source = list(filter(lambda s: s in fileID, sources))
        # continue if current file doesn't come from the specified sources or specified task not in label
        if not file_source or task not in file_labels[fileID]:
            continue

        # find corresponding patient for file ID.
        pID = None
        for (d, curID) in patient_labels:
            # chop off dataset in front of fileID
            if curID in fileID:
                pID = curID
        
        patient_files[pID].append(fileID)

    folds = [[train_labels, train_IDs], [val_labels, val_IDs], [test_labels, test_IDs]]
    
    # Now split patients among the folds.
    keys = list(patient_files.keys())
    random.shuffle(keys)
    for i, pID in enumerate(keys):
        files = patient_files[pID]
        fold_num, [cur_labels, cur_IDs] = get_fold(i, len(patient_files), folds)
        
        for fileID in files:
            cur_labels[fileID] = file_labels[fileID]
            cur_IDs[len(cur_IDs)] = fileID
            if fold_num > 0:
                fold = list(filter(lambda s: s in fileID, sources))[0]
                fold += '_val' if fold_num == 1 else '_test'
                source_labels, source_IDs = source_files[fold][1], source_files[fold][0]
                source_labels[fileID] = file_labels[fileID]
                source_IDs[len(source_IDs)] = fileID
    
    if not augment:
        train_augment_IDs, train_augment_labels = train_IDs.copy(), train_labels.copy()
    else:
        train_augment_IDs, train_augment_labels = {}, {}
        # Now load in the augmented samples for trainset_augment.
        for pklFile in os.listdir(f"{data_dir}/gen"):
            if 'pkl' in pklFile and list(filter(lambda s: s in pklFile, sources)) is not None:
                # load in all generated files.
                with open(os.path.join(f"{data_dir}/gen", pklFile), 'rb') as f:
                    cur_mapping = pickle.load(f)

                for fileID in set(train_IDs.values()):
                    # For fileID in training set, add all corresponding generated files to trainset_augment.
                    for genID in cur_mapping[fileID]:
                        train_augment_labels[genID] = file_labels[fileID]
                        train_augment_IDs[len(train_augment_IDs)] = genID

    # Split into train/val/test sets.
    trainset = MRIDataset(train_IDs, train_labels, task, data_dir)
    trainset_augment = MRIDataset(train_augment_IDs, train_augment_labels, task, data_dir)
    valset = MRIDataset(val_IDs, val_labels, task, data_dir)
    testset = MRIDataset(test_IDs, test_labels, task, data_dir)
    source_sets = dict(map(lambda s: (s, MRIDataset(*source_files[s], task, data_dir)), source_splits))

    return trainset, trainset_augment, valset, testset, source_sets
