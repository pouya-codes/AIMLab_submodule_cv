import os
import json
import time
import sys
import enum

import yaml
from tqdm import tqdm
from pynvml import *
import numpy as np
import torch
import torchvision
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader

import submodule_utils as utils
from submodule_cv.dataset import PatchDataset
import submodule_cv.logger as logger
import submodule_cv.models as models

# Folder permission mode
p_mode = 0o777
oldmask = os.umask(000)
nvmlInit()

class ChunkLookupException(Exception):
    pass

def setup_log_file(log_folder_path, log_name):
    os.makedirs(log_folder_path, exist_ok=True)
    l_path = os.path.join(log_folder_path, "log_{}.txt".format(log_name))
    sys.stdout = logger.Logger(l_path)

def gpu_selector(gpu_to_use=-1, number_of_gpus=1):
    """
    Returns
    -------
    list(int)
        A list of GPU(s) device(s) to use
    """
    gpu_to_use = -1 if gpu_to_use == None else gpu_to_use
    if gpu_to_use < 0 :
        deviceCount = torch.cuda.device_count()
        number_of_gpus = min(number_of_gpus, deviceCount)
        print(f"Auto selecting {number_of_gpus} GPU(s) from exising {deviceCount} GPU(s)")
        list_of_gpus = np.argsort([nvmlDeviceGetMemoryInfo(handle).free for handle in
                                   [nvmlDeviceGetHandleByIndex(gpu_id) for gpu_id in range(deviceCount) ]])[-number_of_gpus:][::-1]
        print(f"Using GPU(s): {list_of_gpus}")
        return list_of_gpus
    else:
        return [gpu_to_use]

class PatchHanger(object):
    """Hanger with functionality to create torch data loaders for patch dataset

    Attributes
    ----------
    patch_pattern : dict
        Directory describing patch path structure

    CategoryEnum : enum.Enum
        Create CategoryEnum used to group patch paths. The key in the CategoryEnum is the string fragment in the patch path to lookup and the value in the CategoryEnum is group ID to group the patch.

    is_binary : bool
        Whether we want to categorize patches by the Tumor/Normal category (true) or by the subtype category (false).

    batch_size : int
        Batch size to use on training, validation and test dataset.

    num_patch_workers : int
        Number of loader worker processes to multi-process data loading.

    chunk_file_location : str
        File path of group or split file (aka. chunks) to use (i.e. /path/to/patient_3_groups.json).
    
    model_config_location : str
        Path to model config JSON (i.e. /path/to/model_config.json).
    """

    def load_model_config(self):
        '''Load the model config JSON file as a dict

        Returns
        -------
        dict
            The model config
        '''
        return utils.load_json(self.model_config_location)
    
    def build_model(self, device=None):
        '''Builds model by reading file specified in model config path

        Returns
        -------
        models.DeepModel
        '''
        return models.DeepModel(self.load_model_config(), device=device)

    def load_chunks(self, chunk_ids):
        """Load patch paths from specified chunks in chunk file

        Parameters
        ----------
        chunks : list of int
            The IDs of chunks to retrieve patch paths from

        Returns
        -------
        list of str
            Patch paths from the chunks
        """
        patch_paths = []
        with open(self.chunk_file_location) as f:
            data = json.load(f)
            chunks = data['chunks']
            for chunk in data['chunks']:
                if chunk['id'] in chunk_ids:
                    patch_paths.extend(chunk['imgs'])
        if len(patch_paths) == 0:
            raise ChunkLookupException(
                    f"chunks {tuple(chunk_ids)} not found in {self.chunk_file_location}")
        return patch_paths
        
    def extract_label_from_patch(self, patch_path):
        """Get the label value according to CategoryEnum from the patch path

        Parameters
        ----------
        patch_path : str

        Returns
        -------
        int
            The label id for the patch
        """
        '''
        Returns the CategoryEnum
        '''
        patch_id = utils.create_patch_id(patch_path, self.patch_pattern)
        label = utils.get_label_by_patch_id(patch_id, self.patch_pattern,
                self.CategoryEnum, is_binary=self.is_binary)
        return label.value

    def extract_labels(self, patch_paths):
        return list(map(self.extract_label_from_patch, patch_paths))

    def create_data_loader(self, chunk_ids, shuffle=False):
        patch_paths = self.load_chunks(chunk_ids)
        labels = self.extract_labels(patch_paths)
        patch_dataset = PatchDataset(patch_paths, labels, self.model_config['normalize'] if "normalize" in self.model_config else None)
        return DataLoader(patch_dataset, batch_size=self.batch_size, 
                shuffle=shuffle, num_workers=self.num_patch_workers)

# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

