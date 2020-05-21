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

# def build_model(model_config):
#     return models.DeepModel(model_config)

def setup_log_file(log_folder_path, log_name):
    os.makedirs(log_folder_path, exist_ok=True)
    l_path = os.path.join(log_folder_path, "log_{}.txt".format(log_name))
    sys.stdout = logger.Logger(l_path)

def gpu_selector(gpu_to_use=-1):
    """

    Returns
    -------
    int
        The GPU device to use

    TODO: it does not make sense to set CUDA_VISIBLE_DEVICES when PyTorch is already imported.
    Must use another way to set GPU device. This could be refactored...
    """
    gpu_to_use = -1 if gpu_to_use == None else gpu_to_use
    deviceCount = nvmlDeviceGetCount()
    if gpu_to_use < 0:
        print("Auto selecting GPU") 
        gpu_free_mem = 0
        for i in range(deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            mem_usage = nvmlDeviceGetMemoryInfo(handle)
            if gpu_free_mem < mem_usage.free:
                gpu_to_use = i
                gpu_free_mem = mem_usage.free
            print("GPU: {} \t Free Memory: {}".format(i, mem_usage.free))
    print("Using GPU {}".format(gpu_to_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)
    return gpu_to_use

class PatchHanger(object):
    '''Hanger with functionality to create torch data loaders for patch dataset

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
    '''

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

    def create_data_loader(self, chunk_ids, color_jitter=False, shuffle=False):
        patch_paths = self.load_chunks(chunk_ids)
        labels = self.extract_labels(patch_paths)
        patch_dataset = PatchDataset(patch_paths, labels, color_jitter=color_jitter)
        return DataLoader(patch_dataset, batch_size=self.batch_size, 
                shuffle=shuffle, num_workers=self.num_patch_workers)
