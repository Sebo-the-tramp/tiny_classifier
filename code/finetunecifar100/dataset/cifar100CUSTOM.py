import torch
import torch.nn as nn
import torchvision

import os.path
import pickle

import numpy as np
from PIL import Image

from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.utils import check_integrity

clustering_mapping = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2, 30: 3, 31: 3, 32: 3, 33: 3, 34: 3, 35: 3, 36: 3, 37: 3, 38: 3, 39: 3, 40: 4, 41: 4, 42: 4, 43: 4, 44: 4, 45: 4, 46: 4, 47: 4, 48: 4, 49: 4, 50: 5, 51: 5, 52: 5, 53: 5, 54: 5, 55: 5, 56: 5, 57: 5, 58: 5, 59: 5, 60: 6, 61: 6, 62: 6, 63: 6, 64: 6, 65: 6, 66: 6, 67: 6, 68: 6, 69: 6, 70: 7, 71: 7, 72: 7, 73: 7, 74: 7, 75: 7, 76: 7, 77: 7, 78: 7, 79: 7, 80: 8, 81: 8, 82: 8, 83: 8, 84: 8, 85: 8, 86: 8, 87: 8, 88: 8, 89: 8, 90: 9, 91: 9, 92: 9, 93: 9, 94: 9, 95: 9, 96: 9, 97: 9, 98: 9, 99: 9}
complete_mapping   = {4: 0, 30: 1, 55: 2, 72: 3, 95: 4, 1: 5, 32: 6, 67: 7, 73: 8, 91: 9, 54: 10, 62: 11, 70: 12, 82: 13, 92: 14, 47: 15, 52: 16, 56: 17, 59: 18, 96: 19, 0: 20, 51: 21, 53: 22, 57: 23, 83: 24, 9: 25, 10: 26, 16: 27, 28: 28, 61: 29, 22: 30, 39: 31, 40: 32, 86: 33, 87: 34, 5: 35, 20: 36, 25: 37, 84: 38, 94: 39, 6: 40, 7: 41, 14: 42, 18: 43, 24: 44, 26: 45, 45: 46, 77: 47, 79: 48, 99: 49, 23: 50, 33: 51, 49: 52, 60: 53, 71: 54, 12: 55, 17: 56, 37: 57, 68: 58, 76: 59, 3: 60, 42: 61, 43: 62, 88: 63, 97: 64, 15: 65, 19: 66, 21: 67, 31: 68, 38: 69, 34: 70, 63: 71, 64: 72, 66: 73, 75: 74, 36: 75, 50: 76, 65: 77, 74: 78, 80: 79, 8: 80, 13: 81, 48: 82, 58: 83, 90: 84, 41: 85, 69: 86, 81: 87, 85: 88, 89: 89, 2: 90, 11: 91, 35: 92, 46: 93, 98: 94, 27: 95, 29: 96, 44: 97, 78: 98, 93: 99}
inverse_complete = {}

class CIFAR100CUSTOM(torchvision.datasets.CIFAR100):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        coarse: bool = True
    ) -> None:

        self.coarse = coarse

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                # map into the correcto order for hierarchy purposes
                entry["fine_labels"] = [complete_mapping[x] for x in entry["fine_labels"]]                           
                if(self.coarse):                   
                    self.targets.extend([clustering_mapping[x] for x in entry["fine_labels"]])                    
                else:                    
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            #print(data['fine_label_names'])
            if(self.coarse):
                #self.classes = data['coarse_label_names']                
                self.classes = ["aquatic", "plants", "food", "houshold", "insects", "outdoor_scenes", "large_animals", "medium_animals", "vehicles", "other"]
            else:
                self.classes = data['fine_label_names']

        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}