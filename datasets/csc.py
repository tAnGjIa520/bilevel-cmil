import pandas as pd
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


############################################# MIL DATASET ##################################################
class MILDataset(Dataset):

    def __init__(self, data_frame, data_root='./local_data', max_instances=-1):

        """Dataset object for MIL embeddings.
        Reads data from a dataframe and loads the patch-level embeddings.

        Args:
            data_frame (pd.DataFrame): DataFrame with ground truth information for each bag.
            max_instances (int, optional): Maximum number of instances per bag. Defaults to 200.
        """
        self.data_frame = data_frame
        self.classes = "GT"
        self.images = self.data_frame['Patient'].values  # WSI identifiers
        self.targets = self.data_frame['GT'].values  # Ground truth
        self.data_root = data_root

        # Adaptation for incremental learning
        order = np.array(['2', '4', '0', '5', '3', '1'])
        for idx, lab in enumerate(self.targets.tolist()):
            self.targets[idx] = np.asarray(np.where(order == lab)[0], dtype=str)[0]

        # Loading the embeddings
        self.patch_embeddings = []
        for wsi_id in tqdm(self.images):
            npy_id = os.path.join(self.data_root, 'embeddings', wsi_id + '.npy')
            npy_embeddings = np.load(npy_id)
            self.patch_embeddings.append(npy_embeddings)

        self.max_instances = max_instances

    def __len__(self):
        """Denotes the total number of samples/patients"""
        return len(self.images)

    def __getitem__(self, idx):
        """Generates one sample of data"""
        x = self.patch_embeddings[idx]
        y = int(self.targets[idx])

        # Limit the number of patches if necessary
        if self.max_instances > 0 and x.shape[0] > self.max_instances:
            idx_random = np.random.choice(x.shape[0], self.max_instances, replace=False)
            x = x[idx_random]

        # Convert to torch tensors
        x = torch.tensor(x.astype('float32'))
        y = torch.tensor(y, dtype=torch.int64)
        return {'features': x, 'label': y, 'slide_id': idx}