import sys
import numpy as np

import pickle
import os
from torch.utils.data import random_split, DataLoader
# from dataset import MNISTDataset, CIFARDataset
from path import Path

import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(
        self,
        subset=None,
        data=None,
        targets=None,
        transform=None,
        target_transform=None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        if (data is not None) and (targets is not None):
            self.data = data.unsqueeze(1)
            self.targets = targets
        elif subset is not None:
            self.data = torch.stack(
                list(
                    map(
                        lambda tup: tup[0]
                        if isinstance(tup[0], torch.Tensor)
                        else torch.tensor(tup[0]),
                        subset,
                    )
                )
            )
            self.targets = torch.stack(
                list(
                    map(
                        lambda tup: tup[1]
                        if isinstance(tup[1], torch.Tensor)
                        else torch.tensor(tup[1]),
                        subset,
                    )
                )
            )
        else:
            raise ValueError(
                "Data Format: subset: Tuple(data: Tensor / Image / np.ndarray, targets: Tensor) OR data: List[Tensor]  targets: List[Tensor]"
            )

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return data, targets

    def __len__(self):
        return len(self.targets)


class CIFARDataset(Dataset):
    def __init__(
        self,
        subset=None,
        data=None,
        targets=None,
        transform=None,
        target_transform=None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        if (data is not None) and (targets is not None):
            self.data = data.unsqueeze(1)
            self.targets = targets
        elif subset is not None:
            self.data = torch.stack(
                list(
                    map(
                        lambda tup: tup[0]
                        if isinstance(tup[0], torch.Tensor)
                        else torch.tensor(tup[0]),
                        subset,
                    )
                )
            )
            self.targets = torch.stack(
                list(
                    map(
                        lambda tup: tup[1]
                        if isinstance(tup[1], torch.Tensor)
                        else torch.tensor(tup[1]),
                        subset,
                    )
                )
            )
        else:
            raise ValueError(
                "Data Format: subset: Tuple(data: Tensor / Image / np.ndarray, targets: Tensor) OR data: List[Tensor]  targets: List[Tensor]"
            )

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return img, targets

    def __len__(self):
        return len(self.targets)

class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):

        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)


class affectnet_mead_seed_DataSet(Dataset): #affectnet=1408 MEAD=2048    SEED=310
        def __init__(self, datas):
            super(affectnet_mead_seed_DataSet, self).__init__()
            self.length = len(datas)
            self.datas = datas
            
        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            
            image=self.datas[idx][0]
            lable=self.datas[idx][1]
            sample=[]
            sample.append(image)
            sample.append(lable)
            return sample


CURRENT_DIR = Path(__file__).parent.abspath()


def get_dataloader(client_dataset, client_id: int, batch_size=20):
    

    
    data_num= torch.load('/home/tjut_lishuai/meta-fed/data/data_split/client_'+str(client_id)+'.pt')
    
    random_indices = np.random.choice(len(data_num), size=len(data_num), replace=False) 
    train_indices = random_indices[:int(0.5 * len(data_num))]
    test_indices = random_indices[int(0.5 * len(data_num)):]

    trainloader = DataLoader(client_dataset, batch_size=batch_size,sampler=SubsetSequentialSampler(train_indices),num_workers=0,pin_memory=False,drop_last=True)
    valloader = DataLoader(client_dataset, batch_size=batch_size,sampler=SubsetSequentialSampler(test_indices),num_workers=0,pin_memory=False,drop_last=True)

    return trainloader, valloader


def get_client_id_indices(dataset):
    dataset_pickles_path = CURRENT_DIR / dataset / "pickles"
    with open(dataset_pickles_path / "seperation.pkl", "rb") as f:
        seperation = pickle.load(f)
    return (seperation["train"], seperation["test"], seperation["total"])


def aggregate_pro_parameters(client_params):
    
    num_clients = len(client_params)
    weights = torch.zeros(num_clients)

    # 
    for i in range(num_clients):
        for j in range(num_clients):
            if i != j:
                #  Frobenius 范数
                distance = torch.norm(client_params[i] - client_params[j], p='fro')
                weights[i] += 1 / (distance.cpu() + 1e-10)  # 

    # 归一化权重
    weights /= weights.sum()

    aggregated_params = sum(w * p for w, p in zip(weights, client_params))
    return aggregated_params

def data_deal():
    print('Starting to read file data.................')
    affectnet_data = torch.load('/data/affectnet/affectnet.pt',map_location='cpu') #affectnet
    mead_data = torch.load('/data/MEAD/mead.pt',map_location='cpu') #mead
    seed_data = torch.load('/data/seed/seed.pt',map_location='cpu') #seed
    print('Reading completed')
    data=[affectnet_data,mead_data,seed_data]

    return data


def angular_distance(client_params, center_param):
    
    # center_param = center_param.unsqueeze(0) 
    difference_vectors = []

    for param in client_params:
        difference_vector = param - center_param 
        difference_vectors.append(difference_vector)

    difference_vectors= torch.stack(difference_vectors)  
    normalized_vectors =  torch.nn.functional.normalize(difference_vectors, p=2, dim=1)
    similarity_matrix = torch.mm(normalized_vectors, normalized_vectors.t())  
    similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
    
    
    inverse_similarity = torch.acos(similarity_matrix) / np.pi
    
    
    angular_dis_matrix= 1 - inverse_similarity

    
    from sklearn.cluster import SpectralClustering
    
    clustering = SpectralClustering(n_clusters=4, affinity='precomputed', random_state=42)
    labels = clustering.fit_predict(angular_dis_matrix)
    return labels

def cluster_parameters(params_list, labels,global_models):
    
    clustered_params = {}

    
    for param, label in zip(params_list, labels):
        if label not in clustered_params:
            clustered_params[label] = []
        clustered_params[label].append(param)
    from fedlab.utils.aggregator import Aggregators
    from fedlab.utils.serialization import SerializationTool
    
    aggregated_params = []
    for key in sorted(clustered_params.keys()):
        aggregated_param = Aggregators.fedavg_aggregate(clustered_params[key])
        aggregated_params.append(aggregated_param)
    
    for i in range(4): #n_clusters
        SerializationTool.deserialize_model(global_models[i], aggregated_params[i])
    return global_models
