import os
import torch
import torch.utils
import torchvision.transforms as transforms
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import MNIST, ImageFolder, DatasetFolder
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import ImageFile, Image

import gzip
import zipfile
import tarfile
from typing import Optional, Callable
from torch.utils.model_zoo import tqdm
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity

from utils.partition import partition_data
from utils.partition import record_net_data_stats
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_data(data_name):
    """
    Get the dataset class from the dataset name.
    """

    datalist = {'off_home': 'img_union_test', 'pacs': 'img_union_test', 'vlcs': 'img_union_test', 
                'medmnist': 'medmnist', 'medmnistA': 'medmnist', 'medmnistC': 'medmnist', 
                'pamap': 'pamap', 'covid19': 'covid', 'cifar10': 'cifar', 'cifar100':'cifar',
                'femnist':'femnist', 
                'imagenet1000':'image_net','imagenet100':'image_net',
                'mnist':'mnist', 'Fashionmnist':'fashionmnist', 'svhn':'svhn'                 
                }
    if datalist[data_name] not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(data_name))
    return globals()[datalist[data_name]]

class MedMnistDataset(Dataset):
    """
    Fuction: 
        A custom dataset class for the MedMNIST dataset.
    Args:
        root (str): The root directory of the dataset.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and returns a transformed version.
    """

    def __init__(self, root='', transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data = np.load(root + 'xdata.npy')
        self.targets = np.load(root + 'ydata.npy')
        self.classes = np.unique(self.targets)
        self.targets = np.squeeze(self.targets)
        self.data = torch.Tensor(self.data)
        self.data = torch.unsqueeze(self.data, dim=1)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.targets)
    
    def get_data_idx(self, dataidxs=None):
        if dataidxs is not None:
            data = self.data[dataidxs]
            target = self.targets[dataidxs]
        return data, target

def medmnist(args):      
    data = MedMnistDataset(args.root_dir + args.dataset + '/')       
    return get_dataloader(args, data, args.n_parties, args.partition, args.beta)


class PamapDataset(Dataset):
    def __init__(self, data_dir='', transform=None, target_transform=None):        
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

        self.data = np.load(data_dir + 'x.npy')
        self.targets = np.load(data_dir + 'y.npy')        
        self.select_class()

        self.data = torch.unsqueeze(torch.Tensor(self.data), dim=1)
        self.data = torch.einsum('bxyz->bzxy', self.data)
        self.classes = np.unique(self.targets)        

    def select_class(self):
        """
        Function:
            Selects specific classes from the dataset and updates the targets and data accordingly.
        Returns:
            None
        """
        # Select the classes to delete
        delete_classes = [0, 5, 12]
        index = []
        # Find the indices of the classes to delete
        for ic in delete_classes:
            index.append(np.where(self.targets == ic)[0])
        # Concatenate the indices
        index = np.hstack(index)
        # Update the targets and data
        all_index = np.arange(len(self.targets))
        all_index = np.delete(all_index, index)
        self.targets = self.targets[all_index]
        self.data = self.data[all_index]
        
        # Convert the labels to continuous integers starting from 0
        ry = np.unique(self.targets)
        ry2 = {}
        for i in range(len(ry)):
            ry2[ry[i]] = i
        for i in range(len(self.targets)):
            self.targets[i] = ry2[self.targets[i]]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x, target = self.data[index], self.targets[index]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return x, target

def pamap(args):   
    data = PamapDataset(args.root_dir + args.dataset + '/')  
    return get_dataloader(args, data, args.n_parties, args.partition, args.beta)
        

class CovidDataset(Dataset):
    def __init__(self, filename='../data/covid19/', transform=None):
        self.data = np.load(filename + 'xdata.npy')
        self.targets = np.load(filename + 'ydata.npy')
        self.targets = np.squeeze(self.targets)
        self.transform = transform
        self.data = torch.Tensor(self.data)
        self.data = torch.einsum('bxyz->bzxy', self.data)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def covid(args):   
    data = CovidDataset(args.root_dir + args.dataset + '/')  
    return get_dataloader(args, data, args.n_parties, args.partition, args.beta)


class Image_Dataset(Dataset):
    """
    row data structure:
        root_dir/
        |    domain1_dir/
        |        ├── class1/
        |        │   ├── img1.jpg
        |        │   ├── img2.png
        |        │   └── ...
        |        ├── class2/
        |        │   ├── img3.jpg
        |        │   ├── img4.png
        |        │   └── ...
        |        └── ...
        |    domain2_dir/
        |        ├── class1/
        |        │   ├── img1.jpg
        |        │   ├── img2.png
        |        │   └── ...
        |        ├── class2/
        |        │   ├── img3.jpg
        |        │   ├── img4.png
        |        │   └── ...
        |        └── ...
    """
    def __init__(self, root_dir, domain_name, transform=None, target_transform = None):
        
        self.target_transform = target_transform
        self.transform = transform
        self.data = None
        self.targets = None
        self.imgs = None

        for domain in range(domain_name):
            self.imgs[domain] = ImageFolder(root_dir + domain).imgs
            self.data[domain] = [item[0] for item in self.imgs[domain]]
            self.targets[domain] = [item[1] for item in self.imgs[domain]]
            self.targets[domain] = np.array(self.targets[domain])
        self.loader = default_loader
    
    def __getitem__(self, domain, index) :
        xPath = self.data[domain][index]
        x = self.loader(xPath)
        if self.transform:
            x = self.transform(x)
        target = self.targets[domain][index]
        if self.target_transform:
            target = self.target_transform(target)
        return x, target

    def __len__(self, domain):
        return len(self.targets[domain])    

class Domain_Image_Dataset(DatasetFolder):
    def __init__(self, root, domain, transform=None, target_transform=None, download=None):
        self.root = root
        self.domain = domain
        self.target_transform = target_transform
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
            ])                
        imagefolder_obj = ImageFolder(self.root + self.domain, self.transform, self.target_transform)
        self.classes = imagefolder_obj.classes        
        self.loader = imagefolder_obj.loader
        self.samples = np.array(imagefolder_obj.samples)              
        self.data = np.array([item[0] for item in self.samples])
        self.targets = np.array([item[1] for item in self.samples])             

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

def img_union(args):
    """
    Load and partition the domian shift dataset for FL.
    A domain's data is assigned to a party, data is not partitioned.
    Two types of partition strategies: train-test split and train-val-test split.
    """
    trd, vad, ted = [], [], []    
    if args.is_val:
        partition_size = [0.7, 0.1, 0.2]
    else:
        partition_size = [0.8, 0.2]
    
    if args.is_val:
        parties_id = 0
        for item in args.domains:        
            data = Domain_Image_Dataset(args.root_dir + args.dataset + '/', item)
            l = len(data)
            index = np.arange(l)
            # np.random.seed(args.seed)
            np.random.shuffle(index)            
            l1, l2, l3 = int(l*partition_size[0]), int(l*partition_size[1]), int(l*partition_size[2])
            tr_data = Subset(data, index[:l1])
            va_data = Subset(data, index[l1:l1+l2])
            te_data = Subset(data, index[l1+l2:l1+l2+l3])
            # add noise to the data when partition strategy including feature noise
            if args.is_noise:
            #     # noise_level = args.noise_level / (args.n_parties - 1) * parties_id                
                 tr_data = add_gaussian_noise(tr_data, 0., 1, parties_id, args.noise_level, args.n_parties)
            #     va_data = add_gaussian_noise(va_data, 0., 1, parties_id, args.noise_level, args.n_parties)
            #     te_data = add_gaussian_noise(te_data, 0., 1, parties_id, args.noise_level, args.n_parties)
            trd.append(DataLoader(tr_data, args.batch_size, shuffle=True))
            vad.append(DataLoader(va_data, args.batch_size, shuffle=True))
            ted.append(DataLoader(te_data, args.batch_size, shuffle=True))
            parties_id += 1
        return trd, vad, ted 
    else:
        parties_id = 0
        for item in args.domains:        
            data = Domain_Image_Dataset(args.root_dir + args.dataset + '/', item)
            l = len(data)
            index = np.arange(l)
            # np.random.seed(args.seed)
            np.random.shuffle(index)            
            l1 = int(l*partition_size[0])
            tr_data = Subset(data, index[:l1])
            te_data = Subset(data, index[l1:])
            # add noise to the data when partition strategy including feature noise
            if args.is_noise:
            #     # noise_level = args.noise_level / (args.n_parties - 1) * parties_id                
                 tr_data = add_gaussian_noise(tr_data, 0., 1, parties_id, args.noise_level, args.n_parties)
            #     te_data = add_gaussian_noise(te_data, 0., 1, parties_id, args.noise_level, args.n_parties)
            trd.append(DataLoader(tr_data, args.batch_size, shuffle=True))
            ted.append(DataLoader(te_data, args.batch_size, shuffle=True))
            parties_id += 1
        return trd, ted   

def img_union_test(args):
    trd, vad, ted = [], [], []
    # 每一个域的数据集需要分给k客户端
    k = args.n_parties
    for item in args.domains:
        # 获取一个域的数据集
        data = Domain_Image_Dataset(args.root_dir + args.dataset + '/', item)
        if args.is_val:
            tr, va, te = get_dataloader(args, data, k, args.partition, args.beta)
            trd.extend(tr)
            vad.extend(va)
            ted.extend(te)
        else:
            tr, te = get_dataloader(args, data, k, args.partition, args.beta)
            trd.extend(tr)
            ted.extend(te)
    if args.is_val:
        return trd, vad, ted
    else:
        return trd, ted  

class CIFAR10_Dataset(Dataset):
    def __init__(self, filename='', is_train=True, transform=None, target_transform=None):        
        self.data = None
        self.targets = None
        self.transform = transform 
        self.target_transform = None        
        if is_train:            
            train_dataset = datasets.CIFAR10(filename, train=True, transform=self.transform, download=True)
            self.data = train_dataset.data
            self.targets = torch.tensor(train_dataset.targets)
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop((32, 32), padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])                         
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),            
            ])
            test_dataset = datasets.CIFAR10(filename, train=False, transform=self.transform, download=True)                    
            self.data = test_dataset.data
            self.targets = torch.tensor(test_dataset.targets)                                       
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]       
        if self.transform is not None:
            img = self.transform(img)  # 应用变换        
        return img, target

    def __len__(self):
        return len(self.data)

class CIFAR100_Dataset(Dataset):
    def __init__(self, filename='', is_train=True, transform=None):        
        self.data = None
        self.targets = None
        self.transform = transform           
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        if is_train:                    
            train_dataset = datasets.CIFAR100(filename, train=True, transform=self.transform, download=True)
            self.data = train_dataset.data
            self.targets = torch.tensor(train_dataset.targets) 
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                normalize
            ])             
        else:          
            test_dataset = datasets.CIFAR100(filename, train=False, transform=self.transform, download=True)                    
            self.data = test_dataset.data
            self.targets = torch.tensor(test_dataset.targets) 
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ]) 

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)  # 转换为 PIL Image        
        if self.transform is not None:
            img = self.transform(img)  # 应用变换        
        return img, target

    def __len__(self):
        return len(self.data)


def cifar(args):
    if args.dataset == 'cifar10':
        train_data = CIFAR10_Dataset(filename=args.root_dir + args.dataset + '/', is_train=True)
        test_data = CIFAR10_Dataset(filename=args.root_dir + args.dataset + '/', is_train=False)
    else:
        train_data = CIFAR100_Dataset(filename=args.root_dir + args.dataset + '/', is_train=True)
        test_data = CIFAR100_Dataset(filename=args.root_dir + args.dataset + '/', is_train=False)       
    return get_dataloader_with_test(args, train_data, test_data, args.n_parties, args.partition, args.beta)


class FEMNIST(MNIST):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    resources = [
        ('https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
         '59c65cec646fc57fe92d27d83afdf0ed')]

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train
        self.dataidxs = dataidxs

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets, self.users_index = torch.load(os.path.join(self.processed_folder, data_file))

        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.targets = self.targets[self.dataidxs]        


    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='F')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def download(self):
        """Download the FEMNIST data if it doesn't exist in processed_folder already."""
        import shutil

        if self._check_exists():
            return

        mkdirs(self.raw_folder)
        mkdirs(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')
        shutil.move(os.path.join(self.raw_folder, self.training_file), self.processed_folder)
        shutil.move(os.path.join(self.raw_folder, self.test_file), self.processed_folder)

    def __len__(self):
        return len(self.data)
    
    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0]+os.path.splitext(os.path.basename(url))[1]))
            for url, _ in self.resources
        )

def femnist(args):
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    train_data = FEMNIST(args.root_dir + args.dataset + '/', train=True, transform=transform_train) 
    test_data = FEMNIST(args.root_dir + args.dataset + '/', train=False, transform=transform_test)  
    return get_dataloader_with_test(args, train_data, test_data, args.n_parties, args.partition, args.beta)


# Get DataLoader
def get_dataloader(args, data, n_parties, partition, beta): 
    targets = np.array(data.targets, dtype=int) 
    num_classes = len(np.unique(targets))
    test_idx = []
    train_idx = []
    for i in range(num_classes):
        temp_y = np.array(targets).flatten().astype(int)                
        idx = np.where(np.equal(temp_y, i))[0]
        idx = np.random.permutation(idx)
        temp_idx = idx[:int(len(idx) * 0.2)]
        test_idx.extend(temp_idx.tolist())
        train_idx.extend(list(set(idx) - set(temp_idx)))

    train_targets = np.array([targets[i] for i in train_idx])
    partid2traindataidx_temp = partition_data(train_targets, num_classes, partition, n_parties, beta, args.main_prop)
    partid2traindataidx = {}
    for i in range(n_parties):
        partid2traindataidx[i] = [train_idx[j] for j in partid2traindataidx_temp[i]]
    len_test_parties = int(len(test_idx) / n_parties)
    net_cls_counts_npy = record_net_data_stats(targets, partid2traindataidx)        
    if args.is_whole_label:
        row_sums = net_cls_counts_npy.sum(axis=1)[:, np.newaxis]
        net_cls_counts_npy = net_cls_counts_npy / row_sums
        net_cls_counts_npy = net_cls_counts_npy + 0.05    
    row_sums = net_cls_counts_npy.sum(axis=1)[:, np.newaxis]
    net_cls_counts_npy = net_cls_counts_npy / row_sums
    net_cls_counts_npy = (net_cls_counts_npy * len_test_parties).astype(int)
    
    test_targets = np.array([targets[i] for i in test_idx])   
    idx_batch = [[] for _ in range(n_parties)]    
    for i in range(num_classes):        
        y = np.array(test_targets).flatten().astype(int)                
        idx = np.where(np.equal(y, i))[0]
        idx = np.random.permutation(idx)
        for party_id in range(n_parties):   
            if net_cls_counts_npy[party_id][i] > len(idx):
                cur_idx = idx      
            else:
                cur_idx = np.random.choice(idx, net_cls_counts_npy[party_id][i] , replace=False)
            idx_batch[party_id] = idx_batch[party_id] + cur_idx.tolist()
    partid2testdataidx = {i:idx_batch[i] for i in range(n_parties)}
    for i in range(n_parties):
        partid2testdataidx[i] = [test_idx[j] for j in partid2testdataidx[i]] 
    return get_parties_dataloader(args, data, data, partid2traindataidx, partid2testdataidx, n_parties)


def get_dataloader_with_test(args, train_data, test_data, n_parties, partition, beta): 
    num_classes = len(np.unique(train_data.targets))
    if args.partition == 'noniid-feature-users' and args.dataset == 'femnist':
        num_users = args.n_parties * 20
        train_users = train_data.users_index[:num_users]
        test_users = test_data.users_index[:num_users]
        partid2traindataidx = partition_data(train_data.targets, num_classes, partition, n_parties, beta, args.main_prop, train_users)
        partid2testdataidx = partition_data(test_data.targets, num_classes, partition, n_parties, beta, args.main_prop, test_users)
    else:
        partid2traindataidx = partition_data(train_data.targets, num_classes, partition, n_parties, beta, args.main_prop)
        net_cls_counts_npy = record_net_data_stats(train_data.targets, partid2traindataidx)
        nums_test_targets = len(test_data.targets)
        nums_per_party = int(nums_test_targets / n_parties)

        if args.is_whole_label:
            row_sums = net_cls_counts_npy.sum(axis=1)[:, np.newaxis]
            net_cls_counts_npy = net_cls_counts_npy / row_sums
            net_cls_counts_npy = net_cls_counts_npy + 0.05
        
        row_sums = net_cls_counts_npy.sum(axis=1)[:, np.newaxis]
        net_cls_counts_npy = net_cls_counts_npy / row_sums
        net_cls_counts_npy = (net_cls_counts_npy * nums_per_party).astype(int)
        
        targets = np.array(test_data.targets)   
        idx_batch = [[] for _ in range(n_parties)]    
        for i in range(num_classes):        
            y = np.array(targets).flatten().astype(int)                
            idx = np.where(np.equal(y, i))[0]
            idx = np.random.permutation(idx)
            for party_id in range(n_parties):    
                if net_cls_counts_npy[party_id][i] > len(idx):
                    cur_idx = idx
                else:      
                    cur_idx = np.random.choice(idx, net_cls_counts_npy[party_id][i] , replace=False)
                idx_batch[party_id] = idx_batch[party_id] + cur_idx.tolist()
        partid2testdataidx = {i:idx_batch[i] for i in range(n_parties)}
    return get_parties_dataloader(args, train_data, test_data, partid2traindataidx, partid2testdataidx, n_parties)


def get_parties_dataloader(args, train_data, test_data, partid2traindataidx, partid2testdataidx, n_parties):
    trd, vad, ted = [], [], []
    if args.is_val:
        if args.dataset == 'femnist':
            partition = [0.1, 0.1]
        elif args.dataset == 'pacs' or 'vlcs':
            partition = [0.25, 0.25]
        else:
            partition = [0.6, 0.4]
        for parties_id in range(n_parties):
            tr_idx = partid2traindataidx[parties_id]
            np.random.shuffle(tr_idx)
            l = len(tr_idx)
            # train_size = int(partition[0] * l)
            # trd_idx, vad_idx = tr_idx[:train_size], tr_idx[train_size:]
            trd_idx = tr_idx[:int(partition[0] *l)]
            vad_idx = tr_idx[int(partition[0] *l):int(partition[0]*l)+int(partition[1]*l)]
            te_idx = partid2testdataidx[parties_id]
            tr_data = Subset(train_data, trd_idx)
            va_data = Subset(train_data, vad_idx)
            te_data = Subset(test_data, te_idx)            
            if args.is_noise:
                tr_dl = DataLoader(tr_data, args.batch_size, shuffle=True, 
                        collate_fn=lambda x: add_gaussian_noise_collate_fn(
                            x, 0., 1, parties_id, args.noise_level, n_parties))
                va_dl = DataLoader(va_data, args.batch_size, shuffle=True, 
                        collate_fn=lambda x: add_gaussian_noise_collate_fn(
                            x, 0., 1, parties_id, args.noise_level, n_parties))
                te_dl = DataLoader(te_data, args.batch_size, shuffle=True, 
                        collate_fn=lambda x: add_gaussian_noise_collate_fn(
                            x, 0., 1, parties_id, args.noise_level, n_parties))
            else:
                tr_dl = DataLoader(tr_data, args.batch_size, shuffle=True)
                va_dl = DataLoader(va_data, args.batch_size, shuffle=True)
                te_dl = DataLoader(te_data, args.batch_size, shuffle=True)
            trd.append(tr_dl)
            vad.append(va_dl)
            ted.append(te_dl)
        return trd, vad, ted
    else:
        if args.dataset == 'femnist':
            partition = 0.2
        elif args.dataset == 'pacs' or 'vlcs':
            partition = 0.5
        else:
            partition = 1.0
        for parties_id in range(n_parties):
            tr_idx = partid2traindataidx[parties_id]
            tr_idx = np.random.permutation(tr_idx)
            trd_idx = tr_idx[:int(partition * len(tr_idx))]
            te_idx = partid2testdataidx[parties_id]
            tr_data = Subset(train_data, trd_idx)
            te_data = Subset(test_data, te_idx)        
            if args.is_noise:
                tr_dl = DataLoader(tr_data, args.batch_size, shuffle=True, 
                        collate_fn=lambda x: add_gaussian_noise_collate_fn(
                            x, 0., 1, parties_id, args.noise_level, n_parties))
                te_dl = DataLoader(te_data, args.batch_size, shuffle=True, 
                        collate_fn=lambda x: add_gaussian_noise_collate_fn(
                            x, 0., 1, parties_id, args.noise_level, n_parties))
            else:
                tr_dl = DataLoader(tr_data, args.batch_size, shuffle=True)
                te_dl = DataLoader(te_data, args.batch_size, shuffle=True) 
            trd.append(tr_dl)
            ted.append(te_dl)            
        return trd, ted


def add_gaussian_noise_collate_fn(batch, mean=0., std=1., parties_id=None, noise_level=1, total=0, is_space=False):
    data_batch, label_batch = zip(*batch)  
    data_batch = torch.stack(data_batch)  
    label_batch = torch.tensor(label_batch)  
    std = noise_level / (total - 1) * parties_id if parties_id is not None else std
    if is_space and parties_id is not None:
        num = int(np.sqrt(total))
        if num * num < total:
            num += 1
        tmp = torch.randn(data_batch.size())
        filt = torch.zeros(data_batch.size())
        size = int(28 / num)
        row = int(parties_id / num)
        col = parties_id % num
        for i in range(size):
            for j in range(size):
                filt[:, :, row * size + i, col * size + j] = 1
        tmp = tmp * filt
        data_batch = data_batch + tmp * std + mean
    else:
        data_batch = data_batch + torch.randn(data_batch.size()) * std + mean
    return data_batch, label_batch


# data process
def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def gen_bar_updater() -> Callable[[int, int, int], None]:
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update

def download_url(url: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None) -> None:
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:   # download the file
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        except (urllib.error.URLError, IOError) as e:  # type: ignore[attr-defined]
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")

def _is_tarxz(filename: str) -> bool:
    return filename.endswith(".tar.xz")

def _is_tar(filename: str) -> bool:
    return filename.endswith(".tar")

def _is_targz(filename: str) -> bool:
    return filename.endswith(".tar.gz")

def _is_tgz(filename: str) -> bool:
    return filename.endswith(".tgz")

def _is_gzip(filename: str) -> bool:
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")

def _is_zip(filename: str) -> bool:
    return filename.endswith(".zip")

def extract_archive(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False) -> None:
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, 'r:xz') as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)

def download_and_extract_archive(
        url: str,
        download_root: str,
        extract_root: Optional[str] = None,
        filename: Optional[str] = None,
        md5: Optional[str] = None,
        remove_finished: bool = False,
    ) -> None:
        download_root = os.path.expanduser(download_root)
        if extract_root is None:
            extract_root = download_root
        if not filename:
            filename = os.path.basename(url)

        download_url(url, download_root, filename, md5)

        archive = os.path.join(download_root, filename)
        print("Extracting {} to {}".format(archive, extract_root))
        extract_archive(archive, extract_root, remove_finished)


