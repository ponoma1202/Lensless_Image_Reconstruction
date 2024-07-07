import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

from torch.utils.data import Dataset

class Mirflickr(Dataset):
    def __init__(self, root_dir, data_list=None, target_list=None, transform=None):
        super().__init__()

        # TODO: Issue. Cannot apply different transforms to train and validation. Need to change this class somehow.
        # TODO: Easiest thing to do is to generate a list of images to use beforehand and feed it in
        self.root_dir = root_dir 

        if data_list == None: 
            self.data_dir = os.path.join(root_dir, "diffuser_images_npy")
            self.data_list = os.listdir(self.data_dir)
        else:
            self.data_list = data_list
        
        if target_list == None:
            self.target_dir = os.path.join(root_dir, "ground_truth_lensed_npy")
            self.target_list = os.listdir(self.target_dir)
        else:
            self.target_list = target_list

        self.train_norm = transforms.Normalize([0.11155567, 0.12113422, 0.1406812], [0.089452974, 0.09483851, 0.10869888])
        self.val_norm = transforms.Normalize([0.24371915, 0.25690535, 0.2630154], [0.32493576, 0.3335854, 0.33773425])
        self.transform = transform

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, index):
        image = np.load(os.path.join(self.data_dir, self.data_list[index]))
        target = np.load(os.path.join(self.target_dir, self.target_list[index]))

        # change channel order from (H, W, C) to (C, H, W)
        image = np.moveaxis(image, 2, 0)
        target = np.moveaxis(target, 2, 0)

        # Normalize both input images and ground truth images
        image = self.train_norm(image)
        target = self.val_norm(target)

        # Center crop according to smallest side
        # _, height, width = image.shape
        # new_side_len = min(height, width)                       # mirflickr has (H, W) = (270, 400)
        # x = (width // 2) - (new_side_len // 2)                   # width // 2 = center_x
        # y = (height // 2) - (new_side_len // 2)                  # new_side_len // 2 = half of image
        # image = image[x:x+new_side_len, y: y+new_side_len]
        # target = target[x:x+new_side_len, y: y+new_side_len]

        if self.transform:
            image = self.transform(image)

        sample = {0: image, 1: target}

        return sample
    

def get_loader(dataset, min_side_len, batch_size, num_workers, root_dir="/home/ponoma/workspace/DATA/mirflickr_dataset/"):
    if dataset=="CIFAR10":
        train_transform = torchvision.transforms.Compose([transforms.RandomCrop(min_side_len, padding=4), 
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandAugment(),  # RandAugment augmentation for strong regularization
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])          # positional embedding cannot take negative values (center around mean not around 0)
                                            # transforms.Normalize([0, 0, 0], [0.2470, 0.2435, 0.2616]),
                                            # Rescale()])                     

        val_transform = torchvision.transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),])
                                                        #  transforms.Normalize([0, 0, 0], [0.2470, 0.2435, 0.2616]),
                                                        #  Rescale()])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=val_transform)

    elif dataset=="Mirflickr":
        train_transform = torchvision.transforms.Compose([transforms.ToTensor()])                       

        val_transform = torchvision.transforms.Compose([transforms.ToTensor()])

        dataset = Mirflickr(root_dir)           # Make it take in a list of 
        generator = torch.Generator().manual_seed(3)        # generator should yield deterministic behavior
        trainset, valset, testset = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15], generator)

        # Apply transforms after doing random split
        trainset.dataset = Mirflickr(root_dir, trainset.dataset.data_list, trainset.dataset.target_list, train_transform)
        valset.dataset = Mirflickr(root_dir, valset.dataset.data_list, valset.dataset.target_list, val_transform)
        
    else:
        raise("Unkown dataset.")
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
    

