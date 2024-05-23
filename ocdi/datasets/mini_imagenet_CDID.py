import pickle
import numpy as np
from abc import ABC, abstractmethod
import os
from datasets.mini_imagenet.dataset_utils import gaussian_noise
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import random
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable



class Mini_ImageNet_CDID:
    def __init__(self, dataset_params, debug=False, save_cache=False, load_cache=False, device='cpu', quick_load=False):
        self.debug = debug
        self.dataset_path = dataset_params['dataset_path']
        self.perturbation_params = dataset_params['perturbation_params']
        self.save_cache = save_cache
        self.load_cache = load_cache
        self.device = device

        # Original data
        self.original_data = None
        if quick_load:
            self.samples_per_class = 600
            # Original labels
            original_labels = []
            for i in range(100):  # 100 classes
                y_train = np.ones((600,)) * i  # 600 samples per class
                original_labels.append(y_train)
            self.original_labels = np.vstack(original_labels)
            self.distribution_labels = np.concatenate(
                [self.original_labels.reshape(-1) + (i * self.num_classes) for i in
                 range(len(self.perturbation_params))],
                axis=0)
        else:
            self.load_data()
            self.samples_per_class = self.original_data.shape[1]
            # Original labels
            original_labels = []
            for i in range(len(self.original_data)): #100
                y_train = np.ones((self.original_data.shape[1],)) * i #600
                original_labels.append(y_train)
            self.original_labels = np.vstack(original_labels)
            self.distribution_labels = np.concatenate(
                [self.original_labels.reshape(-1) + (i * self.num_classes) for i in range(len(self.perturbation_params))],
                axis=0)

        # Add perturbed versions of the data
        self.all_data = []
        self.num_distributions = len(self.perturbation_params)*self.num_classes

        self.all_feats = []
        if quick_load:
            print('You are using QUICK LOAD, skipping image generation and loading features from disk...')
        for pert in self.perturbation_params:
            save_name = ''
            if pert == None:
                print(f'generating perturbation: no pertubation')
                save_name = 'original'
                if not quick_load:
                    self.all_data.append(deepcopy(self.original_data))
            elif pert['type'] == 'gaussian':
                print(f"generating perturbation: gaussian - {pert['noise_factor']}, {pert['sig']}")
                save_name = f"gaussian-{pert['noise_factor']}_{pert['sig']}"
                if not quick_load:
                    self.all_data.append(self.perturb_gaussian(pert['noise_factor'], pert['sig'], cache=self.save_cache))
            elif pert['type'] == 'occlusion':
                save_name = f"occlusion-{pert['occlusion_factor']}"
                print(f"generating perturbation: occlusion - {pert['occlusion_factor']}")
                if not quick_load:
                    self.all_data.append(self.perturb_occlusion(pert['occlusion_factor'], cache=self.save_cache))
            print('done')
            self.all_feats.append(self.extract_resnet_features(len(self.all_data)-1, save_name=save_name, cache=True))
            print(self.all_feats[-1].shape)
        print(f'TOT samples: {np.sum([x.shape[0]*x.shape[1] for x in self.all_data])}')

    def load_data(self):
        data_path = f"{self.dataset_path}/mini-imagenet-cache-all_data_int.pickle"
        if not os.path.isfile(data_path):
            self.generate_all_data(cache=self.save_cache)
            self.original_data = self.original_data.astype(np.float32) / 255.0
        else:
            with open(data_path, "rb") as all_in:
                self.original_data = pickle.load(all_in).astype(np.float32) / 255.0
        if self.debug:
            self.original_data = self.original_data[:,:10,...] # 10 samples per class instead of 600

    def generate_all_data(self, cache=False):
        with open(f"{self.dataset_path}/mini-imagenet-cache-train.pkl", "rb") as train_in:  # TODO:fix this
            train = pickle.load(train_in)["image_data"].reshape([64, 600, 84, 84, 3])  # .astype(np.float16) / 255.0
        with open(f"{self.dataset_path}/mini-imagenet-cache-val.pkl", "rb") as val_in:  # TODO:fix this
            val = pickle.load(val_in)['image_data'].reshape([16, 600, 84, 84, 3])  # .astype(np.float16) / 255.0
        with open(f"{self.dataset_path}/mini-imagenet-cache-test.pkl", "rb") as test_in:  # TODO:fix this
            test = pickle.load(test_in)['image_data'].reshape([20, 600, 84, 84, 3])  # .astype(np.float16) / 255.0
        self.original_data = np.vstack((train, val, test))

        if cache:
            # STORE DATA
            with open(f"{self.dataset_path}/mini-imagenet-cache-all_data_int.pickle", "wb") as handle:
                pickle.dump(self.original_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def add_gaussian_noise(self, x, noise_factor, sig):
        return np.clip(x + noise_factor * np.random.normal(loc=0.0, scale=sig, size=x.shape), 0, 1)

    def perturb_gaussian(self, noise_factor, sig, cache=False):
        file_name = f'{self.dataset_path}/mini-imagenet-cache-all_data_gaussian_{noise_factor}_{sig}.pickle'
        if os.path.exists(file_name):
            with open(file_name, 'rb') as fh:
                x_gaussian = pickle.load(fh)
        else:
            x_gaussian = deepcopy(self.original_data)

            for i in tqdm(range(x_gaussian.shape[0])):
                x_gaussian[i, ...] = self.add_gaussian_noise(self.original_data[i], noise_factor, sig)

            if cache:
                with open(file_name, 'wb') as handle:
                    pickle.dump(x_gaussian, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return x_gaussian

    def perturb_occlusion(self, occlusion_factor, cache=False):
        file_name = f'{self.dataset_path}/mini-imagenet-cache-all_data_occlusion_{occlusion_factor}.pickle'
        if os.path.exists(file_name):
            with open(file_name, 'rb') as fh:
                x_occluded = pickle.load(fh)
        else:
            x_occluded = deepcopy(self.original_data)

            image_size = self.original_data.shape[2]  # height or width (we assume H==W)
            occlusion_size = int(occlusion_factor * image_size)
            half_size = occlusion_size // 2
            occlusion_x = random.randint(min(half_size, image_size - half_size),
                                         max(half_size, image_size - half_size))
            occlusion_y = random.randint(min(half_size, image_size - half_size),
                                         max(half_size, image_size - half_size))

            x_occluded[:, :, max((occlusion_x - half_size), 0):min((occlusion_x + half_size), image_size), \
            max((occlusion_y - half_size), 0):min((occlusion_y + half_size), image_size)] = 1
            if cache:
                with open(f'{self.dataset_path}/mini-imagenet-cache-all_data_occlusion_{occlusion_factor}.pickle',
                          'wb') as handle:
                    pickle.dump(x_occluded, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return x_occluded

    @property
    def img_size(self):
        return self.original_data[0][0].shape

    @property
    def num_classes(self):
        return len(np.unique(self.original_labels))

    def get_train_val_test(self, val_ratio=0.1, test_ratio=0.2, num_val_distributions=0.8, use_resnet=True):
        #TODO: check this! Issues might be caused by the limited amount of samples that I am using for debugging
        train_ratio = 1 - val_ratio - test_ratio
        train_limit = int(self.samples_per_class*train_ratio)
        train_ids = np.arange(0, train_limit)
        val_limit = train_limit + int(self.samples_per_class*val_ratio)
        val_ids = np.arange(train_limit, val_limit)
        test_ids = np.arange(val_limit, self.samples_per_class)

        # split and reshape the data

        # TRAIN
        if use_resnet:
            self.train_data = torch.concat([x.view(100, 600, 512)[:,train_ids].view(-1, 512) for x in self.all_feats], axis=0)
        else:
            self.train_data = np.concatenate([x[:, train_ids, ...].reshape(-1, 84, 84, 3) for x in self.all_data],
                                             axis=0)
        self.train_labels = np.tile(self.original_labels[:, train_ids].reshape(-1), len(self.perturbation_params))
        self.train_distributions = np.concatenate(
            [self.original_labels[:, train_ids].reshape(-1) + (i * self.num_classes) for i in range(len(self.perturbation_params))],
            axis=0)

        # VAL
        if use_resnet:
            self.val_data = torch.concat([x.view(100, 600, 512)[:,val_ids].view(-1, 512) for x in self.all_feats], axis=0)
        else:
            self.val_data = np.concatenate([x[:, val_ids, ...].reshape(-1, 84, 84, 3) for x in self.all_data], axis=0)
        self.val_labels = np.tile(self.original_labels[:, val_ids].reshape(-1), len(self.perturbation_params))
        self.val_distributions = np.concatenate(
            [self.original_labels[:, val_ids].reshape(-1) + (i * self.num_classes) for i in
             range(len(self.perturbation_params))],
            axis=0)

        # TEST
        if use_resnet:
            self.test_data = torch.concat([x.view(100, 600, 512)[:,test_ids].view(-1, 512) for x in self.all_feats], axis=0)
        else:
            self.test_data = np.concatenate([x[:, test_ids, ...].reshape(-1, 84, 84, 3) for x in self.all_data], axis=0)
        self.test_labels = np.tile(self.original_labels[:, test_ids].reshape(-1), len(self.perturbation_params))
        self.test_distributions = np.concatenate(
            [self.original_labels[:, test_ids].reshape(-1) + (i * self.num_classes) for i in
             range(len(self.perturbation_params))],
            axis=0)

        # remove samples from the last distributions from the validation set
        val_dist_limit = int(self.val_data.shape[0]*num_val_distributions)
        self.val_data = self.val_data[:val_dist_limit]
        self.val_labels = self.val_labels[:val_dist_limit]
        self.val_distributions = self.val_distributions[:val_dist_limit]

        X_train = self.train_data
        X_val = self.val_data
        X_test = self.test_data

        if not use_resnet:
            X_train = torch.tensor(X_train).float()
            X_val = torch.tensor(X_val).float()
            X_test = torch.tensor(X_test).float()

        data = {'X_train': X_train.to(self.device),
                'X_val': X_val.to(self.device),
                'X_test': X_test.to(self.device),
                'y_train': torch.tensor(self.train_labels).long().to(self.device),
                'y_val': torch.tensor(self.val_labels).long().to(self.device),
                'y_test': torch.tensor(self.test_labels).long().to(self.device),
                'distributions_train': torch.tensor(self.train_distributions).int(),
                'distributions_val': torch.tensor(self.val_distributions).int(),
                'distributions_test': torch.tensor(self.test_distributions).int(),
                }

        return data

    def extract_resnet_features(self, subset, save_name='', cache=True):
        file_path = f"{self.dataset_path}/mini-imagenet-cache-resnet_feats_{save_name}.pickle"

        if os.path.isfile(file_path) and self.load_cache:
            # load data if already cached
            with open(file_path, "rb") as handle:
                all_resnet_feats = pickle.load(handle)
            return all_resnet_feats

        if not hasattr(self, 'resnet'):
            # Load the pretrained model
            resnet_model = models.resnet18(pretrained=True)
            modules = list(resnet_model.children())[:-1]
            self.resnet = nn.Sequential(*modules).to(self.device)

            # Set model to evaluation mode
            self.resnet.eval()

        # Image transforms
        scaler = transforms.Resize((224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        topil = transforms.ToPILImage()

        if subset == 'train':
            data = self.train_data
        elif subset == 'val':
            data = self.val_data
        elif subset == 'test':
            data = self.test_data
        elif type(subset) == int:
            data = self.all_data[subset].reshape(-1, 84, 84, 3)
        all_resnet_feats = torch.empty((len(data), 512),device=self.device)
        for i, img in tqdm(enumerate(data)):
            t_img = Variable(normalize(to_tensor(scaler(topil((img*255).astype(np.uint8))))).unsqueeze(0)).to(self.device)
            all_resnet_feats[i,:] = self.resnet(t_img).squeeze(-1).squeeze(-1).detach()
        if cache:
            with open(file_path, "wb") as handle:
                pickle.dump(all_resnet_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return all_resnet_feats
