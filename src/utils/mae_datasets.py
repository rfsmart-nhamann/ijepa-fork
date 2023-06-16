# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

import numpy as np
import torch
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR10':
        print("reading CIFAR-10 from data path: ", args.data_path)
        dataset = datasets.CIFAR10(args.data_path, train=True, transform=transform, download=True)

        # generate random subset of data, also do train/val split
        dataset_pct = 0.1
        val_frac = 0.2

        N = len(dataset)
        num_train_samples = int(N * dataset_pct)
        dataset_indices = np.random.choice(N, num_train_samples, replace=False)
        dataset_subset = torch.utils.data.Subset(dataset, dataset_indices)

        # split into train/val
        N_subset = len(dataset_subset)
        V = int(num_train_samples * val_frac)
        dataset_train, dataset_val = torch.utils.data.random_split(dataset_subset, [N_subset - V, V])

        print(f'getting {"train" if is_train else "val"}.  train/val is {N_subset - V}/{V}')

        nb_classes = 10
        dataset = dataset_train if is_train else dataset_val
    elif args.data_set == 'IMNET':
        print("reading ImageNet from datapath: ", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        # root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of classes = %d" % nb_classes)
    print(dataset)

    return dataset, nb_classes


    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
