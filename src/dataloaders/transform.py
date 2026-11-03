"""
Image transformation utilities for data augmentation and preprocessing.

This module provides various transform classes and utilities for image processing,
including resizing, cropping, flipping, blurring, and normalization operations.
"""

import random as rd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math


# -------------------------------------
# Main transform composition functions
# ------------------------------------- 
def get_train_transforms(image_size):
    return Compose([
        RandomCrop(image_size),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        NormalizationMinMax(),
    ])


def get_inference_transforms(image_size, norm_method='minmax'):
    if norm_method == "minmax":
        return Compose([
            CenterCrop(image_size),
            NormalizationMinMax(),
        ])
    else:
        return Compose([
            CenterCrop(image_size),
            Normalization(),
        ])


def get_inference_from_estimates_transforms(image_size):
    return Compose([
        CenterCrop(image_size)
    ])


# -------------------------------------
# Transform classes
# ------------------------------------- 
class Compose(object):
    """Composes multiple transforms together and applies them sequentially to a sample."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class Resize(object):
    """Resizes all images in the sample to a specified size using bilinear interpolation."""

    def __init__(self, dst_size):
        self.dst_size = dst_size

    def __call__(self, sample):
        result_sample = {}
        for key in sample:
            key_sample = sample[key]
            if "dm" in key:
                _, h, w = key_sample.shape
                resize_factor = self.dst_size // h
                key_sample *= resize_factor

            result_sample[key] = F.interpolate(
                key_sample.unsqueeze(0), (self.dst_size, self.dst_size), mode='bilinear', align_corners=True).squeeze()

        return result_sample


def crop_sample(crop_size, sample):
    """Helper function to crop a sample to a specified size."""
    target = sample['pre_post_image']
    _, h, w = target.shape
    th, tw = crop_size, crop_size

    result_sample = {}
    if h <= th:
        for key in sample:
            result_sample[key] = sample[key]
    else:
        y = rd.choice(range(1, target.size(dim=1) - th - 1))
        x = rd.choice(range(1, target.size(dim=2) - tw - 1))

        for key in sample:
            result_sample[key] = sample[key][:, y:y + th, x:x + tw]
    return result_sample

    
class RandomCrop(object):
    """Randomly crops all images in the sample to a specified size."""
    
    def __init__(self, crop_size):
        self.crop_size = crop_size
        
    def __call__(self, sample):
        return crop_sample(self.crop_size, sample)


class RandomCropResize(object):
    """Randomly crops the sample and then resizes it to a specified size."""

    def __init__(self, crop_size, image_size):
        self.crop_size = crop_size
        self.image_size = image_size

        self.random_crop = RandomCrop(crop_size)
        self.resize = Resize(image_size)

    def __call__(self, sample):
        result_sample = self.random_crop(sample)
        if self.crop_size != self.image_size:
            result_sample = self.resize(result_sample)
        return result_sample


class RandomCropResizeWithFactor(object):
    """Randomly crops the sample with a random scaling factor and resizes it back."""

    def __init__(self, crop_size, max_crop_factor=1):
        self.crop_size = crop_size
        self.crop_factors = [int(math.pow(2, i)) for i in range(1, max_crop_factor+1)]
        self.resize = Resize(crop_size)

    def __call__(self, sample):
        resize = rd.random() < 0.5
        if resize:
            crop_factor = rd.choice(self.crop_factors)
            mean_crop_size = self.crop_size // crop_factor
            sample_crop_size = mean_crop_size
            result_sample = crop_sample(sample_crop_size, sample)
            if self.crop_size != sample_crop_size:
                result_sample = self.resize(result_sample)
        else:
            result_sample = crop_sample(self.crop_size, sample)
        return result_sample


class CenterCrop(object):
    """Crops the sample from the center to a specified size."""

    def __init__(self, image_size):
        self.image_size = image_size    
    
    def __call__(self, sample):
        result_sample = {}
        for key in sample:
            image = sample[key]
            _, h, w = image.shape
            
            if h <= self.image_size:
                result_sample[key] = sample[key]
            else:
                x = int(round((w - self.image_size) / 2.0))
                y = int(round((h - self.image_size) / 2.0))
                result_sample[key] = sample[key][:, y:y + self.image_size, x:x + self.image_size]
        return result_sample


class Normalization(object):
    """Normalizes image data in the sample by subtracting mean and dividing by standard deviation."""

    def __call__(self, sample):
        result_sample = {}
        for key in sample:
            if "image" in key:
                key_sample = sample[key]

                normalized_array = []
                for image_id in range(2):
                    mean = torch.mean(key_sample[image_id])
                    std = torch.std(key_sample[image_id])
                    normalized_array.append((key_sample[image_id] - mean) / std)

                result_sample[key] = torch.stack(normalized_array, dim=0)
            else:
                result_sample[key] = sample[key]
        result_sample['pre_post_image_no_normalization'] = sample['pre_post_image']  # copy not done in place
        return result_sample


class NormalizationMinMax(object):
    """Normalizes image data in the sample by subtracting min and dividing by max - min."""

    def __call__(self, sample):
        result_sample = {}
        for key in sample:
            if "image" in key:
                key_sample = sample[key]

                normalized_array = []
                for image_id in range(2):
                    mini = torch.min(key_sample[image_id])
                    maxi = torch.max(key_sample[image_id])
                    denominator = maxi - mini if (maxi - mini) != 0 else 1
                    normalized_array.append((key_sample[image_id] - mini) / denominator)

                result_sample[key] = torch.stack(normalized_array, dim=0)
            else:
                result_sample[key] = sample[key]
        result_sample['pre_post_image_no_normalization'] = sample['pre_post_image']  # copy not done in place
        return result_sample


class RandomHorizontalFlip(object):
    """Randomly flips the sample horizontally with 50% probability."""

    def __call__(self, sample):
        flip = rd.random() < 0.5
        result_sample = {}
        for key in sample:
            if not flip:
                result_sample[key] = sample[key]
            else:
                key_sample = torch.flip(sample[key], dims=[2])
                if "dm" in key:
                    key_sample[0] = -key_sample[0]
                result_sample[key] = key_sample

        return result_sample


class RandomVerticalFlip(object):
    """Randomly flips the sample vertically with 50% probability."""

    def __call__(self, sample):
        flip = rd.random() < 0.5
        result_sample = {}
        for key in sample:
            if not flip:
                result_sample[key] = sample[key]
            else:
                key_sample = torch.flip(sample[key], dims=[1])
                if "dm" in key:
                    key_sample[1] = -key_sample[1]
                result_sample[key] = key_sample

        return result_sample


class RandomSharpenBlurRGB(object):
    """Randomly applies either sharpening or blurring to RGB images with specified probabilities."""

    def __init__(self, max_sharpen_count=5, max_blur_count=5, proba_sharpen=0.25, proba_blur=0.5):
        self.sharpen = RandomSharpen(max_sharpen_count)
        self.blur = RandomBlur(max_blur_count)
        self.proba_sharpen = proba_sharpen
        self.proba_blur = proba_blur

    def __call__(self, sample):
        proba = rd.random()
        if proba < self.proba_sharpen:
            return self.sharpen(sample)
        elif proba < self.proba_blur:
            return self.blur(sample)
        else:
            return sample


class RandomSharpen(object):
    """Applies random sharpening to images using a convolution kernel."""

    def __init__(self, max_sharpen_count=5):
        self.max_sharpen_count = max_sharpen_count
    def __call__(self, sample):
        num_times = rd.randint(1, self.max_sharpen_count)
        result_sample = {}
        for key in sample:
            if key != "pre_post_image":
                result_sample[key] = sample[key]
            else:
                kernel = torch.tensor(
                    [[-1, -2, -1], [-2, 28, -2], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3) / 16

                key_sample = sample[key].clone().unsqueeze(1)
                src_min = torch.min(key_sample)
                src_max = torch.max(key_sample)
                for _ in range(num_times):
                    key_sample = F.conv2d(key_sample, kernel, padding=0)
                    key_sample = clip_normalize(key_sample, src_min, src_max)

                padding_op = nn.ReplicationPad2d(num_times)
                key_sample = padding_op(key_sample)

                result_sample[key] = key_sample.squeeze()

        return result_sample


class RandomBlur(object):
    """Applies random Gaussian-like blurring to images using a convolution kernel."""

    def __init__(self, max_blur_count=5):
        self.max_blur_count = max_blur_count
    def __call__(self, sample):
        num_times = rd.randint(1, self.max_blur_count)
        result_sample = {}
        for key in sample:
            if key != "pre_post_image":
                result_sample[key] = sample[key]
            else:
                kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 16

                key_sample = sample[key].clone().unsqueeze(1)
                src_min = torch.min(key_sample)
                src_max = torch.max(key_sample)
                for _ in range(num_times):
                    key_sample = F.conv2d(key_sample, kernel, padding=0)
                    key_sample = clip_normalize(key_sample, src_min, src_max)

                padding_op = nn.ReplicationPad2d(num_times)
                key_sample = padding_op(key_sample)

                result_sample[key] = key_sample.squeeze()

        return result_sample


class NoneTransform(object):
    """Identity transform that returns the input without any modification."""

    def __call__(self, image):
        return image


def clip_normalize(image, src_min, src_max):
    """Normalizes image values to a specified range while preserving relative differences."""
    mini = torch.min(image)
    maxi = torch.max(image)
    return src_min + (image - mini) / (maxi - mini) * (src_max - src_min)