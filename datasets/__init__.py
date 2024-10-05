import torch.utils.data
import torchvision

from .custom import build as build_custom


def build_dataset(dataset_file: str, image_set: str, args):
    if dataset_file == 'custom':
        return build_custom(image_set, args)
    raise ValueError(f'dataset {dataset_file} not supported')
