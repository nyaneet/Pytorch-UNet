import torchio as tio
import torch
import random

from pathlib import Path
from torch.utils.data import DataLoader

random.seed(42)
num_workers = 5

#dataset_path_base = '/home/bagurgle/mag/iip/iip-filtered-new'
dataset_path_base = './data/iip'
band_0_path = f'{dataset_path_base}/images_clipped/band_0'
band_1_path = f'{dataset_path_base}/images_clipped/band_1'
masks_path = f'{dataset_path_base}/masks_clipped'
masks_prob_path = f'{dataset_path_base}/masks_prob_clipped'

bands_0 = sorted(list(Path(band_0_path).glob('*_HH_*.png')))
bands_1 = sorted(list(Path(band_1_path).glob('*_HH_*.png')))
masks = sorted(list(Path(masks_path).glob('*_HH_*.png')))
probs = sorted(list(Path(masks_prob_path).glob('*_HH_*.png')))

idxs = list(range(0, len(bands_0)))
random.shuffle(idxs)

bands_0 = [bands_0[idx] for idx in idxs]
bands_1 = [bands_1[idx] for idx in idxs]
masks = [masks[idx] for idx in idxs]
probs = [probs[idx] for idx in idxs]

val_len = len(bands_0) // 10
train_len = len(bands_0) - val_len

train_subjects = []
for (band_0, band_1, mask, prob) in zip(bands_0[:train_len], bands_1[:train_len], masks[:train_len], probs[:train_len]):
    file_name = str(band_0).split('/')[-1]

    if file_name != str(band_1).split('/')[-1] or file_name != str(mask).split('/')[-1] or file_name != str(prob).split('/')[-1]:
        print(band_0)
        print(band_1)
        print(mask)
        print(prob)
        break

    subject = tio.Subject(
        band_0=tio.ScalarImage(band_0),
        band_1=tio.ScalarImage(band_1),
        iceberg=tio.LabelMap(mask),
        sampler_probability=tio.LabelMap(prob),
    )

    train_subjects.append(subject)

val_subjects = []
for (band_0, band_1, mask, prob) in zip(bands_0[train_len:], bands_1[train_len:], masks[train_len:], probs[train_len:]):
    file_name = str(band_0).split('/')[-1]

    if file_name != str(band_1).split('/')[-1] or file_name != str(mask).split('/')[-1] or file_name != str(prob).split('/')[-1]:
        print(band_0)
        print(band_1)
        print(mask)
        print(prob)
        break

    subject = tio.Subject(
        band_0=tio.ScalarImage(band_0),
        band_1=tio.ScalarImage(band_1),
        iceberg=tio.LabelMap(mask),
        sampler_probability=tio.LabelMap(prob),
    )

    val_subjects.append(subject)

queue_length = 100
samples_per_volume = 1

probabilities = {
    0: 0,
    1: 0.025,
    2: 0.025,
    3: 0.15,
    4: 0.3,
    5: 0.5,
}

#
# Train dataloader
#

def get_train_iip_dataloader(patch_size, batch_size=2):
    train_sampler = tio.data.LabelSampler(
        (patch_size, patch_size, 1),
        'sampler_probability',
        label_probabilities=probabilities,
    )

    training_transform = tio.Compose([
        tio.RescaleIntensity(
            out_min_max=(0, 1),
            in_min_max=(0, 255),
        ),
        tio.Pad((50, 50, 0), padding_mode=0),
        tio.RandomFlip(
            axes=(1,),
            flip_probability=0.5,
        ),
        tio.RandomAffine(
            scales=0,
            translation=(64, 64, 0),
            degrees=0,
            include=['sampler_probability'],
        ),
        tio.RandomAffine(
            scales=(0.1, 0.1, 0),
            translation=0,
            degrees=(0, 0, 5),
        ),
    ])

    train_dataset = tio.SubjectsDataset(train_subjects, transform=training_transform)

    train_queue = tio.Queue(
        train_dataset,
        queue_length,
        samples_per_volume,
        train_sampler,
        num_workers=num_workers,
    )

    train_dataloader = DataLoader(
        train_queue,
        batch_size=batch_size,
        num_workers=0,
    )

    return train_dataloader

#
# Validation dataset
#

def get_validation_iip_dataset(patch_size):
    val_transform = tio.Compose([
        tio.RescaleIntensity(
            out_min_max=(0, 1),
            in_min_max=(0, 255),
        ),
        tio.Pad((patch_size//2, patch_size//2, 0), padding_mode=0),
    ])

    val_dataset = tio.SubjectsDataset(val_subjects, transform=val_transform)

    return val_dataset


#
# Extract tensors from batch pathces
#

def extract_iip_data(patches_batch):
    band_0 = patches_batch['band_0'][tio.DATA]
    band_1 = patches_batch['band_1'][tio.DATA]
    targets = patches_batch['iceberg'][tio.DATA] / 255
    bands = torch.cat((band_0, band_1), axis=1)

    return bands, targets