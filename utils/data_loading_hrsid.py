import torchio as tio
import random

from pathlib import Path
from torch.utils.data import DataLoader

random.seed(42)
num_workers = 5

#dataset_path_base = '/home/bagurgle/mag/hrsid'
dataset_path_base = './data/hrsid'
band_path = f'{dataset_path_base}/images'
masks_path = f'{dataset_path_base}/masks'
masks_prob_path = f'{dataset_path_base}/masks_prob'

bands = sorted(list(Path(band_path).glob('*.jpg')))
masks = sorted(list(Path(masks_path).glob('*.jpg')))
probs = sorted(list(Path(masks_prob_path).glob('*.jpg')))

idxs = list(range(0, len(bands)))
random.shuffle(idxs)

bands = [bands[idx] for idx in idxs]
masks = [masks[idx] for idx in idxs]
probs = [probs[idx] for idx in idxs]

val_len = len(bands) // 10
train_len = len(bands) - val_len

train_subjects = []
for (band, mask, prob) in zip(bands[:train_len], masks[:train_len], probs[:train_len]):
    file_name = str(band).split('/')[-1]

    if file_name != str(mask).split('/')[-1] or file_name != str(prob).split('/')[-1]:
        print(band)
        print(mask)
        print(prob)
        break

    subject = tio.Subject(
        band=tio.ScalarImage(band),
        ship=tio.LabelMap(mask),
        sampler_probability=tio.LabelMap(prob),
    )

    train_subjects.append(subject)

val_subjects = []
for (band, mask, prob) in zip(bands[train_len:], masks[train_len:], probs[train_len:]):
    file_name = str(band).split('/')[-1]

    if file_name != str(mask).split('/')[-1] or file_name != str(prob).split('/')[-1]:
        print(band)
        print(mask)
        print(prob)
        break

    subject = tio.Subject(
        band=tio.ScalarImage(band),
        ship=tio.LabelMap(mask),
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

def get_train_hrsid_dataloader(patch_size, batch_size=2):
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

def get_validation_hrsid_dataset(patch_size):
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

def extract_hrsid_data(patches_batch):
    bands = patches_batch['band'][tio.DATA]
    targets = patches_batch['ship'][tio.DATA] / 255

    return bands, targets