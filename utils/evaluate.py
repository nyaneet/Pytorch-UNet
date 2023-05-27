import torch
import torchio as tio
import torch.nn.functional as F

import time

from tqdm import tqdm

@torch.inference_mode()
def evaluate(model, device, val_dataset, data_extractor, n_channels, patch_size=128, threshold=0.5):
    model.eval()
    dice_score = 0
    iou_score = 0
    epsilon = 1e-6
    
    num_val_samples = len(val_dataset)
    
    for subject in tqdm(val_dataset):
        time.sleep(0.5)
        grid_sampler = tio.inference.GridSampler(
            subject,
            (patch_size, patch_size, 1),
            patch_overlap=0,
        )

        validation_dataloader = torch.utils.data.DataLoader(grid_sampler, batch_size=4)
        pred_aggregator = tio.inference.GridAggregator(grid_sampler)
        mask_aggregator = tio.inference.GridAggregator(grid_sampler)
        
        with torch.no_grad():
            for patches_batch in validation_dataloader:
                bands, targets = data_extractor(patches_batch)
                
                base_shape = targets.shape
                locations = patches_batch[tio.LOCATION]

                bands = bands.view(bands.shape[0], n_channels, patch_size, patch_size)
                images = bands.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                
                targets = targets.view(targets.shape[0], patch_size, patch_size)

                preds = model(images)
                preds = (F.sigmoid(preds) > threshold).float()

                pred_aggregator.add_batch(preds.view(base_shape), locations)
                mask_aggregator.add_batch(targets.view(base_shape), locations)
        
            preds = pred_aggregator.get_output_tensor()
            masks = mask_aggregator.get_output_tensor()

            dice_score += (2*torch.sum(preds*masks)+epsilon) / (torch.sum(preds)+torch.sum(masks)+epsilon)
            iou_score += (torch.sum(preds*masks)+epsilon) / (torch.sum(preds)+torch.sum(masks)-torch.sum(preds*masks)+epsilon)

    return dice_score / max(num_val_samples, 1), iou_score / max(num_val_samples, 1)