import json
import os
import logging
import torch
import torch.nn.functional as F

import torchio as tio

from pathlib import Path
from torch import optim
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

from unet import UNet, AttentionUnet, UNet3plus, UNet3plusAttention, UNet3plusCBAM
from transunet import TransUNet
from utils.data_loading_iip import get_train_iip_dataloader, get_validation_iip_dataset, extract_iip_data
from utils.data_loading_hrsid import get_train_hrsid_dataloader, get_validation_hrsid_dataset, extract_hrsid_data
from utils.dice_score import dice_loss
from utils.evaluate import evaluate

MODELS = {
    'unet3+': UNet3plus,
    'att_gate_unet3+': UNet3plusAttention,
    'cbam_unet3+': UNet3plusCBAM,
    'transunet': TransUNet,
}
CONFIG_FILE = './models_configs.json'

CHECKPOINT_NAME = None
# Unet3plus training example on the iip iceberg dataset
MODEL_NAME = 'unet3+' # unet3+ | att_gate_unet3+ | cbam_unet3+ | transunet
DATASET_NAME = 'iip' # hrsid | iip
LOG_LEVEL = 'INFO'
GPU_ID = 0

if __name__ == '__main__':
    logging.basicConfig(level=LOG_LEVEL)

    model_name = MODEL_NAME
    model_class = MODELS[model_name]
    dataset = DATASET_NAME

    with open(CONFIG_FILE, 'r') as config_file:
        config = json.load(config_file)
    
    model_path = f'ckpts/{model_name}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    epochs = config[dataset][model_name]['epochs']
    learning_rate = config[dataset][model_name]['learning_rate']
    n_channels = config[dataset][model_name]['n_channels']
    n_classes = config[dataset][model_name]['n_classes']
    weight_decay = config[dataset][model_name]['weight_decay']
    momentum = config[dataset][model_name]['momentum']
    gradient_clipping = config[dataset][model_name]['gradient_clipping']
    amp = config[dataset][model_name]['amp']
    patch_size = config[dataset][model_name]['patch_size']
    batch_size = config[dataset][model_name]['batch_size']

    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

    if CHECKPOINT_NAME is not None:
        checkpoint = torch.load(f'{model_path}/{CHECKPOINT_NAME}.pt', f'cuda:{GPU_ID}')
    
    if model_name == 'transunet':
        model = model_class(
            img_dim=patch_size,
            in_channels=n_channels,
            out_channels=128,
            head_num=4,
            mlp_dim=512,
            block_num=8,
            patch_dim=16,
            class_num=n_classes,
        )
    else:
        model = model_class(
            n_channels=n_channels,
            n_classes=n_classes,
            bilinear=False,
        )

    if CHECKPOINT_NAME is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(memory_format=torch.channels_last)
    model.to(device=device)

    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        foreach=True,
    )
    if CHECKPOINT_NAME is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = BCEWithLogitsLoss()

    best_val_score = checkpoint['best_score'] if CHECKPOINT_NAME is not None else 0

    logging.info(f'Previous best validation score: {best_val_score}')

    global_step = 0
    start_epoch = checkpoint['epoch'] if CHECKPOINT_NAME is not None else 0

    train_dataloader = get_train_iip_dataloader(
        patch_size=patch_size,
        batch_size=batch_size,
    ) if dataset == 'iip' else get_train_hrsid_dataloader(
        patch_size=patch_size,
        batch_size=batch_size,
    )

    validation_dataset = get_validation_iip_dataset(
        patch_size=patch_size,
    ) if dataset == 'iip' else get_validation_hrsid_dataset(
        patch_size=patch_size,
    )

    data_extractor = extract_iip_data if dataset == 'iip' else extract_hrsid_data

    for epoch in range(start_epoch+1, epochs+1):
        epoch_loss = 0
        for patches_batch in train_dataloader:
            bands, targets = data_extractor(patches_batch)

            bands = bands.view(bands.shape[0], n_channels, patch_size, patch_size)
            targets = targets.view(targets.shape[0], patch_size, patch_size)
            
            assert bands.shape[1] == n_channels, \
                        f'Network has been defined with {n_channels} input channels, ' \
                        f'but loaded images have {bands.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'
            
            images = bands.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = targets.to(device=device, dtype=torch.long)

            model.train()
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                preds = model(images)
                bce = criterion(preds.squeeze(1), true_masks.float())
                dice = dice_loss(F.sigmoid(preds.squeeze(1)), true_masks.float(), multiclass=False)
                loss = bce + dice

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            loss_val = loss.item()
            global_step += 1
            epoch_loss += loss_val

            lr_ = optimizer.param_groups[0]['lr']
            logging.info(f'[{epoch}/{epochs}]\n\ttrain loss: {loss_val} | {bce.item()} | {dice.item()}\n\tstep: {global_step}/{global_step%len(train_dataloader)}\n\tlr: {lr_}\n')

        val_score = evaluate(model, device, validation_dataset, data_extractor, n_channels)[0]
        scheduler.step(val_score)
        logging.info(f'[{epoch}/{epochs}]\n\tval score: {val_score}\n\tbest: {best_val_score}\n\tstep: {global_step}\n')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_val,
            'best_score': best_val_score,
        }, f'{model_path}/model-last.pt')

        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_val,
                'best_score': best_val_score,
            }, f'{model_path}/model-{epoch}.pt')

        if val_score > best_val_score:
            print(f'\n\tNew best validation score: {val_score}\n\tOld: {best_val_score}\n')
            best_val_score = val_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_val,
                'best_score': best_val_score,
            }, f'{model_path}/model-best.pt')
