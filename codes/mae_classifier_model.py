import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import models_mae

chkpt_path = 'saved_models/mae_visualize_vit_large.pth'

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    model = getattr(models_mae, arch)()
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def classifier_mae():
    model_mae = prepare_model(chkpt_path)
    print('MAE Model loaded.')
    return model_mae
    