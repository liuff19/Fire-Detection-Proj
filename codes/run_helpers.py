import os
import sys

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument('--mode', type=str, default='test', help='train or test')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='path to directory with fire images')
    parser.add_argument('--epoch', type=int, default=30, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--create_model', type=str, default='vit_base_patch16_224',
                        help='the model we used')
    parser.add_argument('--ckpt_path', type=str, default='./saved_models/FireRes.pth', help='path to save checkpoints')
    parser.add_argument('--backbone_print', action='store_true', help='whether to print backbone')
    # configurations for prediction
    parser.add_argument('--im_path', type=str, default='', help='path of an image to be recognized')

    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda')
    return parser