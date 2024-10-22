from .vit import vit
from .hrnet import hrnet_w48
from .resnet import resnet

def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'vit':
        return vit(cfg)
    elif cfg.MODEL.BACKBONE.TYPE == 'resnet':
        return resnet(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')

def create_hmr_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'hrnet_w48-conv':
        pretrained_ckpt_path = "hmr2_training_data/pose_hrnet_w48_256x192.pth"
        return hrnet_w48(pretrained_ckpt_path=pretrained_ckpt_path, downsample = True, use_conv=True)
    else:
        raise NotImplementedError('Backbone type is not implemented')
