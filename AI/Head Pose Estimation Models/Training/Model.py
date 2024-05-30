import torch.nn as nn
from backbone.repvgg import get_RepVGG_func_by_name
from backbone.efficientnet_lite import build_efficientnet_lite
import torch
import utils
class SixDRepNet(nn.Module):
    def __init__(self,
                 backbone_name, backbone_file, deploy,
                 pretrained=True):
        super(SixDRepNet, self).__init__()
        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k,
                    v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)
            # for param in backbone.parameters():
            #     param.requires_grad = False
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        #self.greyscaletorgb = nn.Conv3d(in_channels=1,out_channels=3,kernel_size=3,padding="same")
        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        fea_dim = last_channel

        self.linear_reg = nn.Linear(fea_dim, 6)

    def forward(self, x):
        #x = self.greyscaletorgb(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear_reg(x)
        return utils.compute_rotation_matrix_from_ortho6d(x)
class SixDENet(nn.Module):
    def __init__(self,
                 backbone_name, backbone_file, deploy,
                 pretrained=True):
        super(SixDENet, self).__init__()
        self.backbone = build_efficientnet_lite(backbone_name,1000)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 6)
        if pretrained and backbone_file:
            self.backbone.load_pretrain(backbone_file)
            self.backbone.eval()
        

    def forward(self, x):
        #x = self.greyscaletorgb(x)
        x = self.backbone(x)
        return utils.compute_rotation_matrix_from_ortho6d(x)
