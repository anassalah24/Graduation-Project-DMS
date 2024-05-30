import numpy as np

import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import torch
import torch.nn as nn
from torch.backends import cudnn
from torchvision import transforms
import utils
from Model import SixDRepNet,SixDENet
from Dataset import BIWI
def run():
    # li_of_backends=['RepVGG-AX',
    #                 'RepVGG-AY',
    #                 'RepVGG-AZ']
    # li_of_backends_path={
    #                 'RepVGG-AX': 'Trainon300w-lpTestonBIWIbackboneRepVGG-AX_epoch_80.tar',
    #                 'RepVGG-AY': 'Trainon300w-lpTestonBIWIbackboneRepVGG-AY_epoch_80.tar',
    #                 'RepVGG-AZ': 'Trainon300w-lpTestonBIWIbackboneRepVGG-AZ_epoch_80.tar',
    #     }
    li_of_backends=['efficientnet_lite0',
                    'efficientnet_lite1',
                    'efficientnet_lite2',
                    'efficientnet_lite3']
    li_of_backends_path={
                   'efficientnet_lite0':'Trainon300w-lpTestonBIWIbackboneefficientnet_lite0_epoch_80.tar',
                    'efficientnet_lite1':'Trainon300w-lpTestonBIWIbackboneefficientnet_lite1_epoch_80.tar',
                    'efficientnet_lite2':'Trainon300w-lpTestonBIWIbackboneefficientnet_lite2_epoch_80.tar',
                    'efficientnet_lite3': 'Trainon300w-lpTestonBIWIbackboneefficientnet_lite3_epoch_80.tar'
        }
    rootRep= 'E:/HeadPose/Training/output/snapshots/SixDRepNet_1715636118_bs64/'
    rootEff= 'E:/HeadPose/Training/output/snapshots/SixDRepNet_1716136846_bs64/'
    batch_size=80
    gpu=0
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

    transformations = transforms.Compose([transforms.RandomResizedCrop(size=224,scale=(0.8,1)),
                                        transforms.ToTensor(),
                                        normalize])
    test_pose_dataset =BIWI("E:/HeadPose/Training/Datasets/BIWI_done.npz",
                                    transform=transformations,
                                    train_mode=False) 
    test_effloader = torch.utils.data.DataLoader(
                dataset=test_pose_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4)
    def load_filtered_state_dict(model, snapshot):
        # By user apaszke from discuss.pytorch.org
        model_dict = model.state_dict()
        snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
        model_dict.update(snapshot)
        model.load_state_dict(model_dict)
    for backendna in li_of_backends:
        # model = SixDRepNet(backbone_name=backendna,
        #                     backbone_file='',
        #                     deploy=False,
        #                     pretrained=False)
        # model = model.cuda(gpu)
        # saved_state_dict = torch.load(root+li_of_backends_path[backendna])
    
        # load_filtered_state_dict(model, saved_state_dict['model_state_dict'])
        # model.eval()
        model = SixDENet(backbone_name=backendna,backbone_file=rootEff+li_of_backends_path[backendna],deploy=False,pretrained=True) 
        model = model.cuda(gpu)
        model.eval()
        total = 0
        yaw_error = pitch_error = roll_error = .0
        # v1_err = v2_err = v3_err = .0

        with torch.no_grad():

            for i, (images, label) in enumerate(test_effloader):
                images = torch.Tensor(images).cuda(gpu)
                total += label.size(0)

                # gt matrix
                R_gt = label

                # gt euler
                pose = utils.compute_euler_angles_from_rotation_matrices(
                    R_gt)*180/np.pi
                # And convert to degrees.
                p_gt_deg = pose[:,0].cpu()
                y_gt_deg = pose[:,1].cpu()
                r_gt_deg = pose[:,2].cpu()

                R_pred = model(images)

                euler = utils.compute_euler_angles_from_rotation_matrices(
                    R_pred)*180/np.pi
                p_pred_deg = euler[:, 0].cpu()
                y_pred_deg = euler[:, 1].cpu()
                r_pred_deg = euler[:, 2].cpu()
                pitch_error += torch.sum(torch.min(torch.stack((torch.abs(p_gt_deg - p_pred_deg), torch.abs(p_pred_deg + 360 - p_gt_deg), torch.abs(
                    p_pred_deg - 360 - p_gt_deg), torch.abs(p_pred_deg + 180 - p_gt_deg), torch.abs(p_pred_deg - 180 - p_gt_deg))), 0)[0])
                yaw_error += torch.sum(torch.min(torch.stack((torch.abs(y_gt_deg - y_pred_deg), torch.abs(y_pred_deg + 360 - y_gt_deg), torch.abs(
                    y_pred_deg - 360 - y_gt_deg), torch.abs(y_pred_deg + 180 - y_gt_deg), torch.abs(y_pred_deg - 180 - y_gt_deg))), 0)[0])
                roll_error += torch.sum(torch.min(torch.stack((torch.abs(r_gt_deg - r_pred_deg), torch.abs(r_pred_deg + 360 - r_gt_deg), torch.abs(
                    r_pred_deg - 360 - r_gt_deg), torch.abs(r_pred_deg + 180 - r_gt_deg), torch.abs(r_pred_deg - 180 - r_gt_deg))), 0)[0])

            print(backendna)
            print('Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (
                yaw_error / total, pitch_error / total, roll_error / total,
                (yaw_error + pitch_error + roll_error) / (total * 3)))
        MAE= (yaw_error + pitch_error + roll_error) / (total * 3)
if __name__=="__main__":
    run()