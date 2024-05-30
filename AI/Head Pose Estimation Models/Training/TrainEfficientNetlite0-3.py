# to understand how pipeline works refer to the TrainRepVGGB1g2.py
# here the same pipeline holds but with alot of models
# each model is trained until the best model snapshot is recorded
# this is used to train models Efficientnet-lite0->3 sequentially

import numpy as np

import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import torch
import torch.nn as nn
from torch.backends import cudnn
from torchvision import transforms
#import matplotlib
#from matplotlib import pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
import utils
from Model import SixDENet
from Dataset import pose_eff_dataset,BIWI

writer = SummaryWriter()
#matplotlib.use('TkAgg')
class GeodesicLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, m1, m2):
        m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3

        cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
        theta = torch.acos(torch.clamp(cos, -1+self.eps, 1-self.eps))

        return torch.mean(theta)



def main():
    li_of_backends={ 'efficientnet_lite0': './efficientnet_lite0.pth',
    'efficientnet_lite1': './efficientnet_lite1.pth',
    'efficientnet_lite2': './efficientnet_lite2.pth',
    'efficientnet_lite3': './efficientnet_lite3.pth',
    'efficientnet_lite4': './efficientnet_lite4.pth'}
    listofmaxscores={}
    cudnn.enabled = True
    snapshot=''
    batch_size = 64
    gpu = 0
    b_scheduler = False
    lr = 1e-4
    if not os.path.exists('./output/snapshots'):
        os.makedirs('./output/snapshots')

    summary_name = '{}_{}_bs{}'.format(
        'SixDRepNet', int(time.time()), batch_size)

    if not os.path.exists('output/snapshots/{}'.format(summary_name)):
        os.makedirs('output/snapshots/{}'.format(summary_name))
    for backendna in li_of_backends.keys():

        model = SixDENet(backbone_name=backendna,
                            backbone_file=li_of_backends[backendna],
                            deploy=False,
                            pretrained=True)

        if not snapshot == '':
            saved_state_dict = torch.load(snapshot)
            model.load_state_dict(saved_state_dict['model_state_dict'])

        print('Loading data.')
        pkla=pd.read_pickle("./Datasets/300W_LP/300W_LP/file.pkl")
        pkla = pkla.sample(frac=1, random_state=42)
        # train=pkla[:int(0.9*len(df_shuffled))]
        # test=pkla[int(0.9*len(df_shuffled)):]
        # test.reset_index(inplace=True)
        # print(test.head())
        normalize = transforms.Normalize(
        mean=[0.498, 0.498, 0.498],
        std=[0.502, 0.502, 0.502])

        transformations = transforms.Compose([transforms.RandomResizedCrop(size=224,scale=(0.8,1)),
                                            transforms.ToTensor(),
                                            normalize])
        train_pose_dataset =pose_eff_dataset('./Datasets/300W_LP/300W_LP',
                                        pkla,
                                        transformations)
        # test_pose_dataset =pose_eff_dataset('./Datasets/300W_LP/300W_LP',
        #                                 test,
        #                                 transformations)
        test_pose_dataset =BIWI("E:/HeadPose/Training/Datasets/BIWI_done.npz",
                                transform=transformations,
                                train_mode=False) 
        train_effloader = torch.utils.data.DataLoader(
            dataset=train_pose_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4)

        test_effloader = torch.utils.data.DataLoader(
            dataset=test_pose_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4)

        model = model.cuda(gpu)
        crit = GeodesicLoss().cuda(gpu) #torch.nn.MSELoss().cuda(gpu)
        optimizer = torch.optim.Adam(model.parameters(), lr)
        num_epochs=80
        milestones = [10, 20]
        b_scheduler = True
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.5)
        datasetname="300w-lp"
        datasettest="BIWI"
        output_string= "Trainon"+datasetname+"Teston"+datasettest+"backbone"+backendna
        print('Starting training.')
        lowest_epoch_crossval= 1000000
        lowest_yaw=1000000
        lowest_pitch=1000000
        lowest_roll=1000000
        lowestepochno=0
        bestmodel=model
        for epoch in range(num_epochs):
            loss_sum = .0
            iter = 0
            for (images, gt_mat) in  tqdm(train_effloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
                iter += 1
                images = torch.Tensor(images).cuda(gpu)
                # Forward pass
                pred_mat = model(images)
                # Calc loss
                loss = crit(gt_mat.cuda(gpu), pred_mat)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
            
                print('Epoch [%d/%d], Iter [%d/%d] Loss: '
                        '%.6f' % (
                            epoch+1,
                            num_epochs,
                            iter,
                            len(train_pose_dataset)//batch_size,
                            loss.item(),
                        )
                        )
                        

            if b_scheduler:
                scheduler.step()
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

                    # R_pred = R_pred.cpu()
                    # v1_err += torch.sum(torch.acos(torch.clamp(
                    #     torch.sum(R_gt[:, 0] * R_pred[:, 0], 1), -1, 1)) * 180/np.pi)
                    # v2_err += torch.sum(torch.acos(torch.clamp(
                    #     torch.sum(R_gt[:, 1] * R_pred[:, 1], 1), -1, 1)) * 180/np.pi)
                    # v3_err += torch.sum(torch.acos(torch.clamp(
                    #     torch.sum(R_gt[:, 2] * R_pred[:, 2], 1), -1, 1)) * 180/np.pi)

                    pitch_error += torch.sum(torch.min(torch.stack((torch.abs(p_gt_deg - p_pred_deg), torch.abs(p_pred_deg + 360 - p_gt_deg), torch.abs(
                        p_pred_deg - 360 - p_gt_deg), torch.abs(p_pred_deg + 180 - p_gt_deg), torch.abs(p_pred_deg - 180 - p_gt_deg))), 0)[0])
                    yaw_error += torch.sum(torch.min(torch.stack((torch.abs(y_gt_deg - y_pred_deg), torch.abs(y_pred_deg + 360 - y_gt_deg), torch.abs(
                        y_pred_deg - 360 - y_gt_deg), torch.abs(y_pred_deg + 180 - y_gt_deg), torch.abs(y_pred_deg - 180 - y_gt_deg))), 0)[0])
                    roll_error += torch.sum(torch.min(torch.stack((torch.abs(r_gt_deg - r_pred_deg), torch.abs(r_pred_deg + 360 - r_gt_deg), torch.abs(
                        r_pred_deg - 360 - r_gt_deg), torch.abs(r_pred_deg + 180 - r_gt_deg), torch.abs(r_pred_deg - 180 - r_gt_deg))), 0)[0])


                print('Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (
                    yaw_error / total, pitch_error / total, roll_error / total,
                    (yaw_error + pitch_error + roll_error) / (total * 3)))
            MAE= (yaw_error + pitch_error + roll_error) / (total * 3)
            writer.add_scalars('run_full'+backendna, {'yaw':yaw_error/total,
                                            'pitch':pitch_error/total,
                                            'roll': roll_error/total,
                                            'total': (yaw_error + pitch_error + roll_error) / (total * 3)}
                                            , i)
            if MAE < lowest_epoch_crossval :
                lowest_epoch_crossval = MAE
                lowest_yaw= yaw_error / total
                lowest_roll=  roll_error / total
                lowest_pitch= pitch_error / total
                lowestepochno = epoch
                bestmodel=model

            # Save models at numbered epochs.
            # if epoch % 1 == 0 and epoch < num_epochs:
            #     print('Taking snapshot...',
            #           torch.save({
            #               'epoch': epoch,
            #               'model_state_dict': model.state_dict(),
            #               'optimizer_state_dict': optimizer.state_dict(),
            #           }, 'output/snapshots/' + summary_name + '/' + output_string +
            #               '_epoch_' + str(epoch+1) + '.tar')
            #           )
        print('Taking snapshot...',
            torch.save({
                'epoch': epoch,
                'model_state_dict': bestmodel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'output/snapshots/' + summary_name + '/' + output_string +
                '_epoch_' + str(epoch+1)+ '.tar')
            )      
        print(f"Best accuracy is MAE of {lowest_epoch_crossval} \n yaw loss of {lowest_yaw} \n pitch loss of{lowest_pitch} \n roll loss of{lowest_roll} at epoch {lowestepochno}")
        listofmaxscores[backendna]={'total error':lowest_epoch_crossval,
                            'yaw error':lowest_yaw,
                            'pitch error':lowest_pitch,
                            'roll error':lowest_roll}
    # for i in li_of_backends.keys():
    #     print(i, end=":\n")
    #     for k in listofmaxscores[i].keys():
    #         print(k,":",listofmaxscores[i][k])
    print(listofmaxscores)
writer.close()
if __name__=="__main__":
    main()