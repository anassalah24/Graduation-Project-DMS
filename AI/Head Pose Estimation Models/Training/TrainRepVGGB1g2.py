# Required libraries to train models
# numpy and pandas used for data collection and vector arithmetics 
# os is used for management of path where model snapshots are saved
# Torch is used as main pytorch training library and to develop the loss function 
# Torchvision is used to provide a preprocessing stage for both the training and validation sets
# TensorBoard is used to log each validation score for models
# TQDM is used to provide a visual representation for progress of models
# utils is used to convert rotational matrix to euler angles when needed
# Model is used to encapsulate different models definitions
# Dataset is used to encapsulate different Datasets definitions
import numpy as np
import pandas as pd
import os
import time
import torch
import torch.nn as nn
from torch.backends import cudnn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import utils
from Model import SixDRepNet
from Dataset import pose_eff_dataset,BIWI
# Starting torchboard Logging
writer = SummaryWriter()
# definition of loss function to use

class GeodesicLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, m1, m2):
        # batch matrix multiplication is used to perform matrix multiplications for various input matrices
        m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3

        cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
        theta = torch.acos(torch.clamp(cos, -1+self.eps, 1-self.eps))

        return torch.mean(theta)
#
def main():
    # it all starts by enabling cuda acceleration 
    # defining the batch size = amount of data used at each epoch
    # learning rate = by how much does the error affects the weights at each step
    cudnn.enabled = True
    snapshot=''
    batch_size = 64
    gpu = 0
    b_scheduler = False
    lr = 1e-4
    # the creation of folder to hold models
    if not os.path.exists('./output/snapshots'):
        os.makedirs('./output/snapshots')
    # inside the snapshots folder each run of this script has its own folder
    # where its name is SixDRepNet_{timeinnumbers}_bs{batch size} 
    summary_name = '{}_{}_bs{}'.format(
        'SixDRepNet', int(time.time()), batch_size)
    
    if not os.path.exists('output/snapshots/{}'.format(summary_name)):
        os.makedirs('output/snapshots/{}'.format(summary_name))
    # declaring model where backbone name is provided with the pretrained model
    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                        backbone_file='./RepVGG-B1g2-train.pth',
                        deploy=False,
                        pretrained=True)
    # this allow for resuming training from another snapshot if needed
    if not snapshot == '':
        saved_state_dict = torch.load(snapshot)
        model.load_state_dict(saved_state_dict['model_state_dict'])
    # loading training data from .pkl file
    print('Loading data.')
    pkla=pd.read_pickle("./Datasets/300W_LP/file.pkl")
    pkla = pkla.sample(frac=1, random_state=42)
    # specifying preprocessing steps as cropping image-turning image into tensor -normalizing image
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

    transformations = transforms.Compose([transforms.RandomResizedCrop(size=224,scale=(0.8,1)),
                                        transforms.ToTensor(),
                                        normalize])
    # here its required to specify root directory for 300W_lP dataset
    train_pose_dataset =pose_eff_dataset('./Datasets/300W_LP/',
                                    pkla,
                                    transformations)
    # then the validation dataset is created
    val_pose_dataset =BIWI("./Datasets/BIWI_done.npz",
                            transform=transformations,
                            train_mode=False)
    # specifying data loaders for each dataset 
    train_effloader = torch.utils.data.DataLoader(
        dataset=train_pose_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    val_effloader = torch.utils.data.DataLoader(
        dataset=val_pose_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    # sending model and loss function to GPU and using 
    # scheduled learning rate to decrease learning rate after 10 and 20 iterations
    model = model.cuda(gpu)
    crit = GeodesicLoss().cuda(gpu) #torch.nn.MSELoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    num_epochs= 30
    milestones = [10, 20]
    b_scheduler = True
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5)
    datasetname="300w-lp"
    datasettest="BIWI"
    output_string= "Trainon"+datasetname+"Teston"+datasettest
    print('Starting training.')

    lowest_epoch_crossval= 1000000
    lowest_yaw=1000000
    lowest_pitch=1000000
    lowest_roll=1000000
    lowestepochno=0
    bestmodel=model
    # at each epoch the model is trained and then validation score of model is compared
    # with previous models and model with best validation score is saved
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

        with torch.no_grad():

            for i, (images, label) in enumerate(val_effloader):
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


            print('Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (
                yaw_error / total, pitch_error / total, roll_error / total,
                (yaw_error + pitch_error + roll_error) / (total * 3)))
        MAE= (yaw_error + pitch_error + roll_error) / (total * 3)
        writer.add_scalars('run_full', {'yaw':yaw_error/total,
                                        'pitch':pitch_error/total,
                                        'roll': roll_error/total,
                                        'total': (yaw_error + pitch_error + roll_error) / (total * 3)}, i)
        if MAE < lowest_epoch_crossval :
            lowest_epoch_crossval = MAE
            lowest_yaw= yaw_error / total
            lowest_roll=  roll_error / total
            lowest_pitch= pitch_error / total
            lowestepochno = epoch
            bestmodel=model
    print('Taking snapshot...',
        torch.save({
            'epoch': epoch,
            'model_state_dict': bestmodel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'output/snapshots/' + summary_name + '/' + output_string +
            '_epoch_' + str(epoch+1)+ '.tar')
        )      
    print(f"Best accuracy is MAE of {lowest_epoch_crossval} \n yaw loss of {lowest_yaw} \n pitch loss of{lowest_pitch} \n roll loss of{lowest_roll} at epoch {lowestepochno}")
    writer.close()
if __name__=="__main__":
    main()