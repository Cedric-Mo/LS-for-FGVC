import os, sys, time

import torch
import torch.nn as nn
import torch.optim as opt

from torchvision import transforms, datasets
from torch.optim import lr_scheduler

from config import config
from utils import imdb
import modellearning
import models

progpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(progpath)

# define hyperparameter
args = config()

##################### Dataset path
datasets_path = os.path.expanduser("~/Datas")
datasetname = args.dataset
datasetpath = os.path.join(datasets_path, datasetname)

os.environ['CUDA_VISIBLE_DEVICES'] = args.device

# parameters
batchsize = 8

osmeflag = args.osme
nparts = args.nparts     # number of parts you want to use for your dataset

gamma1 = args.gamma1
gamma2 = args.gamma2
lr = args.lr
optmeth = 'sgd'

epochs = 30
isckpt = False     # if you want to load model params from checkpoint, set it to True

# define log_path
date = time.strftime('%Y-%m-%d~%H.%M')
modelname = r"{}-parts{}-gamma{}_{}-{}-lr{}-resnet".format(datasetname, nparts, gamma1, gamma2, optmeth, lr)
log_name = date + "-" + modelname + ".txt"
log_path = os.path.join("./results", log_name)

# organizing data
assert imdb.creatDataset(datasetpath, datasetname=datasetname) == True, "Failing to creat train/val/test sets"
data_transform = {
    'trainval': transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.RandomCrop((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.CenterCrop((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# using ground truth data
datasplits = {x: datasets.ImageFolder(os.path.join(datasetpath, x), data_transform[x])
              for x in ['trainval', 'test']}

dataloader = {x: torch.utils.data.DataLoader(datasplits[x], batch_size=batchsize, shuffle=True, num_workers=8)
              for x in ['trainval', 'test']}

datasplit_sizes = {x: len(datasplits[x]) for x in ['trainval', 'test']}
class_names = datasplits['trainval'].classes
num_classes = len(class_names)

# constructing or loading model
model = models.resnet50(num_classes=num_classes, osmeflag=osmeflag, nparts=nparts, pretrained=True)

# creating loss functions
cls_loss = nn.CrossEntropyLoss()
c3s_reg_loss = models.C3S_RegularLoss(gamma=gamma1, nparts=nparts)
mnl_reg_loss = models.MNL_RegularLoss(gamma=gamma2)
criterion = [cls_loss, c3s_reg_loss, mnl_reg_loss]

# creating optimizer
optimizer = opt.SGD(model.parameters(), lr=lr, momentum=0.9)

# creating optimization scheduler
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

# DataParallelism
if torch.cuda.device_count() > 0:
    model = nn.DataParallel(model)
model.cuda()

# training the model
print("{}: {}, gamma1: {}, gamma2: {}, nparts: {}, epochs: {}".format(optmeth, lr, gamma1, gamma2, nparts, epochs))

model, train_rsltparams = modellearning.train(model, dataloader, criterion, optimizer, scheduler,
                                               datasetname=datasetname, isckpt=isckpt,
                                               epochs=epochs, log_path=log_path)

#### save model
modelpath = './models'
modelname = r"{}-parts{}-gamma{}_{}-{}-lr{}-resnet.model".format(datasetname, nparts, gamma1, gamma2, optmeth, lr)
torch.save(model.state_dict(), os.path.join(modelpath, modelname))
