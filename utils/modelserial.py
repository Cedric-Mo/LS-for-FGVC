import os
import torch
from config import config

args = config()
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

def saveCheckpoint(state, datasetname=None):
    """Save checkpoint if a new best is achieved"""

    filename = './ckpt/{}-checkpoint.pth.tar'

    # if is_best:
    print("=> Saving a new best")
    torch.save(state, filename.format(datasetname))  # save checkpoint
    # else:
    #     print("=> Validation Accuracy did not improve")

def loadCheckpoint(datasetname):
    filename = './ckpt/{}-checkpoint.pth.tar'
    checkpoint = torch.load(filename.format(datasetname))
    return checkpoint
