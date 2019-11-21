import copy
import os
import torch
import time
import torch.nn.functional as F

from config import config
from utils import modelserial

args = config()
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

def train(model, dataloader, criterion, optimizer, scheduler, datasetname=None, isckpt=False, epochs=30, log_path=None):

    # get the size of train and evaluation data
    if isinstance(dataloader, dict):
        dataset_sizes = {x: len(dataloader[x].dataset) for x in dataloader.keys()}
        print(dataset_sizes)
    else:
        dataset_sizes = len(dataloader.dataset)

    if not isinstance(criterion, list):
        criterion = [criterion]

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    start_epoch = 1

    if isckpt:
        checkpoint = modelserial.loadCheckpoint(datasetname + '2')
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        best_model_params = checkpoint['best_state_dict']
        best_epoch = checkpoint['best_epoch']

    since = time.time()
    for epoch in range(start_epoch, epochs+1):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)
        with open(log_path, "a") as log_file:
            log_file.writelines('Epoch {}/{}\n'.format(epoch, epochs))
            log_file.writelines('-' * 10 + '\n')

        for phase in ['trainval', 'test']:
            if phase == 'trainval':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_cls_loss = 0.0
            running_c3s_reg_loss = 0.0
            running_mnl_reg_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader[phase]:
                #pdb.set_trace()
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'trainval'):
                    # pdb.set_trace()
                    outputs, parts = model(inputs)
                    log_softmax_out = F.log_softmax(outputs, dim=-1)
                    _, preds = torch.max(outputs, 1)

                    cls_loss = criterion[0](outputs, labels)
                    c3s_reg_loss = criterion[1](parts)
                    mnl_reg_loss = criterion[2](log_softmax_out)

                    total_loss = cls_loss + c3s_reg_loss + mnl_reg_loss

                    # backward + optimize only if in training phase
                    if phase == 'trainval':
                        # pdb.set_trace()
                        total_loss.backward()
                        optimizer.step()

                # statistics
                running_cls_loss += cls_loss.item() * inputs.size(0)
                running_c3s_reg_loss += c3s_reg_loss.item() * inputs.size(0)
                running_mnl_reg_loss += mnl_reg_loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'trainval':
                scheduler.step()

            epoch_loss = (running_cls_loss + running_c3s_reg_loss + running_mnl_reg_loss) / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            with open(log_path, "a") as log_file:
                log_file.writelines('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_params = copy.deepcopy(model.state_dict())

            if phase == 'test' and epoch % 5 == 4:
                modelserial.saveCheckpoint({'epoch': epoch,
                                            'best_epoch': best_epoch,
                                            'state_dict': model.state_dict(),
                                            'best_state_dict': best_model_params,
                                            'best_acc': best_acc}, datasetname+'2')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))
    with open(log_path, "a") as log_file:
        log_file.writelines('Training complete in {:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60))
        log_file.writelines('Best test Acc: {:4f}\n'.format(best_acc))

    rsltparams = dict()
    rsltparams['val_acc'] = best_acc.item()
    rsltparams['gamma1'] = criterion[1].gamma
    rsltparams['lr'] = optimizer.param_groups[0]['lr']
    rsltparams['best_epoch'] = best_epoch

    # load best model weights
    model.load_state_dict(best_model_params)
    return model, rsltparams


def eval(model, dataloader=None):
    model.eval()
    datasize = len(dataloader.dataset)
    running_corrects = 0
    for inputs, labels in dataloader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs, _ = model(inputs)
            preds = torch.argmax(outputs, dim=1)
        running_corrects += torch.sum(preds == labels.data)
    acc = torch.div(running_corrects.double(), datasize).item()
    print("Test Accuracy: {}".format(acc))

    rsltparams = dict()
    rsltparams['test_acc'] = acc
    return rsltparams


