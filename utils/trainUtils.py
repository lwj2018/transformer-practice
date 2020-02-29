import torch
import time
from utils.metricUtils import *
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(model, criterion, optimizer, trainloader, device, epoch, logger, log_interval, writer, TRG):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_bleu = AverageMeter()
    # Set trainning mode
    model.train()

    end = time.time()
    for i, data in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # get the inputs and labels
        src, tgt = data.src.to(device), data.trg.to(device)

        optimizer.zero_grad()
        # forward
        outputs = model(src, tgt[:-1,:])

        # compute the loss
        loss = criterion(outputs.view(-1, outputs.shape[-1]), tgt[1:,:].view(-1))

        # compute the bleu metrics
        bleu = count_bleu(outputs, tgt[1:,:], TRG)

        # backward & optimize
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # update average value
        losses.update(loss)
        avg_bleu.update(bleu)

        if i % log_interval == 0:
            output = ('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'bleu {bleu.val:.4f} ({bleu.avg:.4f})\t'
                    .format(
                        epoch, i, len(trainloader), batch_time=batch_time,
                        data_time=data_time, loss=losses,  bleu=avg_bleu,
                        lr=optimizer.param_groups[-1]['lr']))

            logger.info(output)
    
