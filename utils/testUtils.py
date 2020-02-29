import torch
import time
from utils.metricUtils import *

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

def test(model, criterion, testloader, device, epoch, logger, log_interval, writer, TRG):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # losses = AverageMeter()
    avg_bleu = AverageMeter()
    # Set eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, data in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            # get the inputs and labels
            src, tgt = data.src.to(device), data.trg.to(device)

            # forward
            outputs = model.module.greedy_decode(src, 15)

            # compute the loss
            # loss = criterion(outputs.view(-1, outputs.shape[-1]), tgt.view(-1))

            # compute the bleu metrics
            bleu = count_bleu(outputs, tgt, TRG)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # update average value
            # losses.update(loss)
            avg_bleu.update(bleu)

            if i % log_interval == 0:
                output = ('[Test] Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                        'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                        # 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'bleu {bleu.val:.4f} ({bleu.avg:.4f})\t'
                        .format(
                            epoch, i, len(testloader), batch_time=batch_time,
                            data_time=data_time,  bleu=avg_bleu
                            ))

                logger.info(output)

    return avg_bleu.avg
        
