import os
import sys
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.trainUtils import train
from utils.testUtils import test
from args import Arguments
from utils.ioUtils import save_checkpoint, resume_model
from utils.critUtils import LabelSmoothing
from models import TransformerModel

# get arguments
args = Arguments()

# Log to file
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(args.log_path)])
logger = logging.getLogger('nlp')
logger.info('Logging to file...')

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=args.device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dont use writer temporarily
writer = None

best_wer = 999.00
start_epoch = 0

# Train with Transformer
if __name__ == '__main__':
    # Prepare data
    SRC = Field(tokenize = 'spacy',
            tokenizer_language = 'de',
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

    TRG = Field(tokenize = 'spacy',
                tokenizer_language = 'en',
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True)

    train_data, val_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (SRC, TRG))
    train_iter, val_iter, test_iter = BucketIterator.splits((train_data, val_data, test_data), batch_size = 2)

    print(len(train_iter))

    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)

    # Create model
    model = TransformerModel(len(SRC.vocab),len(TRG.vocab),
        args.d_model,args.n_head,args.num_enc_layers,
        args.num_dec_layers,args.dim_feedforword,
        args.dropout,args.activation).to(device)
    if args.resume_model is not None:
        start_epoch, best_wer = resume_model(model,args.resume_model)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    criterion = nn.CrossEntropyLoss(ignore_index = SRC.vocab.stoi['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Start training
    logger.info("Training Started".center(60, '#'))
    for epoch in range(start_epoch, args.epochs):
        # Train the model
        train(model, criterion, optimizer, trainloader, device, epoch, logger, args.log_interval, writer)
        # Test the model
        wer = test(model, criterion, testloader, device, epoch, logger, args.log_interval, writer)
        # Save model
        # remember best wer and save checkpoint
        is_best = wer<best_wer
        best_wer = min(wer, best_wer)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_wer': best_wer
        }, is_best, args.model_path, args.store_name)
        logger.info("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

    logger.info("Training Finished".center(60, '#'))

