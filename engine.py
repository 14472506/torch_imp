import utils
from matplotlib import pyplot as plt
import torch
import numpy as np
import math
import sys
import shutil

def checkpoint(epoch, model, optimizer, out_dir, epoch_count, iter_count):
    # generating state checkpoint dict
    utils.mkdir(out_dir + "/checkpoints/")
    
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    # file path
    file_name = str(epoch_count) + "_"  + str(iter_count) + ".pth"             
    file_path = out_dir + "/checkpoints/" + file_name

    # saving checkpoint
    torch.save(checkpoint, file_path)

def training_loop(model, device, optimizer, data_loader, start_epoch, num_epochs, print_freq, out_dir):
    # config model for training
    model.train()
    
    # initialising data capture
    loss_list = []
    epoch_count = start_epoch + 1

    for epoch in range(start_epoch, num_epochs):
        # initialise logging
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{epoch}]"

        iter_count = 1

        # iterating through images in data loader 
        for images, targets in metric_logger.log_every(data_loader, print_freq, header):
            
            # loading images to gpu
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # getting loss data from model and processing data for logging
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())  
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())          
            loss_value = losses_reduced.item()

            # if not zero/infinate
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict_reduced)
                sys.exit(1)

            # saving model
            if not loss_list or loss_value < min(loss_list):
                checkpoint(epoch, model, optimizer, out_dir, epoch_count, iter_count)
            # appending loss value to lost list
            loss_list.append(loss_value)

            # zero the gradient stored gadient before carrying out
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
       
            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            iter_count += 1

        epoch_count += 1
