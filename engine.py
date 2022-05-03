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

def train_one_step(images, targets, model, device, optimizer, loss_list ,out_dir, 
                    epoch, epoch_count, iter_count, metric_logger):
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

    return loss_list

def val_one_step(images, targets, model, device, loss_list):
    # loading images to gpu
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    # getting loss data from model and processing data for logging
    loss_dict = model(images, targets) 
    loss_dict_reduced = utils.reduce_dict(loss_dict)
    losses_reduced = sum(loss for loss in loss_dict_reduced.values())          
    loss_value = losses_reduced.item()
    
    # if not zero/infinate
    if not math.isfinite(loss_value):
        print(f"Loss is {loss_value}, stopping training")
        print(loss_dict_reduced)
        sys.exit(1)
    
    # appending zero value to list
    loss_list.append(loss_value)
    
    return loss_list

def train_one_epoch(print_freq, data_loader, model, device, optimizer, loss_list ,out_dir,
                         epoch, epoch_count):
    # config model for training
    model.train()

    # initialise logging
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    
    # initialise itter count
    iter_count = 1
    loop_list = []

    # iterating through images in data loader 
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        loop_list = train_one_step(images, targets, model, device, optimizer, loop_list, 
                                   out_dir, epoch, epoch_count, iter_count, metric_logger)
        
        iter_count += 1
    
    loss_list.append(sum(loop_list)/len(loop_list))
    
    return loss_list

def val_one_epoch(model, device, loss_list, data_loader, epoch, print_freq):
    # config model for training
    model.train()
    loop_list = []
    
    for images, targets in data_loader:
        loop_list = val_one_step(images, targets, model, device, loop_list)
    
    loss_list.append(sum(loop_list)/len(loop_list))

    return loss_list

def training_loop(model, device, optimizer, train_data_loader, val_data_loader, 
                    start_epoch, num_epochs, print_freq, out_dir):    
    # initialising data capture
    train_loss_list = []
    val_loss_list = []
    epoch_list = [] 
    
    # epoch counter
    epoch_count = start_epoch + 1

    for epoch in range(start_epoch, num_epochs):
        train_loss_list = train_one_epoch(print_freq, train_data_loader, model, device,
                                          optimizer, train_loss_list, out_dir, epoch, epoch_count)
        
        with torch.no_grad():
            val_loss_list = val_one_epoch(model, device, val_loss_list, val_data_loader,
                                         epoch, print_freq)

        epoch_list.append(epoch_count)
        epoch_count += 1
    
    return train_loss_list, val_loss_list, epoch_list
    
    