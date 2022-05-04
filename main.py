from models import MaskRCNN_model
from data_loader import COCOLoader
import torch
import utils
from engine import training_loop
import json

def data_loader_config(dir, batch_size):
    # configuring json string
    json = "/" + dir.split("/")[-1] + ".json"
    
    # loading dataset
    dataset = COCOLoader(dir, dir + json)

    # configuring data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    
    # returing data loader
    return(data_loader)

def main(conf_dict):
    # Look into this for fixing random seed
    # https://vandurajan91.medium.com/random-seeds-and-reproducible-results-in-pytorch-211620301eba

    # This line should be ran first to ensure a gpu is being used if possible
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 2 classes, 1 is the background
     
    # retieving data loaders
    if conf_dict["train_ds"] is not "":
        train_data_loader = data_loader_config(conf_dict["train_ds"], conf_dict['batch_size'])
    if conf_dict["val_ds"] is not "": 
        val_data_loader = data_loader_config(conf_dict["val_ds"], conf_dict['batch_size'])
    if conf_dict["test_ds"] is not "":
        test_data_loader = data_loader_config(conf_dict["test_ds"], conf_dict['batch_size'])

    # get the model from model function and load it to device
    model = MaskRCNN_model(conf_dict['num_classes'])  
    model.to(device)  

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    start_epoch = 0
    
    # loading model data if specidied
    if conf_dict["load"] is not "":
        checkpoint = torch.load(conf_dict["load"])
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
    
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    training_loop(model, device, optimizer, train_data_loader, val_data_loader, start_epoch,
                  conf_dict["num_epochs"], conf_dict["print_freq"], conf_dict["out_dir"],
                  conf_dict["val_freq"])

    print("training complete")

if __name__ == "__main__":
    
    # defining configurabels
    conf_dict = {}
    conf_dict["train_ds"] = "data/jersey_royal_ds/train"
    conf_dict["val_ds"] = "data/jersey_royal_ds/val"
    conf_dict["test_ds"] = "data/jersey_royal_ds/test"

    conf_dict["batch_size"] = 2
    conf_dict["num_classes"] = 2 
    conf_dict["num_epochs"] = 20
    conf_dict["print_freq"] = 20
    conf_dict["val_freq"] = 20   
    
    conf_dict["out_dir"] = "output/Mask_RCNN_R50_test"
    conf_dict["load"] = ""

    # call main
    main(conf_dict)