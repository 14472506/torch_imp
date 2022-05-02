# run this first

# =======================================================================================
# Data loader
# =======================================================================================
import os
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO

class PennFudanDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

    
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        #np.savetxt('mask.csv', masks[i])

        return img, target

    def __len__(self):
        return len(self.imgs)

class COCOLoader(torch.utils.data.Dataset):
    """
    Custom coco data loader form pytorch
    """
    def __init__(self, root, json, image_transforms=None, target_transforms=None):
        """
        Details of init
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.imgs.keys())
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms

    def __getitem__(self, idx):
        """
        Detail Get item
        """
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds = img_id)
        
        # generating target
        anns = self.coco.loadAnns(ann_ids)

        labels = []
        boxes = []        
        masks_list = []
        areas = []
        iscrowds = []
        
        for ann in anns:
            
            labels.append(ann['category_id'])
            areas.append(ann['area'])

            bbox = ann['bbox']            
            new_bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            boxes.append(new_bbox)
    
            if ann["iscrowd"]:
                iscrowds.append(1)
            else:
                iscrowds.append(0)

            mask = self.coco.annToMask(ann)
            mask == ann['category_id']
            masks_list.append(torch.from_numpy(mask))

            #pos = np.where(mask == True)
            #xmin = np.min(pos[1])
            #xmax = np.max(pos[1])
            #ymin = np.min(pos[0])
            #ymax = np.max(pos[0])
            #boxes.append([xmin, ymin, xmax, ymax])            

        # to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(areas, dtype=torch.int64)
        masks = torch.stack(masks_list, 0)
        iscrowd = torch.as_tensor(iscrowds, dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # loading image
        image_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, image_path)).convert('RGB')
        im_conv = T.ToTensor()
        img = im_conv(img)

        # applying transforms
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return img, target

    def __len__(self):
        return len(self.ids)

## =======================================================================================
## Defining model --- [Uncomment to use]
## =======================================================================================
#import torchvision
#from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
#
## load pre trained coco model
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#
## modifiy the parameters of the model for fine tuning
## num classes is 2, person + background
#num_classes = 2
## get number of input features from classifier
#in_features = model.roi_heads.box_predictor.cls_score.in_features
## replace pre-trained heads with a new one
#model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

## =======================================================================================
## Modifying the model to add different backbone --- [Uncomment to use]
## =======================================================================================
#import torchvision
#from torchvision.models.detection import FasterRCNN
#from torchvision.models.detection.rpn import AnchorGenerator
#
## loading the pre-trained model for classification and return only the features
#backbone = torchvision.models.mobilenet_v2(pretrained=True).features
#
## FastRCNN requires the number of output channels in the backbone for mobilenet_v2 its
## 1280 so this must be defined
#backbone.out_channels = 1280
#
## The RPN is going to generate 5 x 3 anchors per spatial location with 5 different sizes and
## 3 different aspects rations. We have a [Tuple[Tuple[int]]] because each feature map could
## potentially have different sizes and aspect ratios
#anchor_generator = AnchorGenerator(size=((32, 64, 128, 256, 512),),
#                                   aspect_ratio=((0.5, 1.0, 2.0),))
#
## defining the region of the feature map that is used to perform ROI cropping, as well as
## the size of the crop after rescalling. if the backbone returns a tensor, featuremap_names
## expected to be [0]. more generally backbone should return and OrderedDict[Tensor], and 
## in featuremap_names you can choose which feature map to use.
#roi_pooler = torchvision.ops.MultiScaleRoIAlign(feature_names=['0'],
#                                                output_size=7,
#                                                sample_ratio=2)
#
## putting the pieces together inside a FasterRCNN model
#model = FasterRCNN(backbone,
#                   num_classes=2,
#                   rpn_anchor_generator=anchor_generator,
#                   box_roi_pool=roi_pooler)

# =======================================================================================
# Instance segmentation for PennFudan dataset
# =======================================================================================
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

# =======================================================================================
# For augmentation
# =======================================================================================
from torchvision import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# =======================================================================================
# The main function
# =======================================================================================
from torchvision_files.engine import train_one_epoch, evaluate
import utils
from tqdm import tqdm
from matplotlib import pyplot as plt

def main():
    # This line should be ran first to ensure a gpu is being used if possible
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 2 classes, 1 is the background
    num_classes = 2 
    
    # use the loaded dataset and defined transofrmations provided
    #dataset = PennFudanDataset("data/PennFudanPed", get_transform(train=True))
    #ataset_test = PennFudanDataset("data/PennFudanPed", get_transform(train=False))

    dataset = COCOLoader("data/jersey_royal_ds/train", "data/jersey_royal_ds/train/train.json")
    dataset_test = COCOLoader("data/jersey_royal_ds/val", "data/jersey_royal_ds/val/val.json")


    #im, targ = dataset.__getitem__(int(0))
    # split the dataset in train and test set
    #indices = torch.randperm(len(dataset)).tolist()
    #dataset = torch.utils.data.Subset(dataset, indices[:-50])
    #dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    #for epoch in range(num_epochs):
    #    # train for one epoch, printing every 10 iterations
    #    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    #    # update the learning rate
    #    lr_scheduler.step()
    #    # evaluate on the test dataset
    #    evaluate(model, data_loader_test, device=device)

    # training loops
    loss_list = []
    n_epochs = 10
    model.train()
    for epochs in tqdm(range(n_epochs)):
        loss_epoch = []
        iter = 1
        
        for images, targets in tqdm(data_loader):
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            model = model.float()
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            loss_epoch.append(losses.item())

            # Plotting every 10th iteration
            #plt.plot(list(range(iter)), loss_epoch)
            #plt.xlabel('Iteration')
            #plt.ylabel('Loss')
            #plt.show()
            #plt.close()
            iter += 1
        
        loss_epoch_mean = np.mean(loss_epoch)
        loss_list.append(loss_epoch_mean)
        print("Average loss for epoch = {:.4f}".format(loss_epoch_mean))
    
        #https://bjornkhansen95.medium.com/mask-r-cnn-for-segmentation-using-pytorch-8bbfa8511883

    print("That's it!")
    
if __name__ == "__main__":
    main()