import torch
import torchvision.transforms as T
import torch.utils.data as data
import os 
from pycocotools.coco import COCO
from PIL import Image

class COCOLoader(data.Dataset):
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
        
        # generating target/s
        target = self.coco.loadAnns(ann_ids)

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
        print(len(self.ids))
        
#if __name__ == "__main__":
#    
#    root_dir = "data/jersey_royal_ds/val"
#    json_root = "data/jersey_royal_ds/val/val.json"
#
#    loader = COCOLoader(root_dir, json_root)
#
#    img, target = loader.__getitem__(1)
#
#    print(target)
