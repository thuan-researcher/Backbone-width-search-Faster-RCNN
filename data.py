from torch.utils.data import Dataset
import pydicom
import torch
from pycocotools.coco import COCO
from PIL import Image
from torchvision.transforms import ToTensor, Compose, PILToTensor
import os

class myDataset(Dataset): #Class to load Training Data
    def __init__(self, root, json, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.transform = Compose([PILToTensor()])

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        ann_id = self.ids[index]
        bbox = [i for i in coco.anns[ann_id]['bbox']]
        area = [bbox[2]*bbox[3]]
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        bbox = [bbox]
        img_id = coco.anns[ann_id]['image_id']
        label = [coco.anns[ann_id]['category_id']-1]
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path))
        image = ToTensor()(image)

        bbox = torch.as_tensor(bbox, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.int64)
        img_id = torch.as_tensor(img_id, dtype=torch.int64)
        area = torch.as_tensor(area)

        target = {"labels": label, "boxes": bbox, "image_id": img_id, "area": area, "iscrowd": torch.Tensor(1)}

        return image, target

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, batch):
        return tuple(zip(*batch))