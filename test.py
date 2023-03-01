from data import myDataset
from torch.utils.data import DataLoader
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from util.utils import make_divisible

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def label_to_name(id):
    id = int(id)
    if id == 0:
        return "creeping"
    if id == 1:
        return "crawling"
    if id == 2:
        return "stooping"
    if id == 3:
        return "climbing"
    if id == 4:
        return "other"
    else:
        return str(id)
    
def collate_fn(batch):
    return tuple(zip(*batch))


test_dataset = myDataset('./dataset/testTIDI', './dataset/testTIDI.json')
test_data_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=4,collate_fn=collate_fn)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 5  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model=torch.load('/media/hsh/Workspace/Nguyen_Duc_Thuan/Project Mang neuron/myProject/0.5-20230301-043838602099/model-0.5-epoch-11-mAP-0.45235079526901245.pth')
model.eval()
model.to(device)
idx = 0
for images, targets in tqdm.tqdm(test_data_loader):
    images = list(img.to(device) for img in images)

    outputs = model(images)
    outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

    boxes = outputs[0]['boxes'].cpu().detach().numpy().astype(np.int32)
    img = images[0].permute(1,2,0).cpu().detach().numpy()
    labels= outputs[0]['labels'].cpu().detach().numpy().astype(np.int32)
    score = outputs[0]['scores']

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    img = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB)
    for i in range(len(boxes)):
        if score[i] > 0.5:
            img = cv2.rectangle(img,(boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3]),(0,255,0),1)
            #print(le.inverse_transform([labels[i]-1])[0])
            #print(label_to_name(labels[i]), (boxes[i][0]+paddingSize,boxes[i][1]+paddingSize),(boxes[i][2]+paddingSize,boxes[i][3]+paddingSize))
            img = cv2.putText(img, label_to_name(labels[i]), (int(boxes[i][0]), int(boxes[i][1])), cv2.FONT_HERSHEY_DUPLEX, 0.5, color = (0,255,0))
    idx += 1
    ax.set_axis_off()
    ax.imshow(img)
    plt.savefig("./img/" + str(idx) + ".jpg", bbox_inches='tight')