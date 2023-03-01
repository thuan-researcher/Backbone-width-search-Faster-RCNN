import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from data import myDataset
from torch.utils.data import DataLoader
from utils import *
import tqdm
from util.engine import evaluate
from util.utils import make_divisible
from mean_average_precision import MetricBuilder
import numpy as np
import os
from datetime import datetime
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import json


num_classes = 5

width_mult = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
for w in range(len(width_mult)):
    timestr = datetime.now().strftime("%Y%m%d-%H%M%S%f")
    model_save_dir = timestr
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    backbone = torchvision.models.restnet50(width_mult=width_mult[w]).features

    backbone.out_channels = make_divisible(1280*width_mult[w], 8)
    print(backbone)


    original_sizes = [32, 64, 128, 256, 512]
    sizes = [make_divisible(i*width_mult[w], 8) for i in original_sizes]
    anchor_generator = AnchorGenerator(sizes=(sizes,),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone,
                    num_classes=num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)

    # # load a model pre-trained on COCO
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # # replace the classifier with a new one, that has
    # # num_classes which is user-defined
    # num_classes = 5  # 1 class (person) + background
    # # get number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # replace the pre-trained head with a new one
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_dataset = myDataset('./dataset/trainTIDI', './dataset/trainTIDI.json')
    valid_dataset = myDataset('./dataset/testTIDI', './dataset/testTIDI.json')
    train_data_loader = DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=4,collate_fn=collate_fn)
    valid_data_loader = DataLoader(valid_dataset,batch_size=1,shuffle=False,num_workers=4,collate_fn=collate_fn)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    num_epochs =  20

    loss_hist = Averager()
    itr = 1
    lossHistoryiter = []
    lossHistoryepoch = []
    loss_log = {'epoch': [], 'loss_classifier': [], 'loss_box_reg': [], 'loss_objectness': [],'loss_rpn_box_reg': [], 'total_loss': [], 'mAP_50': [], 'mAP_75': [], 'mAP_COCO': []}
    import time
    start = time.time()

    for epoch in range(num_epochs):
        loss_hist.reset()
        loss_classifier = 0
        loss_box_reg = 0
        loss_objectness = 0
        loss_rpn_box_reg = 0

    ##################################################TRAINING################################   
        for images, targets in tqdm.tqdm(train_data_loader):
            model.train()
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)  
            loss_classifier += loss_dict['loss_classifier'].item()
            loss_box_reg += loss_dict['loss_box_reg'].item()
            loss_objectness += loss_dict['loss_objectness'].item()
            loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
            
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)
            lossHistoryiter.append(loss_value)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        loss_log['epoch'].append(epoch)
        loss_log['loss_classifier'].append(loss_classifier/len(train_data_loader))
        loss_log['loss_box_reg'].append(loss_box_reg/len(train_data_loader))
        loss_log['loss_objectness'].append(loss_objectness/len(train_data_loader))
        loss_log['loss_rpn_box_reg'].append(loss_rpn_box_reg/len(train_data_loader))
        loss_log['total_loss'].append(loss_hist.value)
        
        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()
        lossHistoryepoch.append(loss_hist.value)
        print(f"Epoch #{epoch} loss: {loss_hist.value}")
        
    ############################################VALIDATION################################
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode = True, num_classes = 5)
        for images, targets in tqdm.tqdm(valid_data_loader):
            model.eval()
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

            boxes = outputs[0]['boxes'].cpu().detach().numpy().astype(np.int32)
            img = images[0].permute(1,2,0).cpu().detach().numpy()
            labels= outputs[0]['labels'].cpu().detach().numpy().astype(np.int32)
            score = outputs[0]['scores'].cpu().detach().numpy()

            preds = np.insert(boxes, 4, labels, axis=1)
            preds = np.insert(preds, 5, score, axis=1)

            gt = np.insert(targets[0]['boxes'].cpu().detach().numpy(), 4, targets[0]['labels'].cpu().detach().numpy(), axis=1)
            gt = np.insert(gt, 5, np.zeros_like(targets[0]['labels'].cpu().detach().numpy()), axis=1)
            gt = np.insert(gt, 6, np.zeros_like(targets[0]['labels'].cpu().detach().numpy()), axis=1)

            metric_fn.add(preds, gt)

        mAP_50 = metric_fn.value(iou_thresholds = 0.5)['mAP']
        mAP_75 = metric_fn.value(iou_thresholds = 0.75)['mAP']
        mAP_COCO = metric_fn.value(iou_thresholds = np.arange(0.5, 0.95, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
        print("Validation mAP at epoch " + str(epoch) + " is " + str(mAP_50))

        loss_log['mAP_50'].append(float(mAP_50))
        loss_log['mAP_75'].append(float(mAP_75))
        loss_log['mAP_COCO'].append(float(mAP_COCO))

        torch.save(model, model_save_dir + 
            "/model-{}-epoch-{}-mAP-{}.pth".format(width_mult[w], epoch, mAP_50))

        with open(model_save_dir + "/lossHistory.json", "w") as outfile:
            json.dump(loss_log, outfile)
    ######################################################################################################3
            
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time taken to Train the model :{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
