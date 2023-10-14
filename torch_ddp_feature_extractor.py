#!/usr/bin/env python
import sys
import os
import cv2
import models
import skimage.transform

import json
import numpy as np
import torch.nn as nn
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


class YFCC100m(Dataset):
    def __init__(self, image_dir, path, transform=None):
        self.transform = transform
        self.image = image_dir
        self.metadata = self.read_json(path)["images"]
        
    def read_json(self, path):
        with open(path, 'rb') as file:
            data=json.load(file)
            file.close()
        return data
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        try:
            indices = np.array([index]).astype("int")
            image_hash = self.metadata[index]["id"]
            image_path = "../" + self.metadata[index]["file_path"]
            image = Image.open(image_path).convert("RGB")
            transform_image = self.transform(image).unsqueeze(0)
            
        except Exception as exp:
            print(exp)
            return None

        return indices, transform_image
    

class MyCollate:
    
    def __call__(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        idx, att_feats = zip(*batch)
        
        indices = np.stack(idx, axis=0).reshape(-1)
        image_feats = torch.stack(att_feats, 0)  # [B, 3, 384, 384]
        
        return indices, image_feats
    
def fetch_hashes(index, fn):
    f = index.tolist()
    g = []
    for i in f:
        name = fn.metadata[i]["id"]
        g.append(name)
    return g

    
class ImageFeatures(nn.Module):
    def __init__(self, layer_num):
        super(ImageFeatures, self).__init__()
        model = torch.hub.load('facebookresearch/WSL-Images', "resnext101_32x48d_wsl")
        self.ig_resnext = torch.nn.Sequential(*list(model.children())[:layer_num])

    def forward(self, img):
        img = img.to("cuda")
        with torch.no_grad():
            x = self.ig_resnext(img)
        return x


def feature_extractor_ddp(rank, world_size):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    # create default process group
    dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)
    group = dist.new_group(list(range(world_size)))
    
    cuda_device = rank

    batch_size = 2
    num_workers = 8
    world_size = dist.get_world_size()
    
    train_transform =  transforms.Compose([transforms.Resize((256)),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    dataset = YFCC100m(image_dir="../ParlAI/images/train_images",
                      path="data/personcap_added1.json",
                      transform=train_transform)
    
    train_sampler = DistributedSampler(dataset=dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              collate_fn=MyCollate(),
                              shuffle=False,
                              pin_memory=True,
                              sampler=train_sampler,
                              num_workers=num_workers)
    
    mean_feats = ImageFeatures(-1).to(cuda_device)
    spatial_feats = ImageFeatures(-2).to(cuda_device)
    
    mean_model = DDP(mean_feats, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    spatial_model = DDP(spatial_feats, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    
    for i, (idx, img) in enumerate(train_loader):
        img = img.to(cuda_device, non_blocking=True)
        ofc_feats, oatt_feats = mean_model(img), spatial_model(img)
        images_hashes = fetch_hashes(idx, dataset)
        fc_feats = ofc_feats.detach().cpu().numpy()
        att_feats = oatt_feats.detach().cpu().numpy()
        
        for i, hashes in enumerate(images_hashes):
            with open(f"data/yfcc_images/resnext101_32x48d_wsl/{hashes}.npy", "wb") as f:
                np.save(f, fc_feats[i])
            with open(f"data/yfcc_images/resnext101_32x48d_wsl_spatial_att/{hashes}.npy", "wb") as f:
                np.save(f, att_feats[i])

        print_freq = 5000
        if rank == 0 and i % print_freq == 0:  # print only for rank 0
            print(f"Completed Inference on batch {i} for GPU 0")
        if rank == 1 and i % print_freq == 0:  # print only for rank 1
            print(f"Completed Inference on batch {i} for GPU 1")
        if rank == 2 and i % print_freq == 0:  # print only for rank 2
            print(f"Completed Inference on batch {i} for GPU 2")
        if rank == 3 and i % print_freq == 0:  # print only for rank 3
            print(f"Completed Inference on batch {i} for GPU 3")

                
def run_ddp(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size, ), nprocs=world_size, join=True)
    # dist.destroy_process_group()
    
    
if __name__ == "__main__":

    world_size = 4  # number of gpus to parallize over
    run_ddp(feature_extractor_ddp, world_size)
