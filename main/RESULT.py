import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision
import os
import cfg
from tqdm import tqdm

def pred(loader, model, return_file_names=cfg.return_file_name):
    model.eval()
    res = []
    with torch.no_grad():
        for x in tqdm(loader, desc="model prediction"):
            pred = model(x)[0]
            pred = [1 if i == max(pred) else 0 for i in pred]
            pred = "floor" if pred==[1,0,0] else "ship" if pred==[0,1,0] else "plane"
            res.append(pred)
    if return_file_names:
        df = pd.DataFrame(data=[res, file_names], columns=["Class", "file_names"])
    else:
        df = pd.DataFrame(data=res, columns=["Class"])
    df.to_csv(cfg.output_path+cfg.output_name, index="Id")
    

def get_data():
    img_list = os.listdir(cfg.input_path)
    img_list = sorted(img_list, key=lambda x: int(x[:-4]))

    input_list = []
    for img_name in tqdm(img_list, desc="loading images"):
        input_list.append([np.asarray(Image.open(cfg.input_path + img_name)) for i in range(3)])

    input_list = np.array(input_list)

    global loader
    tens_input = torch.FloatTensor(input_list)/255
    tens_input.to(device)
    loader = DataLoader(tens_input)


if cfg.device == "none":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = cfg.device

model = cfg.model(num_classes=3)
model.load_state_dict(torch.load(cfg.model_wigth,map_location=torch.device('cpu')))
model.eval()

get_data()
pred(loader, model)