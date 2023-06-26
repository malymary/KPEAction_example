# dataloaders and datasets etc
import csv
import os.path
import json
import math

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt 


# dataset = KPEAction(file_path = "D:/......")


# ok SO this is gonna be a bitch to work on
class KPEAction(Dataset):
    '''
    Preliminary Experiments are based on Duan et al's Revisiting Skeleton Based Action Recognition.
    Heatmap volumes are denoted as K*H*W where K represents the number of joints.
    Coord triplets (x, y, c) are used to compose K gaussian maps centered at every joint.
    '''
    def __init__(self,
        file_path:str
    ):
        with open(file_path) as f:
            # DONT FUCKING TOUCH THIS OR EVERYTHIGN WILL BREAK
            self.data = json.load(f)
        self.ids = list(set([obj["id"] for obj in self.data])) # this is what we should be iterating based on
        self.lbls = [obj["action"] for obj in self.data]
        class_names = { '2':   'Attending',
                        '38':  'Drinking', 
                        '40':  'Eating' ,   
                        '45':  'Exploring',
                        '68':  'Keeping still', 
                        '100': 'Running',
                        '102': 'Sensing', 
                        '133': 'Walking' }
        self.heatmap_size = (56,56)

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        return self._make_heatmaps(idx), self.lbls[idx]
    
    def _get_sample_joints_id(self, idx):
        # should this be index based or sample id based
        frames_kps = [obj["joints"] for obj in self.data if obj["id"]==self.ids[idx]]
        label = self.lbls[idx]
        return frames_kps, label

    def _make_heatmaps(self, idx):
        # lets make a fucking ducking heatmap
        frames_kps, label = self._get_sample_joints_id(idx)
        sigma_sq = 1
        for frame in frames_kps:
            x_values = np.array([coord[0] for coord in frame if coord>[0,0]])
            y_values = np.array([coord[1] for coord in frame if coord>[0,0]])
            x_max, x_min = max(x_values), min(x_values)
            y_max, y_min = max(y_values), min(y_values)
            x_values = x_values - x_min
            y_values = y_values - y_min
            frame = zip(x_values, y_values)

            for x,y in frame: # x,y not cropped and scaled
                heatmap = np.zeros(self.heatmap_size)
                for i in range(self.heatmap_size[0]):
                    for j in range(self.heatmap_size[1]):
                        heatmap[i][j] = np.power(math.e, (-((i-x)**2 + (j-y)**2)/2*sigma_sq))
                plt.imshow(heatmap)
                plt.show()


def read_annotations(path, num):
    count = 0
    with open(path) as f:
        for line in f:
            print(line)
            if count == num: break
            else: count+=1

def read_trainval(
        path=r"D:\data\animal\AnimalKingdom\Animal_Kingdom\action_recognition\annotation\\",
        trainval="train"):
    ''''''
    path = path + trainval+".csv"
    # read the action recognition train and val csvs
    ids_lbls = []
    
    with open(path) as f:
        file = csv.reader(f)
        for row in file:
            ids_lbls.append((row[0].split(" ")[0], row[0].split(" ")[4]))
            
            
    return set(ids_lbls) # many many many duplicates in ids, and of course lbls

def read_kpe_file(
        path=r"D:\data\animal\AnimalKingdom\Animal_Kingdom\pose_estimation\annotation\ak_P3_mammal\\", 
        traintest="train"):
    path = path + traintest+".json"

    with open(path) as f:
        file = json.load(f)
    data = []
    exclude = ["Monkey", "Bird", "Amphibian", "Rodent", "Squirrel", "Fish", "Frog", "Hare", "Hedgehog", "Loris"]
    for obj in file:
        for animal in exclude:
            if animal in obj["animal_subclass"] or animal in obj["animal_class"]: 
                break
        else:
            data.append(obj)

    ids = set([obj["image"].split("/")[0]for obj in data])
    animals = set(obj["animal"] for obj in data)
    # print(animals)
    return ids, animals
   
def get_intersection(a, b):
    intersection = [x for x in a if x in b]
    print(f"IDs in a: {len(a)}")
    print(f"IDs in b: {len(b)}")
    print(f"IDs in intersection: {len(intersection)}")
    return intersection

def read_lbl_names(
        path = r"D:\data\animal\AnimalKingdom\Animal_Kingdom\action_recognition\annotation\df_action.xlsx"):
    df = pd.read_excel(path)

def make_save_ds(
        path=r"D:\data\animal\AnimalKingdom\Animal_Kingdom\pose_estimation\annotation\ak_P3_mammal\\", 
        traintest="train",
        filter=[], # the sample IDs we want
        id_action=set(), # id_action is passed from both_k
        save_path="data"): 
    path = path + traintest+".json"

    with open(path) as f:
        file = json.load(f)
    data = []
    for obj in file:
        id = obj["image"].split("/")[0]
        if id in filter:
            sample = {"id": id,
                      "action":     [x[1] for x in id_action if x[0]==id][0],
                      "joints_vis": obj["joints_vis"],
                      "joints":     obj["joints"],
                      "scale":      obj["scale"],
                      "center":     obj["center"],
                      "animal":     obj["animal"]}
            data.append(sample)
    with open(f'{save_path}.json', 'w') as f:
        json.dump(data, f, indent=4)

def run_analytics_ds_generation():
    ar_ids_lbls = read_trainval(trainval = "train") | read_trainval(trainval = "val")
    ar_ids = set([x[0] for x in ar_ids_lbls])
    kpe_ids = set(read_kpe_file(traintest="train")[0] | read_kpe_file(traintest="test")[0])
    both_kpe_action = ar_ids & kpe_ids

    print(f"Both KP and AR data: {len(both_kpe_action)}")

    videos = os.listdir(r"D:\data\animal\AnimalKingdom\video")
    video_ids = set([vid.split(".")[0] for vid in videos])

    kpe_ar_videos = both_kpe_action & video_ids
    print(f"KPE AR video data: {len(kpe_ar_videos)}")

    # make_save_ds(traintest="train", filter=kpe_ar_videos, id_action=ar_ids_lbls, save_path="kpe_ar1")
    # make_save_ds(traintest="test", filter=kpe_ar_videos, id_action=ar_ids_lbls, save_path="kpe_ar2")

    distribution = {}
    ids = []
    lbls = []
    ar_lbls = [x[1] for x in ar_ids_lbls]
    for id in kpe_ar_videos:
        lbl = [x[1] for x in ar_ids_lbls if x[0] == id][0]
        ids.append(id)
        lbls.append(lbl)
    for lbl in lbls:
        try:
            distribution[lbl] +=1
        except KeyError:
            distribution[lbl] = 1

    print(distribution)


if __name__ == "__main__":
    train_set = KPEAction("kpe_ar1.json")
    for idx in range(1):
        train_set._make_heatmaps(idx)
    print("All done!")