# based on https://github.com/plusgood-steven/ID-Blau/blob/main/dataloader.py

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from time import sleep
from torchvision import transforms
from glob import glob
import random
from PIL import Image


def get_img(path, max_retries=5, delay=1):
    for attempt in range(max_retries):
        try:
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return np.array(img, dtype=np.float32)
        except BrokenPipeError:
            print(f"[{attempt + 1}/{max_retries}] BrokenPipeError")
            sleep(delay)
            delay *= 2  # exponential backoff
    raise RuntimeError(f"Failed to load {path} after {max_retries} retries due to BrokenPipeError")


def get_npy(path, max_retries=5, delay=1):
    for attempt in range(max_retries):
        try:
            img = np.load(path).transpose(1, 2, 0)
            return img.astype(np.float32)
        except BrokenPipeError:
            print(f"[{attempt + 1}/{max_retries}] BrokenPipeError")
            sleep(delay)
            delay *= 2  # exponential backoff
    raise RuntimeError(f"Failed to load {path} after {max_retries} retries due to BrokenPipeError")



def rotation_matrix(angle):
    rad = np.radians(angle)

    cos_theta = np.cos(rad)
    sin_theta = np.sin(rad)
    rot_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    return rot_matrix


class RandomRotate90(object):
    def __call__(self, data):
        dirct = random.randint(0, 3)
        for key in data.keys():
            if key != "flow":
                data[key] = np.rot90(data[key], dirct).copy()
            else:
                vectors = data[key][:, :, :2].copy()

                vectors_origin_shape = vectors.shape
                vectors = vectors.reshape((-1, 2))
                rot_matrix = rotation_matrix(90 * dirct)

                rotated_vectors = (rot_matrix @ vectors.T).T
                rotated_vectors = rotated_vectors.reshape(vectors_origin_shape)

                data[key][:, :, :2] = rotated_vectors

        return data


class RandomFlip(object):
    def __call__(self, data):
        if random.randint(0, 1) == 1:
            for key in data.keys():
                if key != "flow":
                    data[key] = np.fliplr(data[key]).copy()
                else:
                    data[key][:, :, 0] = -data[key][:, :, 0]
        if random.randint(0, 1) == 1:
            for key in data.keys():
                if key != "flow":
                    data[key] = np.flipud(data[key]).copy()
                else:
                    data[key][:, :, 1] = -data[key][:, :, 1]
        return data


"""
class RandomRotate90(object):
    def __call__(self, data):
        dirct = random.randint(0, 3)
        for key in data.keys():
            data[key] = np.rot90(data[key], dirct).copy()
            if key == "flow":
                u_v = data[key][:, :, :2].copy()
                if dirct == 1:
                    data[key][:, :, 0] = -u_v[:, :, 1]
                    data[key][:, :, 1] = u_v[:, :, 0]
                elif dirct == 2:
                    data[key][:, :, 0] = -u_v[:, :, 0]
                    data[key][:, :, 1] = -u_v[:, :, 1]
                elif dirct == 3:
                    data[key][:, :, 0] = u_v[:, :, 1]
                    data[key][:, :, 1] = -u_v[:, :, 0]

        return data


class RandomFlip(object):
    def __call__(self, data):
        if random.randint(0, 1) == 1:
            for key in data.keys():
                data[key] = np.fliplr(data[key]).copy()
                if key == "flow":
                    data[key][:, :, 0] = -data[key][:, :, 0]

        if random.randint(0, 1) == 1:
            for key in data.keys():
                data[key] = np.flipud(data[key]).copy()
                if key == "flow":
                    data[key][:, :, 1] = -data[key][:, :, 1]
        return data
"""


class RandomCrop(object):
    def __init__(self, Hsize, Wsize):
        super(RandomCrop, self).__init__()
        self.size = Hsize, Wsize

    def __call__(self, data):
        H, W, _ = np.shape(list(data.values())[0])
        h, w = self.size
        top = random.randint(0, H - h)
        left = random.randint(0, W - w)
        for key in data.keys():
            data[key] = data[key][top : top + h, left : left + w].copy()
        return data


class Normalize(object):
    def __init__(self):
        super(Normalize, self).__init__()

    def __call__(self, data):
        data["sharp"] = ((data["sharp"] / 255) * 2 - 1.0).copy()
        data["blur"] = ((data["blur"] / 255) * 2 - 1.0).copy()

        magnitude = data["flow"][:, :, 2] / 147
        magnitude[magnitude > 1] = 1
        data["flow"][:, :, 2] = magnitude
        
        data["sharp"] = data["sharp"].transpose(2, 0, 1)
        data["blur"] = data["blur"].transpose(2, 0, 1)
        data["flow"] = data["flow"].transpose(2, 0, 1)

        return data


class GoProLoader(Dataset):
    def __init__(self, mode, patch_size=None, portion=1.0):
        self.data_path = os.path.abspath("../datasets/GoPro")
        self.flow_path = os.path.abspath("../datasets/GoPro_flow")
        
        blur_list = []
        sharp_list = []
        #flow_list = []

        if patch_size is not None:
            self.transform = transforms.Compose([
                RandomCrop(patch_size, patch_size),
                RandomFlip(),
                RandomRotate90(),
            ])
        else:
            self.transform = transforms.Compose([Normalize()])
            
        for video in sorted(os.listdir(os.path.join(self.data_path, mode))):
            sharp_video_path = os.path.join(self.data_path, mode, video, "sharp")
            blur_video_path = os.path.join(self.data_path, mode, video, "blur")
            #flow_video_path = os.path.join(self.flow_path, mode, video, "sharp_flow")

            # all frames from single video (e.g. datasets/GoPro_flow/test/GOPR0384_11_00)
            sharp_video_data_path = sorted(glob(os.path.join(sharp_video_path, "*.png")))
            blur_video_data_path = sorted(glob(os.path.join(blur_video_path, "*.png")))
            #flow_video_data_path = sorted(glob(os.path.join(flow_video_path, "*.npy")))

            # add frames from current video to lists of all frames
            sharp_list.extend(sharp_video_data_path)
            blur_list.extend(blur_video_data_path)
            #flow_list.extend(flow_video_data_path)
            
        assert len(sharp_list) == len(blur_list) #== len(flow_list)
        
        preferred_len = max(1, int(len(sharp_list) * portion))
        self.sharp_list = sharp_list[:preferred_len]
        self.blur_list = blur_list[:preferred_len]
        #self.flow_list = flow_list[:preferred_len]


    def __len__(self):
        return len(self.sharp_list)
    
    def __getitem__(self, idx):
        blur = get_img(self.blur_list[idx])
        sharp = get_img(self.sharp_list[idx])
        #flow = get_npy(self.flow_list[idx])
        
        sample = {
            "blur": blur,
            "sharp": sharp,
            #"flow": flow
        }
        sample = self.transform(sample)
        return sample
    

"""[TODO] make bidirectional OF dataset + finish this class

class RealBlurLoader(Dataset):
    def __init__(self, mode, patch_size=None, raw=False):
        if raw:
            self.data_path = os.path.abspath("../datasets/RealBlur/RealBlur-R_BM3D_ECC_IMCORR_centroid_itensity_ref")
            self.flow_path = os.path.abspath("../datasets/...")
        else:
            self.data_path = os.path.abspath("../datasets/RealBlur/RealBlur-J_ECC_IMCORR_centroid_itensity_ref")
            self.flow_path = os.path.abspath("../datasets/...")
        
        self.blur_list = []
        self.sharp_list = []
        self.flow_list = []

        if patch_size is not None:
            self.transform = transforms.Compose([
                RandomCrop(patch_size, patch_size),
                RandomFlip(),
                RandomRotate90(),
                Normalize(),
                ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([Normalize(), ToTensor()])

        if mode == 'train':
            _range = range(1, 183)
        elif mode == 'test':
            _range = range(183, 233)
        else:
            raise ValueError
            
        for video_idx in _range:
            video_name = f"scene{video_idx:3d}"
            video_path = os.path.join(self.data_path, video_name)
            flow_video_path = os.path.join(self.flow_path, video_name)

            sharp_video_data_path = sorted(glob(os.path.join(video_path, "gt_*.png")))
            blur_video_data_path = sorted(glob(os.path.join(video_path, "blur_*.png")))
            flow_video_data_path = sorted(glob(os.path.join(flow_video_path, "*.npy")))

            self.sharp_list.extend(sharp_video_data_path)
            self.blur_list.extend(blur_video_data_path)
            self.flow_list.extend(flow_video_data_path)

        assert len(self.sharp_list) == len(self.blur_list) == len(self.flow_list)

    def __len__(self):
        return len(self.flow_list)
    
    def __getitem__(self, idx):
        blur = get_img(self.blur_list[idx])
        sharp = get_img(self.sharp_list[idx])
        flow = np.load(self.flow_list[idx]).transpose(1, 2, 0).astype(np.float32)

        sample = {"blur": blur, "sharp": sharp, "flow": flow}
        sample = self.transform(sample)
        return sample

"""
