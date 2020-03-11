import os
import os.path
import numpy as np
from glob import glob
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import data_augmentation
from random import shuffle


def normalize(data):
    return data/255.


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endh = img.shape[1]
    endw = img.shape[2]
    patch = img[:, 0:endh-win+0+1:stride, 0:endw-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endh-win+i+1:stride, j:endw-win+j+1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def prepare_data(data_path, patch_size, stride, aug_times=1):
    # '''
    # train
    print('process training data')
    scales = [1]
    files = glob.glob(os.path.join('D:', 'NH-HAZE_train', 'HAZY', '*.png'))
    # mix = list(range(len(files)))
    # random.shuffle(mix)
    # mix_train = mix[:int(len(files)*0.96)]
    # mix_val = mix[int(len(files)*0.96):]
    files.sort()
    h5f = h5py.File('D:/train_input.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        Img = cv2.imread(files[i])
        h, w, c = Img.shape
        for k in range(len(scales)):
            # Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            # Img = np.expand_dims(Img[:, :, :].copy(), 0)
            Img = np.swapaxes(Img, 0, 2)
            Img = np.swapaxes(Img, 1, 2)
            Img = np.float32(normalize(Img))
            # print(Img.shape)
            patches = Im2Patch(Img, patch_size, stride)
            # print(i)
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], aug_times*patches.shape[3]))
            for n in range(patches.shape[3]):
                data = patches[:, :, :, n].copy()
                # print(data.shape)
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1, 8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    train_num += 1
    h5f.close()
    print('process training gt')
    scales = [1]
    files = glob.glob(os.path.join('D:', 'NH-HAZE_train', 'GT', '*.png'))
    files.sort()
    h5f = h5py.File('D:/train_gt.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        Img = cv2.imread(files[i])
        h, w, c = Img.shape
        for k in range(len(scales)):
            # Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            # Img = np.expand_dims(Img[:, :, :].copy(), 0)
            Img = np.swapaxes(Img, 0, 2)
            Img = np.swapaxes(Img, 1, 2)
            Img = np.float32(normalize(Img))
            patches = Im2Patch(Img, patch_size, stride)
            # print(i)
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], aug_times*patches.shape[3]))
            for n in range(patches.shape[3]):
                data = patches[:, :, :, n].copy()
                # print(data.shape)
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1, 8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    # files.clear()
    files = glob.glob(os.path.join('D:', 'NH-HAZE_validation', 'HAZY', '*.png'))
    files.sort()
    h5f = h5py.File('D:/val_input.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        # img = np.expand_dims(img[:, :, :], 0)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = np.float32(normalize(img))
        # print(i)
        # print(img.shape)
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    # '''
    print('\nprocess validation gt')
    # files.clear()
    files = glob.glob(os.path.join('D:', 'NH-HAZE_validation', 'GT', '*.png'))
    files.sort()
    h5f = h5py.File('D:/val_gt.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        # img = np.expand_dims(img[:, :, :], 0)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = np.float32(normalize(img))
        # print(i)
        # print(img.shape)
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    # print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)
    # '''


class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            h5f = []
            for im in range(7):
                h5 = h5py.File('/home/user/depthMap/ksm/CVPR/demoire/data/train_input' + str(im) + '.h5', 'r')
                h5f.append(h5)
            h5f_gt = h5py.File('/home/user/depthMap/ksm/CVPR/demoire/data/train_gt.h5', 'r')
        else:
            h5f = []
            for im in range(7):
                h5 = h5py.File('/home/user/depthMap/ksm/CVPR/demoire/data/val_input' + str(im) + '.h5', 'r')
                h5f.append(h5)
            h5f_gt = h5py.File('/home/user/depthMap/ksm/CVPR/demoire/data/val_gt.h5', 'r')
        self.keys = []
        for im in range(7):
            h5 = h5f[im]
            self.keys.append(list(h5.keys()))
            h5.close()
        self.keys_gt = list(h5f_gt.keys())
        h5f_gt.close()

    def __len__(self):
        return len(self.keys_gt)

    def __getitem__(self, index):
        if self.train:
            h5f = []
            for im in range(7):
                h5 = h5py.File('/home/user/depthMap/ksm/CVPR/demoire/data/train_input' + str(im) + '.h5', 'r')
                h5f.append(h5)
            h5f_gt = h5py.File('/home/user/depthMap/ksm/CVPR/demoire/data/train_gt.h5', 'r')
        else:
            h5f = []
            for im in range(7):
                h5 = h5py.File('/home/user/depthMap/ksm/CVPR/demoire/data/val_input' + str(im) + '.h5', 'r')
                h5f.append(h5)
            h5f_gt = h5py.File('/home/user/depthMap/ksm/CVPR/demoire/data/val_gt.h5', 'r')
        data = []
        for im in range(7):
            k = self.keys[im][index]
            h5 = h5f[im]
            kk = h5[k]
            data.append(torch.Tensor(np.array(kk)).unsqueeze(0))
            h5.close()
        key_gt = self.keys_gt[index]
        gt = np.array(h5f_gt[key_gt])
        h5f_gt.close()
        return torch.cat(data, 0), torch.Tensor(gt)


class DatasetBurst(udata.Dataset):
    def __init__(self, train=True):
        super(DatasetBurst, self).__init__()
        self.train = train
        if self.train:
            self.input_list = glob.glob("/home/user/depthMap/ksm/CVPR/demoire/data/train/input/*.png")
            self.gt_list = glob.glob("/home/user/depthMap/ksm/CVPR/demoire/data/train/gt/*.png")
        else:
            self.input_list = glob.glob("/home/user/depthMap/ksm/CVPR/demoire/data/val/input/*.png")
            self.gt_list = glob.glob("/home/user/depthMap/ksm/CVPR/demoire/data/val/gt/*.png")
        self.frame = int(len(self.input_list)/len(self.gt_list))
        self.crop = 128
        self.th = 50

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, index):
        order = list(range(len(self.gt_list)))
        shuffle(order)
        index = order[index]
        data_list = []
        self.input_list.sort(key=str.lower)
        self.gt_list.sort(key=str.lower)
        origin = cv2.imread(self.input_list[index * self.frame + 3])
        for im in range(self.frame):
            data = cv2.imread(self.input_list[index * self.frame + im])
            if im != 3:
                _, bin2 = cv2.threshold(data, self.th, 255, cv2.THRESH_BINARY)
                _, bin3 = cv2.threshold(data, self.th, 255, cv2.THRESH_BINARY_INV)
                final2 = cv2.bitwise_and(data, bin2, mask=None)
                final3 = cv2.bitwise_and(origin, bin3, mask=None)
                data = cv2.bitwise_or(final3, final2, mask=None)
            data = np.float32(normalize(data))
            data = np.transpose(data, (2, 0, 1))
            data = torch.Tensor(data).unsqueeze(0)
            data_list.append(data)
        gt = cv2.imread(self.gt_list[index])
        gt = np.float32(normalize(gt))
        gt = np.transpose(gt, (2, 0, 1))
        return torch.cat(data_list, 0), torch.Tensor(gt)
