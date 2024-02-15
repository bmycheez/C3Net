import glob
import torch
import natsort
import pytorch_lightning as pl

from PIL import Image
from torchvision.transforms import Compose, ToTensor


class CFAMoire(torch.utils.data.Dataset):
    def __init__(self,
                 root: str = 'D:/data/NTIRE2020/moire/train_burst/',
                 transform=None):
        super().__init__()

        self.root = {'input': None, 'gt': None}
        for key in self.root:
            self.root[key] = root + key
        self.n_input = len(glob.glob(f"{self.root['input']}/000000*.png"))
        self._get_image_list()

        self.transform = transform

    def _get_image_list(self):
        for key in self.root:
            if self.root[key] is not None:
                self.root[key] = \
                    natsort.natsorted(glob.glob(f"{self.root[key]}/*.png"))

    def _get_image(self,
                   key: str = 'input',
                   index: int = 0):
        image = Image.open(self.root[key][index]).convert('RGB')
        return self.transform(image)

    def __len__(self):
        return int(len(self.root['input']) / self.n_input)

    def __getitem__(self,
                    index: int = 0):
        path = self.root['input'][index]
        if 'burst' in path:
            input_image = []
            for i in range(self.n_input):
                input_image.append(
                    self._get_image(index=index * self.n_input + i))
        elif 'single' in path:
            input_image = [self._get_image(index=index)]
        else:
            raise NotImplementedError

        if 'train' in path:
            gt_image = [self._get_image(key='gt', index=index)]
            return input_image, gt_image
        elif 'val' in path or 'test' in path:
            return input_image
        else:
            raise NotImplementedError


class LightningDataset(pl.LightningDataModule):
    def __init__(self,
                 root: dict[str] = {'train': None, 'val': None, 'test': None},
                 batch_size: dict[int] = {'train': 1, 'val': 1, 'test': 1},
                 num_workers: dict[int] = {'train': 4, 'val': 4, 'test': 4},
                 shuffle: dict[bool] =
                 {'train': False, 'val': False, 'test': False},
                 drop_last: dict[bool] =
                 {'train': False, 'val': False, 'test': False}
                 ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last

    def prepare_data(self):
        pass

    def setup(self,
              stage: str = None):
        if 'moire' in self.root['train']:
            self.transform = Compose([ToTensor()])
            self.ds_train = CFAMoire(root=self.root['train'],
                                     transform=Compose([ToTensor()]))
            self.ds_val = CFAMoire(root=self.root['val'],
                                   transform=Compose([ToTensor()]))
            self.ds_test = CFAMoire(root=self.root['test'],
                                    transform=Compose([ToTensor()]))
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_train,
            batch_size=self.batch_size['train'],
            num_workers=self.num_workers['train'],
            shuffle=self.shuffle['train'],
            drop_last=self.drop_last['train']
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_val,
            batch_size=self.batch_size['val'],
            num_workers=self.num_workers['val'],
            shuffle=self.shuffle['val'],
            drop_last=self.drop_last['val']
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_test,
            batch_size=self.batch_size['test'],
            num_workers=self.num_workers['test'],
            shuffle=self.shuffle['test'],
            drop_last=self.drop_last['test']
        )


if __name__ == '__main__':
    # Dataset
    train_dataset = CFAMoire(root='D:/data/NTIRE2020/moire/train_burst/')
    val_dataset = CFAMoire(root='D:/data/NTIRE2020/moire/val_single/')
    test_dataset = CFAMoire(root='D:/data/NTIRE2020/moire/test_burst/')
    print(len(train_dataset.root['input']))
    print(train_dataset.__len__(), test_dataset.__len__())

    # DataLoader
    dm = LightningDataset(
        root={'train': 'D:/data/NTIRE2020/moire/train_burst/',
              'val': 'D:/data/NTIRE2020/moire/val_burst/',
              'test': 'D:/data/NTIRE2020/moire/test_burst/'}
        )
    dm.setup()

    train_dataloader = dm.train_dataloader()
    print(len(train_dataloader))
    for i, (noisy, denoised) in enumerate(train_dataloader):
        print(len(noisy), noisy[3].size(), torch.mean(noisy[3]))
        print(len(denoised), denoised[0].size(), torch.mean(denoised[0]))
        break

    val_dataloader = dm.val_dataloader()
    print(len(val_dataloader))
    for i, (noisy) in enumerate(val_dataloader):
        print(len(noisy), noisy[0].size(), torch.mean(noisy[0]))
        break
