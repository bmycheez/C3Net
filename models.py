import torch
import numpy as np

from torch.nn import (Module, Sequential, Conv2d, PReLU, ReLU, Sigmoid,
                      AdaptiveAvgPool2d, PixelShuffle)
from lightning import LightningModule, Trainer
from kornia.color import rgb_to_yuv
from skimage.metrics import peak_signal_noise_ratio

# https://github.com/VainF/pytorch-msssim
from pytorch_msssim import SSIM, MS_SSIM

from datasets import LightningDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# GPU 11GB
single_model_config = {
    'num_frames': 1,
    'features': 64,
    'kernel_size': 3,
    'num_global': 22,
    'num_residual': 2,
    'num_attention': 1,
    'scale_residual': 0.2,
}

# GPU 8GB
burst_model_config = {
    'num_frames': 7,
    'features': 48,
    'kernel_size': 3,
    'num_global': 5,
    'num_residual': 2,
    'num_attention': 1,
    'scale_residual': 1.0,
}


class MyLightning(LightningModule):
    def __init__(self,
                 model_mode: str = 'burst',
                 loss_mode: str = 'l1+color'):
        super().__init__()
        self.to(device)

        self.model_mode = model_mode
        if self.model_mode == 'single':
            self.model = C3Net(config=single_model_config).to(device)
        elif self.model_mode == 'burst':
            self.model = C3Net(config=burst_model_config).to(device)
        else:
            raise NotImplementedError
        # print(next(self.model.parameters()).is_cuda)

        self.loss_mode = loss_mode

        self.validation_psnr = []
        self.validation_ssim = []

    def training_step(self, batch, batch_idx):
        noisy, clean = batch

        if self.model_mode == 'single':
            inp = noisy[0]
        elif self.model_mode == 'burst':
            inp = torch.stack(noisy, dim=1)
        else:
            raise NotImplementedError
        outp = self.model(inp)
        gt = clean[0]

        if self.loss_mode == 'l1+color':
            loss = torch.nn.functional.l1_loss(outp, gt) + \
                self.color_loss(outp, gt)
            loss /= 2
        elif self.loss_mode == 'l1+ms_ssim':
            loss = torch.nn.functional.l1_loss(outp, gt) + \
                self.ms_ssim_loss(outp, gt)
            loss /= 2
        elif self.loss_mode == 'l1+ssim':
            loss = torch.nn.functional.l1_loss(outp, gt) + \
                self.ssim_loss(outp, gt)
            loss /= 2
        elif self.loss_mode == 'l1':
            loss = torch.nn.functional.l1_loss(outp, gt)
        elif self.loss_mode == 'l2':
            loss = torch.nn.functional.l2_loss(outp, gt)
        else:
            raise NotImplementedError

        self.log('train_loss', loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.validation_psnr.append(peak_signal_noise_ratio(outp, gt))
        # self.validation_ssim.append(ssim(outp, gt, 1., size_average=False))

        return loss

    def color_loss(self, outp, clean):
        out_yuv = rgb_to_yuv(outp)
        out_u = out_yuv[:, 1, :, :]
        out_v = out_yuv[:, 2, :, :]
        target_yuv = rgb_to_yuv(clean)
        target_u = target_yuv[:, 1, :, :]
        target_v = target_yuv[:, 2, :, :]

        return torch.div(
            torch.mean((out_u - target_u).pow(1)).abs() +
            torch.mean((out_v - target_v).pow(1)).abs(), 2)

    def ms_ssim_loss(self, outp, clean):
        ms_ssim_module = MS_SSIM(data_range=1., size_average=True, channel=3)
        return 1 - ms_ssim_module(outp, clean)

    def ssim_loss(self, outp, clean):
        ssim_module = SSIM(data_range=1., size_average=True, channel=3)
        return 1 - ssim_module(outp, clean)

    def on_training_epoch_end(self):
        pass
        # mean_psnr = sum(self.validation_psnr) / len(self.validation_psnr)
        # mean_ssim = sum(self.validation_psnr) / len(self.validation_psnr)
        # self.log('psnr', mean_psnr,
        #          on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('ssim', mean_ssim,
        #          on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def psnr(self, outp, clean, data_range=1.):
        outp = outp.data.cpu().numpy().astype(np.float32)
        clean = clean.data.cpu().numpy().astype(np.float32)
        PSNR = 0
        for i in range(outp.shape[0]):
            PSNR += peak_signal_noise_ratio(
                clean[i, :, :, :], outp[i, :, :, :], data_range=data_range)
        return (PSNR/outp.shape[0])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=10, gamma=0.5)
        return [optimizer], [scheduler]


class RB(Module):
    def __init__(self,
                 features: int = 48,
                 kernel_size: int = 3,
                 reduction: int = 16):
        super(RB, self).__init__()

        self.features = features
        self.kernel_size = kernel_size
        self.reduction = reduction

        self.seq_residual = self._residual()
        self.ca = self._channel_attention()

    def _residual(self):
        return Sequential(
            Conv2d(
                in_channels=self.features,
                out_channels=self.features,
                kernel_size=self.kernel_size,
                padding=self.kernel_size//2,
                stride=1,
                bias=True),
            PReLU(),
            Conv2d(
                in_channels=self.features,
                out_channels=self.features,
                kernel_size=self.kernel_size,
                padding=self.kernel_size//2,
                stride=1,
                bias=True)
        )

    def _channel_attention(self):
        return Sequential(
            AdaptiveAvgPool2d(1),
            Conv2d(in_channels=self.features,
                   out_channels=self.features // self.reduction,
                   kernel_size=1,
                   padding=0,
                   stride=1,
                   bias=True),
            ReLU(inplace=True),
            Conv2d(in_channels=self.features // self.reduction,
                   out_channels=self.features,
                   kernel_size=1,
                   padding=0,
                   stride=1,
                   bias=True),
            Sigmoid()
        )

    def forward(self, x):
        out = self.seq_residual(x)
        out *= self.ca(out)
        out += x
        return out


class AB(Module):
    def __init__(self,
                 features: int = 48,
                 num_residual: int = 2):
        super(AB, self).__init__()
        self.features = features
        self.num_residual = num_residual

        self.seq_residual1b = self._make_layer(RB, self.features,
                                               self.num_residual)
        self.down12 = self._make_layer(self._down, self.features, 1)
        self.seq_residual2b = self._make_layer(RB, self.features * 2,
                                               self.num_residual)
        self.down23 = self._make_layer(self._down, self.features * 2, 1)
        self.seq_residual3 = self._make_layer(RB, self.features * 4,
                                              self.num_residual)
        self.up32 = self._make_layer(self._up, self.features * 8, 1)
        self.seq_residual2a = self._make_layer(RB, self.features * 4,
                                               self.num_residual)
        self.up21 = self._make_layer(self._up, self.features * 4, 1)
        self.seq_residual1a = self._make_layer(RB, self.features * 2,
                                               self.num_residual)

        self.conv_m = Conv2d(in_channels=self.features * 2,
                             out_channels=self.features,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.relu_m = PReLU()

    def _make_layer(self,
                    block: Module,
                    features: int,
                    num: int):
        layers = []
        for _ in range(num):
            layers.append(block(features=features).to(device))
        return Sequential(*layers)

    def _down(self,
              features: int):
        return Sequential(
            Conv2d(in_channels=features,
                   out_channels=2 * features,
                   kernel_size=3,
                   padding=1,
                   stride=2,
                   bias=True),
            PReLU(),
        )

    def _up(self,
            features=None):
        return Sequential(
            PixelShuffle(2),
            PReLU(),
        )

    def forward(self, x):
        conc1 = self.seq_residual1b(x)
        out = self.down12(conc1)

        conc2 = self.seq_residual2b(out)
        conc3 = self.down23(conc2)

        out = self.seq_residual3(conc3)
        out = torch.cat([conc3, out], 1)

        out = self.up32(out)
        out = torch.cat([conc2, out], 1)
        out = self.seq_residual2a(out)

        out = self.up21(out)
        out = torch.cat([conc1, out], 1)
        out = self.seq_residual1a(out)

        out = self.relu_m(self.conv_m(out))
        out += x

        return out


class GAB(Module):
    def __init__(self,
                 num_frames: int = 7,
                 features: int = 48,
                 kernel_size: int = 3,
                 num_residual: int = 2,
                 num_attention: int = 1,
                 scale_residual: float = 1.0):
        super(GAB, self).__init__()
        self.num_frames = num_frames
        self.features = features
        self.kernel_size = kernel_size
        self.num_residual = num_residual
        self.num_attention = num_attention
        self.scale_residual = scale_residual

        self.res_i = self._make_layer(RB, self.features, self.num_residual)

        self.RB = self._make_layer(RB, self.features,
                                   self.num_residual, l=True)
        self.AB = self._make_layer(AB, self.features,
                                   self.num_attention, l=True)

        self.res_fusion = self._fusion(self.num_residual)
        self.att_fusion = self._fusion(self.num_attention)

        self.res_m = self._make_layer(RB, self.features * 2, 2)
        self.conv_m = Conv2d(in_channels=self.features * 2,
                             out_channels=self.features,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.relu_m = PReLU()

    def _make_layer(self,
                    block: Module,
                    features: int,
                    num: int,
                    l: bool = False):
        layers = []
        for _ in range(num):
            layers.append(block(features=features).to(device))
        return Sequential(*layers) if not l else layers

    def _residual(self,
                  num: int):
        return Sequential(
            Conv2d(
                in_channels=self.features,
                out_channels=self.features,
                kernel_size=self.kernel_size,
                padding=self.kernel_size//2,
                stride=1,
                bias=True),
            PReLU(),
            Conv2d(
                in_channels=self.features,
                out_channels=self.features,
                kernel_size=self.kernel_size,
                padding=self.kernel_size//2,
                stride=1,
                bias=True)
        )

    def _fusion(self,
                num: int):
        return Sequential(
            Conv2d(
                in_channels=num * self.features,
                out_channels=self.features,
                kernel_size=1,
                padding=0,
                stride=1),
            Conv2d(in_channels=self.features,
                   out_channels=self.features,
                   kernel_size=self.kernel_size,
                   padding=self.kernel_size//2,
                   stride=1),
        )

    def forward(self, x):
        out = self.res_i(x)

        RB_outs = []
        for i in range(self.num_residual):
            outR = self.RB[i](out)
            RB_outs.append(outR)
        outR = torch.cat(RB_outs, 1)
        outR = self.res_fusion(outR)

        AB_outs = []
        for i in range(self.num_attention):
            outA = self.AB[i](out)
            AB_outs.append(outA)
        outA = torch.cat(AB_outs, 1)
        outA = self.att_fusion(outA)

        out = torch.cat([outR, outA], 1)
        out = self.res_m(out)

        out = self.relu_m(self.conv_m(out))
        out *= self.scale_residual
        out += x

        return out


class C3Net(Module):
    def __init__(self, config):
        super(C3Net, self).__init__()
        self.num_frames = config['num_frames']
        self.features = config['features']
        self.kernel_size = config['kernel_size']
        self.num_global = config['num_global']
        self.num_residual = config['num_residual']
        self.num_attention = config['num_attention']
        self.scale_residual = config['scale_residual']

        self.conv_i = Conv2d(in_channels=3,
                             out_channels=self.features,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.relu_i = PReLU()

        self.global_attention = self._make_layer(GAB, self.num_global, l=True)

        self.conv_m = Conv2d(in_channels=self.features * 2,
                             out_channels=self.features,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.relu_m = PReLU()

        self.fusion = self._fusion(self.num_global)

        self.conv_o = Conv2d(in_channels=self.features,
                             out_channels=3,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.relu_o = PReLU()

    def _make_layer(self,
                    block: Module,
                    num: int,
                    l: bool = False):
        layers = []
        for _ in range(num):
            layers.append(
                block(features=self.features,
                      kernel_size=self.kernel_size,
                      num_residual=self.num_residual,
                      num_attention=self.num_attention,
                      scale_residual=self.scale_residual).to(device)
            )
        return Sequential(*layers) if not l else layers

    def _global_maxpool(self, input):
        output = torch.max(input, dim=1)[0]
        return torch.unsqueeze(output, 1)

    def _fusion(self, num):
        return Sequential(
            Conv2d(
                in_channels=num * self.features * self.num_frames,
                out_channels=self.features,
                kernel_size=1,
                padding=0,
                stride=1),
            Conv2d(in_channels=self.features,
                   out_channels=self.features,
                   kernel_size=self.kernel_size,
                   padding=self.kernel_size//2,
                   stride=1),
        )

    def forward(self, x):
        if self.num_frames != 1:
            residual = x[:, 3, ...]
            b, im, c, h, w = x.size()
            x = x.view((b*im, c, h, w))
        else:
            residual = x
        out = self.relu_i(self.conv_i(x))

        outs = []
        for i in range(self.num_global):
            out = self.global_attention[i](out)
            if self.num_frames != 1:
                out_max = self._global_maxpool(out.view((b, im, -1, h, w)))
                out_max = out_max.repeat(1, im, 1, 1, 1).view(b*im, -1, h, w)
                out = self.relu_m(self.conv_m(torch.cat([out, out_max], 1)))
            outs.append(out)
        out = torch.cat(outs, 1)

        if self.num_frames != 1:
            out = out.view((b, -1, h, w))
        out = self.fusion(out)
        out = self.relu_o(self.conv_o(out))
        out += residual
        return out


if __name__ == "__main__":
    # DataLoader
    dataset = LightningDataset(
        root={'train': 'D:/data/NTIRE2020/moire/train_burst/',
              'val': 'D:/data/NTIRE2020/moire/val_burst/',
              'test': 'D:/data/NTIRE2020/moire/test_burst/'}
        )
    dataset.setup()
    train_dataloader = dataset.train_dataloader()
    val_dataloader = dataset.val_dataloader()

    # Trainer
    model = MyLightning(model_mode='burst', loss_mode='l1+color')
    trainer = Trainer(max_epochs=1, accelerator='cuda', devices=[0])
    trainer.fit(model=model,
                train_dataloaders=train_dataloader)
