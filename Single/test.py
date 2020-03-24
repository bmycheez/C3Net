import cv2
import os
import argparse
import glob
import time
from torch.autograd import Variable
from models_v2 import Net
from utils import *


parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num", type=int, default=3, help="Number of total layers")
parser.add_argument("--logdir", type=str, default=".", help='path of log files')
parser.add_argument("--gpu", type=str, default='0', help='test on Set12 or Set68')
parser.add_argument("--inputdir", type=str, default='DemoireingTestInputSingle', help='noise level used on test set')
opt = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu


def normalize(data):
    return data/255.


def self_ensemble(out, mode, forward):
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        if forward == 1:
            out = np.rot90(out)
        else:
            out = np.rot90(out, k=3)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        if forward == 1:
            out = np.rot90(out)
            out = np.flipud(out)
        else:
            out = np.flipud(out)
            out = np.rot90(out, k=3)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        if forward == 1:
            out = np.rot90(out, k=2)
            out = np.flipud(out)
        else:
            out = np.flipud(out)
            out = np.rot90(out, k=2)
    elif mode == 6:
        if forward == 1:
            out = np.rot90(out, k=3)
        else:
            out = np.rot90(out)
    elif mode == 7:
        # rotate 270 degree and flip
        if forward == 1:
            out = np.rot90(out, k=3)
            out = np.flipud(out)
        else:
            out = np.flipud(out)
            out = np.rot90(out)
    return out


def self_ensemble_v2(out, mode, forward):
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        out = np.fliplr(out)
    elif mode == 3:
        out = np.flipud(out)
        out = np.fliplr(out)
    return out


def main():
    # Build model
    print('Loading model ...\n')
    model = Net().cuda()
    # device_ids = [0]
    # model = nn.DataParallel(net, device_ids=device_ids).cuda()
    a = torch.load(glob.glob(os.path.join(opt.logdir, '*.pth'))[0])
    print(glob.glob(os.path.join(opt.logdir, '*.pth'))[0])
    ok = input("Right model? ")
    if ok == 'n':
        return
    model.load_state_dict(a)
    DHDN_flag = 4
    ensemble_flag = 4
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('D:/', opt.inputdir, '*_%d.png'
                                          % opt.num))
    files_source.sort()
    # process data
    psnr_test = 0
    c = 0
    for f in files_source:
        # image
        start = time.time()
        final = np.zeros(cv2.imread(f).shape)
        for mode in range(ensemble_flag):
            Img = cv2.imread(f)
            hh, ww, cc = Img.shape
            Img = self_ensemble_v2(Img, mode, 1)
            Img = np.swapaxes(Img, 0, 2)
            Img = np.swapaxes(Img, 1, 2)
            Img = np.float32(normalize(Img))
            a = Img.shape[1]
            b = Img.shape[2]
            if a % DHDN_flag != 0 or b % DHDN_flag != 0:
                h = DHDN_flag - (a % DHDN_flag)
                w = DHDN_flag - (b % DHDN_flag)
                Img = np.pad(Img, [(0, 0), (h//2, h-h//2), (w//2, w-w//2)], mode='edge')
            Img = np.expand_dims(Img, 0)
            ISource = torch.Tensor(Img)
            INoisy = Variable(ISource.cuda())
            with torch.no_grad(): # this can save much memory
                Out = torch.clamp(model(INoisy), 0., 1.)
            if a % DHDN_flag != 0 or b % DHDN_flag != 0:
                h = DHDN_flag - (a % DHDN_flag)
                w = DHDN_flag - (b % DHDN_flag)
                Out = Out[:, :, h//2:Img.shape[0]-(h-h//2+1), w//2:Img.shape[1]-(w-w//2+1)]
            name = str(c)
            if str(c) != 6:
                for i in range(6 - len(str(c))):
                    name = '0' + name
            out = Out.squeeze(0).permute(1, 2, 0) * 255
            out = out.cpu().numpy()
            out = self_ensemble_v2(out, mode, 0)
            final += out
        cv2.imwrite(name + "_gt.png", final/ensemble_flag)
        mytime = time.time() - start
        psnr_test += mytime
        print("%s" % f)
        c += 1
    psnr_test /= len(files_source)
    print("\nRuntime on test data %.2f" % psnr_test)


if __name__ == "__main__":
    main()
