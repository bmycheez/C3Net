import cv2
import os
import argparse
import glob
import time
from torch.autograd import Variable
from models import *
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--logdir", type=str, default="/home/user/depthMap/ksm/CVPR/demoire/logs", help='path of log files')
opt = parser.parse_args()


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


def main(model=0):
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

    # DHDN_flag = 4
    frame = 7
    ensemble_flag = 1
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('/home/user/depthMap/ksm/CVPR/demoire', 'ValidationInput', '*.png'))
    files_source.sort()
    # process data
    psnr_test = 0
    c = 0
    for f in range(len(files_source)//frame):
        # image
        start = time.time()
        ISource = []
        # final = np.zeros(cv2.imread(f).shape)
        origin = cv2.imread(files_source[f * frame + 3])
        for mode in range(ensemble_flag):
            for im in range(frame):
                data = cv2.imread(files_source[f * frame + im])
                if im != 3:
                    _, bin2 = cv2.threshold(data, 50, 255, cv2.THRESH_BINARY)
                    _, bin3 = cv2.threshold(data, 50, 255, cv2.THRESH_BINARY_INV)
                    final2 = cv2.bitwise_and(data, bin2, mask=None)
                    final3 = cv2.bitwise_and(origin, bin3, mask=None)
                    data = cv2.bitwise_or(final3, final2, mask=None)
                data = np.float32(normalize(data))
                data = np.transpose(data, (2, 0, 1))
                data = torch.Tensor(data).unsqueeze(0)
                ISource.append(data)
                """
                data = cv2.imread(files_source[f * frame + 3])
                data = np.float32(normalize(data))
                data = np.transpose(data, (2, 0, 1))
                data = torch.Tensor(data).unsqueeze(0)
                ISource.append(data)
                """
            ISource = torch.cat(ISource, 0)
            """
            hh, ww, cc = Img.shape
            for ch in range(cc):
                pl = Img[:, :, ch]
                Img[:, :, ch] = self_ensemble(pl, mode, 1)
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
            """
            INoisy = Variable(ISource.unsqueeze(0).cuda())
            # print(INoisy.size())
            with torch.no_grad(): # this can save much memory
                Out = torch.clamp(model(INoisy), 0., 1.)
            """
            if a % DHDN_flag != 0 or b % DHDN_flag != 0:
                h = DHDN_flag - (a % DHDN_flag)
                w = DHDN_flag - (b % DHDN_flag)
                Out = Out[:, :, h//2:Img.shape[0]-(h-h//2+1), w//2:Img.shape[1]-(w-w//2+1)]
            """
            c = f
            name = str(c)
            if str(c) != 6:
                for i in range(6 - len(str(c))):
                    name = '0' + name
            out = Out.squeeze(0).permute(1, 2, 0) * 255
            out = out.cpu().numpy()
            """
            for ch in range(cc):
                out[:, :, ch] = self_ensemble(out[:, :, ch], mode, 0)
            final += out
            """
        cv2.imwrite("/home/user/depthMap/ksm/CVPR/demoire/" + name + "_gt.png", out/ensemble_flag)
        mytime = time.time() - start
        psnr_test += mytime
        print("%s" % f)
        c += 1
    psnr_test /= (len(files_source)//frame)
    print("\nRuntime on test data %.2f" % psnr_test)


if __name__ == "__main__":
    main()
