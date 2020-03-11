import argparse
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import *
from dataset import *
from utils import *
from datetime import datetime
from ssim import *


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--patch", type=int, default=128, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
parser.add_argument("--start_epochs", type=int, default=60, help="Number of training epochs")
parser.add_argument("--start_iters", type=int, default=0, help="Number of training epochs")
parser.add_argument("--resume", type=str, default="/home/user/depthMap/ksm/CVPR/demoire/logs/48_40.50146.pth",
                    help="Number of training epochs")
parser.add_argument("--step", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--decay", type=int, default=10, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="/home/user/depthMap/ksm/CVPR/demoire/checkpoint",
                    help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
opt = parser.parse_args()


def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = DatasetBurst(train=True)
    dataset_val = DatasetBurst(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=False)
    loader_val = DataLoader(dataset=dataset_val, num_workers=4, batch_size=1, shuffle=False)
    # print(opt.batchSize)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    # net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    model = Net().cuda()
    # s = MSSSIM()
    criterion = nn.L1Loss().cuda()
    burst = BurstLoss().cuda()
    # vgg = Vgg16(requires_grad=False).cuda()
    # vgg = VGG('54').cuda()
    # Move to GPU
    # model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # '''
    if opt.resume:
        model.load_state_dict(torch.load(opt.resume))
        # test.main(model)
        # return
    # '''
    # summary(model, (3, 128, 128))
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    psnr_max = 0
    loss_min = 1
    for epoch in range(opt.start_epochs, opt.epochs):
        # current_lr = opt.lr * ((1 / opt.decay) ** ((epoch - opt.start_epochs) // opt.step))
        current_lr = opt.lr * ((1 / opt.decay) ** (epoch // opt.step))
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, (imgn_train, img_train) in enumerate(loader_train, 0):
            if i < opt.start_iters:
                continue
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            out_train = model(imgn_train)
            # feat_x = vgg(imgn_train)
            # feat_y = vgg(out_train)
            # perceptual_loss = criterion(feat_y.relu2_2, feat_x.relu2_2)
            # perceptual_loss = vgg(out_train, img_train)
            loss_color = color_loss(out_train, img_train)
            loss_content = criterion(out_train, img_train)
            loss_burst = burst(out_train, img_train)
            m = [5, 5, 0]
            loss = torch.div(m[0] * loss_color.cuda() + m[1] * loss_content.cuda() + m[2] * loss_burst.cuda(), 10)
            loss.backward()
            optimizer.step()
            # '''
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if i % int(len(loader_train)//5) == 0:
                # the end of each epoch
                model.eval()
                # validate
                psnr_val = 0
                for _, (imgn_val, img_val) in enumerate(loader_val, 0):
                    with torch.no_grad():
                        img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
                        out_val = torch.clamp(model(imgn_val), 0., 1.)
                    psnr_val += batch_PSNR(out_val, img_val, 1.)
                psnr_val /= len(dataset_val)
                now = datetime.now()
                print("[epoch %d][%d/%d] loss: %.6f PSNR_val: %.4f" %
                      (epoch+1, i+1, len(loader_train), loss.item(), psnr_val), end=' ')
                print(now.year, now.month, now.day, now.hour, now.minute, now.second)
                if psnr_val > psnr_max or loss < loss_min:
                    psnr_max = psnr_val
                    loss_min = loss
                    torch.save(model.state_dict(), os.path.join(opt.outf, 'net_' + str(round(psnr_val, 4)) + '.pth'))
            # '''
    torch.save(model.state_dict(), os.path.join(opt.outf, 'net_' + str(round(psnr_val, 4)) + '.pth'))


if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=opt.patch, stride=opt.patch, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    main()
