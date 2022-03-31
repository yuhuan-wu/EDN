import os, shutil
import torch
import pickle
from models import model as net
import numpy as np
import transforms as myTransforms
from dataset import Dataset
from parallel import DataParallelModel, DataParallelCriterion
import time
from argparse import ArgumentParser
from saleval import SalEval
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
from torch.nn.parallel import gather
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def BCEDiceLoss(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    #print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    return bce + 1 - dice

class DSBCEDiceLoss(nn.Module):
    def __init__(self):
        super(DSBCEDiceLoss, self).__init__()

    def forward(self, inputs, target, teacher=False):
        #pred1, pred2, pred3, pred4, pred5 = tuple(inputs)
        if isinstance(target, tuple):
           target = target[0]
        #target = target[:,0,:,:]
        loss1 = BCEDiceLoss(inputs[:,0,:,:], target)
        loss2 = BCEDiceLoss(inputs[:,1,:,:], target)
        loss3 = BCEDiceLoss(inputs[:,2,:,:], target)
        loss4 = BCEDiceLoss(inputs[:,3,:,:], target)
        loss5 = BCEDiceLoss(inputs[:,4,:,:], target)
        
        return loss1+loss2+loss3+loss4+loss5

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, inputs, target):
        if isinstance(target, tuple):
            target = target[0]
        assert inputs.shape[1] == 5

        loss1 = F.binary_cross_entropy(inputs[:, 0, :, :], target)
        loss2 = F.binary_cross_entropy(inputs[:, 1, :, :], target)
        loss3 = F.binary_cross_entropy(inputs[:, 2, :, :], target)
        loss4 = F.binary_cross_entropy(inputs[:, 3, :, :], target)
        loss5 = F.binary_cross_entropy(inputs[:, 4, :, :], target)
        return loss1 + loss2 + loss3 + loss4 + loss5

class FLoss(nn.Module):
    def __init__(self, beta=0.3, log_like=False):
        super(FLoss, self).__init__()
        self.beta = beta
        self.log_like = log_like

    def _compute_loss(self, prediction, target):
        EPS = 1e-10
        N = prediction.size(0)
        TP = (prediction * target).view(N, -1).sum(dim=1)
        H = self.beta * target.view(N, -1).sum(dim=1) + prediction.view(N, -1).sum(dim=1)
        fmeasure = (1 + self.beta) * TP / (H + EPS)
        if self.log_like:
            loss = -torch.log(fmeasure)
        else:
            loss  = 1 - fmeasure
        return loss.mean()

    def forward(self, inputs, target):
        loss1 = self._compute_loss(inputs[:, 0, :, :], target)
        loss2 = self._compute_loss(inputs[:, 1, :, :], target)
        loss3 = self._compute_loss(inputs[:, 2, :, :], target)
        loss4 = self._compute_loss(inputs[:, 3, :, :], target)
        loss5 = self._compute_loss(inputs[:, 4, :, :], target)
        return 1.0*loss1 + 1.0*loss2 + 1.0*loss3 + 1.0*loss4 + 1.0*loss5

@torch.no_grad()
def val(args, val_loader, model, criterion):
    # switch to evaluation mode
    model.eval()
    sal_eval_val = SalEval()
    epoch_loss = []
    total_batches = len(val_loader)

    for iter, (input, target) in enumerate(tqdm(val_loader)):
        start_time = time.time()
        if args.gpu:
            input = input.cuda()
            target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target).float()

        # run the model
        output = model(input_var)
        loss = criterion(output, target_var)
        #torch.cuda.synchronize()
        time_taken = time.time() - start_time

        epoch_loss.append(loss.data.item())

        # compute the confusion matrix
        if args.gpu and torch.cuda.device_count() > 1:
            output = gather(output, 0, dim=0)
        sal_eval_val.add_batch(output[:, 0, :, :],  target_var)
        #if iter % 10 == 0:
        #    print('[%d/%d] loss: %.3f time: %.3f' % (iter, total_batches, loss.data.item(), time_taken))

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    F_beta, MAE = sal_eval_val.get_metric()

    return average_epoch_loss_val, F_beta, MAE

def train(args, train_loader, model, criterion, optimizer, epoch, max_batches, cur_iter=0):
    # switch to train mode
    model.train()
    #sal_eval_train = SalEval()
    epoch_loss = []
    total_batches = len(train_loader)
    iter_time = 0
    optimizer.zero_grad()
    bar = tqdm(train_loader)
    for iter, (input, target) in enumerate(bar):
        start_time = time.time()
        

        # adjust the learning rate
        lr = adjust_learning_rate(args, optimizer, epoch, iter + cur_iter, max_batches)

        if args.gpu == True:
            input = input.cuda()
            target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target).float()
        if args.ms1:
            resize = np.random.choice([320,352,384])
            input_var = F.interpolate(input_var, size=(resize, resize), mode='bilinear', align_corners=False)
            target_var = F.interpolate(target_var.unsqueeze(dim=1), size=(resize, resize), mode='bilinear', align_corners=False).squeeze(dim=1)

        # run the model
        output = model(input_var)
        loss = criterion(output, target_var) / args.iter_size

        loss.backward()
        
        iter_time += 1
        if iter_time % args.iter_size == 0:            
            optimizer.step()
            optimizer.zero_grad()
            
        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time

        # compute the confusion matrix
        bar.set_description("loss: {:.5f}, lr: {:.7f}".format(loss.data.item(), lr))

        if args.gpu and torch.cuda.device_count() > 1:
           output = gather(output, 0, dim=0)

        #if iter % 10 == 0:
        #    print('[%d/%d] iteration: [%d/%d] lr: %.7f loss: %.3f time:%.3f' % (iter, \
        #            total_batches, iter+cur_iter, max_batches*args.max_epochs, lr, \
        #            loss.data.item() * args.iter_size, time_taken))
    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    F_beta, MAE = 0, 0# sal_eval_train.get_metric()

    return average_epoch_loss_train, F_beta, MAE, lr

def adjust_learning_rate(args, optimizer, epoch, iter, max_batches):
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step_loss))
    elif args.lr_mode == 'poly':
        max_iter = max_batches * args.max_epochs
        lr = args.lr * (1 - iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
    if epoch == 0 and iter < 200: # warm up
        lr = args.lr * 0.99 * (iter + 1) / 200 + 0.01 * args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train_validate_saliency(args):
    # load the model
    model = net.EDN(arch=args.arch, pretrained=True, freeze_s1=args.freeze_s1)

    args.savedir = args.savedir + '/'
    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    
    print('copying train.py, train.sh, model.py to snapshots dir')
    shutil.copy('scripts/train.py', args.savedir + 'train.py')
    shutil.copy('scripts/train.sh', args.savedir + 'train.sh')
    os.system("scp -r {} {}".format("scripts", args.savedir))
    os.system("scp -r {} {}".format("models", args.savedir))
    
    if args.gpu and torch.cuda.device_count() > 1:
        #model = nn.DataParallel(model)
        model = DataParallelModel(model)

    if args.gpu:
        model = model.cuda()

    total_paramters = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters: ' + str(total_paramters))

    NORMALISE_PARAMS = [np.array([0.406, 0.456, 0.485], dtype=np.float32).reshape((1, 1, 3)), # MEAN, BGR
                        np.array([0.225, 0.224, 0.229], dtype=np.float32).reshape((1, 1, 3))] # STD, BGR
    
    # compose the data with transforms
    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(*NORMALISE_PARAMS),
        myTransforms.Scale(args.width, args.height),
        myTransforms.RandomCropResize(int(7./224.*args.width)),
        myTransforms.RandomFlip(),
        #myTransforms.GaussianNoise(),
        myTransforms.ToTensor(BGR=False)
    ])

    trainDataset_scale1 = myTransforms.Compose([
        myTransforms.Normalize(*NORMALISE_PARAMS),
        myTransforms.Scale(320, 320),
        myTransforms.RandomCropResize(int(7./224.*320)),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor(BGR=False)
    ])
    trainDataset_scale2 = myTransforms.Compose([
        myTransforms.Normalize(*NORMALISE_PARAMS),
        myTransforms.Scale(352, 352),
        myTransforms.RandomCropResize(int(7./224.*352)),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor(BGR=False)
    ])

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(*NORMALISE_PARAMS),
        myTransforms.Scale(args.width, args.height),
        myTransforms.ToTensor(BGR=False)
    ])

    val_names = ["DUTS-TE", "DUT-OMRON", "HKU-IS", "ECSSD", "SOD", "PASCAL-S"]
    
    trainLoader_main = torch.utils.data.DataLoader(
        Dataset(args.data_dir, 'DUTS-TR', transform=trainDataset_main),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)

    trainLoader_scale1 = torch.utils.data.DataLoader(
        Dataset(args.data_dir, 'DUTS-TR', transform=trainDataset_scale1),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
    trainLoader_scale2 = torch.utils.data.DataLoader(
        Dataset(args.data_dir, 'DUTS-TR', transform=trainDataset_scale2),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)

    valLoader = torch.utils.data.DataLoader(
        Dataset(args.data_dir, val_names[0], transform=valDataset),
        batch_size=12, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    valLoader1 = torch.utils.data.DataLoader(
        Dataset(args.data_dir, val_names[1], transform=valDataset),
        batch_size=12, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    valLoader2 = torch.utils.data.DataLoader(
        Dataset(args.data_dir, val_names[2], transform=valDataset),
        batch_size=12, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    valLoader3 = torch.utils.data.DataLoader(
        Dataset(args.data_dir, val_names[3], transform=valDataset),
        batch_size=12, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    valLoader4 = torch.utils.data.DataLoader(
        Dataset(args.data_dir, val_names[4], transform=valDataset),
        batch_size=12, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    valLoader5 = torch.utils.data.DataLoader(
        Dataset(args.data_dir, val_names[5], transform=valDataset),
        batch_size=12, shuffle=False, num_workers=args.num_workers, pin_memory=False)        
    if args.ms:
        max_batches = len(trainLoader_main) + len(trainLoader_scale1) + len(trainLoader_scale2)
    else:
        max_batches = len(trainLoader_main)
    print('max_batches {}'.format(max_batches))
    cudnn.benchmark = True

    start_epoch = 0

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            #args.lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    log_file = args.savedir + args.log_file
    if os.path.isfile(log_file):
        logger = open(log_file, 'a')
    else:
        logger = open(log_file, 'w')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t\t%s\t%s\t%s\t%s\t%s\tlr" % ('Epoch', \
                'Loss(Tr)', 'F_beta (tr)', 'MAE (tr)', 'F_beta (val)', 'MAE (val)'))
    logger.flush()

    normal_parameters = []; picked_parameters = []
    if args.group_lr:
        ### use smaller lr in backbone
            for pname, p in model.named_parameters():
                    if 'backbone' in pname:
                        picked_parameters.append(p)
                        print("lr/10", pname)
                    else:
                        normal_parameters.append(p)
            optimizer = torch.optim.Adam([
                {
                    'params': normal_parameters,
                    'lr': args.lr,
                    'weight_decay': 1e-4
                },
                {
                    'params': picked_parameters,
                    'lr': args.lr / 10,
                    'weight_decay': 1e-4
                },
            ],
             lr=args.lr,
             betas=(0.9, args.adam_beta2),
             eps=1e-08,
             weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, args.adam_beta2), eps=1e-08, weight_decay=1e-4)
    cur_iter = 0

    criteria = CrossEntropyLoss()
    if args.bcedice:
        criteria = DSBCEDiceLoss()
    if args.gpu and torch.cuda.device_count() > 1:
        print("using mutliple gpus")
        criteria = DataParallelCriterion(criteria)

    epoch_idxes = []
    F_beta_vals = []
    F_beta_val1s = []
    F_beta_val2s = []
    F_beta_val3s = []
    F_beta_val4s = []
    F_beta_val5s = []
    MAE_vals = []
    MAE_val1s = []
    MAE_val2s = []
    MAE_val3s = []
    MAE_val4s = []
    MAE_val5s = []

    for epoch in range(start_epoch, args.max_epochs):
        # train for one epoch
        if args.ms:
            train(args, trainLoader_scale1, model, criteria, optimizer, epoch, max_batches, cur_iter)
            cur_iter += len(trainLoader_scale1)
            torch.cuda.empty_cache()
            train(args, trainLoader_scale2, model, criteria, optimizer, epoch, max_batches, cur_iter)
            cur_iter += len(trainLoader_scale2)
            torch.cuda.empty_cache()

        loss_tr, F_beta_tr, MAE_tr, lr = \
            train(args, trainLoader_main, model, criteria, optimizer, epoch, max_batches, cur_iter)
        cur_iter += len(trainLoader_main)
        torch.cuda.empty_cache()

        # evaluate on validation set
        print("start to evaluate on epoch {}".format(epoch+1))
        import time
        start_time = time.time()
        loss_val, F_beta_val, MAE_val = val(args, valLoader, model, criteria)
        torch.cuda.empty_cache()
        if epoch > args.max_epochs * 0.5:
            loss_val1, F_beta_val1, MAE_val1 = val(args, valLoader1, model, criteria)
            torch.cuda.empty_cache()
            loss_val2, F_beta_val2, MAE_val2 = val(args, valLoader2, model, criteria)
            torch.cuda.empty_cache()
            loss_val3, F_beta_val3, MAE_val3 = val(args, valLoader3, model, criteria)
            torch.cuda.empty_cache()
            loss_val4, F_beta_val4, MAE_val4 = val(args, valLoader4, model, criteria)
            torch.cuda.empty_cache()
            loss_val5, F_beta_val5, MAE_val5 = val(args, valLoader5, model, criteria)
            F_beta_vals.append(F_beta_val)
            F_beta_val1s.append(F_beta_val1)
            F_beta_val2s.append(F_beta_val2)
            F_beta_val3s.append(F_beta_val3)
            F_beta_val4s.append(F_beta_val4)
            F_beta_val5s.append(F_beta_val5)
            MAE_vals.append(MAE_val)
            MAE_val1s.append(MAE_val1)
            MAE_val2s.append(MAE_val2)
            MAE_val3s.append(MAE_val3)
            MAE_val4s.append(MAE_val4)
            MAE_val5s.append(MAE_val5)
            epoch_idxes.append(epoch+1)

        
        print("elapsed evaluation time: {} hours".format((time.time()-start_time)/3600.0))
        torch.cuda.empty_cache()

        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_val': loss_val,
            'iou_val': F_beta_val,
        }, args.savedir + 'checkpoint.pth.tar')

        # save the model also
        
        if epoch > args.max_epochs * 0.5:
            model_file_name = args.savedir + '/model_' + str(epoch + 1) + '.pth'
            print("saving state dict to {}".format(model_file_name))
            torch.save(model.state_dict(), model_file_name)
        
        log_str = "\n{} {:.4f} {:.4f}".format(epoch+1, F_beta_val, MAE_val)
        try:
            log_str = log_str + " {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(
                F_beta_val1, MAE_val1, F_beta_val2, MAE_val2, F_beta_val3, MAE_val3, F_beta_val4, MAE_val4, F_beta_val5, MAE_val5)
        except:
            pass
        logger.write(log_str)
        logger.flush()
        print("Epoch " + str(epoch) + ': Details')
        print("\nEpoch No. %d: \t Val Loss = %.4f\t F_beta(val) = %.4f" \
                % (epoch, loss_val, F_beta_val))
        torch.cuda.empty_cache()
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="./data/", help='Data directory')
    parser.add_argument('--width', type=int, default=384, help='Width of RGB image')
    parser.add_argument('--height', type=int, default=384, help='Height of RGB image')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=10, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='step', help='Learning rate policy, step or poly')
    parser.add_argument('--savedir', default='./results', help='Directory to save the results')
    parser.add_argument('--resume', default=None, help='Use this checkpoint to continue training')
    parser.add_argument('--log_file', default='trainValLog.txt', help='File that stores the training and validation logs')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--iter_size', default=1, type=int)
    parser.add_argument('--arch', default='vgg16', type=str)
    parser.add_argument('--ms', default=1, type=int) # normal multi-scale training
    parser.add_argument('--ms1', default=0, type=int) # hybrid multi-scale training. It has comparable performance with normal multi-scale training in my experiments. But I think hybrid multi-scale training may be a better choice.
    parser.add_argument('--adam_beta2', default=0.999, type=float) # The value of 0.99 can introduce slightly higher performance (0.1%~0.2%)
    parser.add_argument('--bcedice', default=0, type=int)
    parser.add_argument('--group_lr', default=0, type=int)
    parser.add_argument('--freeze_s1', default=0, type=int)

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    train_validate_saliency(args)

