import os
from datetime import datetime
from utils import *
from torch.autograd import Variable
from loss.contrast_loss import SupConLoss, CORALLoss
from network.unet2d import UNet2D_classification
from dataset.chd import CHD
from dataset.acdc import ACDC
from myconfig import get_config
from batchgenerators.utilities.file_and_folder_operations import *
from lr_scheduler import LR_Scheduler
from torch.utils.tensorboard import SummaryWriter
from experiment_log import PytorchExperimentLogger

def main():
    # initialize config
    args = get_config()

    if args.save == '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.experiment_name + args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logger = PytorchExperimentLogger(save_path, "elog", ShowTerminal=True)
    model_result_dir = join(save_path, 'model')
    maybe_mkdir_p(model_result_dir)
    args.model_result_dir = model_result_dir

    logger.print(f"saving to {save_path}")
    writer = SummaryWriter('runs/' + args.experiment_name + args.save)

    # setup cuda
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # logger.print(f"the model will run on device {args.device}")

    # create model
    logger.print("creating model ...")
    model = UNet2D_classification(in_channels=1, initial_filter_size=args.initial_filter_size, kernel_size=3, classes=args.classes, do_instancenorm=True)

    if args.restart:
        logger.print('loading from saved model'+args.pretrained_model_path)
        dict = torch.load(args.pretrained_model_path,
                          map_location=lambda storage, loc: storage)
        save_model = dict["net"]
        model.load_state_dict(save_model)

    model.to(args.device)
    model = torch.nn.DataParallel(model)

    num_parameters = sum([l.nelement() for l in model.module.parameters()])
    logger.print(f"number of parameters: {num_parameters}")

    if args.dataset == 'chd':
        training_keys = os.listdir(os.path.join(args.data_dir,'train'))
        training_keys.sort()
        train_dataset = CHD(keys=training_keys, purpose='train', args=args)
    elif args.dataset == 'acdc':
        train_dataset = ACDC(keys=list(range(1,101)), purpose='train', args=args)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_works, drop_last=True)

    # define loss function (criterion) and optimizer
    if args.contrastive_method == 'coral':
        feature_dim = args.classes
        criterion = CORALLoss(
            feature_dim=feature_dim, 
            temperature=args.temp,
            contrast_mode='all',
            alpha=args.alpha,
            apply_soft_weights=args.apply_soft_weights,
            sigmoid_scale=args.sigmoid_scale
        ).to(args.device)
        logger.print(f"CORAL Loss initialized with: alpha={args.alpha}, "
                    f"soft_weights={'enabled' if args.apply_soft_weights else 'disabled'}, "
                    f"sigmoid_scale={args.sigmoid_scale}")
        optimizer = torch.optim.SGD(
            list(model.parameters()) + list(criterion.parameters()), 
            lr=args.lr, 
            momentum=0.9, 
            weight_decay=1e-5
        )
    else:
        criterion = SupConLoss(
            threshold=args.slice_threshold, 
            temperature=args.temp, 
            contrastive_method=args.contrastive_method
        ).to(args.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_loader))

    for epoch in range(args.epochs):
        # train for one epoch
        train_loss = train(train_loader, model, criterion, epoch, optimizer, scheduler, logger, args)

        logger.print('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     .format(epoch + 1, train_loss=train_loss))

        writer.add_scalar('training_loss', train_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # save model
        save_dict = {"net": model.module.state_dict()}
        torch.save(save_dict, os.path.join(args.model_result_dir, "latest.pth"))

def train(data_loader, model, criterion, epoch, optimizer, scheduler, logger, args):
    model.train()
    losses = AverageMeter()
    ral_losses = AverageMeter()
    oal_losses = AverageMeter()
    
    for batch_idx, tup in enumerate(data_loader):
        scheduler(optimizer, batch_idx, epoch)
        img1, img2, slice_position, partition = tup
        image1_var = Variable(img1.float(), requires_grad=False).to(args.device)
        image2_var = Variable(img2.float(), requires_grad=False).to(args.device)
        f1_1 = model(image1_var)
        f2_1 = model(image2_var)
        bsz = img1.shape[0]
        features = torch.cat([f1_1.unsqueeze(1), f2_1.unsqueeze(1)], dim=1)
        
        if args.contrastive_method == 'coral':
            loss_dict = criterion(features, labels=slice_position)
            loss = loss_dict['loss']
            ral_losses.update(loss_dict['ral_loss'], bsz)
            oal_losses.update(loss_dict['oal_loss'], bsz)
            losses.update(loss.item(), bsz)
            logger.print(f"epoch:{epoch}, batch:{batch_idx}/{len(data_loader)}, "
                        f"lr:{optimizer.param_groups[0]['lr']:.6f}, "
                        f"loss:{losses.avg:.4f}, "
                        f"RAL:{ral_losses.avg:.4f}, "
                        f"OAL:{oal_losses.avg:.4f}")
        elif args.contrastive_method == 'pcl':
            loss = criterion(features, labels=slice_position)
            losses.update(loss.item(), bsz)
            logger.print(f"epoch:{epoch}, batch:{batch_idx}/{len(data_loader)}, "
                        f"lr:{optimizer.param_groups[0]['lr']:.6f}, "
                        f"loss:{losses.avg:.4f}")
        elif args.contrastive_method == 'gcl':
            loss = criterion(features, labels=partition)
            losses.update(loss.item(), bsz)
            logger.print(f"epoch:{epoch}, batch:{batch_idx}/{len(data_loader)}, "
                        f"lr:{optimizer.param_groups[0]['lr']:.6f}, "
                        f"loss:{losses.avg:.4f}")
        else: # simclr
            loss = criterion(features)
            losses.update(loss.item(), bsz)
            logger.print(f"epoch:{epoch}, batch:{batch_idx}/{len(data_loader)}, "
                        f"lr:{optimizer.param_groups[0]['lr']:.6f}, "
                        f"loss:{losses.avg:.4f}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return losses.avg

if __name__ == '__main__':
    main()