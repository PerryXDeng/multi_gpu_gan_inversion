import argparse
# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.data as data
import yaml

# from PIL import Image
from tqdm import tqdm
# from torchvision import transforms, utils
# from tensorboard_logger import Logger

from utils.datasets import *
from utils.functions import clip_img
from ranger import Ranger

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


# below settings cause error when calling iresnet in distributed training on PyTorch 1.7
# probably due to calling cudnn instructions on the wrong gpu
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True


torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
# device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
parser.add_argument('--config', type=str, default='001', help='Path to the config file.')
parser.add_argument('--real_dataset_path', type=str, default='./data/ffhq-dataset/images/', help='dataset path')
parser.add_argument('--dataset_path', type=str, default='./data/stylegan2-generate-images/ims/', help='dataset path')
parser.add_argument('--label_path', type=str, default='./data/stylegan2-generate-images/seeds_pytorch_1.8.1.npy', help='laebl path')
parser.add_argument('--stylegan_model_path', type=str, default='./pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt', help='pretrained stylegan2 model')
parser.add_argument('--arcface_model_path', type=str, default='./pretrained_models/backbone.pth', help='pretrained arcface model')
parser.add_argument('--parsing_model_path', type=str, default='./pretrained_models/79999_iter.pth', help='pretrained parsing model')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file path')

# https://pytorch.org/docs/1.8.0/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
# above only distributes forward(), not custom functions according to https://discuss.pytorch.org/t/multiple-forward-functions-in-dp-and-ddp/135029/3
# this means we need to put all the logic with custom functions in train.py into the trainer class forward() function
# https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
# set up distributed training
import torch.distributed as dist
def setup(rank, num_gpus):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=num_gpus)


def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()


from torch.utils.data.distributed import DistributedSampler
def prepare_dataloaders(rank, num_gpus, batch_size, img_size, train_data_split, noise_example, opts, pin_memory=False, num_workers=0):
    dataset_A = MyDataSet(image_dir=opts.dataset_path, label_dir=opts.label_path, output_size=img_size,
                          noise_in=noise_example, training_set=True, train_split=train_data_split)
    dataset_B = MyDataSet(image_dir=opts.real_dataset_path, label_dir=None, output_size=img_size,
                          noise_in=noise_example, training_set=True, train_split=train_data_split)
    sampler_A = DistributedSampler(dataset_A, num_replicas=num_gpus, rank=rank)
    sampler_B = DistributedSampler(dataset_B, num_replicas=num_gpus, rank=rank)
    dataloader_A = data.DataLoader(dataset_A, batch_size=batch_size, sampler=sampler_A, num_workers=num_workers,
                                   drop_last=False, shuffle=False, pin_memory=pin_memory)
    dataloader_B = data.DataLoader(dataset_B, batch_size=batch_size, sampler=sampler_B, num_workers=num_workers,
                                   drop_last=False, shuffle=False, pin_memory=pin_memory)
    return dataloader_A, dataloader_B



# backward() call automatically synchronized across multiple processes
def parallel_train(rank, world_size, opts):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    print('starting training on process: ', rank)
    config = yaml.load(open('./configs/' + opts.config + '.yaml', 'r'), Loader=yaml.FullLoader)
    from dist_trainer import Trainer
    trainer = Trainer(config, opts).cuda()
    noise_example = trainer.noise_inputs
    train_data_split = 0.9 if 'train_split' not in config else config['train_split']
    epochs = config['epochs']
    iter_per_epoch = config['iter_per_epoch']
    img_size = (config['resolution'], config['resolution'])
    img_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    process_batch_size = config['batch_size']//world_size # should be multiple of num_gpus
    loader_A, loader_B = prepare_dataloaders(rank, world_size, process_batch_size, img_size, train_data_split, noise_example, opts, pin_memory=True, num_workers=4)
    ddp_model = DDP(trainer, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # TODO: parameter specfication possibly incorrect - might need to access them through ddp_model.parameters()
    if 'optimizer' in config and config['optimizer'] == 'ranger':
        enc_opt = Ranger(ddp_model.module.enc_params, lr=config['lr'], betas=(config['beta_1'], config['beta_2']),
                         weight_decay=config['weight_decay'])
    else:
        enc_opt = torch.optim.Adam(ddp_model.module.enc_params, lr=config['lr'], betas=(config['beta_1'], config['beta_2']),
                                   weight_decay=config['weight_decay'])
    enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_opt, step_size=config['step_size'], gamma=config['gamma'])

    # Start Training
    epoch_0 = 0

    # check if checkpoint exist
    log_dir = os.path.join(opts.log_path, opts.config) + '/'
    if 'checkpoint.pth' in os.listdir(log_dir):
        ckpt_path = os.path.join(log_dir, 'checkpoint.pth')
        state_dict = torch.load(ckpt_path)
        enc_opt.load_state_dict(state_dict['enc_opt_state_dict'])
        enc_scheduler.load_state_dict(state_dict['enc_scheduler_state_dict'])
        epoch_0 = trainer.load_checkpoint(ckpt_path)

    if opts.resume:
        epoch_0 = trainer.load_checkpoint(os.path.join(opts.log_path, opts.checkpoint))

    for n_epoch in tqdm(range(epoch_0, epochs)):
        print('process', rank, 'starting epoch: ', n_epoch)
        iter_A = iter(loader_A)
        iter_B = iter(loader_B)
        iter_0 = n_epoch * iter_per_epoch

        for n_iter in range(iter_0, iter_0 + iter_per_epoch):

            if opts.dataset_path is None:
                z, noise = next(iter_A)
                img_A = None
            else:
                z, img_A, noise = next(iter_A)
                # img_A = img_A.to(device)

            # z = z.to(device)
            # noise = [ee.to(device) for ee in noise]

            if 'fixed_noise' in config and config['fixed_noise']:
                img_A, noise = None, None

            img_B = None
            if 'use_realimg' in config and config['use_realimg']:
                try:
                    img_B = next(iter_B)
                    if img_B.size(0) != process_batch_size:
                        iter_B = iter(loader_B)
                        img_B = next(iter_B)
                except StopIteration:
                    iter_B = iter(loader_B)
                    img_B = next(iter_B)
                # img_B = img_B.to(device)

            ############################################
            # # following logic should be put in trainer.forward()
            # w = trainer.mapping(z) # forward propagation on stylegan
            #
            # # another forward propagation on stylegan, with w in a continued forwarding
            # # also includes loss calc and back prop
            # trainer.update(w=w, img=img_A, noise=noise, real_img=img_B, n_iter=n_iter)
            ############################################
            enc_opt.zero_grad()
            # print("process: ", rank, "input shape", z.shape, img_A.shape, noise[0].shape, img_B.shape)
            z = z.to(rank)
            img_A = img_A.to(rank, non_blocking=True) if img_A is not None else None
            noise = [n.to(rank, non_blocking=True) for n in noise]
            img_B = img_B.to(rank, non_blocking=True) if img_B is not None else None
            # print("process: ", rank, ", devices", z.device, img_A.device if img_A is not None else -1,
            #       noise[0].device, img_B.device if img_B is not None else -1)
            loss = ddp_model(z, img_A, noise, img_B, n_iter)
            print("process: ", rank, ", iteration: ", n_iter, ", loss: ", loss.item())
            loss.backward()
            enc_opt.step()

            # if (n_iter+1) % config['log_iter'] == 0:
            #     trainer.log_loss(logger, n_iter, prefix='train')
            # if (n_iter+1) % config['image_save_iter'] == 0:
            #     trainer.save_image(log_dir, n_epoch, n_iter, prefix='/train/', w=w, img=img_A, noise=noise)
            #     trainer.save_image(log_dir, n_epoch, n_iter+1, prefix='/train/', w=w, img=img_B, noise=noise, training_mode=False)

        enc_scheduler.step()
        # trainer.save_checkpoint(n_epoch, log_dir, enc_opt, enc_scheduler) # TODO: implement checkpoint for distributed training

        # Test the model on celeba hq dataset on gpu0
        #if rank == 0:
        if True:
            with torch.no_grad():
                for i in range(10):
                    image_A = img_to_tensor(Image.open('./data/celeba_hq/%d.jpg' % i)).unsqueeze(0).to(rank)
                    output = ddp_model.module.test(img=image_A)
                    out_img = torch.cat(output, 3)
                    utils.save_image(clip_img(out_img[:1]), log_dir + 'validation/' + 'epoch_' +str(n_epoch+1) + '_' + str(i) + 'cuda' + str(rank) + '.jpg')
                # trainer.compute_loss(w=w, img=img_A, noise=noise, real_img=img_B)
                # trainer.log_loss(logger, n_iter, prefix='validation')
        dist.barrier()
    if rank == 0:
        ddp_model.module.save_model(log_dir)
    cleanup()


def main():
    opts = parser.parse_args()
    print('opt parsed')
    world_size = opts.num_gpus
    log_dir = os.path.join(opts.log_path, opts.config) + '/'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir + 'validation/', exist_ok=True)
    print('spawning processes')
    mp.spawn(parallel_train, args=(world_size, opts,), nprocs=world_size)


if __name__ == '__main__':
    main()


# log_dir = os.path.join(opts.log_path, opts.config) + '/'
# os.makedirs(log_dir, exist_ok=True)
# logger = Logger(log_dir)

# config = yaml.load(open('./configs/' + opts.config + '.yaml', 'r'), Loader=yaml.FullLoader)
#
# batch_size = config['batch_size'] # seems to be 1 by default
# epochs = config['epochs']
# iter_per_epoch = config['iter_per_epoch']
# img_size = (config['resolution'], config['resolution'])


# img_to_tensor = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
# img_to_tensor_car = transforms.Compose([
#     transforms.Resize((384, 512)),
#     transforms.Pad(padding=(0, 64, 0, 64)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# Initialize trainer
# trainer = Trainer(config, opts)
# if 'optimizer' in config and config['optimizer'] == 'ranger':
#     enc_opt = Ranger(self.enc_params, lr=config['lr'], betas=(config['beta_1'], config['beta_2']),
#                           weight_decay=config['weight_decay'])
# else:
#     enc_opt = torch.optim.Adam(self.enc_params, lr=config['lr'], betas=(config['beta_1'], config['beta_2']),
#                                     weight_decay=config['weight_decay'])
# enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_opt, step_size=config['step_size'], gamma=config['gamma'])

# noise_exemple = trainer.noise_inputs
# train_data_split = 0.9 if 'train_split' not in config else config['train_split']
#
# # Load synthetic dataset
# dataset_A = MyDataSet(image_dir=opts.dataset_path, label_dir=opts.label_path, output_size=img_size, noise_in=noise_exemple, training_set=True, train_split=train_data_split)
# loader_A = data.DataLoader(dataset_A, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
# # Load real dataset
# dataset_B = MyDataSet(image_dir=opts.real_dataset_path, label_dir=None, output_size=img_size, noise_in=noise_exemple, training_set=True, train_split=train_data_split)
# loader_B = data.DataLoader(dataset_B, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# # Start Training
# epoch_0 = 0
#
# # check if checkpoint exist
# if 'checkpoint.pth' in os.listdir(log_dir):
#     ckpt_path = os.path.join(log_dir, 'checkpoint.pth')
#     state_dict = torch.load(ckpt_path)
#     enc_opt.load_state_dict(state_dict['enc_opt_state_dict'])
#     enc_scheduler.load_state_dict(state_dict['enc_scheduler_state_dict'])
#     epoch_0 = trainer.load_checkpoint(ckpt_path)
#
# if opts.resume:
#     epoch_0 = trainer.load_checkpoint(os.path.join(opts.log_path, opts.checkpoint))

# torch.manual_seed(0)
# os.makedirs(log_dir + 'validation/', exist_ok=True)

# print("Start!")

# ddp_loss_model = DDP(trainer, device_ids=[rank])

# for n_epoch in tqdm(range(epoch_0, epochs)):
#
#     iter_A = iter(loader_A)
#     iter_B = iter(loader_B)
#     iter_0 = n_epoch*iter_per_epoch
#
#
#     for n_iter in range(iter_0, iter_0 + iter_per_epoch):
#
#         if opts.dataset_path is None:
#             z, noise = next(iter_A)
#             img_A = None
#         else:
#             z, img_A, noise = next(iter_A)
#             # img_A = img_A.to(device)
#
#         # z = z.to(device)
#         # noise = [ee.to(device) for ee in noise]
#
#         if 'fixed_noise' in config and config['fixed_noise']:
#             img_A, noise = None, None
#
#         img_B = None
#         if 'use_realimg' in config and config['use_realimg']:
#             try:
#                 img_B = next(iter_B)
#                 if img_B.size(0) != batch_size:
#                     iter_B = iter(loader_B)
#                     img_B = next(iter_B)
#             except StopIteration:
#                 iter_B = iter(loader_B)
#                 img_B = next(iter_B)
#             # img_B = img_B.to(device)
#
#         ############################################
#         # # following logic should be put in trainer.forward()
#         # w = trainer.mapping(z) # forward propagation on stylegan
#         #
#         # # another forward propagation on stylegan, with w in a continued forwarding
#         # # also includes loss calc and back prop
#         # trainer.update(w=w, img=img_A, noise=noise, real_img=img_B, n_iter=n_iter)
#         ############################################
#         enc_opt.zero_grad()
#         loss = trainer(z, img_A, noise, img_B, n_iter)
#         loss.backward()
#         enc_opt.step()
#
#         # if (n_iter+1) % config['log_iter'] == 0:
#         #     trainer.log_loss(logger, n_iter, prefix='train')
#         # if (n_iter+1) % config['image_save_iter'] == 0:
#         #     trainer.save_image(log_dir, n_epoch, n_iter, prefix='/train/', w=w, img=img_A, noise=noise)
#         #     trainer.save_image(log_dir, n_epoch, n_iter+1, prefix='/train/', w=w, img=img_B, noise=noise, training_mode=False)
#
#     enc_scheduler.step()
#     # trainer.save_checkpoint(n_epoch, log_dir, enc_opt, enc_scheduler)
#
#     # Test the model on celeba hq dataset
#     with torch.no_grad():
#         continue
#         # trainer.enc.eval()
#         # for i in range(10):
#         #     image_A = img_to_tensor(Image.open('./data/celeba_hq/%d.jpg' % i)).unsqueeze(0)#.to(device)
#         #     output = trainer.test(img=image_A)
#         #     out_img = torch.cat(output, 3)
#         #     utils.save_image(clip_img(out_img[:1]), log_dir + 'validation/' + 'epoch_' +str(n_epoch+1) + '_' + str(i) + '.jpg')
#         # trainer.compute_loss(w=w, img=img_A, noise=noise, real_img=img_B)
#         # trainer.log_loss(logger, n_iter, prefix='validation')
