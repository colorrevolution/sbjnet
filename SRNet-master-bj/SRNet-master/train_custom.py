# Training script for the SRNet. Refer README for instructions.
# author: Niwhskal
# github : https://github.com/Niwhskal/SRNet

import numpy as np
import os
import torch
from utils import *
import cfg
from tqdm import tqdm
from model_custom import Generator, Discriminator,Vgg19
from loss_custom import build_generator_loss
from datagen_custom import datagen_srnet, example_dataset, To_tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from bce import WeightedBCELoss

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def custom_collate(batch):
    i_t_batch, i_s_batch = [], []
    t_sk_batch, t_t_batch = [], []
    w_sum = 0

    # for item in batch:
    #     t_t = item[3]
    #     h, w = t_t.shape[:2]
    #     scale_ratio = cfg.data_shape[0] / h
    #     w_sum += int(w * scale_ratio)
    #
    # to_h = cfg.data_shape[0]
    # to_w = w_sum // cfg.batch_size
    # to_w = int(round(to_w / 8)) * 8
    # to_scale = (to_h, to_w)

    for item in batch:
        i_t, i_s, t_sk, t_t = item

        # i_t = resize(i_t, to_scale, preserve_range=True)
        # i_s = resize(i_s, to_scale, preserve_range=True)
        # t_sk = np.expand_dims(resize(t_sk, to_scale, preserve_range=True), axis=-1)
        # t_t = resize(t_t, to_scale, preserve_range=True)

        t_sk = np.expand_dims(t_sk,-1)
        i_t = i_t.transpose((2, 0, 1))
        i_s = i_s.transpose((2, 0, 1))
        t_sk = t_sk.transpose((2, 0, 1))
        t_t = t_t.transpose((2, 0, 1))


        i_t_batch.append(i_t)
        i_s_batch.append(i_s)
        t_sk_batch.append(t_sk)
        t_t_batch.append(t_t)


    i_t_batch = np.stack(i_t_batch)
    i_s_batch = np.stack(i_s_batch)
    t_sk_batch = np.stack(t_sk_batch)
    t_t_batch = np.stack(t_t_batch)


    i_t_batch = torch.from_numpy(i_t_batch.astype(np.float32) / 127.5 - 1.)
    i_s_batch = torch.from_numpy(i_s_batch.astype(np.float32) / 127.5 - 1.)
    t_sk_batch = torch.from_numpy(t_sk_batch.astype(np.float32) / 255.)
    t_t_batch = torch.from_numpy(t_t_batch.astype(np.float32) / 127.5 - 1.)

    print("i_t_batch", i_t_batch.shape)
    print("t_sk_batch", t_sk_batch.shape)


    return [i_t_batch, i_s_batch, t_sk_batch, t_t_batch]


def clip_grad(model):
    for h in model.parameters():
        h.data.clamp_(-0.01, 0.01)


def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

    writer = SummaryWriter("./log/", comment="newdataset1")
    train_name = get_train_name()

    print_log('Initializing SRNET', content_color=PrintColor['yellow'])

    train_data = datagen_srnet(cfg)

    train_data = DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True,pin_memory=True)

    trfms = To_tensor()
    example_data = example_dataset(transform=trfms)

    example_loader = DataLoader(dataset=example_data, batch_size=1, shuffle=False)

    print_log('training start.', content_color=PrintColor['yellow'])

    G = Generator(in_channels=3).cuda()

    D1 = Discriminator(in_channels=6).cuda()

    weighted_bce_loss = WeightedBCELoss().cuda()

    # vgg_features = Vgg19().cuda()

    G_solver = torch.optim.Adam(G.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))
    D1_solver = torch.optim.Adam(D1.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))
    vgg_features = Vgg19().cuda()

    # g_scheduler = torch.optim.lr_scheduler.MultiStepLR(G_solver, milestones=[30, 200], gamma=0.5)

    # d1_scheduler = torch.optim.lr_scheduler.MultiStepLR(D1_solver, milestones=[30, 200], gamma=0.5)

    # d2_scheduler = torch.optim.lr_scheduler.MultiStepLR(D2_solver, milestones=[30, 200], gamma=0.5)

    # try:
    #
    #   checkpoint = torch.load(cfg.ckpt_path)
    #   G.load_state_dict(checkpoint['generator'])
    #   D1.load_state_dict(checkpoint['discriminator1'])
    #   D2.load_state_dict(checkpoint['discriminator2'])
    #   # G_solver.load_state_dict(checkpoint['g_optimizer'])
    #   # D1_solver.load_state_dict(checkpoint['d1_optimizer'])
    #   # D2_solver.load_state_dict(checkpoint['d2_optimizer'])
    #   # print(len(checkpoint['g_optimizer']["param_groups"][0]['params']))
    #   # print(list(G.parameters()))
    #   #
    #   '''
    #   g_scheduler.load_state_dict(checkpoint['g_scheduler'])
    #   d1_scheduler.load_state_dict(checkpoint['d1_scheduler'])
    #   d2_scheduler.load_state_dict(checkpoint['d2_scheduler'])
    #   '''
    #
    #   print('Resuming after loading...')
    #
    # except FileNotFoundError:
    #
    #   print('checkpoint not found')
    #   pass

    requires_grad(G, True)

    requires_grad(D1, False)


    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    trainiter = train_data
    example_iter = example_loader

    K = torch.nn.ZeroPad2d((0, 1, 1, 0))

    for step in tqdm(range(cfg.max_iter)):

        if ((step + 1) % cfg.save_ckpt_interval == 0):
            torch.save(
                {
                    'generator': G.state_dict(),
                    'discriminator1': D1.state_dict(),
                    'g_optimizer': G_solver.state_dict(),
                    'd1_optimizer': D1_solver.state_dict(),
                    # 'g_scheduler' : g_scheduler.state_dict(),
                    # 'd1_scheduler':d1_scheduler.state_dict(),
                    # 'd2_scheduler':d2_scheduler.state_dict(),
                },
                cfg.checkpoint_savedir + f'train_step-{step + 1}.model',
            )
        # try:
        #
        #     i_t, i_s, t_sk, t_t = trainiter.next()
        #
        # except StopIteration:
        #
        #     trainiter = iter(train_data)
        #     i_t, i_s, t_sk, t_t = trainiter.next()
        # print("22222222222222222")
        for i_t, i_s, t_sk, t_t in trainiter:
            # D1_solver.zero_grad()
            # print("1111111111111111")
            i_t = i_t.cuda()
            i_s = i_s.cuda()
            t_sk = t_sk.cuda()
            t_t = t_t.cuda()
            # inputs = [i_t, i_s]
            labels = [t_sk, t_t]

            # o_sk, o_t = G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))  # Adding dim info
            #
            # o_sk = K(o_sk)
            # o_t = K(o_t)
            #
            # # print("o_sk",o_sk.shape)
            #
            # i_t_true = torch.cat((t_t, i_s), dim=1)
            # i_t_pred = torch.cat((o_t, i_s), dim=1)
            #
            # o_t_true = D1(i_t_true)
            # o_t_pred = D1(i_t_pred)
            #
            # t_loss = build_discriminator_loss(o_t_true, o_t_pred)
            # writer.add_scalar("D_Loss/t_loss", t_loss.item(), step)
            # t_loss.backward()
            # D1_solver.step()

            # clip_grad(D1)

            # if ((step + 1) % 2 == 0):
            #
            #     requires_grad(G, True)
            #     requires_grad(D1, False)
            #     G_solver.zero_grad()
            #     o_sk, o_t = G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))
            #     o_sk = K(o_sk)
            #     o_t = K(o_t)
            #
            #
            #     # print(o_sk.shape, o_t.shape, o_b.shape, o_f.shape)
            #     # print('------')
            #     # print(i_s.shape)
            #     i_t_true = torch.cat((t_t, i_s), dim=1)
            #     i_t_pred = torch.cat((o_t, i_s), dim=1)
            #
            #     o_t_true = D1(i_t_true)
            #     o_t_pred = D1(i_t_pred)
            #
            #     t_loss = build_discriminator_loss(o_t_true, o_t_pred)
            #
            #     i_vgg = torch.cat((t_t, o_t), dim=0)
            #
            #     # out_vgg = vgg_features(i_vgg)
            #
            #     out_g = [o_sk, o_t]
            #     # g_loss = build_generator_loss(out_g, out_vgg, labels) + t_loss
            #     g_loss = build_generator_loss(out_g, labels) + t_loss
            #     writer.add_scalar("G_Loss/g_loss", g_loss.item(), step)
            #     g_loss.backward()
            #
            #     G_solver.step()
            #
            #     # g_scheduler.step()
            #
            #     requires_grad(G, False)
            #
            #     requires_grad(D1, True)


        # if ((step + 1) % cfg.write_log_interval == 0):
        #     print('Iter: {}/{} | Gen: {} | t_loss: {} '.format(step + 1, cfg.max_iter, g_loss.item(),
        #                                                                 t_loss.item()))
            # if ((step + 1) % cfg.gen_example_interval == 0):
            #
            #     savedir = os.path.join(cfg.example_result_dir, train_name,
            #                            'iter-' + str(step + 1).zfill(len(str(cfg.max_iter))))
            #
            #     with torch.no_grad():
            #
            #         try:
            #
            #             inp = example_iter.next()
            #
            #         except StopIteration:
            #
            #             example_iter = iter(example_loader)
            #             inp = example_iter.next()
            #
            #         i_t = inp[0].cuda()
            #         i_s = inp[1].cuda()
            #         name = str(inp[2][0])
            #
            #         o_sk, o_t = G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))
            #
            #         o_sk = o_sk.squeeze(0).to('cpu')
            #         o_t = o_t.squeeze(0).to('cpu')
            #
            #         if not os.path.exists(savedir):
            #             os.makedirs(savedir)
            #
            #         o_sk = F.to_pil_image(o_sk)
            #         o_t = F.to_pil_image((o_t + 1) / 2)
            #
            #         o_sk.save(os.path.join(savedir, name + 'o_sk.png'))
            #         o_t.save(os.path.join(savedir, name + 'o_t.png'))

            G_solver.zero_grad()
            o_sk, o_t = G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))
            i_vgg = torch.cat((t_t, o_t), dim=0)
            out_vgg = vgg_features(i_vgg)
            # o_sk = K(o_sk)
            # o_t = K(o_t)
            out_g = [o_sk, o_t]
            # g_loss = build_generator_loss(out_g, out_vgg, labels) + t_loss
            bce_loss = weighted_bce_loss(o_t,t_t)
            L_tv_loss = 0.5 * ((o_t[:,:,1:,:]-o_t[:,:,:-1,:]).abs().mean()+(o_t[:,:,:,1:]-o_t[:,:,:,:-1]).abs().mean())
            generator_loss = build_generator_loss(out_g, labels, out_vgg)
            g_loss = 5 *generator_loss  + 30 * bce_loss  + 30 * L_tv_loss
            # print(L_tv_loss,"    ",g_loss ,"  ",bce_loss, "   ",generator_loss)
            writer.add_scalar("G_Loss/g_loss", g_loss.item(), step)
            g_loss.backward()
            G_solver.step()

            # g_scheduler.step()


if __name__ == '__main__':
    main()
