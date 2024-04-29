import argparse
import os
import warnings
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
import imgaug as iaa
from data.mvtec_period1_dataset import MVTecDataset_test, get_data_transforms
from data.mvtec_period2_dataset import MVTecDataset
from RD_eval import evaluation_multi_proj
from model.losses import Revisit_RDLoss
from model.ProjLayer import MultiProjectionLayer
from model.ReviewKD import ReviewKD, hcl_loss
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50


warnings.filterwarnings("ignore")

def setup_seed(seed):
    random.seed(seed)
    iaa.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args, category, rotate_90=False, random_rotate=0):
    if not os.path.exists(args.save_path + '/' + CLASS):
        os.makedirs(args.save_path + '/' + CLASS)

    save_model_path  = args.save_path + '/' + CLASS + '/' + 'res18_'+ CLASS +'.pth'

    # 布置组件
    encoder, bn = resnet18(pretrained=True)
    encoder = encoder.cuda()
    bn = bn.cuda()

    decoder = de_resnet18(pretrained=False)
    decoder = decoder.cuda()

    in_channels = [128, 64]
    out_channels = [128, 64]
    mid_channels = [256, 128]

    outputS = (ReviewKD(in_channels, out_channels, mid_channels))
    outputS = outputS.cuda()

    proj_layer = MultiProjectionLayer(base=64).cuda()

    proj_loss = Revisit_RDLoss()

    # 布置梯度处理器
    proj_optimizer = torch.optim.Adam(list(proj_layer.parameters()), lr=args.proj_lr, betas=(0.5, 0.999))

    distill_optimizer = torch.optim.Adam(
        list(decoder.parameters()) + list(bn.parameters()) + list(outputS.parameters()), lr=args.distill_lr,
        betas=(0.5, 0.999))

    # 布置训练数据集
    train_dataset = MVTecDataset(
        is_train=True,
        mvtec_dir=args.mvtec_path + category + "/train/good/",
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=args.dtd_path,
        rotate_90=rotate_90,
        random_rotate=random_rotate,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_bs,
        shuffle=True,
        drop_last=True,
    )

    # 布置结果输出相关
    # period 1 
    best_score = 0
    best_step = 0
    best_auroc_px = 0
    best_auroc_sp = 0
    best_aupro_px = 0

    global_step = 0

    flag = True

    while flag:
        for _, sample_batched in enumerate(train_dataloader):

            distill_optimizer.zero_grad()
            proj_optimizer.zero_grad()

            img_origin = sample_batched["img_origin"].cuda()
            img_aug = sample_batched["img_aug"].cuda()

            encoder.eval()
            bn.train()
            proj_layer.train()
            decoder.train()
            outputS.train()

            inputs = encoder(img_origin)
            inputs_noise = encoder(img_aug)

            (feature_space_noise, feature_space) = proj_layer(inputs, features_noise=inputs_noise)

            L_proj = proj_loss(inputs_noise, feature_space_noise, feature_space)

            outputs = decoder(bn(feature_space))  # bn(inputs))
            OUTPUTS = outputS(outputs)


            L_distill = hcl_loss(OUTPUTS, inputs)

            # RD loss
            RD_loss = L_distill + args.weight_proj * L_proj
            RD_loss.backward()
            proj_optimizer.step()
            distill_optimizer.step()

            global_step += 1

            # 结果输出
            if global_step % args.log_per_steps == 0:

                data_transform, gt_transform = get_data_transforms(args.image_size, args.image_size)
                test_path = args.mvtec_path + CLASS
                test_data = MVTecDataset_test(root=test_path, transform=data_transform, gt_transform=gt_transform)
                test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False)

                auroc_px, auroc_sp, aupro_px = evaluation_multi_proj(encoder, proj_layer, bn, decoder, outputS,
                                                                         test_dataloader, device='cuda')
                print(
                    'Epoch {}, Sample Auroc: {:.4f}, Pixel Auroc:{:.4f}, Pixel Aupro: {:.4f}'.format(global_step, auroc_sp, auroc_px, aupro_px))
                                                      
                if (auroc_px + auroc_sp + aupro_px) / 3 > best_score:
                    best_score = (auroc_px + auroc_sp + aupro_px) / 3

                    best_auroc_sp = auroc_sp
                    best_auroc_px = auroc_px
                    best_aupro_px = aupro_px
                    best_step = global_step

                    torch.save({'proj': proj_layer.state_dict(),
                                'decoder': decoder.state_dict(),
                                'bn':bn.state_dict(),
                                'outputS':outputS.state_dict()},
                                save_model_path)

                if global_step == args.period1_steps:
                    print(
                        'Period1 : Best score of class: {}, Auroc sample: {:.4f}, Auroc pixel:{:.4f}, Pixel Aupro: {:.4f}, best step: {}'.format(
                            CLASS, best_auroc_sp, best_auroc_px, best_aupro_px, best_step))
                    f = open('..\SegRRD_result.txt', 'a')
                    f.writelines(
                        '\n Period1 : Best score of class: {}, Auroc sample: {:.4f}, Auroc pixel:{:.4f}, Pixel Aupro: {:.4f}, best step: {}'.format(
                            CLASS, best_auroc_sp, best_auroc_px, best_aupro_px, best_step))
                    f.close()

                if global_step >= args.period1_steps:
                    flag = False
                    break

 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--mvtec_path", type=str, default="/data1/zhangjr/dataset/MVtec/")
    parser.add_argument("--dtd_path", type=str, default="/data1/zhangjr/dataset/dtd/images/")
    parser.add_argument("--save_path", type=str, default="/data1/zhangjr/Final/SegRD++/save_path")
    parser.add_argument('--proj_lr', default=0.001, type=float)
    parser.add_argument('--distill_lr', default=0.005, type=float)
    parser.add_argument('--weight_proj', default=0.2, type=float)
    parser.add_argument("--train_bs", type=int, default=16)
    parser.add_argument("--test_bs", type=int, default=1)
    parser.add_argument("--period1_steps", type=int, default=5000)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument("--log_per_steps", type=int, default=10)

    # classes是训练的类型
    args = parser.parse_args()

    no_rotation_category = [
        "leather",
        "cable",
        "capsule",
        "metal_nut",
        "transistor",
        "bottle"
    ]

    rotation_category = [
        "hazelnut",
        "carpet"
    ]

    slight_rotation_category = [
        "screw",
        "wood",
        "toothbrush",
        "pill",
        "grid",
        "tile",
        "zipper"
    ]


    classes = ['leather','cable','capsule','metal_nut','transistor','bottle','hazelnut','carpet','screw','wood','toothbrush','pill','grid','tile','zipper']

    with torch.cuda.device(args.gpu_id):
        for i in range(0, 15):
            setup_seed(111)
            CLASS = classes[i]
               
            if CLASS in no_rotation_category:
                print(CLASS)
                train(args, CLASS)

            if CLASS in rotation_category:
                print(CLASS)
                train(args, CLASS, rotate_90=True, random_rotate=5)

            if CLASS in slight_rotation_category:
                print(CLASS)
                train(args, CLASS, rotate_90=False, random_rotate=5)
