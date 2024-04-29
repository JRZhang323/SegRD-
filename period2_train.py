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
from Seg_eval import Seg_evaluate
from RD_eval import evaluation_multi_proj
from model.Segmentation import Segmentation
from model.losses import focal_loss, l1_loss
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

    Seg = Segmentation(encoder, proj_layer, bn, decoder, outputS).cuda()

    # 读取保存参数
    checkpoint_class  = args.save_path + '/' + CLASS + '/' + 'res18_'+ CLASS +'.pth'

    ckp = torch.load(checkpoint_class, map_location='cpu')
    proj_layer.load_state_dict(ckp['proj'])
    bn.load_state_dict(ckp['bn'])
    decoder.load_state_dict(ckp['decoder'])
    outputS.load_state_dict(ckp['outputS'])

    # 布置梯度处理器
    seg_optimizer = torch.optim.SGD(
        [
            {"params": Seg.segmentation_net.res.parameters(), "lr": args.lr_res},
            {"params": Seg.segmentation_net.head.parameters(), "lr": args.lr_seghead},
        ],
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )

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
    # period 2
    best_seg_score = 0
    best_seg_step = 0
    best_seg_auroc_px = 0
    best_seg_auroc_sp = 0
    best_seg_aupro_px = 0

    # 重现最优次数的一阶段最终结果
    data_transform, gt_transform = get_data_transforms(args.image_size, args.image_size)
    test_path = args.mvtec_path + CLASS
    test_data = MVTecDataset_test(root=test_path, transform=data_transform, gt_transform=gt_transform)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False)

    auroc_px, auroc_sp, aupro_px = evaluation_multi_proj(encoder, proj_layer, bn, decoder, outputS, test_dataloader, device='cuda')
    print('Period1 results, Sample Auroc: {:.4f}, Pixel Auroc:{:.4f}, Pixel Aupro: {:.4f}'.format(
        auroc_sp,auroc_px,aupro_px))

    f = open('..\SegRRD_result.txt', 'a')
    f.writelines(
        '\n Period1 : best score of class: {}, Auroc sample: {:.4f}, Auroc pixel:{:.4f}, Pixel Aupro: {:.4f}'.format(
            CLASS, auroc_sp, auroc_px,  aupro_px))
    f.close()

    #进入二阶段训练。

    global_step = 0

    flag = True

    while flag:
        for _, sample_batched in enumerate(train_dataloader):

            seg_optimizer.zero_grad()

            img_origin = sample_batched["img_origin"].cuda()
            img_aug = sample_batched["img_aug"].cuda()
            mask = sample_batched["mask"].cuda()

            encoder.eval()
            bn.eval()
            proj_layer.eval()
            decoder.eval()
            outputS.eval()
            Seg.segmentation_net.train()

            output_segmentation = Seg(img_aug)

            mask = F.interpolate(
                mask,
                size=output_segmentation.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            mask = torch.where(
                mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
            )

            # Seg loss
            Focalloss = focal_loss(output_segmentation, mask, gamma=args.gamma)
            L1loss = l1_loss(output_segmentation, mask)
            Seg_loss = Focalloss + L1loss
            Seg_loss.backward()
            seg_optimizer.step()


            global_step += 1

            # 结果输出
            if global_step % args.log_per_steps == 0:
                
                seg_auroc_sp, seg_aupro_px, seg_auroc_px = Seg_evaluate(args, category, Seg, global_step)
                print(
                    'Epoch {}, Sample Auroc: {:.4f}, Pixel Auroc:{:.4f}, Pixel Aupro: {:.4f}'.format(global_step, seg_auroc_sp, seg_auroc_px, seg_aupro_px))

                if (seg_auroc_sp + seg_aupro_px + seg_auroc_px) / 3 > best_seg_score:
                    best_seg_score = (seg_auroc_sp + seg_aupro_px + seg_auroc_px) / 3

                    best_seg_auroc_sp = seg_auroc_sp
                    best_seg_auroc_px = seg_auroc_px
                    best_seg_aupro_px = seg_aupro_px
                    best_seg_step = global_step
               
                if global_step == args.period2_steps:
                    print(
                        'Period2 : Best score of class: {}, Auroc sample: {:.4f}, Auroc pixel:{:.4f}, Pixel Aupro: {:.4f}, best step: {}'.format(
                            CLASS, best_seg_auroc_sp, best_seg_auroc_px, best_seg_aupro_px, best_seg_step))

                    f = open('..\SegRRD_result.txt', 'a')
                    f.writelines(
                        '\n Period2 : Best score of class: {}, Auroc sample: {:.4f}, Auroc pixel:{:.4f}, Pixel Aupro: {:.4f}, best step: {} ,global step:{}'.format(
                            CLASS, best_seg_auroc_sp, best_seg_auroc_px, best_seg_aupro_px, best_seg_step, global_step))
                    f.close()

            if global_step >= args.period2_steps:
                flag = False
                break

 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--mvtec_path", type=str, default="/data1/zhangjr/dataset/MVtec/")
    parser.add_argument("--dtd_path", type=str, default="/data1/zhangjr/dataset/dtd/images/")
    parser.add_argument("--save_path", type=str, default="/data1/zhangjr/Final/SegRD++/save_path")
    parser.add_argument("--train_bs", type=int, default=16)
    parser.add_argument("--test_bs", type=int, default=1)
    parser.add_argument("--lr_res", type=float, default=0.1)
    parser.add_argument("--lr_seghead", type=float, default=0.01)
    parser.add_argument("--period2_steps", type=int, default=2000)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument("--log_per_steps", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=4)  # for focal loss
    parser.add_argument("--T", type=int, default=1)  
 
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
