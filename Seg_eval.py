import argparse
from torch.utils.data import DataLoader
from torchmetrics import AUROC, AveragePrecision
from model.metrics import AUPRO, IAPS
import warnings
import torch
from torch.nn import functional as F
from data.mvtec_period2_dataset import MVTecDataset
from model.Segmentation import Segmentation
from model.ProjLayer import MultiProjectionLayer
from model.ReviewKD import ReviewKD
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50



# Period 2 : Seg的inference设计(Seg_evaluate)

warnings.filterwarnings("ignore")

def Seg_evaluate(args, category, model, global_step):
    model.eval()
    with torch.no_grad():
        test_dataset = MVTecDataset(
            is_train=False,
            mvtec_dir=args.mvtec_path + category + "/test/",
            resize_shape=[256, 256],
            normalize_mean=[0.485, 0.456, 0.406],
            normalize_std=[0.229, 0.224, 0.225],
            )

        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=args.test_bs, 
            shuffle=False
        )

        seg_AUPRO = AUPRO().cuda()
        seg_AUROC = AUROC().cuda()
        seg_detect_AUROC = AUROC().cuda()

        for _, sample_batched in enumerate(test_dataloader):
            img = sample_batched["img"].cuda()
            mask = sample_batched["mask"].to(torch.int64).cuda()

            output_segmentation = model(img)

            output_segmentation = F.interpolate(
                output_segmentation,
                size=mask.size()[2:],
                mode="bilinear",
                align_corners=False,
            )

            mask_sample = torch.max(mask.view(mask.size(0), -1), dim=1)[0]
            output_segmentation_sample, _ = torch.sort(
                output_segmentation.view(output_segmentation.size(0), -1),
                dim=1,
                descending=True,
            )
            output_segmentation_sample = torch.mean(
                output_segmentation_sample[:, : args.T], dim=1
            )

            seg_AUPRO.update(output_segmentation, mask)
            seg_AUROC.update(output_segmentation.flatten(), mask.flatten())
            seg_detect_AUROC.update(output_segmentation_sample, mask_sample)

        aupro_seg,  auc_seg, auc_detect_seg = (
            seg_AUPRO.compute(),
            seg_AUROC.compute(),
            seg_detect_AUROC.compute(),
        )


        print("Eval at step", global_step)
        print("================================")
        print("pixel_AUC:", round(float(auc_seg), 4))
        print("PRO:", round(float(aupro_seg), 4))
        print("image_AUC:", round(float(auc_detect_seg), 4))
        print()

        sample_AUROC = auc_detect_seg
        pixel_AUPRO = aupro_seg
        pixel_AUROC = auc_seg
        seg_AUPRO.reset()
        seg_AUROC.reset()
        seg_detect_AUROC.reset()
        return sample_AUROC,pixel_AUPRO,pixel_AUROC


