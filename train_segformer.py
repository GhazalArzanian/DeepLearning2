
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os

from dlvc.models.segformer import  SegFormer
from dlvc.models.segment_model import DeepSegmenter
from dlvc.dataset.cityscapes import CityscapesCustom
from dlvc.dataset.oxfordpets import OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.trainer import ImgSemSegTrainer


def train(args):

    train_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])

    train_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])#,
    
    val_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    val_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])

    if args.dataset == "oxford":
        train_data = OxfordPetsCustom(root="data", 
                                split="trainval",
                                target_types='segmentation', 
                                transform=train_transform,
                                target_transform=train_transform2,
                                download=True)

        val_data = OxfordPetsCustom(root="data", 
                                split="test",
                                target_types='segmentation', 
                                transform=val_transform,
                                target_transform=val_transform2,
                                download=True)
    if args.dataset == "city":
        train_data = CityscapesCustom(root="data/cityscapes_assg2", 
                                split="train",
                                mode="fine",
                                target_type='semantic', 
                                transform=train_transform,
                                target_transform=train_transform2)
        val_data = CityscapesCustom(root="data/cityscapes_assg2", 
                                split="val",
                                mode="fine",
                                target_type='semantic', 
                                transform=val_transform,
                                target_transform=val_transform2)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(train_data.classes_seg)
    backbone = SegFormer(num_classes=num_classes)
    model = DeepSegmenter(backbone)
    # If you are in the fine-tuning phase:
    if args.dataset == 'oxford' and args.pretrained is not None:
        full_state = torch.load(args.pretrained, map_location='cpu')
        if any(k.startswith('net.encoder.') for k in full_state):
            prefix = 'net.encoder.'
        elif any(k.startswith('encoder.') for k in full_state):
            prefix = 'encoder.'
        else:
            raise RuntimeError("no encoder weights found")

        enc_state = {k.replace(prefix, ''): v
                    for k, v in full_state.items() if k.startswith(prefix)}

        missing, unexpected = model.net.encoder.load_state_dict(enc_state, strict=False)
        print(f"{len(enc_state) - len(missing)} tensors loaded, "
            f"{len(missing)} missing, {len(unexpected)} unexpected")

        if args.freeze_encoder:
            model.net.encoder.requires_grad_(False)

    model.to(device)

    lr = 0.0001
    params_to_opt = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=lr, amsgrad=True)


    ignore_val = 255 if args.dataset == "city" else -100

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_val)
    
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2 # for 

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    trainer = ImgSemSegTrainer(model, 
                    optimizer,
                    loss_fn,
                    lr_scheduler,
                    train_metric,
                    val_metric,
                    train_data,
                    val_data,
                    device,
                    args.num_epochs, 
                    model_save_dir,
                    batch_size=64,
                    val_frequency = val_frequency)
    trainer.train()
    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    trainer.dispose() 

    if args.model_out is not None:
        torch.save(model.state_dict(), args.model_out)
        print(f'\nModel saved ➜  {args.model_out}')

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str,
                      help='index of which GPU to use')
    
    
    args.add_argument('--num_epochs', type=int, default=31)
    args.add_argument('--dataset', choices=['oxford', 'city'],
                  default='oxford', help='data set')
    args.add_argument('--pretrained', type=str, default=None,
                  help='path to encoder weights')
    args.add_argument('--freeze_encoder', action='store_true',
                  help='freeze encoder during finetune')
    args.add_argument('--model_out', type=str, default=None,
                  help='save model with custom name')
    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0

    train(args)