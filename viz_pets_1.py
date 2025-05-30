import argparse, os, torch, torchvision
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path



# --------------------------------------------------------------------------- #
def save_grid(tensor, fname, nrow=4, is_mask=False):
    """
    Saves a BCHW tensor to *fname* as a grid.

    If *is_mask* is True the tensor contains class-IDs (long) and is scaled to 0-1.
    """
    if is_mask:                                            # class-IDs → greyscale
        grid = torchvision.utils.make_grid(tensor.float() / tensor.max(), nrow=nrow)
        grid = grid.expand(3, *grid.shape[1:])             # greyscale → 3-channel
    else:                                                  # already RGB float 0-1
        grid = torchvision.utils.make_grid(tensor, nrow=nrow)

    npimg = grid.cpu().numpy()
    plt.imsave(fname, np.transpose(npimg, (1, 2, 0)))


@torch.no_grad()
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- transforms (must match training!) ---------------------------------- #
    img_tf = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((64, 64), interpolation=v2.InterpolationMode.NEAREST),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])
    tgt_tf = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.long, scale=False),
        v2.Resize((64, 64), interpolation=v2.InterpolationMode.NEAREST),
    ])

    val_ds = OxfordPetsCustom(root=args.root,
                              split="test",
                              target_types="segmentation",
                              transform=img_tf,
                              target_transform=tgt_tf,
                              download=True)

    val_loader = torch.utils.data.DataLoader(val_ds,
                                             batch_size=args.n,
                                             shuffle=True,
                                             num_workers=2)

    # ----------------- load *whole* model object ---------------------------- #
    net = torch.load(args.ckpt, map_location=device)
    net = net.to(device).eval()

    # ----------------- one random batch ------------------------------------- #
    imgs, _ = next(iter(val_loader))
    imgs = imgs.to(device)

    logits = net(imgs)         # (B,3,H,W)
    preds = logits.argmax(1)   # (B,H,W) long

    # ----------------- write grids ------------------------------------------ #
    Path("img").mkdir(exist_ok=True)

    # un-normalise RGB for nicer viewing
    mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)
    imgs_vis = imgs * std + mean
    imgs_vis = torch.clamp(imgs_vis, 0, 1)

    save_grid(imgs_vis.cpu(), "img/val_inputs.png", nrow=4, is_mask=False)
    save_grid(preds.cpu(),    "img/val_preds.png",  nrow=4, is_mask=True)

    print("Done → img/val_inputs.png   &   img/val_preds.png")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Path to .pth file that contains the whole DeepSegmenter model")
    ap.add_argument("--root", default="data/oxford-iiit-pet",
                    help="Oxford-IIIT-Pet root directory (parent of 'images', 'annotations', …)")
    ap.add_argument("--n", type=int, default=8,
                    help="Number of validation samples to visualise (multiple of 4 works best)")
    args = ap.parse_args()
    main(args)