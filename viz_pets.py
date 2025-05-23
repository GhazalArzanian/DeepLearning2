import os
import torch
import torchvision
import torchvision.transforms.v2 as v2
import os
import matplotlib.pyplot as plt
import numpy as np
os.chdir(os.getcwd() + "change to your working directory if necessary")


from train import OxfordPetsCustom


def imshow(img, filename='img/test.png'):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imsave(filename,np.transpose(npimg, (1, 2, 0)))


if __name__ == '__main__': 

    train_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])

    train_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])

    train_data = OxfordPetsCustom(root="change to the path were your dataset is stored", 
                            split="trainval",
                            target_types='segmentation', 
                            transform=train_transform,
                            target_transform=train_transform2,
                            download=True)
    train_data_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=8,
                                            shuffle=False,
                                            num_workers=2)

    # get some random training images
    dataiter = iter(train_data_loader)
    images, labels = next(dataiter)
    images_plot = torchvision.utils.make_grid(images, nrow=4)
    labels_plot = torchvision.utils.make_grid((labels-1)/2, nrow=4)#.to(torch.uint8)

    # show/plot images
    imshow(images_plot, filename="img/input_test_pets.png")
    imshow(labels_plot,filename="img/seg_mask_test_pets.png")

