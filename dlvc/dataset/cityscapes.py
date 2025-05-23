from collections import namedtuple
from torchvision.datasets import Cityscapes

CityscapesLabels = namedtuple(
        "CityscapesLabels",
        ["name", "id", "color"],
    )
class CityscapesCustom(Cityscapes):
     classes_seg = [
        
        CityscapesLabels("road",  0,  (128, 64, 128)),
        CityscapesLabels("sidewalk", 1,  (244, 35, 232)),
        CityscapesLabels("building",  2,  (70, 70, 70)),
        CityscapesLabels("wall",  3,  (102, 102, 156)),
        CityscapesLabels("fence",  4, (190, 153, 153)),
        CityscapesLabels("pole",  5, (153, 153, 153)),
        CityscapesLabels("traffic light",  6,  (250, 170, 30)),
        CityscapesLabels("traffic sign",  7,  (220, 220, 0)),
        CityscapesLabels("vegetation",  8,  (107, 142, 35)),
        CityscapesLabels("terrain",  9,  (152, 251, 152)),
        CityscapesLabels("sky",  10,  (70, 130, 180)),
        CityscapesLabels("person",  11,  (220, 20, 60)),
        CityscapesLabels("rider",  12,  (255, 0, 0)),
        CityscapesLabels("car",  13,  (0, 0, 142)),
        CityscapesLabels("truck",  14,  (0, 0, 70)),
        CityscapesLabels("bus",  15,  (0, 60, 100)),
        CityscapesLabels("train",  16,  (0, 80, 100)),
        CityscapesLabels("motorcycle",  17,  (0, 0, 230)),
        CityscapesLabels("bicycle",  18,  (119, 11, 32))
    ]
     def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == "instance":
            return f"{mode}_instanceIds.png"
        elif target_type == "semantic":
            #return f"{mode}_labelIds.png"
            return f"{mode}_labelTrainIds.png"
       
        elif target_type == "color":
            return f"{mode}_color.png"
        else:
            return f"{mode}_polygons.json"