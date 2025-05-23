from collections import namedtuple
from torchvision.datasets import OxfordIIITPet

OxfordpetsLabels = namedtuple(
        "OxfordpetsLabels",
        ["name", "id"],
    )



class OxfordPetsCustom(OxfordIIITPet):
    classes_seg=[
        OxfordpetsLabels("pet",  0,),
        OxfordpetsLabels("background",  1 ),
        OxfordpetsLabels("border",  2 )
    ]