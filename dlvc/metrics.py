from abc import ABCMeta, abstractmethod
import torch

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass


class SegMetrics(PerformanceMeasure):
    '''
    Mean Intersection over Union.
    '''

    def __init__(self, classes):
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.num_classes = len(self.classes)
        
        self._cm = torch.zeros(self.num_classes, self.num_classes,
                               dtype=torch.int64)



    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''

        if prediction.ndim != 4:
            raise ValueError("prediction must have shape (B,C,H,W)")
        if target.ndim != 3:
            raise ValueError("target must have shape (B,H,W)")
        if prediction.shape[0] != target.shape[0] \
           or prediction.shape[2:] != target.shape[1:]:
            raise ValueError("prediction and target shape mismatch")

        # -------- logits → class-ids --------------------------------------
        pred_ids = prediction.argmax(dim=1).to(torch.int64)   # (B,H,W)

        # -------- ignore void label (255) -------------------------------
        mask = (target != 255)
        pred_ids = pred_ids[mask]
        tgt_ids  = target[mask].to(torch.int64)

        # -------- accumulate confusion-matrix -----------------------------
        # map (gt, pred) pairs to a 1-D index:  idx = gt * C + pred
        idx = tgt_ids * self.num_classes + pred_ids
        bincount = torch.bincount(idx,
                                  minlength=self.num_classes**2)

        self._cm += bincount.reshape(self.num_classes, self.num_classes)

   

    def __str__(self):
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        return f"mIoU: {self.mIoU():.4f}"
          

    
    def mIoU(self) -> float:
        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''
        if self._cm.sum() == 0:
            return 0.0

        cm = self._cm.to(torch.float32)
        tp = torch.diag(cm)
        fp = cm.sum(0) - tp          # predicted as class c but GT ≠ c
        fn = cm.sum(1) - tp          # GT class c but predicted ≠ c
        denom = tp + fp + fn

        # avoid division by zero – IoU for such a class counts as 0
        iou = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))

        return iou.mean().item()





