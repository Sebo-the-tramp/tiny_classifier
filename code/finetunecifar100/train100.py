"""
This code runs the image classification training loop. It tries to support as much
as timm's functionalities as possible.

For compatibility the prefetcher, re_split and JSDLoss are disabled.

To run the training script, use this command:
    python train.py cfg/phinet.py

You can change the configuration or override the parameters as you see fit.

Authors:
    - Francesco Paissan, 2023
"""

import torch
import torch.nn as nn
from prepare_data import create_loaders, setup_mixup
from torchinfo import summary
from timm.loss import (
    BinaryCrossEntropy,
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
)

import micromind as mm
from micromind.networks import PhiNet, XiNet
from micromind.utils import parse_configuration
import sys

import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
elif torch.backends.mps.is_available: 
    device = torch.device("mps")
    print("Running on the MPS")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

clustering_mapping = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2, 30: 3, 31: 3, 32: 3, 33: 3, 34: 3, 35: 3, 36: 3, 37: 3, 38: 3, 39: 3, 40: 4, 41: 4, 42: 4, 43: 4, 44: 4, 45: 4, 46: 4, 47: 4, 48: 4, 49: 4, 50: 5, 51: 5, 52: 5, 53: 5, 54: 5, 55: 5, 56: 5, 57: 5, 58: 5, 59: 5, 60: 6, 61: 6, 62: 6, 63: 6, 64: 6, 65: 6, 66: 6, 67: 6, 68: 6, 69: 6, 70: 7, 71: 7, 72: 7, 73: 7, 74: 7, 75: 7, 76: 7, 77: 7, 78: 7, 79: 7, 80: 8, 81: 8, 82: 8, 83: 8, 84: 8, 85: 8, 86: 8, 87: 8, 88: 8, 89: 8, 90: 9, 91: 9, 92: 9, 93: 9, 94: 9, 95: 9, 96: 9, 97: 9, 98: 9, 99: 9}

def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class ImageClassification(mm.MicroMind):
    """Implements an image classification class. Provides support
    for timm augmentation and loss functions."""

    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams, *args, **kwargs)

        if hparams.model == "phinet":
            self.modules["feature_extractor"] = PhiNet(
                input_shape=hparams.input_shape,
                alpha=hparams.alpha,
                num_layers=hparams.num_layers,
                beta=hparams.beta,
                t_zero=hparams.t_zero,
                compatibility=False,
                divisor=hparams.divisor,
                downsampling_layers=hparams.downsampling_layers,
                return_layers=hparams.return_layers,
                # classification-specific
                include_top=False,
                #num_classes=hparams.num_classes,
            )

            # i need to find the dimensions of last layer
            # this only works if the include_top is False
            input_features = self.modules["feature_extractor"]._layers[-1]._layers[-1].num_features

            self.modifier_weights = torch.randn(10, input_features, requires_grad=True, device=device)
            
            # Taking away the classifier from pretrained model
            pretrained_dict = torch.load(hparams.ckpt_pretrained, map_location=device)
            
            self.modules['feature_extractor'].load_state_dict(pretrained_dict["feature_extractor"])
            for _, param in self.modules["feature_extractor"].named_parameters():
                param.requires_grad = False

            self.modules["flattener"] = nn.Sequential(                
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),                  
            )

            self.modules["classifier"] = nn.Sequential(
                nn.Linear(in_features=input_features, out_features=10)
            )
            self.modules["classifier"].load_state_dict(pretrained_dict["classifier"])
            for _, param in self.modules["classifier"].named_parameters():    
                param.requires_grad = False

        elif hparams.model == "xinet":
            self.modules["classifier"] = XiNet(
                input_shape=hparams.input_shape,
                alpha=hparams.alpha,
                gamma=hparams.gamma,
                num_layers=hparams.num_layers,
                return_layers=hparams.return_layers,
                # classification-specific
                include_top=True,
                num_classes=hparams.num_classes,
            )

        tot_params = 0
        for m in self.modules.values():
            temp = summary(m, verbose=0)
            tot_params += temp.total_params

        self.mixup_fn, _ = setup_mixup(hparams)

        print(f"Total parameters of model: {tot_params * 1e-6:.2f} M")

    def setup_criterion(self):
        """Setup of the loss function based on augmentation strategy."""
        # setup loss function
        if (
            self.hparams.mixup > 0
            or self.hparams.cutmix > 0.0
            or self.hparams.cutmix_minmax is not None
        ):
            print("Entering mixup")
            # smoothing is handled with mixup target transform which outputs sparse,
            # soft targets
            if self.hparams.bce_loss:
                train_loss_fn = BinaryCrossEntropy(
                    target_threshold=self.hparams.bce_target_thresh
                )
            else:
                train_loss_fn = SoftTargetCrossEntropy()
        elif self.hparams.smoothing:            
            if self.hparams.bce_loss:
                train_loss_fn = BinaryCrossEntropy(
                    smoothing=self.hparams.smoothing,
                    target_threshold=self.hparams.bce_target_thresh,
                )
            else:
                train_loss_fn = LabelSmoothingCrossEntropy(
                    smoothing=self.hparams.smoothing
                )
        else:
            train_loss_fn = nn.CrossEntropyLoss()

        return train_loss_fn

    def forward(self, batch):
        """Computes forward step for image classifier.

        Arguments
        ---------
        batch : List[torch.Tensor, torch.Tensor]
            Batch containing the images and labels.

        Returns
        -------
        Predicted class and augmented class. : Tuple[torch.Tensor, torch.Tensor]
        """
        img, target = batch
        if not self.hparams.prefetcher:
            img, target = img.to(self.device), target.to(self.device)
            if self.mixup_fn is not None:
                img, target = self.mixup_fn(img, target)

        feature_vector = self.modules["feature_extractor"](img)
        feature_vector = self.modules["flattener"](feature_vector)        
        x = self.modules["classifier"](feature_vector)
        indices_1 = torch.argmax(x, dim=1)

        indices_np = indices_1.to('cpu').numpy()
        test = batch[1].to('cpu').numpy()
        # print("Gorund truth", test)
        # print("Predicted", indices_1)
        test2 = np.array([clustering_mapping[y] for y in test])
        # print("Gorund truth Clusters", test2)
        # print("predicted", indices_np)
        # print(test2 == indices_np)
        # print((test2 == indices_np).sum()/len(indices_1))        
        # print(test2)
        # print(indices_np)
        # print(test2 == indices_np)
        # print((test2 == indices_np).sum(0))
        # print(torch.tensor(indices_1.tolist() == test2).sum()/len(indices_1))        

        # broadcasted_bias = self.modifier_weights.unsqueeze(0).expand(len(batch[0]),-1,-1)
        # out_expanded = x.unsqueeze(2)
        # result = out_expanded * broadcasted_bias
        # shifted = result.sum(dim=1)

        # print(indices_1)
        weights = torch.index_select(self.modifier_weights, 0, indices_1)  
        # print(weights)
        # print(weights.shape)
        # print(feature_vector.shape)
        shifted = weights * feature_vector

        last = self.modules["classifier"](shifted)
        softmax2 = DiffSoftmax(last, tau=1.0, hard=True, dim=1)

        # Calculate the ranges using vectorized operations
        start = indices_1 * 10
        end = (indices_1 + 1) * 10        

        output_tensor = torch.zeros(len(batch[0]), 100, device=device)
        to_add = torch.stack([torch.arange(start[i], end[i], device=device) for i in range(len(softmax2))])
        output_tensor.scatter_(1, to_add, softmax2)  

        # print(self.modifier_weights)
        # print(self.modules["classifier"][0].weight)

        # print(torch.argmax(output_tensor, dim=1), target)

        return (output_tensor, target)

    def compute_loss(self, pred, batch):
        """Sets up the loss function and computes the criterion.

        Arguments
        ---------
        pred : Tuple[torch.Tensor, torch.Tensor]
            Predicted class and augmented class.
        batch : List[torch.Tensor, torch.Tensor]
            Same batch as input to the forward step.

        Returns
        -------
        Cost function. : torch.Tensor
        """
        self.criterion = self.setup_criterion()

        #print(pred[0], pred[1])

        # taking it from pred because it might be augmented
        return self.criterion(pred[0], pred[1])

    def configure_optimizers(self):
        """Configures the optimizes and, eventually the learning rate scheduler."""
        opt = torch.optim.Adam([self.modifier_weights], lr=0.1)
        return opt


def top_k_accuracy(k=1):
    """
    Computes the top-K accuracy.

    Arguments
    ---------
    k : int
       Number of top elements to consider for accuracy.

    Returns
    -------
        accuracy : Callable
            Top-K accuracy.
    """

    def acc(pred, batch):
        if pred[1].ndim == 2:
            target = pred[1].argmax(1)
        else:
            target = pred[1]
        _, indices = torch.topk(pred[0], k, dim=1)
        correct = torch.sum(indices == target.view(-1, 1))
        accuracy = correct.item() / target.size(0)

        return torch.Tensor([accuracy]).to(pred[0].device)

    return acc


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please pass the configuration file to the script."
    hparams = parse_configuration(sys.argv[1])

    train_loader, val_loader = create_loaders(hparams, coarse=False)

    exp_folder = mm.utils.checkpointer.create_experiment_folder(
        hparams.output_folder, hparams.experiment_name
    )

    checkpointer = mm.utils.checkpointer.Checkpointer(
        exp_folder, hparams=hparams, key="loss"
    )

    mind = ImageClassification(hparams=hparams)

    top1 = mm.Metric("top1_acc", top_k_accuracy(k=1), eval_only=False)
    top5 = mm.Metric("top5_acc", top_k_accuracy(k=5), eval_only=False)

    mind.train(
        epochs=hparams.epochs,
        datasets={"train": train_loader, "val": val_loader},
        metrics=[top5, top1],
        checkpointer=checkpointer,
        debug=hparams.debug,
    )

    mind.test(datasets={"test": val_loader}, metrics=[top1, top5])