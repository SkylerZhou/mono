import torch
from torch import nn
import torch.nn.functional as F


from common import config as cfg
from .base_model import BaseModel
from .pretrain_model import PretrainModel
from sklearn.metrics import f1_score


class FinetuneModel(BaseModel):
    def __init__(self):
        super(FinetuneModel, self).__init__()
        self.pretrain = PretrainModel(pretrain=False)
        inplane = self.pretrain.d_model
        self.d_model = 256
        
        self.fc = nn.Sequential(
            nn.LayerNorm(inplane),
            nn.Dropout(0.1),
            nn.Linear(inplane, inplane * 4),
            nn.ReLU(),
            nn.Linear(inplane * 4, cfg.n_cls),
        )

    def forward(self, data):

        x = self.pretrain(data)
        x = self.fc(x[:, 0])

        out = {"pred": x}
        out.update(self._compute_loss(out, data))
        out.update(self._compute_metric(out, data))
        return out

    def _compute_loss(self, out, data):
        ce_loss = F.cross_entropy(out["pred"], data["perturb"])
        losses = {"ce_loss": ce_loss}
        losses["update_loss"] = ce_loss
        return losses

    def _compute_metric(self, out, data):
        def accuracy(output, target, topk=(1,)):
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []

            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0)
                res.append(correct_k.mul_(1.0 / batch_size))
            return res

        with torch.no_grad():
            metrics = {}
            topk = [1, 5, 10, 50, 100]
            acc = accuracy(out["pred"], data["perturb"], topk)
            for x, v in zip(topk, acc):
                metrics[f"top{x}_acc"] = v
            return metrics
