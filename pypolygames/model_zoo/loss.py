import torch
from torch import nn
import torch.nn.functional as F

from typing import Tuple

def mcts_loss(
   self,
   model,
   batch,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    predicts = getattr(self, "predicts", 0)

    x = batch["s"]
    v = batch["v"]
    pi = batch["pi"]
    pi_mask = batch["pi_mask"]
    predict_pi = batch["predict_pi"] if predicts > 0 else None
    predict_pi_mask = batch["predict_pi_mask"] if predicts > 0 else None

    pi = pi.flatten(1)
    if predicts > 0:
        pred_v, pred_logit, pred_predict_logit = model._forward(x, return_logit=True)
    else:
        pred_v, pred_logit, *_ = model._forward(x, return_logit=True)

    pi_mask = pi_mask.view(pred_logit.shape);
    pred_logit = pred_logit * pi_mask - 400 * (1 - pi_mask)
    if predicts > 0:
        predict_pi_err = (F.mse_loss(pred_predict_logit, predict_pi, reduction="none") * predict_pi_mask).flatten(2).sum(2).flatten(1).mean(1)

    v_err = F.mse_loss(pred_v, v, reduction="none").squeeze(1)
    pred_log_pi = nn.functional.log_softmax(pred_logit.flatten(1), dim=1).view_as(pred_logit) * pi_mask
    pi_err = -(pred_log_pi * pi).sum(1)

    err = v_err * 1.5 + pi_err + (predict_pi_err * 0.1 if predicts > 0 else 0)

    return err.mean(), v_err.detach().mean(), pi_err.detach().mean(), (predict_pi_err.detach().mean() if predicts > 0 else None)


