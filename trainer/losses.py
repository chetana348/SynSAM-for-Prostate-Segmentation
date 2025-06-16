import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

class EWC:
    """
    Elastic Weight Consolidation (EWC) loss for continual learning.
    Penalizes changes to parameters important for previous tasks.
    """
    def __init__(self, lambda_ewc=0.4, fisher_information=None, saved_params=None):
        self.lambda_ewc = lambda_ewc
        self.fisher_information = fisher_information if fisher_information else {}
        self.saved_parameters = saved_params if saved_params else {}

    def compute(self, current_parameters):
        """
        Compute the EWC penalty given the current model parameters.

        Args:
            current_parameters (Iterator[named_parameters]): The model's current parameters.

        Returns:
            torch.Tensor: The total EWC loss.
        """
        total_penalty = 0.0
        for param_name, param_value in current_parameters:
            # Retrieve Fisher and previous param value
            if param_name not in self.fisher_information or param_name not in self.saved_parameters:
                continue  
            fisher_score = self.fisher_information[param_name]
            reference_param = self.saved_parameters[param_name]
            
            # EWC loss: lambda/2 * sum(F * (θ - θ_old)^2)
            penalty = fisher_score * (param_value - reference_param).pow(2)
            total_penalty += 0.5 * self.lambda_ewc * penalty.sum()

        return total_penalty


class vEWC:
    """
    Variational EWC Regularization Loss for Continual Learning.
    Combines Fisher information and parameter importance.
    """
    def __init__(self, reg_strength=0.4, fisher_scores=None, prev_params=None, importance_scores=None):
        self.reg_strength = reg_strength
        self.past_tasks = list(fisher_scores.keys())[:-1]  # Exclude current task
        self.fisher_scores = fisher_scores if fisher_scores else {}
        self.reference_params = prev_params if prev_params else {}
        self.importance_scores = importance_scores if importance_scores else {}

    def compute(self, model_params):
        """
        Compute the vEWC loss term given the current model parameters.

        Args:
            model_params (Iterator[named_parameters]): The model's current parameter set.

        Returns:
            torch.Tensor: Scalar vEWC loss penalty.
        """
        penalty = 0.0
        for task in self.past_tasks:
            for param_name, current_value in model_params:
                fisher_val = self.fisher_scores[task][param_name]
                prev_val = self.reference_params[task][param_name]
                score_val = self.importance_scores[task][param_name]

                deviation = (current_value - prev_val).pow(2)
                weighted_penalty = (fisher_val + score_val) * deviation

                penalty += self.reg_strength * weighted_penalty.sum()

        return penalty


class DiceLoss2D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = 1.

    def forward(self, pred, target):
        target = target.squeeze(1)
        if self.num_classes == 1:
            pred = torch.sigmoid(pred)
            pred = pred.view(-1)
            target = target.view(-1)
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            loss = 1 -  (2. * intersection + self.smooth) / (union + self.smooth)
        else:
            pred = F.softmax(pred, dim=1)
            loss = 0
            for c in range(self.num_classes):
                pred_c = pred[:, c, :, :]
                target_c = (target == c).float()
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                loss += 1 -  (2. * intersection + self.smooth) / (union + self.smooth)
            loss /= self.num_classes
        return loss
