import torch
import torch.nn as nn
import torch.nn.functional as F

class RNDModule(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.target = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        # freeze target
        for p in self.target.parameters():
            p.requires_grad = False

    def bonus(self, obs):
        with torch.no_grad():
            target_feat = self.target(obs)
        pred_feat = self.predictor(obs)
        return F.mse_loss(pred_feat, target_feat, reduction='none').mean(-1)
