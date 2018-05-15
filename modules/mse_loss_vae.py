from torch import nn
import torch


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse_loss = nn.MSELoss(size_average=False)

    def forward(self, out, target, mu, logvar):
        MSE = self.mse_loss(out, target)

        # see Appendix B from VAE paper:    https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        print('Total: %.4f, MSE: %.4f, KLD: %.4f' %(MSE + KLD, MSE, KLD))
        return MSE, KLD
