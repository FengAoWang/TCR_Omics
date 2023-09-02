import sys
import torch
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F


def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def contrastive_Loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency"""
    _, k = x_out.size() # k is the feature dimension
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_j) - lamb * torch.log(p_i))

    loss = loss.sum()

    return loss


def KL_loss(mu, logvar, beta, c=0.0):
    # KL divergence loss
    KLD_1 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    ############################

    # KLD_2 = torch.distributions.kl.kl_divergence(z_dist, prior_dist)
    # KLD_2 = KLD_2.sum(1).mean()
    # KLD_2 = beta * (KLD_2 - c).abs()
    return beta * KLD_1


def KL_divergence(mu1, mu2, log_sigma1, log_sigma2):
    p = Normal(mu1, torch.exp(log_sigma1))
    q = Normal(mu2, torch.exp(log_sigma2))

    # 计算KL损失
    kl_loss = kl_divergence(p, q).mean()

    # sigma1 = torch.exp(log_sigma1)
    # sigma2 = torch.exp(log_sigma2)
    #
    # sigma1_inv = torch.inverse(sigma1)
    # sigma2_det = torch.det(sigma2)
    # sigma1_det = torch.det(sigma1)
    # mu_diff = mu2 - mu1
    #
    # tr_term = torch.trace(torch.matmul(sigma1_inv, sigma2))
    # quad_term = torch.matmul(torch.matmul(mu_diff.T, sigma1_inv), mu_diff)
    # logdet_term = torch.log(sigma2_det / sigma1_det)
    #
    # kl_div = - 0.5 * (tr_term + quad_term - mu1.shape[0] - logdet_term)

    return kl_loss


def reconstruction_loss(recon_x, x, recon_param, dist):
    BCE = torch.nn.BCEWithLogitsLoss(reduction="mean")
    batch_size = recon_x.shape[0]
    if dist == 'bernoulli':
        recons_loss = BCE(recon_x, x) / batch_size
    elif dist == 'gaussian':
        x_recons = recon_x
        recons_loss = F.mse_loss(x_recons, x, reduction='mean') / batch_size

    elif dist == 'F2norm':
        recons_loss = torch.norm(recon_x-x, p=2)
    else:
        raise AttributeError("invalid dist")

    return recon_param * recons_loss
