import random
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.loss_function import KL_loss, reconstruction_loss, KL_divergence
from functools import reduce


def reparameterize(mean, logvar):
    std = torch.exp(logvar / 2)
    epsilon = torch.randn_like(std).cuda()
    return epsilon * std + mean


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def un_dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        un_dfs_freeze(child)





class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim[0]),
                                     nn.BatchNorm1d(hidden_dim[0]),
                                     # nn.Dropout(0.2),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[0], hidden_dim[1]),
                                     nn.BatchNorm1d(hidden_dim[1]),
                                     # nn.Dropout(0.2),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[1], hidden_dim[2]),
                                     nn.BatchNorm1d(hidden_dim[2]),
                                     # nn.Dropout(0.2),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[2], latent_dim),
                                     nn.BatchNorm1d(latent_dim),
                                     # nn.Dropout(0.2),
                                     nn.ReLU())

        self.mu_predictor = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                          nn.ReLU()
                                          )
        self.log_var_predictor = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                               nn.ReLU()
                                               )

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    def decode(self, latent_z):
        cross_recon_x = self.decoder(latent_z)
        return cross_recon_x

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu_predictor(x)
        log_var = self.log_var_predictor(x)
        latent_z = self.reparameterize(mu, log_var)
        # recon_x = self.decoder(latent_z)
        return latent_z, mu, log_var


class decoder(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dim):
        super(decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim[2]),
                                     nn.BatchNorm1d(hidden_dim[2]),
                                     # nn.Dropout(0.2),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[2], hidden_dim[1]),
                                     nn.BatchNorm1d(hidden_dim[1]),
                                     # nn.Dropout(0.2),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[1], hidden_dim[0]),
                                     nn.BatchNorm1d(hidden_dim[0]),
                                     # nn.Dropout(0.2),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[0], input_dim),
                                     )

    def forward(self, latent_z):
        return self.decoder(latent_z)


class MLP_block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_block, self).__init__()
        self.mlp_block = nn.Sequential(nn.Linear(input_dim, output_dim),
                                       nn.BatchNorm1d(output_dim),
                                       # nn.Dropout(0.2),
                                       nn.ReLU())

    def forward(self, input_x):
        return self.mlp_block(input_x)


class MLP_encoder(nn.Module):
    def __init__(self, model_dim, hidden_dim, latent_dim):
        super(MLP_encoder, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(model_dim, hidden_dim[0]),
                                 nn.BatchNorm1d(hidden_dim[0]),
                                 nn.ReLU())
        self.encoders = nn.ModuleList(
            [MLP_block(hidden_dim[i], hidden_dim[i + 1]) for i in range(0, len(hidden_dim) - 1)])
        self.fc2 = nn.Linear(hidden_dim[-1], latent_dim)

    def forward(self, input_x):
        x = self.fc1(input_x)
        for mlp_block in self.encoders:
            x = mlp_block(x)
        x = self.fc2(x)
        return x


class TCR_model(nn.Module):
    def __init__(self, modal_num, modal_dim, latent_dim, hidden_dim, pretrain=False):
        super(TCR_model, self).__init__()

        self.k = modal_num
        self.encoders = nn.ModuleList([encoder(modal_dim[i], latent_dim, hidden_dim) for i in range(self.k)])

        self.decoders = nn.ModuleList([decoder(latent_dim, modal_dim[i], hidden_dim) for i in range(self.k)])

        self.omic_to_tra_encoder = MLP_encoder(latent_dim, [latent_dim, latent_dim], latent_dim)
        self.omic_to_trb_encoder = MLP_encoder(latent_dim, [latent_dim, latent_dim], latent_dim)

        #   modal align
        self.discriminator = nn.Sequential(nn.Linear(latent_dim, 16),
                                           nn.BatchNorm1d(16),
                                           nn.ReLU(),
                                           nn.Linear(16, modal_num))

        #   infer modal and real modal align
        self.infer_discriminator = nn.ModuleList(nn.Sequential(nn.Linear(latent_dim, 16),
                                                               nn.BatchNorm1d(16),
                                                               nn.ReLU(),
                                                               nn.Linear(16, 2))
                                                 for i in range(self.k))

        if pretrain:
            dfs_freeze(self.encoders)

    #   incomplete omics input
    def forward(self, input_x, input_label):
        output = [0 for i in range(self.k)]
        for j in input_label:
            output[j] = self.encoders[j](input_x[j])
        self_elbo_loss = self.self_elbo(input_x, output, input_label)
        cross_infer_loss = self.cross_infer_loss(output, input_label)
        contrastive_loss = self.generate_contrastive_loss(output, input_label)
        return self_elbo_loss, cross_infer_loss, contrastive_loss

    def self_elbo(self, input_data, input_x, input_label):
        self_vae_elbo = 0
        for i in input_label:
            latent_z, mu, log_var = input_x[i]
            reconstruct_omic = self.decoders[i](latent_z)
            self_vae_elbo += 0.001 * KL_loss(mu, log_var, 1.0) + reconstruction_loss(input_data[i], reconstruct_omic, 1.0, 'gaussian')
        return self_vae_elbo

    def cross_infer_loss(self, input_x, input_label):
        infer_loss = 0
        cross_reconstruct_loss = 0
        if 0 in input_label:
            omics_to_tra = self.omic_to_tra_encoder(input_x[1][1])
            infer_loss += reconstruction_loss(input_x[0][1], omics_to_tra, 1.0, 'gaussian')
        if 2 in input_label:
            omics_to_trb = self.omic_to_trb_encoder(input_x[1][1])
            infer_loss += reconstruction_loss(input_x[2][1], omics_to_trb, 1.0, 'gaussian')

        return infer_loss / len(input_label)

    def adversarial_loss(self, batch_size, output, omics, mask_k):
        dsc_loss = 0
        values = list(omics.values())
        for i in range(self.k):
            if (i in values) and (i != mask_k):
                latent_z, mu, log_var = output[i][i]
                shared_fe = self.share_encoder(mu)

                real_modal = (torch.tensor([i for j in range(batch_size)])).cuda()
                pred_modal = self.discriminator(shared_fe)
                # print(i, pred_modal)
                dsc_loss += F.cross_entropy(pred_modal, real_modal, reduction='none')

        dsc_loss = dsc_loss.sum(0) / (self.k * batch_size)
        return dsc_loss

    def get_embedding(self, input_x, input_label):
        modality_embedding = [torch.Tensor([]) for i in range(self.k)]
        for i in input_label:
            modality_embedding[i] = self.encoders[i](input_x[i])[0]
        if 0 not in input_label:
            modality_embedding[0] = self.omic_to_tra_encoder(modality_embedding[1])
        if 2 not in input_label:
            modality_embedding[2] = self.omic_to_trb_encoder(modality_embedding[1])
        multi_representation = torch.concat(modality_embedding, dim=1)
        return multi_representation

    def generate_contrastive_loss(self, input_x, input_label):
        embedding_list = [input_x[0][1], input_x[1][1], input_x[2][1]]
        modality_label = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        contrastive_loss = self.contrastive_loss(embedding_list, modality_label)
        return contrastive_loss

    @staticmethod
    def contrastive_loss(outputs, labels, margin=1.0):
        """
        """

        total_loss = 0.0
        num_modalities = len(outputs)
        batch_size = outputs[0].shape[0]

        # computing cross modal align loss
        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                euclidean_distance = F.pairwise_distance(outputs[i], outputs[j])
                loss_contrastive = torch.mean((1 - labels[i, j]) * torch.pow(euclidean_distance, 2) +
                                              (labels[i, j]) * torch.pow(
                    torch.clamp(margin - euclidean_distance, min=0.0), 2))
                total_loss += loss_contrastive

        # computing the average loss
        return total_loss / 3
