import sys
sys.path.append('/home/wfa/project/TCR_omics')
import torch
from dataset.TCR_Omics_dataset import TCR_Omics_Dataset
from torch.utils.data import DataLoader
from model.model import TCR_model
from tqdm import tqdm
import random
import numpy as np


def set_seed(seed):
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(66)

tra_file = '../data/RNATCREmbedding/TCREmbedding/bcc_paired_tra_embedding.csv'
trb_file = '../data/RNATCREmbedding/TCREmbedding/bcc_paired_trb_embedding.csv'
omics_file = '../data/RNATCREmbedding/CountEmbedding/geneformer_mean_output_BCC_RNA_Count_Paired.csv'
tcr_meta_info = '../data/RNATCREmbedding/CellMetaInfo/BCC_Cell_MetaInfo_Paired.csv'

TCR_dataset = TCR_Omics_Dataset(tra_file, trb_file, omics_file, tcr_meta_info)
TCR_dataloader = DataLoader(TCR_dataset, batch_size=128)

TCR_Model = TCR_model(3, [768, 256, 768], 16, [128, 64, 16]).cuda()
epochs = 20
optimizer = torch.optim.Adam(TCR_Model.parameters(), lr=0.0001)

for epoch in range(epochs):
    with tqdm(TCR_dataloader, unit='batch') as tepoch:
        total_loss = 0
        all_training_embedding = torch.Tensor([]).cuda()

        for batch, data in enumerate(tepoch):
            tepoch.set_description(f" Epoch {epoch}: ")
            input_data = []
            for i in range(len(data) - 1):
                input_data.append(data[i].cuda())
            cell_label = data[-1]
            # print(cell_label)
            input_label = [0, 1, 2]

            self_elbo_loss, cross_infer_loss, contrastive_loss = TCR_Model(input_data, input_label)
            # loss = self_elbo_loss + 0.1 * cross_infer_loss + 0.1 * contrastive_loss
            loss = self_elbo_loss + 0.1 * contrastive_loss

            TCR_embedding = TCR_Model.get_embedding(input_data, input_label)
            all_training_embedding = torch.concat((all_training_embedding, TCR_embedding), dim=0)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item(), self_elbo_loss=self_elbo_loss.item(), cross_infer_loss=cross_infer_loss.item())
        torch.save(all_training_embedding, f'training_process/training_embedding/TCR_embedding_fold{epoch}.pt')

        print(f"epoch {epoch} ", f"loss: {total_loss/len(TCR_dataloader)}")

