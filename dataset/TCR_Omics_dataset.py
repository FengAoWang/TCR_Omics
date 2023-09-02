from torch.utils.data import Dataset
import pandas as pd
import torch


class TCR_Omics_Dataset(Dataset):
    def __init__(self, TCR_tra_file_path, TCR_trb_file_path, Omics_file, TCR_Meta_info):
        super(TCR_Omics_Dataset, self).__init__()
        self.omics_embeddings = pd.read_csv(Omics_file).T
        self.tcr_tra_embeddings = pd.read_csv(TCR_tra_file_path, header=None) if TCR_trb_file_path else None
        self.tcr_trb_embeddings = pd.read_csv(TCR_trb_file_path, header=None) if TCR_trb_file_path else None
        self.tcr_meta_info = pd.read_csv(TCR_Meta_info)
        self.tcr_meta_info.fillna('null', inplace=True)

    def __len__(self):
        return self.omics_embeddings.shape[0]

    def __getitem__(self, item):

        omics_embedding = torch.Tensor(self.omics_embeddings.iloc[item])
        tcr_tra_embedding = torch.Tensor(self.tcr_tra_embeddings.iloc[item, 1:].values.tolist()) if self.tcr_trb_embeddings is not None else None
        tcr_trb_embedding = torch.Tensor(self.tcr_trb_embeddings.iloc[item, 1:].values.tolist()) if self.tcr_trb_embeddings is not None else None
        tcr_cell_label = self.tcr_meta_info.loc[item, ['cluster']].values.tolist()

        return [tcr_tra_embedding, omics_embedding, tcr_trb_embedding, tcr_cell_label]


