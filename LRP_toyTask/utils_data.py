import torch

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dataset,datasetf, labels):
        'Initialization'
        self.labels = labels
        self.dataset = dataset
        self.datasetf=datasetf

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        # Load data and get label
        X = self.dataset[index]
        Xf=self.datasetf[index]
        y = self.labels[index]

        return (X, Xf,y)