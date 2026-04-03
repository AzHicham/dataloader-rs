from dataloader_rs import PyDataloader as DataLoader, PyDataset


class ToyDataset(PyDataset):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return {"x": index, "y": index * 2}


class ListDataset(PyDataset):
    def __init__(self, values):
        super().__init__()
        self._values = list(values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, idx):
        return self._values[idx]


def materialize(loader: DataLoader):
    return list(loader)
