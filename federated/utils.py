from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold

def split_dataset(dataset, n_clients=3):
    """
    Splits dataset into n_clients stratified subsets.
    Returns list of Subset objects for each client.
    """
    labels = dataset.metadata['target'].values
    skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=42)
    subsets = []

    for _, idx in skf.split(range(len(dataset)), labels):
        subsets.append(Subset(dataset, idx))

    return subsets
