import torch

def fed_avg(updates):
    """
    FedAvg: Weighted averaging of model updates.
    updates: list of tuples (state_dict, num_samples)
    """
    total_samples = sum(num_samples for _, num_samples in updates)
    avg_state_dict = {}

    for key in updates[0][0].keys():
        avg_state_dict[key] = torch.zeros_like(updates[0][0][key])

    for state, num_samples in updates:
        for k in avg_state_dict.keys():
            avg_state_dict[k] += state[k] * (num_samples / total_samples)

    return avg_state_dict
