import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score


def precision(predictions, labels):
    return precision_score(labels, predictions, average="macro")


def recall(predictions, labels):
    return recall_score(labels, predictions, average="macro")


def f1(predictions, labels):
    return f1_score(labels, predictions, average="macro")


def attention_mask(input_data) -> list:
    attention = []
    for sent in input_data:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        attention.append(att_mask)
    return attention


def prepare_dataset(inputs, labs, attentions, batch_size) -> DataLoader:
    # Create the DataLoader for our training set.
    train_data = TensorDataset(torch.tensor(inputs), torch.tensor(attentions), torch.tensor(labs))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return train_dataloader
