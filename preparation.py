import json
import pprint

import numpy
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset, RandomSampler


def metrics(pred_flat, labels_flat):
    """Function to various metrics of our predictions vs labels"""
    print(json.dumps(classification_report(labels_flat, pred_flat, output_dict=True)))
    print("\n**** Classification report")
    print(classification_report(labels_flat, pred_flat))
    macro_average_accuracy = 0
    weighted_average_accuracy = 0

    for i in range(4):
        correct_sum = 0
        for elem1, elem2 in zip(pred_flat, labels_flat):
            if elem1 == elem2 and elem1 == i:
                correct_sum += 1

        class_accuracy = correct_sum / len(labels_flat)
        weighted_average_accuracy += class_accuracy * (numpy.sum(labels_flat == i) / len(labels_flat))
        macro_average_accuracy += class_accuracy * 0.25

        print(f"Accuracy class {i}: {class_accuracy}")
    print(f"Macro Average accuracy class: {macro_average_accuracy}")
    print(f"Weighted Average accuracy class: {weighted_average_accuracy}")
    print("\n***Confusion matrix")
    pprint.pprint(confusion_matrix(pred_flat, labels_flat))

    return numpy.sum(pred_flat == labels_flat) / len(labels_flat), classification_report(labels_flat, pred_flat, output_dict=True)["macro avg"]["f1-score"]


def flat_accuracy(preds, labs):
    """Function to calculate the accuracy of our predictions vs labels"""
    pred_flat = numpy.argmax(preds, axis=1).flatten()
    labels_flat = labs.flatten()
    return numpy.sum(pred_flat == labels_flat) / len(labels_flat)


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
