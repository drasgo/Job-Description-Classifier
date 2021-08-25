from typing import Tuple, Any
import os

import numpy
import pandas as pd
import csv
import torch
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import time
import json
import random
import pprint
import datetime
import matplotlib.pyplot as plt

MAX_LEN = 264
batch_size = 32
epochs = 4
device = "cuda" if torch.cuda.is_available() else "cpu"


def plot(loss_values):
    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    plt.plot(loss_values, 'b-o')

    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labs):
    """Function to calculate the accuracy of our predictions vs labels"""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labs.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def encode_vector(original_input, token) -> list:
    encoded = []
    for tweet in original_input:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = token.encode(
            tweet,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        )
        encoded.append(encoded_sent)
    return encoded


def attention_mask(input_data) -> list:
    attention = []
    for sent in input_data:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        attention.append(att_mask)
    return attention


def prepare_dataset(inputs, labs, attentions) -> Tuple[DataLoader, DataLoader]:
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(inputs, labs,
                                                                                        random_state=2018,
                                                                                        test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attentions, labs,
                                                           random_state=2018, test_size=0.1)

    # Create the DataLoader for our training set.
    train_data = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_masks), torch.tensor(train_labels))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(torch.tensor(validation_inputs), torch.tensor(validation_masks),
                                    torch.tensor(validation_labels))
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader


def validation(nn, validation_dataloader, dev):
    print("")
    print("Running Validation...")
    t0 = time.time()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    nn.eval()

    for batch in validation_dataloader:
        batch = tuple(t.to(dev) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = nn(b_input_ids,
                         token_type_ids=None,
                         attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy

        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))



def training(nn, train_dataloader, validation_dataloader, learn_rate, dev="cuda") -> Tuple[Any, list]:
    loss_values = []
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    optimizer = AdamW(nn.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_loss = 0
        nn.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(dev)
            b_input_mask = batch[1].to(dev)
            b_labels = batch[2].to(dev)
            nn.zero_grad()

            outputs = nn(b_input_ids,
                         token_type_ids=None,
                         attention_mask=b_input_mask,
                         labels=b_labels)

            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(nn.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        validation(nn, validation_dataloader, device)
        print("Training Done!")

    return nn, loss_values


def testing(prediction_dataloader, dev="cuda"):
    model.eval()
    predictions, true_labels = [], []

    for batch in prediction_dataloader:
        batch = tuple(t.to(dev) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions += np.argmax(logits, axis=1).tolist()
        true_labels += label_ids.tolist()

    _, f1_macro_avg = metrics(numpy.array(predictions), numpy.array(true_labels))
    print(f"\n\n******\nF1 MACRO AVERAGE is: {f1_macro_avg}")
    print('    DONE.')
    return f1_macro_avg


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
        weighted_average_accuracy += class_accuracy * (np.sum(labels_flat == i) / len(labels_flat))
        macro_average_accuracy += class_accuracy * 0.25

        print(f"Accuracy class {i}: {class_accuracy}")
    print(f"Macro Average accuracy class: {macro_average_accuracy}")
    print(f"Weighted Average accuracy class: {weighted_average_accuracy}")
    print("\n***Confusion matrix")
    pprint.pprint(confusion_matrix(pred_flat, labels_flat))

    return np.sum(pred_flat == labels_flat) / len(labels_flat), classification_report(labels_flat, pred_flat, output_dict=True)["macro avg"]["f1-score"]


def test_performance(test_tweets, test_labels):
    batch = 32
    test_vector = encode_vector(test_tweets, tokenizer)
    test_vector = pad_sequences(test_vector, maxlen=MAX_LEN, dtype="long",
                                value=0, truncating="post", padding="post")

    test_attention_mask = attention_mask(test_vector)

    prediction_inputs = torch.tensor(test_vector)
    prediction_masks = torch.tensor(test_attention_mask)
    prediction_labels = torch.tensor(test_labels)

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch)
    print("Starting testing")
    return testing(prediction_dataloader)


def save_model(mod, name):
    output_dir = f'./model_save_{name}/'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = mod.module if hasattr(mod, 'module') else mod  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    training_phase = None
    untrained_bert = False

    if training_phase is True:
        learning_rates = [1e-3, 5e-4, 1e-4, 5e-5, 3e-5, 2e-5]
        f1_scores = {}
        for learn in learning_rates:
            print(f"Starting training with learning rate {learn}")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
                num_labels=4,  # The number of output labels--4 for multi-class classification.
                output_attentions=False,  # Whether the model returns attentions weights.
                output_hidden_states=False,  # Whether the model returns all hidden-states.
            )
            model.cuda()

            df = pd.read_csv('datasets/train/SemEval2018-T3-train-taskB.txt', delimiter="\t", quoting=csv.QUOTE_NONE)
            tweets = df["Tweet text"].values
            labels = df["Label"].values

            input_ids = encode_vector(tweets, tokenizer)
            input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                                      value=0, truncating="post", padding="post")

            attention_masks = attention_mask(input_ids)
            train_dataset, validation_dataset = prepare_dataset(input_ids, labels, attention_masks)
            total_steps = len(train_dataset) * epochs

            model, losses = training(model, train_dataset, validation_dataset, learn, device)
            save_model(model, str(learn))
            plot(losses)

            test = pd.read_csv("datasets/goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt", delimiter="\t",
                               quoting=csv.QUOTE_NONE)
            tweets = test["Tweet text"].values
            labels = test["Label"].values

            # Load a trained model and vocabulary that you have fine-tuned
            model = BertForSequenceClassification.from_pretrained("model/")
            tokenizer = BertTokenizer.from_pretrained("model/")
            print("Model loaded")
            # Copy the model to the GPU.
            model.to(device)
            f1_scores[learn] = test_performance(tweets, labels)

    elif training_phase is False:
        test = pd.read_csv("datasets/goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt", delimiter="\t", quoting=csv.QUOTE_NONE)
        tweets = test["Tweet text"].values
        labels = test["Label"].values

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained("model/")
        tokenizer = BertTokenizer.from_pretrained("model/")
        print("Model loaded")
        # Copy the model to the GPU.
        model.to(device)
        test_performance(tweets, labels)

    else:
        print("Checking similarities")
        if untrained_bert is True:
            MODEL_NAME = 'bert-base-uncased'
            # Use the bert tokenizer\n",

        else:
            MODEL_NAME = "model/"

        config = BertConfig.from_pretrained(MODEL_NAME, output_hidden_states=True)
        model = BertModel.from_pretrained(MODEL_NAME, config=config)
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        test = pd.read_csv("datasets/goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt", delimiter="\t",
                           quoting=csv.QUOTE_NONE)
        tweets = test["Tweet text"].values
        labels = test["Label"].values
        sentence_vectors = []

        for index in range(20):
            print(f"Tweet nr. {index+1}: {tweets[index]}. Label: {labels[index]}")

            tweet = tweets[index]
            tokens = [tokenizer.cls_token] + tokenizer.tokenize(tweet) + [tokenizer.sep_token]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            tokens_tensor = torch.tensor(token_ids).unsqueeze(0)
            model.eval()  # turn off dropout layers
            output = model(tokens_tensor)
            layers = output.hidden_states
            layer = 12
            sentence_vector = layers[layer][0].detach().numpy()
            sentence_vectors.append(sentence_vector[0])

        similarity_matrix = cosine_similarity(np.asarray(sentence_vectors))
        max_3 = []
        max_3_id = []
        min_3 = []
        min_3_id = []
        for index in range(len(similarity_matrix)):
            for index2 in range(len(similarity_matrix[index])):
                if index == index2:
                    continue
                current_tweet = similarity_matrix[index][index2]
                if len(max_3) < 3:
                    max_3.append(current_tweet)
                    min_3.append(current_tweet)
                    max_3_id.append((index+1, index2+1))
                    min_3_id.append((index+1, index2+1))
                    continue

                if any(current_tweet > elem for elem in max_3) and (index2+1, index+1) not in max_3_id:
                    lowest_elem = min(max_3)
                    lowest_elem_index = max_3.index(lowest_elem)
                    max_3.pop(lowest_elem_index)
                    max_3.append(current_tweet)
                    max_3_id.pop(lowest_elem_index)
                    max_3_id.append((index+1, index2+1))

                if any(current_tweet  < elem for elem in min_3) and (index2+1, index+1) not in min_3_id:
                    highest_elem = max(min_3)
                    highest_elem_index = min_3.index(highest_elem)
                    min_3.pop(highest_elem_index)
                    min_3.append(current_tweet)
                    min_3_id.pop(highest_elem_index)
                    min_3_id.append((index+1, index2+1))

        max_3_id = max_3_id[-3:]
        min_3_id = min_3_id[-3:]
        print(similarity_matrix)
        print(f"HIGHEST SIMILARITIES: {max_3}")
        print(f"Coordinates: {max_3_id}")

        print(f"LOWEST SIMILARITIES: {min_3}")
        print(f"Coordinates: {min_3_id}")
        # Plot a heatmap
        ax = sns.heatmap(similarity_matrix, linewidth=0.5, cmap="YlGnBu")
        ids = list(range(1, 20 + 1))
        ax.set_xticklabels(ids)
        ax.set_yticklabels(ids)

        # Remove the ticks, but keep the labels
        ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labeltop=True,
                       labelbottom=False)
        ax.set_title("Similarity between sentence pairs")
        plt.show()