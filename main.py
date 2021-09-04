from typing import Tuple, Any
import os

import numpy
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertForSequenceClassification, AdamW
from transformers import  BertTokenizer
from transformers import get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
import time
import random

import text_clean_up
import preparation



def training(nn, train_dataloader, learn_rate, eps=1e-8, dev="cuda") -> Tuple[Any, list]:
    loss_values = []
    seed_val = 42
    random.seed(seed_val)
    numpy.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    optimizer = AdamW(nn.parameters(),
                      lr=learn_rate,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=eps  # args.adam_epsilon  - default is 1e-8.
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
                elapsed = time.time() - t0
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
        print("  Training epcoh took: {:}".format(time.time() - t0))
        print("Training Done!")

    return nn, loss_values


def model_testing(prediction_dataloader, dev="cuda"):
    model.eval()
    predictions, true_labels = [], []

    for batch in prediction_dataloader:
        batch = tuple(t.to(dev) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions += numpy.argmax(logits, axis=1).tolist()
        true_labels += label_ids.tolist()

    _, f1_macro_avg = preparation.metrics(numpy.array(predictions), numpy.array(true_labels))
    print(f"\n\n******\nF1 MACRO AVERAGE is: {f1_macro_avg}")
    print('    DONE.')
    return f1_macro_avg


def model_test_performance(test_tweets, test_labs, tokens, max_len):
    batch = 32
    test_vector = text_clean_up.encode_vector(test_tweets, tokens, max_len)
    test_vector = pad_sequences(test_vector, maxlen=max_len, dtype="long",
                                value=0, truncating="post", padding="post")

    test_attention_mask = preparation.attention_mask(test_vector)

    prediction_inputs = torch.tensor(test_vector)
    prediction_masks = torch.tensor(test_attention_mask)
    prediction_labels = torch.tensor(test_labs)

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch)
    print("Starting testing")
    return model_testing(prediction_dataloader)


def save_model(mod, tokenizer):
    output_dir = f'./model/'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # They can then be reloaded using `from_pretrained()`
    model_to_save = mod.module if hasattr(mod, 'module') else mod  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    training_phase = True

    batch_size = 32
    epochs = 4
    num_labels = 5
    max_post_size = 512

    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_dataset = "dataset/train_set.csv"
    testing_dataset = "dataset/test_set.csv"

    test = pd.read_csv(testing_dataset)
    test_descriptions = list(test["Job_offer"].values)
    test_labels = list(test["Label"].values)
    converted_labels = text_clean_up.convert_labels(test_labels)

    df = pd.read_csv(training_dataset)
    descriptions = list(df["Job_offer"].values)
    labels = list(df["Label"].values)

    text_clean_up.clean_texts(test_descriptions)
    text_clean_up.prepare_labels(test_labels, converted_labels)

    text_clean_up.clean_texts(descriptions)
    text_clean_up.prepare_labels(labels, converted_labels)

    if training_phase is True:
        learning_rates = [1e-3, 5e-4, 1e-4, 5e-5, 3e-5, 2e-5]

        print("descriptions cleaned:")
        print("labels cleaned: ")
        print("post clean up")

        token = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        input_ids = text_clean_up.encode_vector(descriptions, token, max_post_size)

        print("input encoded")

        input_ids = pad_sequences(input_ids, maxlen=max_post_size, dtype="long",
                                  value=0, truncating="post", padding="post")
        print("input padded")

        attention_masks = preparation.attention_mask(input_ids)
        train_dataset = preparation.prepare_dataset(input_ids, labels, attention_masks, batch_size)
        total_steps = len(train_dataset) * epochs

        print("prepared training dataset")

        for learn in learning_rates:
            print(f"Starting training with learning rate {learn}")

            model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
                num_labels=num_labels,  # The number of output labels--4 for multi-class classification.
                output_attentions=False,  # Whether the model returns attentions weights.
                output_hidden_states=False,  # Whether the model returns all hidden-states.
            )
            if device == "cuda":
                model.cuda()
            # model = None

            print("finished preprocessing")
            quit()
            model, losses = training(model, train_dataset, learn, dev=device)
            save_model(model, token)


    # Load a trained model and vocabulary that you have fine-tuned
    if os.path.isdir("model/"):

        model = BertForSequenceClassification.from_pretrained("model/")
        tokenizer_test = BertTokenizer.from_pretrained("model/")
        print("Model loaded")
        # Copy the model to the GPU.
        model.to(device)
        model_test_performance(test_descriptions, test_labels, tokenizer_test, max_post_size)
