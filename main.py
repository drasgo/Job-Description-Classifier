import os

import numpy
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification, AdamW
from transformers import  BertTokenizer
from transformers import get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
import time
import random

from text_clean_up import clean_texts, encode_vector, prepare_labels, encode_vector_testing
from preparation import recall, precision, f1, attention_mask, prepare_dataset



def training(nn, train_dataloader, steps, learn_rate, eps=1e-8, dev="cuda"):
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
                                                num_training_steps=steps)

    print("Starting Training")
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_loss = 0
        nn.train()

        for step, batch in enumerate(train_dataloader):
            if step % 50 == 0 and not step == 0:
                elapsed = time.time() - t0
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                print("Current average loss: " + str(total_loss/step))
            b_input_ids = batch[0].to(dev)
            b_input_mask = batch[1].to(dev)
            b_labels = batch[2].to(dev)
            nn.zero_grad()

            outputs = nn(b_input_ids,
                         token_type_ids=None,
                         attention_mask=b_input_mask,
                         labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(nn.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(time.time() - t0))

    return nn


def save_model(mod, tokenizer, path=""):
    output_dir = "." + path + '/model/'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    model_to_save = mod.module if hasattr(mod, 'module') else mod  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def model_testing(prediction_dataloader, dev="cuda"):
    model.eval()
    predictions, true_labels = [], []

    # For each element in the backlog, pass the input, attention mask, and label tensors as bidimensional tensors of 
    # dimension [1, whatever] to the network, and extract the result
    for index in tqdm(range(len(prediction_dataloader))):
        b_input_ids = prediction_dataloader[index]["tensor_input"].view(1, -1).to(dev)
        b_input_mask = prediction_dataloader[index]["tensor_attention"].view(1, -1).to(dev)
        b_labels = prediction_dataloader[index]["tensor_label"].view(1, -1).to(dev)

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs.logits.detach().cpu().numpy()
        res = list(numpy.argmax(logits, axis=1))
        label_ids = b_labels.to('cpu').numpy()
        
        # Add the predicted label and the true label to the lists, so they can be later used for computing the precision,
        # recall, and F1 scores
        predictions += res
        true_labels += label_ids.tolist()
        
        # Add the predicted label to the backlog
        prediction_dataloader[index]["predicted"] = res[0]

    print(f"Precision score: {precision(predictions, true_labels)}")
    print(f"Recall score: {recall(predictions, true_labels)}")
    print(f"F1 score: {f1(predictions, true_labels)}")
    return prediction_dataloader


def model_test_performance(elems, test_labs, tokens, max_len, dev):
    # Tokenize input vector
    test_vector = encode_vector_testing(elems, tokens, max_len)
    # Pad input vector
    test_vector = pad_sequences(test_vector, maxlen=max_len, dtype="long",
                                value=0, truncating="post", padding="post")
    # Create attention mask
    test_attention_mask = attention_mask(test_vector)

    # Add data to backlog
    for index in range(len(test_vector)):
        elems[index]["tokenized_input"] = test_vector[index]
        elems[index]["tensor_input"] = torch.tensor(test_vector[index])
        elems[index]["tensor_attention"] = torch.tensor(test_attention_mask[index])
        elems[index]["tensor_label"] = torch.tensor(test_labs[index])

    return model_testing(elems, dev)


if __name__ == "__main__":
    # If you want to restart the training process, change training_phase to True
    training_phase = True

    # Max sequence lenght to be inputted in BERT
    max_post_size = 512

    # Arbitrary conversion of job label in class
    converted_labels = {
        'Web Developer': 0,
        'Java Developer': 1,
        'System Analyst': 2,
        'Software Engineer': 3,
        'Programmer': 4
    }

    # And its reverse
    reverse_converted_labels = {0: "Web Developer",
                                1: "Java Developer",
                                2: "System Analyst",
                                3: "Software Engineer",
                                4: "Programmer"
                                }
    num_labels = len(converted_labels)

    # device = "cpu"
    # If want to use only the CPU, comment the following line
    device = "cuda" if torch.cuda.is_available() else "cpu"

    testing_dataset = "dataset/test_set.csv"
    training_dataset = "dataset/train_set.csv"
    
    output_csv = "predicted_result.csv"

    if training_phase is True:
        batch_size = 1
        epochs = 10
        learning_rate = 2e-5

        df = pd.read_csv(training_dataset)
        descriptions = list(df["Job_offer"].values)
        labels = list(df["Label"].values)

        temp_description = []
        temp_labels = []
        # To artificially increase the dataset, by stocastically adding twice the same pair description-label for the
        # classes with less accuracy
        for idx in range(len(descriptions)):
          if random.random() > 0.75 and labels[idx] in [0,3,4]:
            temp_description.append(descriptions[idx])
            temp_labels.append(labels[idx])
        descriptions = descriptions + temp_description
        labels = labels + temp_labels

        # Preparation of the text and labels
        clean_texts(descriptions)
        prepare_labels(labels, converted_labels)

        # Loading the pre-trained italian version of BERT
        model = BertForSequenceClassification.from_pretrained(
            'dbmdz/bert-base-italian-xxl-uncased',
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False,
        )
        token = BertTokenizer.from_pretrained('dbmdz/bert-base-italian-xxl-uncased', do_lower_case=True)

        # Tokenization of the input
        input_ids = encode_vector(descriptions, token, max_post_size)
        # Padding of the input, so each input sequence has the same lenght
        input_ids = pad_sequences(input_ids, maxlen=max_post_size, dtype="long",
                                  value=0, truncating="post", padding="post")
        # Creating the attention mask, so it's clear if a token in an input sequence is valid or is padding
        attention_masks = attention_mask(input_ids)

        # Create the training dataset
        train_dataset = prepare_dataset(input_ids, labels, attention_masks, batch_size)
        total_steps = len(train_dataset) * epochs

        if device == "cuda":
            model.cuda()

        model = training(model, train_dataset, total_steps, learning_rate, dev=device)
        save_model(model, token)

    # Load the trained model and vocabulary that you have fine-tuned
    if os.path.isdir("model/"):
        test = pd.read_csv(testing_dataset)
        test_descriptions = list(test["Job_offer"].values)
        test_labels = list(test["Label"].values)
        
        cleaned_description = test_descriptions.copy()
        
        # Prepare labels and job descriptions
        prepare_labels(test_labels, converted_labels)
        clean_texts(cleaned_description)

        elements = []
        # Create a backlog that will contain every information of the inputs and labels, including original job description,
        # cleaned up job description, original label, input as tensor, attention mask as tensor, original label as tensor,
        # predicted label
        for idx in range(len(test_descriptions)):
            elements.append({
                "original": test_descriptions[idx],
                "cleaned": cleaned_description[idx],
                "label": test_labels[idx]
            })

        # Load fine-tuned modeland tokenizer
        model = BertForSequenceClassification.from_pretrained("model/")
        # Copy the model to the chosen device
        model.to(device)
        tokenizer_test = BertTokenizer.from_pretrained("model/")
        print("Model loaded")
        
        # Proceed with testing
        finished = model_test_performance(elements, test_labels, tokenizer_test, max_post_size, device)

        result = []
        # Saves the elements we are interested in (i.e. Job_description, true label and predicted label) to a separate
        # list of dictionaries, that will be saved as csv
        for elem in finished:
            result.append({
                "Job_description": elem["original"],
                "Label_true": reverse_converted_labels[elem["label"]],
                "Label_pred": reverse_converted_labels[elem["predicted"]]
            })
        table = pd.DataFrame(result)
        table.to_csv(path_or_buf=output_csv, sep=";")
