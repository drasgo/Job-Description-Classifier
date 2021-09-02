from typing import Tuple, Any, List
import os

import numpy
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW
from transformers import  BertTokenizer
from transformers import get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import simplemma
import nltk
import time
import json
import random
import pprint


ITALIAN_STOPWORDS = ['ad', 'al', 'allo', 'ai', 'agli', 'all', 'agl', 'alla', 'alle', 'con', 'col', 'coi', 'da', 'dal', 'dallo', 'dai', 'dagli', 'dall', 'dagl', 'dalla', 'dalle', 'di', 'del', 'dello', 'dei', 'degli', 'dell', 'degl', 'della', 'delle', 'in', 'nel', 'nello', 'nei', 'negli', 'nell', 'negl', 'nella', 'nelle', 'su', 'sul', 'sullo', 'sui', 'sugli', 'sull', 'sugl', 'sulla', 'sulle', 'per', 'tra', 'contro', 'io', 'tu', 'lui', 'lei', 'noi', 'voi', 'loro', 'mio', 'mia', 'miei', 'mie', 'tuo', 'tua', 'tuoi', 'tue', 'suo', 'sua', 'suoi', 'sue', 'nostro', 'nostra', 'nostri', 'nostre', 'vostro', 'vostra', 'vostri', 'vostre', 'mi', 'ti', 'ci', 'vi', 'lo', 'la', 'li', 'le', 'gli', 'ne', 'il', 'un', 'uno', 'una', 'ma', 'ed', 'se', 'perché', 'anche', 'come', 'dov', 'dove', 'che', 'chi', 'cui', 'non', 'più', 'quale', 'quanto', 'quanti', 'quanta', 'quante', 'quello', 'quelli', 'quella', 'quelle', 'questo', 'questi', 'questa', 'queste', 'si', 'tutto', 'tutti', 'a', 'c', 'e', 'i', 'l', 'o', 'ho', 'hai', 'ha', 'abbiamo', 'avete', 'hanno', 'abbia', 'abbiate', 'abbiano', 'avrò', 'avrai', 'avrà', 'avremo', 'avrete', 'avranno', 'avrei', 'avresti', 'avrebbe', 'avremmo', 'avreste', 'avrebbero', 'avevo', 'avevi', 'aveva', 'avevamo', 'avevate', 'avevano', 'ebbi', 'avesti', 'ebbe', 'avemmo', 'aveste', 'ebbero', 'avessi', 'avesse', 'avessimo', 'avessero', 'avendo', 'avuto', 'avuta', 'avuti', 'avute', 'sono', 'sei', 'è', 'siamo', 'siete', 'sia', 'siate', 'siano', 'sarò', 'sarai', 'sarà', 'saremo', 'sarete', 'saranno', 'sarei', 'saresti', 'sarebbe', 'saremmo', 'sareste', 'sarebbero', 'ero', 'eri', 'era', 'eravamo', 'eravate', 'erano', 'fui', 'fosti', 'fu', 'fummo', 'foste', 'furono', 'fossi', 'fosse', 'fossimo', 'fossero', 'essendo', 'faccio', 'fai', 'facciamo', 'fanno', 'faccia', 'facciate', 'facciano', 'farò', 'farai', 'farà', 'faremo', 'farete', 'faranno', 'farei', 'faresti', 'farebbe', 'faremmo', 'fareste', 'farebbero', 'facevo', 'facevi', 'faceva', 'facevamo', 'facevate', 'facevano', 'feci', 'facesti', 'fece', 'facemmo', 'faceste', 'fecero', 'facessi', 'facesse', 'facessimo', 'facessero', 'facendo', 'sto', 'stai', 'sta', 'stiamo', 'stanno', 'stia', 'stiate', 'stiano', 'starò', 'starai', 'starà', 'staremo', 'starete', 'staranno', 'starei', 'staresti', 'starebbe', 'staremmo', 'stareste', 'starebbero', 'stavo', 'stavi', 'stava', 'stavamo', 'stavate', 'stavano', 'stetti', 'stesti', 'stette', 'stemmo', 'steste', 'stettero', 'stessi', 'stesse', 'stessimo', 'stessero', 'stando']


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


def clean_texts(texts: List[str]) -> None:
    for idx in range(len(texts)):
        texts[idx] = clean_text(texts[idx])


def clean_text(text: str) -> str:
    stemmer = nltk.snowball.ItalianStemmer()
    text = text.lower()
    text = remove_stop_words(text)
    text = lemmetize_text(text)
    text = stem_text(text, stemmer)
    return text


def lemmetize_text(text: str) -> str:
    new_text = []
    lemmatizer = simplemma.load_data("it")
    for word in text.split(" "):
        new_text.append(simplemma.lemmatize(word, lemmatizer))
    return " ".join(new_text)


def remove_stop_words(text: str) -> str:
    return " ".join([word for word in text.split() if word.lower() not in ITALIAN_STOPWORDS])


def stem_text(text: str, stemmer) -> str:
    return stemmer.stem(text)


def encode_vector(original_input, tokens) -> list:
    encoded = []
    for phrase in original_input:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokens.encode(
            phrase,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        )
        encoded.append(encoded_sent)
    return encoded


def convert_labels(labs: List[str]) -> dict:
    converted = {}
    for word in labs:
        if word not in converted:
            converted[word] = str(len(converted))
    return converted


def prepare_labels(labs: List[str], converted: dict) -> None:
    for index in range(len(labs)):
        labs[index] = converted[labs[index]]


def attention_mask(input_data) -> list:
    attention = []
    for sent in input_data:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        attention.append(att_mask)
    return attention


def prepare_dataset(inputs, labs, attentions) -> DataLoader:
    # Create the DataLoader for our training set.
    train_data = TensorDataset(torch.tensor(inputs), torch.tensor(attentions), torch.tensor(labs))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return train_dataloader

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

    _, f1_macro_avg = metrics(numpy.array(predictions), numpy.array(true_labels))
    print(f"\n\n******\nF1 MACRO AVERAGE is: {f1_macro_avg}")
    print('    DONE.')
    return f1_macro_avg


def model_test_performance(test_tweets, test_labs, tokens):
    batch = 32
    test_vector = encode_vector(test_tweets, tokens)
    test_vector = pad_sequences(test_vector, maxlen=MAX_LEN, dtype="long",
                                value=0, truncating="post", padding="post")

    test_attention_mask = attention_mask(test_vector)

    prediction_inputs = torch.tensor(test_vector)
    prediction_masks = torch.tensor(test_attention_mask)
    prediction_labels = torch.tensor(test_labs)

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch)
    print("Starting testing")
    return model_testing(prediction_dataloader)


def save_model(mod, name, tokenizer):
    output_dir = f'./model_save_{name}/'

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
    MAX_LEN = 264
    batch_size = 32
    epochs = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_labels = 5
    training_dataset = "dataset/train_set.csv"
    testing_dataset = "dataset/test_set.csv"
    
    test = pd.read_csv(testing_dataset)
    test_descriptions = test["Job_offer"].values
    test_labels = list(test["Label"].values)
    
    converted_labels = convert_labels(test_labels)

    if training_phase is True:
        learning_rates = [1e-3, 5e-4, 1e-4, 5e-5, 3e-5, 2e-5]
        f1_scores = {}
        for learn in learning_rates:
            print(f"Starting training with learning rate {learn}")
            token = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
                num_labels=num_labels,  # The number of output labels--4 for multi-class classification.
                output_attentions=False,  # Whether the model returns attentions weights.
                output_hidden_states=False,  # Whether the model returns all hidden-states.
            )
            if device == "cuda":
                model.cuda()

            df = pd.read_csv(training_dataset)
            descriptions = list(df["Job_offer"].values)
            labels = list(df["Label"].values)

            clean_texts(descriptions)
            print("descriptions cleaned:")
            print(descriptions)
            prepare_labels(labels, converted_labels)
            print("labels cleaned: ")
            print(labels)
            print("post clean up")

            input_ids = encode_vector(descriptions, token)
            input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                                      value=0, truncating="post", padding="post")

            attention_masks = attention_mask(input_ids)
            train_dataset = prepare_dataset(input_ids, labels, attention_masks)
            total_steps = len(train_dataset) * epochs

            print("prepared training dataset")
            input()
            model, losses = training(model, train_dataset, learn, dev=device)
            save_model(model, str(learn), token)

    clean_texts(test_descriptions)
    prepare_labels(test_labels, converted_labels)

    # Load a trained model and vocabulary that you have fine-tuned
    if os.path.isdir("model/"):
        model = BertForSequenceClassification.from_pretrained("model/")
        tokenizer_test = BertTokenizer.from_pretrained("model/")
        print("Model loaded")
        # Copy the model to the GPU.
        model.to(device)
        model_test_performance(test_descriptions, test_labels, tokenizer_test)
