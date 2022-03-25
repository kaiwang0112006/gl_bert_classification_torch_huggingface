# coding: UTF-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import *
import numpy as np
import random
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn import metrics
import re
import time
import pickle
import argparse
import logging
import logging.config
import os
import sys
import shutil
import i3_word_tokenizer

class cleanTokenv1_0_2:
    def __init__(self):
        jiebaobj = i3_word_tokenizer.jieba_cut.jiebaTools(filter_syb=False,env='dev')
        jiebaobj.reset_jieba()
        jiebaobj.jieba_setup()
        self.jieba = jiebaobj

    def cleanToken(self,text):
        #text = re.sub(r'(@.*?)[\s]', ' ', text)
        # Replace '&amp;' with '&'
        text = text_preprocessing(text)

        wordlist = self.jieba.jieba_custom_lcut(text)
        return wordlist


##########################################
## Options and defaults
##########################################
def getOptions():
    parser = argparse.ArgumentParser(description='python *.py [option]')
    parser.add_argument('--run', dest='run', help="run name", default='test1')
    parser.add_argument('--pretrain', dest='pretrain', help="pretrain huggingface path", default='')
    parser.add_argument('--device', dest='device', help="device", default='')
    parser.add_argument('--batch_size', dest='batch_size',type=int, help="batch_size", default=128)
    parser.add_argument('--pad_size', dest='pad_size',type=int, help="pad_size", default=512)
    parser.add_argument('--class_list', dest='class_list', help="class_list", default="")
    parser.add_argument('--num_epochs', dest='num_epochs',type=int, help="num_epochs", default=3)
    parser.add_argument('--train', dest='train', help="train",required=True)
    parser.add_argument('--eval', dest='eval', help="eval",required=True)
    args = parser.parse_args()

    return args

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)
    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Create a function to tokenize a set of texts
def preprocessing_for_bert(textdata, tokenizer, pad,cutword=None):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []
    # For every sentence...
    for sent in textdata:
        if cutword:
            sent = tuple(cutword(sent))
            is_split_into_words = True
        else:
            is_split_into_words = False
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=sent,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=pad,                  # Max length to truncate/pad
            padding='max_length',         # Pad sentence to max length
            truncation=True,
            return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,      # Return attention mask
            is_split_into_words=is_split_into_words
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids').tolist()[0])
        attention_masks.append(encoded_sent.get('attention_mask').tolist()[0])
        # try:
        #     torch.tensor(input_ids)
        # except:
        #     print(input_ids)
        #     raise
    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, model_path, D_in=768, H=50, D_out=2, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(model_path)

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return F.softmax(logits, dim=1)

def train(model, train_dataloader, val_dataloader=None, optimizer=None, scheduler=None, epochs=4, evaluation=False,device=None,loss_fn=None):
    """Train the BertClassifier model.
    """
    # Start training loop
    # Specify loss function
    logger = logging.getLogger(__name__)
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not loss_fn:
        loss_fn = nn.CrossEntropyLoss()

    history_value = {"train":[],"eval":[]}
    logger.info("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        logger.info(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        logger.info("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                logger.info(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        logger.info("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy, report, confusion = evaluate(model, val_dataloader,device=device,loss_fn=loss_fn)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            history_value["train"].append(avg_train_loss)
            history_value["eval"].append(val_loss)
            logger.info(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            logger.info("-"*70)
    logger.info("Precision, Recall and F1-Score...")
    logger.info(str(report))
    logger.info("Confusion Matrix...")
    logger.info(str(confusion))
    logger.info("\n")
    logger.info("Training complete!")
    logger.info(str(history_value))
    return model


def evaluate(model, val_dataloader,device=None,loss_fn=None):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not loss_fn:
        loss_fn = nn.CrossEntropyLoss()
    model.eval()

    # Tracking variables
    val_accuracy = 0
    val_loss = []
    predict_all = []
    labels_all = []
    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        predict_all = predict_all+preds.tolist()
        labels_all = labels_all+b_labels.tolist()

        # Calculate the accuracy rate
        #accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        #val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = (torch.tensor(predict_all) == torch.tensor(labels_all)).cpu().numpy().mean() * 100
    target_names = [str(i) for i in sorted(list(set(labels_all)))]
    report = metrics.classification_report(labels_all, predict_all, target_names=target_names, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    return val_loss, val_accuracy, report, confusion

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = sklearn.metrics.accuracy_score(y_true=labels, y_pred=pred)
    recall = sklearn.metrics.recall_score(y_true=labels, y_pred=pred)
    precision = sklearn.metrics.precision_score(y_true=labels, y_pred=pred)
    f1 = sklearn.metrics.f1_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}



def main():
    options = getOptions()
    print(options)
    runname = options.run
    if  os.path.exists(runname):
        shutil.rmtree(runname)
    os.mkdir(runname)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    output_file_handler = logging.FileHandler(os.path.join(runname,'runlog.log'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)

    seed = 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #bert_pretrain = "/work/pretrain/huggingface/roberta-base-finetuned-chinanews-chinese/"
    if not options.pretrain:
        raise
    bert_pretrain = options.pretrain
    max_length = options.pad_size
    batch_size = options.batch_size
    epochs = options.num_epochs
    if not options.device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(options.device)
    train_path = options.train
    valid_path = options.eval

    # load dataset
    traindf = pd.read_csv(train_path,sep='\t')
    validdf = pd.read_csv(valid_path,sep='\t')

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_pretrain)

    # Run function `preprocessing_for_bert` on the train set and the validation set
    logger.info('Tokenizing data...')
    tokenobj1_0_2 = cleanTokenv1_0_2()
    wordfun = tokenobj1_0_2.cleanToken
    train_inputs, train_masks = preprocessing_for_bert(list(traindf['text']), tokenizer, max_length,cutword=wordfun)
    val_inputs, val_masks = preprocessing_for_bert(list(validdf['text']), tokenizer, max_length,cutword=wordfun)

    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(traindf['label'], device='cpu')
    val_labels = torch.tensor(validdf['label'], device='cpu')

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    # load the model and pass to CUDA
    model = BertClassifier(model_path=bert_pretrain,
                           D_in=768, H=50,
                           D_out=len(set(traindf['label'])),
                           freeze_bert=False).to(device)


    # Create the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)

    model = train(model=model, train_dataloader=train_dataloader,
          val_dataloader=val_dataloader,
          optimizer=optimizer, scheduler=scheduler,
          epochs=epochs, evaluation=True,device=device)
    #model.save_pretrained('./%s/model' % runname)
    model.to(torch.device("cpu"))
    tokenizer.save_pretrained('./%s/model' % runname)
    torch.save(model,'./%s/model/model.pt' % runname)
    with open('./%s/model/model.pkl' % runname,'wb') as mf:
        pickle.dump(model,mf)
    torch.save(model.state_dict(), './%s/model/weight.ckpt' % runname)

if __name__ == "__main__":
    main()

