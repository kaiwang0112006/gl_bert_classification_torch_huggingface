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
from lightgbm.sklearn import LGBMClassifier
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
import re
import string
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import lightgbm as lgb

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
def preprocessing_for_bert(textdata, tokenizer, pad):
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
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=pad,                  # Max length to truncate/pad
            padding='max_length',         # Pad sentence to max length
            truncation=True,
            return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
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

    def forward(self, input_ids, attention_mask,weight):
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
        last_hidden_state_cls = torch.cat((last_hidden_state_cls,weight), dim=1)
        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)
        probs_detail = F.softmax(logits, dim=1)
        return probs_detail


class cleanToken:
    def __init__(self):
        jiebaobj = i3_word_tokenizer.jieba_cut.jiebaTools(filter_syb=True)
        jiebaobj.reset_jieba()
        jiebaobj.jieba_setup()
        self.jieba = jiebaobj

    def cleanToken(self,text):
        text = re.sub(r'(@.*?)[\s]', ' ', text)
        # Replace '&amp;' with '&'
        text = re.sub(r'&amp;', '&', text)
        # Remove trailing whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        zh_sp = """[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+！，？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
        en_sp = string.punctuation

        wordlist = self.jieba.jieba_custom_lcut(text)
        textbag = [word for word in wordlist if word!=' ' \
                   and word not in zh_sp \
                   and word not in en_sp]
        return textbag

def train(teacher, traindf, validdf, tokenizer, max_length=512, batch_size=64, optimizer=None, epochs=4, evaluation=False,device=None,loss_fn=None):
    """Train the BertClassifier model.
    """
    # Start training loop
    # Specify loss function
    params = {'num_class':len(set(traindf['label'])),'colsample_bytree': 0.1974275059545383, 'drop_rate': 0.3100849740535157, 'learning_rate': 0.059536991142383075, 'max_bin': 899, 'max_depth': 10, 'min_child_samples': 2677, 'min_split_gain': 0.1, 'num_leaves': 40, 'reg_alpha': 0.1, 'reg_lambda': 142.02674856733813, 'sigmoid': 0.1, 'subsample': 1.0, 'subsample_for_bin': 1235, 'subsample_freq': 1, 'is_unbalance': True, 'random_state': 24, 'n_jobs': 10, 'objective': 'multiclass'}
    logger = logging.getLogger(__name__)
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not loss_fn:
        loss_fn = nn.CrossEntropyLoss()
    logger.info('Tokenizing data...')
    val_inputs, val_masks = preprocessing_for_bert(list(validdf['text']), tokenizer, max_length)
    val_labels = torch.tensor(validdf['label']).to(device)
    valsampleweight =  torch.ones([len(validdf), len(set(traindf['label']))], dtype=torch.float32).to(device)

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks, valsampleweight, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    tokenobj = cleanToken()
    custtokenizer = tokenobj.cleanToken
    vect = CountVectorizer(tokenizer=custtokenizer)
    train_vect = vect.fit_transform(traindf["text"])
    train_vect_df = pd.DataFrame(train_vect.toarray())
    logger.info('Student init model...')
    train_dataset = lgb.Dataset(train_vect_df, label=traindf["label"], free_raw_data=False)
    student = lgb.train(params, train_dataset, num_boost_round = 1)

    valid_vect = vect.transform(validdf["text"])
    valid_data = pd.DataFrame(valid_vect.toarray())
    valid_dataset = lgb.Dataset(valid_data, label=validdf["label"], free_raw_data=False)
    train_inputs, train_masks = preprocessing_for_bert(list(traindf['text']), tokenizer, max_length)

    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(traindf['label']).to(device)

    sampleweight =  torch.ones([len(traindf), len(set(traindf['label']))], dtype=torch.float32).to(device)
    logger.info("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        # Create the DataLoader for our training set
        train_data = TensorDataset(train_inputs, train_masks, sampleweight, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        train_pred = []
        # Put the model into the training mode
        teacher.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, sample_weight, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            teacher.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = teacher(b_input_ids, b_attn_mask, sample_weight)
            preds = torch.argmax(logits, dim=1).flatten()
            train_pred += preds.tolist()
            # train_weight = torch.tensor(train_pred).to(device)
            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()

        # Calculate the average loss over the entire training data
        avg_teacher_train_loss = total_loss / len(train_dataloader)
        traindf['teacher_pred_label'] = [1 if i>0.5 else 0 for i in train_pred]
        teacher_tr_acc = sklearn.metrics.accuracy_score(traindf['label'], traindf['teacher_pred_label'])
        logger.info("valid teacher...")
        teacher_val_loss, teacher_val_acc = evaluate(teacher, val_dataloader,
                                                     device=device)

        #train_vect_df = pd.DataFrame(train_vect.toarray())
        train_data_weight = traindf.apply(lambda x:10 if x['teacher_pred_label']!=x['label'] else 1, axis=1)

        params["weight_column"] = "name:weight"
        train_dataset = lgb.Dataset(train_vect_df, weight=train_data_weight,label=traindf["label"], free_raw_data=False)
        student = lgb.train(params, train_dataset, num_boost_round = 50,
                            init_model = student, valid_sets=valid_dataset,
                            early_stopping_rounds=10,keep_training_booster=True,
                            verbose_eval=False)

        tr_stu_pred_proba = student.predict(train_vect_df)

        traindf["stu_pred"] = np.argmax(tr_stu_pred_proba,axis=1)
        student_tr_acc = sklearn.metrics.accuracy_score(traindf['label'], traindf["stu_pred"])

        stu_pred = student.predict(valid_data)
        validdf["stu_pred"] = np.argmax(stu_pred,axis=1)
        student_val_acc = sklearn.metrics.accuracy_score(validdf['label'], validdf["stu_pred"])
        sampleweight = torch.tensor(tr_stu_pred_proba,dtype=torch.float32).to(device)
        logger.info(f"{'Epoch':^7} | {'teacher_val_loss':^7} | {'teacher_tr_acc':^12} | {'teacher_val_acc':^10} | {'student_tr_acc':^9} | {'student_val_acc':^9}")
        logger.info(f"{epoch_i + 1:^7} | {teacher_val_loss:^9.2f}| {teacher_tr_acc:^9.2f} | {teacher_val_acc:^9.2f} | {student_tr_acc:^9.2f}| {student_val_acc:^9.2f}")

    report = sklearn.metrics.classification_report(validdf["label"] , validdf["stu_pred"] , digits=4)
    confusion = sklearn.metrics.confusion_matrix(validdf["label"] , validdf["stu_pred"])
    logger.info("Precision, Recall and F1-Score...")
    logger.info(str(report))
    logger.info("Confusion Matrix...")
    logger.info(str(confusion))
    logger.info("\n")
    logger.info("Training complete!")


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
        b_input_ids, b_attn_mask, sample_weight, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask, sample_weight)

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
    val_accuracy = (torch.tensor(predict_all) == torch.tensor(labels_all)).cpu().numpy().mean()
    #target_names = [str(i) for i in sorted(list(set(labels_all)))]
    #report = sklearn.metrics.classification_report(labels_all, predict_all, target_names=target_names, digits=4)
    #confusion = sklearn.metrics.confusion_matrix(labels_all, predict_all)
    return val_loss, val_accuracy

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

    # load the model and pass to CUDA
    temodel = BertClassifier(model_path=bert_pretrain,
                             D_in=768+len(set(traindf['label'])), H=50,
                             D_out=len(set(traindf['label'])),
                             freeze_bert=False).to(device)


    # Create the optimizer
    optimizer = AdamW(temodel.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )


    train(teacher=temodel, traindf=traindf,
          validdf=validdf, tokenizer=tokenizer, batch_size=batch_size,
          optimizer=optimizer,
          epochs=epochs, evaluation=True,device=device)


if __name__ == "__main__":
    main()

