import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import re
import pickle
from transformers import BertModel, AutoTokenizer
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import numpy as np
import os
import json
import shutil

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
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

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

        return logits

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

def bert_predict(model, tokenizer, text, pad, device):
    encoded_sent = tokenizer.encode_plus(
        text=text_preprocessing(text),  # Preprocess sentence
        add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
        max_length=pad,                  # Max length to truncate/pad
        padding='max_length',         # Pad sentence to max length
        return_tensors='pt',           # Return PyTorch tensor
        return_attention_mask=True      # Return attention mask
    )

    input_ids = encoded_sent.get('input_ids').to(device)
    attention_masks = encoded_sent.get('attention_mask').to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_masks)

    probs_detail = F.softmax(logits, dim=1).cpu()
    probs_list = probs_detail.tolist()[0]
    probs_dict = {i:probs_list[i] for i in range(len(probs_list))}
    probs = torch.max(probs_detail, 1)[1].tolist()[0]
    return probs_dict,probs

class ModelConfig(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'TextCNN'
        self.class_list = ['0','1','2']            # 类别名单
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.embedfreeze = False
        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.hidden_size = 768
        self.filter_num = len(self.filter_sizes)


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        class_num = args.num_classes
        chanel_num = 1
        filter_num = args.filter_num
        filter_sizes = args.filter_sizes

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)

        self.embedding2 = None
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (size, embedding_dimension)) for size in filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)

    def forward(self, x):
        if self.embedding2:
            x = torch.stack([self.embedding(x), self.embedding2(x)], dim=1)
        else:
            x = self.embedding(x)
            x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) output of the model
        labels: (np.ndarray) [0, 1, ..., num_classes-1]
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params.alpha
    T = params.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss

def train_kd(model, teacher_model, optimizer, dataloader, metrics, params):
    model.train()
    teacher_model.eval()
    summ = []

    for step, batch in enumerate(dataloader):
        b_input_ids, b_attn_mask, b_labels = batch

        wbag = []
        for vect in b_input_ids:
            temp = []
            for i in range(params.vocab_len):
                if i in b_input_ids:
                    temp.append(1)
                else:
                    temp.append(0)
            wbag.append(temp)
        output_batch = model(torch.tensor(wbag).to(params.device))
        with torch.no_grad():
            output_teacher_batch = teacher_model(b_input_ids.to(params.device), b_attn_mask.to(params.device))
        loss = loss_fn_kd(output_batch, b_labels.to(params.device), output_teacher_batch, params)
        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()
        # performs updates using calculated gradients
        optimizer.step()
        preds = torch.argmax(output_batch, dim=1).flatten().to(params.device)
        accuracy = (preds == b_labels.to(params.device)).cpu().numpy().mean() * 100


        # compute all metrics on this batch
        summary_batch = {'accuracy':accuracy}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    print("- Train metrics : " + metrics_string)
    return metrics_mean

def evaluate_kd(model, dataloader, metrics, params):
    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for step, batch in enumerate(dataloader):
        b_input_ids, b_attn_mask, b_labels = batch
        wbag = []
        for vect in b_input_ids:
            temp = []
            for i in range(params.vocab_len):
                if i in b_input_ids:
                    temp.append(1)
                else:
                    temp.append(0)
            wbag.append(temp)
        model = model.to(params.device)
        output_batch = model(torch.tensor(wbag).to(params.device))
        loss = 0.0  #force validation loss to zero to reduce computation time

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        preds = torch.argmax(output_batch, dim=1).flatten()
        accuracy = (preds == b_labels.to(params.device)).cpu().numpy().mean() * 100
        loss = F.cross_entropy(output_batch, b_labels.to(params.device))

        # compute all metrics on this batch
        summary_batch = {'accuracy':accuracy}
        # summary_batch['loss'] = loss.data[0]
        summary_batch['loss'] = loss
        summ.append(summary_batch)
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    print("- Eval metrics : " + metrics_string)
    return metrics_mean


def train_and_evaluate_kd(model, teacher_model, train_dataloader, val_dataloader, optimizer,
                          metrics, params):
    """Train the model and evaluate every epoch.
    student_model, teacher_model, train, dev, optimizer,metrics, params
    """

    best_val_acc = 0.0

    # Tensorboard logger setup
    # board_logger = utils.Board_Logger(os.path.join(model_dir, 'board_logs'))

    # learning rate schedulers for different models:

    scheduler = StepLR(optimizer, step_size=100, gamma=0.2)

    for epoch in range(params.num_epochs):

        # Run one epoch
        print("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_metrics_mean = train_kd(model, teacher_model, optimizer, train_dataloader, metrics, params)
        scheduler.step()
        # Evaluate for one epoch on validation set
        val_metrics = evaluate_kd(model, val_dataloader, metrics, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc

        # If best_eval, best_save_path
        if is_best:
            #logging.info("- Found new best accuracy")
            best_val_acc = val_acc
            torch.save(model.state_dict(), params.best_weight_save)
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(params.model_dir, "metrics_val_best_weights.json")
            with open(best_json_path, 'w') as f:
                # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
                d = {k: float(v) for k, v in val_metrics.items()}
                json.dump(d, f, indent=4)
                #msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'


        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(params.model_dir, "metrics_val_last_weights.json")
        with open(last_json_path, 'w') as f:
            # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
            d = {k: float(v) for k, v in val_metrics.items()}
            json.dump(d, f, indent=4)
        torch.save(model.state_dict(), params.last_weight_save)

def main():
    max_length = 512
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    #### load teacher model and test it
    bert_pretrain = "/work/pretrain/huggingface/ernie"
    tokenizer = AutoTokenizer.from_pretrained(bert_pretrain)
    with open('/work/kw/yuqing/torch_test/gl_bert_classification_torch_huggingface/bert/ernie/model/model.pkl','rb') as mf:
        teacher = pickle.load(mf).to(device)

    #teacher = torch.load('/work/kw/yuqing/torch_test/gl_bert_classification_torch_huggingface/bert/ernie/model/model.pt')
    teacher.load_state_dict(torch.load('/work/kw/yuqing/torch_test/gl_bert_classification_torch_huggingface/bert/ernie/model/weight.ckpt'))
    result = bert_predict(teacher,tokenizer,"操控性舒服、油耗低，性价比高",max_length,device)

    #### load dataset
    train_path = "/work/kw/yuqing/torch_test/data/train_3c.tsv"
    valid_path = "/work/kw/yuqing/torch_test/data/test_3c.tsv"

    traindf = pd.read_csv(train_path,sep='\t')
    validdf = pd.read_csv(valid_path,sep='\t')

    # Run function `preprocessing_for_bert` on the train set and the validation set
    print('Tokenizing data...')
    train_inputs, train_masks = preprocessing_for_bert(list(traindf['text']), tokenizer, max_length)
    val_inputs, val_masks = preprocessing_for_bert(list(validdf['text']), tokenizer, max_length)

    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(traindf['label'])
    val_labels = torch.tensor(validdf['label'])

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    #### load vocab
    vocabtxt = []
    with open("/work/kw/yuqing/torch_test/gl_bert_classification_torch_huggingface/bert/ernie/model/vocab.txt") as fin:
        for eachline in fin:
            vocabtxt.append(eachline.strip())

    #### load student
    print('load student')
    mcobj = ModelConfig()
    mcobj.vocabulary_size = len(vocabtxt)
    mcobj.batch_size = batch_size
    mcobj.embedding_dim = 300
    mcobj.temperature = 1
    mcobj.learning_rate = 0.1
    mcobj.runname = 'distil_test1'
    mcobj.modelname = 'textcnn'
    mcobj.device = device
    mcobj.vocab_len = len(vocabtxt)
    mcobj.alpha = 0
    student = TextCNN(mcobj).to(device)


    if  os.path.exists(mcobj.runname):
        shutil.rmtree(mcobj.runname)
    os.mkdir(mcobj.runname)
    mcobj.model_save = os.path.join(mcobj.runname, mcobj.model_name+'.pkl')
    mcobj.best_weight_save = os.path.join(mcobj.runname, mcobj.model_name + '_best.ckpt')
    mcobj.last_weight_save = os.path.join(mcobj.runname, mcobj.model_name + '_last.ckpt')
    with open(mcobj.model_save,'wb') as mf:
        pickle.dump(student,mf)

    print('Traiing student')
    optimizer = torch.optim.Adam(student.parameters(), lr=mcobj.learning_rate)
    metrics = {
        'accuracy': accuracy,
        # could add more metrics such as accuracy for each token type
    }


    train_and_evaluate_kd(student, teacher, train_dataloader, val_dataloader, optimizer,metrics, mcobj)


if __name__ == "__main__":
    main()
