import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable
import re
import pickle
from transformers import *
from transformers import BertModel
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import numpy as np

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
        #D_in, H, D_out = 768, 50, 2

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

class SimpleLSTM(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, batch_size, device=None):
        super(SimpleLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = self.init_device(device)
        self.hidden = self.init_hidden()

    @staticmethod
    def init_device(device):
        if device is None:
            return torch.device('cuda')
        return device

    def init_hidden(self):
        return (Variable(torch.zeros(2 * self.n_layers, self.batch_size, self.hidden_dim).to(self.device)),
                Variable(torch.zeros(2 * self.n_layers, self.batch_size, self.hidden_dim).to(self.device)))

    def forward(self, text):
        self.hidden = self.init_hidden()
        x = self.embedding(text)
        x, self.hidden = self.rnn(x, self.hidden)
        hidden, cell = self.hidden
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        x = self.fc(hidden)

        return x

    def loss(self, output, bert_prob, real_label):
        criterion = torch.nn.BCEWithLogitsLoss()
        return criterion(output, real_label.float())

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

    def add_embeding(self,embedding):
        self.embedding_pretrained = torch.tensor(embedding.astype('float32')) # 预训练词向量
        self.n_vocab = embedding.shape[0]                                     # 词表大小
        self.embed = self.embedding_pretrained.size(1)                        # 字向量维度


def main():
    max_length = 512
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_pretrain = "/work/pretrain/huggingface/ernie"
    tokenizer2 = AutoTokenizer.from_pretrained(bert_pretrain)
    with open('/work/kw/yuqing/torch_test/gl_bert_classification_torch_huggingface/bert/ernie/model/model.pkl','rb') as mf:
        teacher = pickle.load(mf).to(device)

    #teacher = torch.load('/work/kw/yuqing/torch_test/gl_bert_classification_torch_huggingface/bert/ernie/model/model.pt')
    teacher.load_state_dict(torch.load('/work/kw/yuqing/torch_test/gl_bert_classification_torch_huggingface/bert/ernie/model/weight.ckpt'))
    result = bert_predict(teacher,tokenizer2,"操控性舒服、油耗低，性价比高",max_length,device)
    print(result)

    loss_fn = nn.CrossEntropyLoss()

    train_path = "/work/kw/yuqing/torch_test/data/train_3c.tsv"
    valid_path = "/work/kw/yuqing/torch_test/data/test_3c.tsv"

    # load dataset
    traindf = pd.read_csv(train_path,sep='\t')
    validdf = pd.read_csv(valid_path,sep='\t')

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_pretrain)

    # Run function `preprocessing_for_bert` on the train set and the validation set
    print('Tokenizing data...')
    train_inputs, train_masks = preprocessing_for_bert(list(traindf['text']), tokenizer2, max_length)
    val_inputs, val_masks = preprocessing_for_bert(list(validdf['text']), tokenizer2, max_length)

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


    vocabtxt = []
    with open("/work/kw/yuqing/torch_test/gl_bert_classification_torch_huggingface/bert/ernie/model/vocab.txt") as fin:
        for eachline in fin:
            vocabtxt.append(eachline.strip())

    mcobj = ModelConfig()
    mcobj.vocabulary_size = len(vocabtxt)
    mcobj.embedding_dim = 64

    student = TextCNN(mcobj).to(device)

    temperature = 36
    epoch = 10

    KD_loss = nn.KLDivLoss(reduction="batchmean")

    optimizer = AdamW(student.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    for i in range(epoch):
        teacher.eval()
        student.train()
        accuracy_list = []
        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits_t = teacher(b_input_ids, b_attn_mask)
            wbag = []
            for vect in b_input_ids:
                temp = []
                for i in range(len(vocabtxt)):
                    if i in b_input_ids:
                        temp.append(1)
                    else:
                        temp.append(0)
                wbag.append(temp)
            #print(wbag)
            logits_s = student(torch.tensor(wbag).to(device))
            loss = KD_loss(input=F.log_softmax(logits_s/temperature,dim=-1),
                           target=F.softmax(logits_s/temperature,dim=-1))
            loss_s = loss_fn(logits_s, b_labels)
            preds = torch.argmax(logits_s, dim=1).flatten()
            accuracy = (preds == b_labels).cpu().numpy().tolist()
            print('step', step, np.nanmean(accuracy))
            accuracy_list += accuracy
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        acc = np.nanmean(accuracy_list)
        print('epoch', i, acc)
    runname = 'distil'
    torch.save(student,'./%s/model/model.pt' % runname)
    with open('./%s/model/model.pkl' % runname,'wb') as mf:
        pickle.dump(student,mf)
    torch.save(student.state_dict(), './%s/model/weight.ckpt' % runname)



if __name__ == "__main__":
    main()


