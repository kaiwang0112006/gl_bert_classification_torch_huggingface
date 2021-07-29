import torch.nn.functional as F
import torch.nn as nn
import torch
import re
import pickle
from transformers import *

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

def bert_predict(model, tokenizer, text, pad):
    encoded_sent = tokenizer.encode_plus(
        text=text_preprocessing(text),  # Preprocess sentence
        add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
        max_length=pad,                  # Max length to truncate/pad
        padding='max_length',         # Pad sentence to max length
        return_tensors='pt',           # Return PyTorch tensor
        return_attention_mask=True      # Return attention mask
    )

    input_ids = encoded_sent.get('input_ids')
    attention_masks = encoded_sent.get('attention_mask')
    with torch.no_grad():
        logits = model(input_ids, attention_masks)

    probs_detail = F.softmax(logits, dim=1).cpu()
    probs_list = probs_detail.tolist()[0]
    probs_dict = {i:probs_list[i] for i in range(len(probs_list))}
    probs = torch.max(probs_detail, 1)[1].tolist()[0]
    return probs_dict,probs




def main():
    runname = "bert_test1"
    max_length = 512

    bert_pretrain = "/work/pretrain/huggingface/roberta-base-finetuned-chinanews-chinese"
    tokenizer2 = AutoTokenizer.from_pretrained(bert_pretrain)
    with open('./%s/model/model.pkl' % runname,'rb') as mf:
        model2 = pickle.load(mf).to(torch.device("cpu"))
    model2.load_state_dict(torch.load('./%s/model/weight.ckpt' % runname))
    result = bert_predict(model2,tokenizer2,"操控性舒服、油耗低，性价比高",max_length)
    print(result)


if __name__ == "__main__":
    main()