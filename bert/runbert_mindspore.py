# coding: UTF-8
##########################################
## data: wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate/examples/data/IMDB.csv
## bineray /multi-classification, df[label]=1
## python run_bert_mindspore.py --train=demo_train.tsv --test=demo_train.tsv --eval=demo_train.tsv --pretrain=bert-base-cased --run=bert_base_cased
##########################################
import argparse
import mindspore
from mindspore.dataset import GeneratorDataset, transforms
from mindnlp.engine import Trainer
from mindnlp.transformers import BertTokenizer
from mindnlp.transformers import BertForSequenceClassification, BertModel
from mindnlp.engine import TrainingArguments
from mindnlp import evaluate
import numpy as np

##########################################
## Options and defaults
##########################################
def getOptions():
    parser = argparse.ArgumentParser(description='python *.py [option]')
    parser.add_argument('--run', dest='run', help="run name", default='test1')
    parser.add_argument('--pretrain', dest='pretrain', help="pretrain huggingface path", default='')
    #parser.add_argument('--device', dest='device', help="device", default='')
    #parser.add_argument('--batch_size', dest='batch_size',type=int, help="batch_size", default=128)
    #parser.add_argument('--pad_size', dest='pad_size',type=int, help="pad_size", default=512)
    parser.add_argument('--num_class', dest='num_class', type=int, help="num_class", default=2)
    parser.add_argument('--lr', dest='lr', type=float, help="learning rate", default=2e-5)
    parser.add_argument('--num_epochs', dest='num_epochs',type=int, help="num_epochs", default=3)
    parser.add_argument('--train', dest='train', help="train",required=True)
    parser.add_argument('--test', dest='test', help="test", required=True)
    parser.add_argument('--eval', dest='eval', help="eval",required=True)
    args = parser.parse_args()

    return args

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
# prepare dataset
class SentimentDataset:
    """Sentiment Dataset"""

    def __init__(self, path):
        self.path = path
        self._labels, self._text_a = [], []
        self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            dataset = f.read()
        lines = dataset.split("\n")
        for line in lines[1:-1]:
            label, text_a = line.split("\t")
            self._labels.append(int(label))
            self._text_a.append(text_a)

    def __getitem__(self, index):
        return self._labels[index], self._text_a[index]

    def __len__(self):
        return len(self._labels)


def process_dataset(source, tokenizer, max_seq_len=64, batch_size=32, shuffle=True):
    is_ascend = mindspore.get_context('device_target') == 'Ascend'

    column_names = ["label", "text_a"]

    dataset = GeneratorDataset(source, column_names=column_names, shuffle=shuffle)
    # transforms
    type_cast_op = transforms.TypeCast(mindspore.int32)

    def tokenize_and_pad(text):
        if is_ascend:
            tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=max_seq_len)
        else:
            tokenized = tokenizer(text)
        return tokenized['input_ids'], tokenized['attention_mask']

    # map dataset
    dataset = dataset.map(operations=tokenize_and_pad, input_columns="text_a",
                          output_columns=['input_ids', 'attention_mask'])
    dataset = dataset.map(operations=[type_cast_op], input_columns="label", output_columns='labels')
    # # batch dataset
    if is_ascend:
        dataset = dataset.batch(batch_size)
    else:
        dataset = dataset.padded_batch(batch_size, pad_info={'input_ids': (None, tokenizer.pad_token_id),
                                                             'attention_mask': (None, 0)})

    return dataset

def main():
    options = getOptions()
    print(options)
    tokenizer = BertTokenizer.from_pretrained(options.pretrain)
    dataset_train = process_dataset(SentimentDataset(options.train), tokenizer)
    dataset_val = process_dataset(SentimentDataset(options.eval), tokenizer)
    dataset_test = process_dataset(SentimentDataset(options.test), tokenizer, shuffle=False)

    model = BertForSequenceClassification.from_pretrained(options.pretrain, num_labels=options.num_class)

    training_args = TrainingArguments(
        output_dir=options.run+"_output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=options.num_epochs,
        learning_rate=options.lr
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        compute_metrics=compute_metrics
    )

    # start training
    trainer.train()
    model.save_pretrained(options.run)
    tokenizer.save_pretrained(options.run)

if __name__ == "__main__":
    main()