# coding: UTF-8
from transformers import *
import re
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time
import os, sys
import pandas as pd
from sklearn import metrics
from pyhive import hive
import argparse
import datetime
import logging
import re
from sqlalchemy import create_engine

sys.path.append("/work/kw/ppackage")
from miloTools.tools.multi_apply import *
from urllib import parse


##########################################
## Options and defaults
##########################################
def getOptions():
    parser = argparse.ArgumentParser(description='python *.py [option]')
    requiredgroup = parser.add_argument_group('required arguments')
    requiredgroup.add_argument('--dt', dest='dt', help='dt', default='')
    args = parser.parse_args()

    return args


def get_hivedf(sql_select):
    connection = hive.Connection(host='10.24.16.96',
                                 port=10000,
                                 auth="CUSTOM",
                                 database='i3',
                                 username='hive',
                                 password='*Hive123')

    cur = connection.cursor()
    cur.execute(sql_select)
    columns = [col[0] for col in cur.description]
    result = [dict(zip(columns, row)) for row in cur.fetchall()]
    hivedf = pd.DataFrame(result)
    hivedf.columns = columns
    return hivedf


def flatten_columns(df, cols):
    """Flattens multiple columns in a data frame, cannot specify all columns!"""
    flattened_cols = {}
    for col in cols:
        flattened_cols[col] = pd.DataFrame(
            [(index, value) for (index, values) in df[col].iteritems() for value in values],
            columns=['index', col]).set_index('index')
    flattened_df = df.drop(cols, axis=1)
    for col in cols:
        flattened_df = flattened_df.join(flattened_cols[col])
    return flattened_df


def removeUrl(text):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for u in urls:
        text = text.replace(u, '')
    return text


def cut_sentences(content, ml=-1):
    # 结束符号，包含中文和英文的
    end_flag = ['?', '!', '？', '！', '。', '…', '。']

    content = removeUrl(content)
    content_len = len(content)
    sentences = []
    tmp_char = ''
    for idx, char in enumerate(content):
        # 拼接字符
        if char != ' ':
            tmp_char += char

        # 判断是否已经到了最后一位
        if (idx + 1) == content_len:
            sentences.append(tmp_char)
            break
        # 判断此字符是否为结束符号
        if char in end_flag:
            # 再判断下一个字符是否为结束符号，若是不是结束符号，则切分句子
            next_idx = idx + 1
            if not content[next_idx] in end_flag:
                sentences.append(tmp_char)
                tmp_char = ''
    sentences = [s.strip() for s in sentences if len(s) > 10]
    return tuple(sentences)


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

    text = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a?!？！。…。])", "", text)
    return text


class BertClassifier1(nn.Module):
    """Bert Model for Classification Tasks.
    """

    def __init__(self, model_path, n_classes, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier1, self).__init__()

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(model_path)

        # Instantiate an one-layer feed-forward classifier
        self.classifier_list = []
        for c in n_classes:
            self.classifier_list.append(nn.Linear(self.bert.config.hidden_size, c))
        self.fc1 = nn.Linear(self.bert.config.hidden_size, n_classes[0])
        self.fc2 = nn.Linear(self.bert.config.hidden_size, n_classes[1])
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

        # Feed input to classifier to compute logits
        out1 = self.fc1(outputs.pooler_output)
        out2 = self.fc2(outputs.pooler_output)
        return out1, out2

class BertClassifier2(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, model_path, D_in=768, H=50, D_out=2, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier2, self).__init__()

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


def get_mix(model, tokenizer, text, device, max_length=128):
    try:
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(text),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=max_length,  # Max length to truncate/pad
            padding='max_length',  # Pad sentence to max length
            truncation=True,
            return_tensors='pt',  # Return PyTorch tensor
            return_attention_mask=True  # Return attention mask
        )

        input_ids = encoded_sent.get('input_ids').to(device)
        attention_masks = encoded_sent.get('attention_mask').to(device)
        with torch.no_grad():
            probs_detail = model(input_ids, attention_masks)
        return (probs_detail[0][0].tolist(), probs_detail[1][0].tolist())
    except:
        return (-1, -1)

def get_property(model, tokenizer, text, device, max_length=128):
    try:
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(text),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=max_length,  # Max length to truncate/pad
            padding='max_length',  # Pad sentence to max length
            truncation=True,
            return_tensors='pt',  # Return PyTorch tensor
            return_attention_mask=True  # Return attention mask
        )

        input_ids = encoded_sent.get('input_ids').to(device)
        attention_masks = encoded_sent.get('attention_mask').to(device)
        with torch.no_grad():
            probs_detail = model(input_ids, attention_masks)
        prob = probs_detail[0].tolist()
        return get_label(prob)
    except:
        return ""


def get_label(r, label=1):
    if label == 1:
        lmap = {0: '上下车便利性', 1: '个人维修维护便利性', 2: '二手车保值率', 3: '交车服务', 4: '企业形象', 5: '侧面', 6: '保养服务', 7: '保有量',
                8: '保险及其他增值服务', 9: '信息/按键标识便利性', 10: '充电便利性', 11: '充电安全', 12: '充电时间', 13: '内饰布局', 14: '内饰材质',
                15: '内饰细节特征', 16: '内饰风格', 17: '刹车/制动性能', 18: '前排座椅舒适性', 19: '前排空间大小', 20: '加速动力性能', 21: '动力参数',
                22: '厂家售后服务', 23: '合格证及上牌上户', 24: '后备厢/后备箱空间', 25: '后备箱取放物品便利性', 26: '后排座椅空间灵活性', 27: '品牌价值',
                28: '售后服务态度', 29: '售后诚信', 30: '售后费用', 31: '声音品质感', 32: '备件服务', 33: '外观协调性', 34: '外观品质感', 35: '安全性配置',
                36: '导航系统', 37: '尺寸参数', 38: '底盘高度/离地间隙', 39: '座椅舒适性', 40: '异响', 41: '影音娱乐系统', 42: '悬架/减振舒适性',
                43: '战略规划', 44: '按键、调节操作便利性', 45: '换挡操作性能', 46: '接近角/离去角', 47: '操作品质感', 48: '整体便利性', 49: '整体信息娱乐性',
                50: '整体充电', 51: '整体内饰', 52: '整体动力', 53: '整体外观', 54: '整体安全性', 55: '整体性价比', 56: '整体操控性', 57: '整体智能化',
                58: '整体电池', 59: '整体空间', 60: '整体舒适性', 61: '整体质量', 62: '整体通过性', 63: '智能远程控制', 64: '智能驾驶辅助', 65: '服务其他',
                66: '爬坡动力性能', 67: '电池衰减/里程衰减', 68: '碰撞测试星级', 69: '空调舒适性', 70: '第三排座椅空间灵活性', 71: '第三排座椅舒适性',
                72: '第三排空间大小', 73: '第二排座椅舒适性', 74: '第二排空间大小', 75: '经营表现', 76: '续航里程', 77: '维修质量', 78: '综合油耗',
                79: '行驶稳定性', 80: '试乘试驾', 81: '质量投诉', 82: '购车价格', 83: '购车费用', 84: '起步动力性能', 85: '车内互联系统', 86: '车内储物空间',
                87: '车内气味', 88: '车内静音效果', 89: '车头', 90: '车尾', 91: '车标', 92: '车系评价', 93: '车联网', 94: '车身结构安全', 95: '车轮',
                96: '转向性能', 97: '配置丰富性', 98: '配置实用性', 99: '销售专业性', 100: '销售政策', 101: '销售服务态度', 102: '销售诚信', 103: '销量',
                104: '门店布局', 105: '门店环境', 106: '门店设施', 107: '驾驶操作舒适性', 108: '驾驶视觉安全'}
    else:
        lmap = {0: '负面', 1: '中性', 2: '正面'}
    k = np.argmax(r)
    return lmap[k]


def main():
    # get data
    options = getOptions()
    d = datetime.datetime.now()

    if options.dt == '':
        dt = str(d)[:10]
    else:
        dt = options.dt

    d_before_1 = datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.timedelta(days=1)
    d_before_1 = str(d_before_1)[:10]
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    output_file_handler = logging.FileHandler('/tmp/yuqing_batchrun_runlog_%s.log' % (dt), mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)

    daysql = '''
        select nt.news_uuid news_uuid, nt.news_content news_content
        FROM
            (
                select news_uuid, news_content
                from i3.news_info
                where dt='%s'
            ) nt
        join 
            (
                select news_uuid
                from i3.algo_correlation
                where dt='%s' and label in ("吉利","友商","行业")
            ) cor 
        on nt.news_uuid = cor.news_uuid
        join 
            (
                select news_uuid
                from i3.algo_classified
                where dt='%s' and (class2_label=1 or class3_label=1 or class4_label=1 or class5_label=1)
            ) cls 
        on nt.news_uuid = cls.news_uuid
    ''' % (d_before_1, d_before_1, d_before_1)

    daydf = get_hivedf(daysql).head(100)
    logger.info("hive sql done， samples: %s" % str(len(daydf)))
    daydf["text"] = daydf["news_content"].apply(lambda x: cut_sentences(x))
    daydf["len"] = daydf["text"].apply(lambda x: len(x))
    daydf = daydf[daydf['len'] != 0].drop_duplicates(subset=["text"],keep="first")
    # daydf = apply_by_multiprocessing(daydf, cut_sentences, key='news_content', target='cl',workers=10)
    logger.info("cut_sentences done")
    # daydf["data"] = daydf.apply(lambda r: [{"text": i, "news_uuid": r["news_uuid"]} for i in r["cl"]], axis=1)
    # logger.info("form data done")
    # datadf = pd.DataFrame(np.sum(daydf["data"]))
    datadf = flatten_columns(daydf, ["text"])[["text", "news_uuid"]].fillna('')
    datadf["len"] = datadf["text"].apply(lambda x: len(x))
    datadf = datadf[datadf['len'] > 5]
    logger.info("sum data done， samples: %s" % str(len(datadf)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = BertClassifier1(model_path="/work/pretrain/huggingface/ernie_pretrain",
                           n_classes=[109, 3],
                           freeze_bert=False).to(device)
    model1.load_state_dict(
        torch.load("/work/kw/yuqing/gl_bert_classification_torch_huggingface/bert/mix/model/weight.ckpt",
                   map_location=device))
    tokenizer1 = AutoTokenizer.from_pretrained("/work/kw/yuqing/gl_bert_classification_torch_huggingface/bert/mix/model")

    model2 = BertClassifier2(model_path="/work/pretrain/huggingface/ernie_pretrain",
                           D_in=768, H=50,
                           D_out=109,
                           freeze_bert=False).to(device)
    model2.load_state_dict(
        torch.load("/work/kw/yuqing/gl_bert_classification_torch_huggingface/bert/property/model/weight.ckpt",
                   map_location=device))
    tokenizer2 = AutoTokenizer.from_pretrained("/work/kw/yuqing/gl_bert_classification_torch_huggingface/bert/property/model")

    datadf["pred_mix"] = datadf["text"].apply(lambda t: get_mix(model1, tokenizer1, t, device))
    datadf["property"] = datadf["text"].apply(lambda t: get_property(model2, tokenizer2, t, device))
    print(datadf[datadf["pred_mix"] == (-1, -1)].head())
    logger.info("predict done, -1 count: %s" % len(datadf[datadf["pred_mix"] == (-1, -1)]))

    datadf = datadf[datadf["pred_mix"] != (-1, -1)].drop_duplicates(subset=["text"],keep="first")
    #datadf["property"] = datadf["pred_mix"].apply(lambda x: get_label(x[0], 1))
    datadf["emotion"] = datadf["pred_mix"].apply(lambda x: get_label(x[1], 2))
    datadf["comb"] = datadf.apply(lambda r: (r["property"], r["emotion"]), axis=1)
    combset = set(datadf["comb"])
    sn = int(30000 / len(combset))
    dlist = []
    for comb in combset:
        subdf = datadf[datadf['comb'] == comb]
        if len(subdf) > sn:
            subdf = subdf.sample(n=sn, random_state=24)
        dlist.append(subdf)
    resultdf = pd.concat(dlist)
    resultdf = resultdf.drop(["comb", "pred_mix"], axis=1)
    resultdf["date"] = d_before_1
    logger.info("resultdf done， samples: %s" % str(len(resultdf)))

    resultdf.to_csv("model_batch_pred_%s.csv" % d_before_1, index=False, encoding='utf_8_sig')

if __name__ == "__main__":
    main()