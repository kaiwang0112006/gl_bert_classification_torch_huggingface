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
from lime.lime_text import LimeTextExplainer
import i3_word_tokenizer
import rule_engine

sys.path.append("/work/kw/ppackage")
from miloTools.tools.multi_apply import *
from urllib import parse
import random


def get_hivedf(sql_select):
    connection = hive.Connection(host='10.240.16.114',
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
            # nn.Dropout(0.5),
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


class model_Pred():
    def __init__(self, model, tokenizer, max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenfun(self, text):
        encoded_sent = self.tokenizer.encode_plus(
            text=text_preprocessing(text),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=self.max_length,  # Max length to truncate/pad
            padding='max_length',  # Pad sentence to max length
            truncation=True,
            return_tensors='pt',  # Return PyTorch tensor
            return_attention_mask=True  # Return attention mask
        )

        input_ids = encoded_sent.get('input_ids')
        attention_masks = encoded_sent.get('attention_mask')
        return input_ids, attention_masks

    def get_proba_list(self, tlist):
        gpuid = random.randint(0, 1)
        device = torch.device('cuda:%s' % str(gpuid))
        self.model = self.model.to(device)
        predlist = []
        for text in tlist:
            input_ids, attention_masks = self.tokenfun(text)
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            with torch.no_grad():
                probs_detail = self.model(input_ids, attention_masks)
            prob = probs_detail[0].tolist()
            predlist.append(prob)
        return np.array(predlist)

    def get_pred_list(self, tlist):
        gpuid = random.randint(0, 1)
        device = torch.device('cuda:%s' % str(gpuid))
        self.model = self.model.to(device)
        predlist = []
        for text in tlist:
            input_ids, attention_masks = self.tokenfun(text)
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            with torch.no_grad():
                probs_detail = self.model(input_ids, attention_masks)
            prob = np.argmax(probs_detail[0].tolist())
            predlist.append(prob)
        return np.array(predlist)


def getkeywords(r, explainer, mobj, rlmap):
    w = r["pharse2_batchrun.text"]
    exp = explainer.explain_instance(w, mobj.get_proba_list, num_features=6,
                                     labels=[rlmap[r["pharse2_batchrun.property"]]])
    explist = exp.as_list(rlmap[r["pharse2_batchrun.property"]])
    return [[i[0], i[1]] for i in explist]


class cleanTokenv1_0_2:
    def __init__(self):
        jiebaobj = i3_word_tokenizer.jieba_cut.jiebaTools(filter_syb=False, env='dev')
        jiebaobj.reset_jieba()
        jiebaobj.jieba_setup()
        self.jieba = jiebaobj

    def cleanToken(self, text):
        # text = re.sub(r'(@.*?)[\s]', ' ', text)
        # Replace '&amp;' with '&'
        text = text_preprocessing(text)

        wordlist = self.jieba.jieba_custom_lcut(text)
        return [w for w in wordlist if len(w) > 0]


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


def main():
    df = get_hivedf("select * from i3_log.pharse2_batchrun where dt='2022-03-08' limit 2800")

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
            104: '门店布局', 105: '门店环境', 106: '门店设施', 107: '驾驶操作舒适性', 108: '驾驶视觉安全', 109: 'general'}
    rlmap = dict(zip(lmap.values(), lmap.keys()))

    model = BertClassifier(model_path="/work/pretrain/huggingface/ernie_pretrain",
                           D_in=768, H=50,
                           D_out=110,
                           freeze_bert=False)
    model.load_state_dict(
        torch.load("/work/kw/yuqing/gl_bert_classification_torch_huggingface/bert/property_ernie/model/weight.ckpt"))
    tokenizer = AutoTokenizer.from_pretrained(
        "/work/kw/yuqing/gl_bert_classification_torch_huggingface/bert/property_ernie/model")

    mobj = model_Pred(model, tokenizer)
    tokenobj = cleanTokenv1_0_2()
    wordfun = tokenobj.cleanToken

    class_names = [lmap[i] for i in range(110)]
    explainer = LimeTextExplainer(class_names=class_names, split_expression=wordfun)
    df["expwordslist"] = df.apply(lambda row: getkeywords(row, explainer, mobj, rlmap), axis=1)
    df = flatten_columns(df, ["expwordslist"]).fillna('')
    df["expwords"] = df["expwordslist"].apply(lambda x: x[0])
    df["len"] = df["expwords"].apply(lambda x:len(x))
    df = df[df["len"]>=2]
    df["expweight"] = df["expwordslist"].apply(lambda x: x[1])
    df.sort_values(by=["expwords", "expweight"], ascending=False).drop_duplicates(subset=["expwords"], keep="first")
    df.to_csv("expwords.csv", index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    main()