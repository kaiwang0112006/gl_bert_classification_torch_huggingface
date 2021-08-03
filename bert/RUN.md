# run
    python runbert.py --run=roberta_news --pretrain=/work/pretrain/huggingface/roberta-base-finetuned-chinanews-chinese/ --device=cuda:1;python /work/kw/send_feishu.py --msg="done"
    python runbert.py --run=roberta_dp --pretrain=/work/pretrain/huggingface/roberta-base-finetuned-dianping-chinese/  ;python /work/kw/send_feishu.py --msg="done"
    python runbert.py --run=ernie --pretrain=/work/pretrain/huggingface/ernie/  ;python /work/kw/send_feishu.py --msg="done"
    python runbert.py --run=bertbase --pretrain=/work/pretrain/huggingface/bert-base-chinese/ --device=cuda:0  ;python /work/kw/send_feishu.py --msg="base done"


