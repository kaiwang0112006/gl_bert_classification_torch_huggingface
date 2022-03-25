# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf, HiveContext
import pyspark.sql.functions as F
import pyspark.sql.types as T
from sqlalchemy import create_engine
import pymysql
import datetime
import argparse
import pandas as pd
from pyhive import hive
from urllib import parse
import subprocess
##########################################
## Options and defaults
##########################################
def getOptions():
    parser = argparse.ArgumentParser(description='python *.py [option]')
    requiredgroup = parser.add_argument_group('required arguments')
    requiredgroup.add_argument('--dt', dest='dt', help='dt', default='')
    args = parser.parse_args()

    return args

def main():
    options = getOptions()
    d = datetime.datetime.now()

    if options.dt == '':
        dt = str(d)[:10]
    else:
        dt = options.dt

    d_before_1 = datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.timedelta(days=1)
    d_before_1 = str(d_before_1)[:10]

    get_sql = '''select * from emotion_batchrun where datatime='%s' ''' % d_before_1

    db_info = {
        "host": "10.240.16.33",
        "port": 3306,
        "user": "sentiment",
        "password": parse.quote_plus("ysu839@Am%8sq"),
        "database": "i3"
    }

    engine = create_engine('mysql+pymysql://%(user)s:%(password)s@%(host)s:%(port)s/%(database)s?charset=utf8' % db_info,encoding='utf-8')
    datadf = pd.read_sql(sql=get_sql, con=engine).rename(columns={"text":"sentence"}).fillna("")
    print(datadf.head())
    print(datadf.shape)
    spark= SparkSession \
        .builder \
        .appName("model_monitor_feature_cal") \
        .config("hive.exec.dynamic.partition", 'true') \
        .config("hive.exec.max.dynamic.partitions", '2048') \
        .config("hive.exec.dynamic.partition.mode", 'nonstrict') \
        .enableHiveSupport() \
        .getOrCreate()

    sparkContext = spark.sparkContext
    dropcmd = '''hive -e "drop table if exists i3_log.pharse2_batchrun_%s"''' % (d_before_1.replace("-","_"))
    dropoutput = subprocess.getstatusoutput(dropcmd)

    create_temp = '''
        create table i3_log.pharse2_batchrun_%s
        (
            sentence   string COMMENT  'text'
            ,news_uuid	 string COMMENT  '文章id'
            ,property	  string COMMENT  '属性预测'
            ,emotion	  string COMMENT  '情感预测'
            ,hits	  string COMMENT  '情感预测'
            ,datatime	  string COMMENT  '时间'
        )
    ''' % (d_before_1.replace("-","_"))
    cmd = '''hive -e "%s"''' % create_temp
    createoutput = subprocess.getstatusoutput(cmd)
    datadf["dt"] = d_before_1
    datadf_sp = spark.createDataFrame(datadf)

    datadf_sp.write.mode("overwrite").saveAsTable("i3_log.pharse2_batchrun_%s" % (d_before_1).replace("-","_"))
    hivecmd = '''hive -e "insert overwrite table i3_log.pharse2_batchrun partition (dt='%s') select sentence, news_uuid, property, emotion, hits, datatime from i3_log.pharse2_batchrun_%s" ''' % (d_before_1, d_before_1.replace("-","_"))
    print(hivecmd)
    createoutput = subprocess.getstatusoutput(hivecmd)
    print(createoutput[-1].replace("\\n",'\n'))
    dropoutput = subprocess.getstatusoutput(dropcmd)



if __name__ == '__main__':
    main()