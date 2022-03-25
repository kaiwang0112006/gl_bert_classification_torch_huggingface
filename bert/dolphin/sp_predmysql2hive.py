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
        "host":"10.24.16.144",
        "port":3306,
        "user":"sentiment",
        "password":"uys9W%6ysx#s7p",
        "database":"i3"
    }

    engine = create_engine('mysql+pymysql://%(user)s:%(password)s@%(host)s:%(port)s/%(database)s?charset=utf8' % db_info,encoding='utf-8')
    datadf = pd.read_sql(sql=get_sql, con=engine)

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
    #sqlContext = HiveContext(sparkContext)
    logdf_sp = spark.createDataFrame(datadf)
    logdf_sp.createOrReplaceTempView("datadf")

    spark.sql('''
        insert overwrite table i3_dev.pharse2_batchrun partition  (dt='%s')
        select text, news_uuid, property, emotion, hits, datatime
        from datadf
    ''' % d_before_1).show()



if __name__ == '__main__':
    main()