drop table if exists i3_log.pharse2_batchrun;
create EXTERNAL  table i3_log.pharse2_batchrun
(
    text   string COMMENT  'text'
    ,news_uuid	 string COMMENT  '文章id'
    ,property	  string COMMENT  '属性预测'
    ,emotion	  string COMMENT  '情感预测'
    ,hits	  string COMMENT  '情感预测'
    ,datatime	  string COMMENT  '时间'
) partitioned by (dt String)
ROW FORMAT SERDE
  'org.apache.hadoop.hive.ql.io.orc.OrcSerde'
WITH SERDEPROPERTIES (
  'collection.delim'=',',
  'field.delim'='',
  'serialization.format'='')
STORED AS INPUTFORMAT
  'org.apache.hadoop.hive.ql.io.orc.OrcInputFormat'
OUTPUTFORMAT
  'org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat'
 TBLPROPERTIES (
   'bucketing_version'='2',
   'transactional'='FALSE');

create EXTERNAL  table i3_etl.pharse2_batchrun
(
    text   string COMMENT  'text'
    ,news_uuid	 string COMMENT  '文章id'
    ,property	  string COMMENT  '属性预测'
    ,emotion	  string COMMENT  '情感预测'
    ,hits	  string COMMENT  '情感预测'
    ,datatime	  string COMMENT  '时间'
) partitioned by (dt String)
ROW FORMAT SERDE
  'org.apache.hadoop.hive.ql.io.orc.OrcSerde'
WITH SERDEPROPERTIES (
  'collection.delim'=',',
  'field.delim'='',
  'serialization.format'='')
STORED AS INPUTFORMAT
  'org.apache.hadoop.hive.ql.io.orc.OrcInputFormat'
OUTPUTFORMAT
  'org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat'
 TBLPROPERTIES (
   'bucketing_version'='2',
   'transactional'='FALSE');

CREATE EXTERNAL TABLE `i3_log.pharse2_batchrun`(
  `text` string COMMENT 'text',
  `news_uuid` string COMMENT 'news_uuid',
  `property` string COMMENT 'property',
  `emotion` string COMMENT 'emotion',
  `hits` string COMMENT 'hits',
  `datatime` string COMMENT 'datatime')
PARTITIONED BY (
  `dt` string)
ROW FORMAT SERDE
  'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
WITH SERDEPROPERTIES (
  'collection.delim'=',',
  'field.delim'='\t',
  'serialization.format'='')
STORED AS INPUTFORMAT
  'org.apache.hadoop.mapred.TextInputFormat'
OUTPUTFORMAT
  'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION
  'hdfs://bmr-cluster/warehouse/tablespace/external/hive/i3_log.db/pharse2_batchrun'


CREATE EXTERNAL TABLE `i3_log.pharse2_batchrun`(
  text string COMMENT  'text',
  news_uuid string COMMENT  '文章id',
  property string COMMENT  '属性预测',
  emotion string COMMENT  '情感预测',
  hits  string COMMENT  '情感预测',
  datatime string COMMENT  '时间'
  )
PARTITIONED BY (
  `dt` string)
ROW FORMAT SERDE
  'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
WITH SERDEPROPERTIES (
  'collection.delim'=',',
  'field.delim'='\t',
  'serialization.format'=',')
STORED AS INPUTFORMAT
  'org.apache.hadoop.mapred.TextInputFormat'
OUTPUTFORMAT
  'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'