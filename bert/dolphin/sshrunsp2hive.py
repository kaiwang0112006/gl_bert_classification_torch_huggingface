import json

import paramiko
import os
import datetime
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

    host = '10.24.16.97'

    port = 22
    username = "root"

    pkey = "/root/.ssh/id_rsa"  # 本服务器的私钥文件路径[此文件服务器上~/.ssh/id_rsa可下载到本地]

    key = paramiko.RSAKey.from_private_key_file(pkey)

    # 创建并连接远程主机
    ssh = paramiko.SSHClient()

    # 允许连接不在know_hosts文件中的主机
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # ssh.load_system_host_keys() #如通过known_hosts 方式进行认证可以用这个,如果known_hosts 文件未定义还需要定义 known_hosts

    ssh.connect(host, port, username=username, pkey=key)
    #
    # # 远程执行shell脚本，获取输入输出流
    stdin, stdout, stderr = ssh.exec_command('su - hdfs -c "/opt/bmr/spark2/bin/spark-submit --master yarn  --num-executors 5 --deploy-mode client --driver-memory 5g --executor-memory 10g --conf spark.rpc.message.maxSize=2046 --conf spark.sql.sources.partitionOverwriteMode=dynamic /work/kw/yuqing/predmysql2hive.py --dt=%s"' % d_before_1)
    #
    print(json.dumps(stdout.readlines()))
    ssh.close()

if __name__ == '__main__':
    main()