import paramiko
import threading
from multiprocessing.dummy import Pool as ThreadPool

cmd = ['echo hello!']
username = "dy20"
passwd = "dy20"
success_IP = []

def ssh(ip):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip,22,username,passwd,timeout=1)
        for m in cmd:
            stdin, stdout, stderr = ssh.exec_command(m)
            out = stdout.readlines()
            for o in out:
                print (o)
        print( '%s\tOK\n'%(ip)  )
        success_IP.extend([ip])
        #ssh.close()  
    except Exception as e:
        print(ip,e)
        1==1

ippool= []
for i in range(52,56):
    for j in range(0,255):
        ip = '166.111.{}.{}'.format(i,j)
        ippool.extend([ip])

if __name__=='__main__':
    
    print( "Begin......" )
    pool = ThreadPool()
    pool.map(ssh, ippool)
    pool.close()
    pool.join()
    print('ALL:',success_IP)

