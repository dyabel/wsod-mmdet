from multiprocessing import Process
import time
a = {}
def work(i):
    global a
    a[i] = 1
process_list = []
for n in range(100):
    p = Process(target=work,args=(n,))
    process_list.append(p)
    p.start()
# time.sleep(10)
for p in process_list:
    p.join()
    print('one process is over +++++')

print(a)
