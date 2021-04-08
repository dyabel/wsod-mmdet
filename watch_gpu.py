import GPUtil
import os
import time
import psutil

stopped_num = 10000000  # （设置一个最大获取次数，防止记录文本爆炸）
delay = 10  # 采样信息时间间隔
Gpus = GPUtil.getGPUs()


def get_gpu_info():
    '''
    :return:
    '''
    gpulist = []
    GPUtil.showUtilization()

    # 获取多个GPU的信息，存在列表里

    gpu = Gpus[1]
    # for gpu in Gpus:
    #     print('gpu.id:', gpu.id)
    #     print('GPU总量：', gpu.memoryTotal)
    #     print('GPU使用量：', gpu.memoryUsed)
    #     print('gpu使用占比:', gpu.memoryUtil * 100)
    #     按GPU逐个添加信息
        # gpulist.append([gpu.id, gpu.memoryTotal, gpu.memoryUsed, gpu.memoryUtil * 100])
    return gpu.memoryUtil



def get_cpu_info():
    ''' :return:
    memtotal: 总内存
    memfree: 空闲内存
    memused: Linux: total - free,已使用内存
    mempercent: 已使用内存占比
    cpu: 各个CPU使用占比
    '''
    mem = psutil.virtual_memory()
    memtotal = mem.total
    memfree = mem.free
    mempercent = mem.percent
    memused = mem.used
    cpu = psutil.cpu_percent(percpu=True)

    return memtotal, memfree, memused, mempercent, cpu


# 主函数
def main():
    times = 0
    gpu_percentage = 1
    while gpu_percentage>0.3:
        # 最大循环次数
        # 打印当前时间
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        # 获取CPU信息
        # cpu_info = get_cpu_info()
        # 获取GPU信息
        # gpu_info = get_gpu_info()
        gpu_percentage = get_gpu_info()
        # print(gpu_percentage)
        # 添加时间间隙
        # print(cpu_info)
        # print(gpu_info, '\n')
        time.sleep(delay)
        times += 1

    os.system('./exec.sh')


if __name__ == '__main__':
    main()

