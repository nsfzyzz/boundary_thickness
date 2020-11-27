#!/usr/bin/python
#!/usr/bin/python3

# This script assume exclusive usage of the GPUs. 
# If you have limited usage of GPUs, you can limit the range of gpu indices you are using.


import threading
import time
import os
import os.path
import numpy as np
import argparse

import gpustat
import logging

import itertools

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'
FORMAT_MINIMAL = '%(message)s'

logger = logging.getLogger('runner')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description='Measure boundary thickness')
parser.add_argument('--available-gpus', type=int, nargs='+', default=[0],
                    help='Which GPUs can you use?')
parser.add_argument('--gpu-memory-threshold', type=int, default = 500, 
                    help='GPU memory threshold in MB')

args = parser.parse_args()

exitFlag = 0
GPU_MEMORY_THRESHOLD = args.gpu_memory_threshold

def get_free_gpu_indices():
    '''
        Return an available GPU index.
    '''
    while True:
        stats = gpustat.GPUStatCollection.new_query()
        # print('stats length: ', len(stats))
        return_list = []
        for i, stat in enumerate(stats.gpus):
            memory_used = stat['memory.used']
            if memory_used < GPU_MEMORY_THRESHOLD and i in args.available_gpus:
                return i

        logger.info("Waiting on GPUs")
        time.sleep(10)

def check_result(result_names):
    
    for result_name in result_names:
    
        if not os.path.isfile(result_name):
            
            return False
        
    return True
        
class DispatchThread(threading.Thread):
    def __init__(self, threadID, name, counter, bash_command_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.bash_command_list = bash_command_list

    def run(self):
        logger.info("Starting " + self.name)
        # print_time(self.name, self.counter, 5)
        threads = []
        for i, (bash_command, result_names) in enumerate(self.bash_command_list):
            
            time.sleep(1)
            
            if check_result(result_names):
                print("Result already exists! {0}".format(result_names))
                continue
                
            else:
                print("Result not ready yet. Running it for a second time: {0}".format(result_names))

            cuda_device = get_free_gpu_indices()
            thread1 = ChildThread(1, f"{i}th + {bash_command}", 1, cuda_device, bash_command)
            thread1.start()
            
            time.sleep(10)
            threads.append(thread1)
            

        # join all.
        for t in threads:
            t.join()
        logger.info("Exiting " + self.name)


class ChildThread(threading.Thread):
    def __init__(self, threadID, name, counter, cuda_device, bash_command):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.cuda_device = cuda_device
        self.bash_command = bash_command

    def run(self):
        # os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.cuda_device[0]},{self.cuda_device[1]}'
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.cuda_device}'
        bash_command = self.bash_command

        logger.info(f'executing {bash_command} on GPU: {self.cuda_device}')
        # ACTIVATE
        os.system(bash_command)
        import time
        import random
        time.sleep(random.random() % 5)

        logger.info("Finishing " + self.name)      

def return_command(FILE_NAME, LOG_NAME, PREFIX_NAME, IND, ADV_PARAMS=""):
    
    if not ADV_PARAMS=="":
        ADV_PARAMS = ADV_PARAMS+" "
            
    return "python {0}.py --exp-ind {1} --file-prefix {2} {3}1>./logs/thickness_model_zoo/{4}_ind_{5}.log 2>./logs/thickness_model_zoo/{4}_ind_{5}.err".format(FILE_NAME, IND, PREFIX_NAME, ADV_PARAMS, LOG_NAME, IND)

def return_results(RESULT_DIR, PREFIX_NAME, IND):

    result_names = []
    for ind in range(IND, IND+1):
        result_names.append(RESULT_DIR+PREFIX_NAME+'_ind_'+str(ind)+'.pkl')
        
    return result_names

            
BASH_COMMAND_LIST = []

RESULT_DIR = "./results/thickness_model_zoo/"

if not os.path.exists('./logs/thickness_model_zoo'):
    os.makedirs('./logs/thickness_model_zoo')

for IND in range(15):
    
    FILE_NAME="measure_thickness"
    
    # Thickness on adv direction with default beta
    
    PREFIX_NAME="thickness_adv_200"
    
    LOG_NAME=PREFIX_NAME
    
    ADV_PARAMS="--alpha 0 --beta 0.75 --class-pair --reproduce-model-zoo"
    
    RESULT_NAME=return_results(RESULT_DIR, PREFIX_NAME, IND)

    BASH_COMMAND_LIST.append((return_command(FILE_NAME, LOG_NAME, PREFIX_NAME, IND, ADV_PARAMS), RESULT_NAME))
    
    """
    # Thickness on large attack size
    
    PREFIX_NAME="thickness_adv_200_large"
    
    LOG_NAME=PREFIX_NAME
    
    ADV_PARAMS="--alpha 0 --beta 0.75 --class-pair --eps 2.0 --step-size 0.4 --reproduce-model-zoo"
    
    RESULT_NAME=return_results(RESULT_DIR, PREFIX_NAME, IND)

    BASH_COMMAND_LIST.append((return_command(FILE_NAME, LOG_NAME, PREFIX_NAME, IND, ADV_PARAMS), RESULT_NAME))
    
    # Thickness on small attack size
    
    PREFIX_NAME="thickness_adv_200_small"
    
    LOG_NAME=PREFIX_NAME
    
    ADV_PARAMS="--alpha 0 --beta 0.75 --class-pair --eps 0.6 --step-size 0.12 --reproduce-model-zoo"
    
    RESULT_NAME=return_results(RESULT_DIR, PREFIX_NAME, IND)

    BASH_COMMAND_LIST.append((return_command(FILE_NAME, LOG_NAME, PREFIX_NAME, IND, ADV_PARAMS), RESULT_NAME))
    
    # Thickness on adversarial direction with different betas
    
    # beta = 0.9
    PREFIX_NAME="thickness_adv_200_beta_09"

    LOG_NAME=PREFIX_NAME
    
    RESULT_NAME=return_results(RESULT_DIR, PREFIX_NAME, IND)

    ADV_PARAMS="--alpha 0 --beta 0.9 --class-pair --reproduce-model-zoo"

    BASH_COMMAND_LIST.append((return_command(FILE_NAME, LOG_NAME, PREFIX_NAME, IND, ADV_PARAMS), RESULT_NAME))
    
    # beta = 0.8
    PREFIX_NAME="thickness_adv_200_beta_08"

    LOG_NAME=PREFIX_NAME
    
    RESULT_NAME=return_results(RESULT_DIR, PREFIX_NAME, IND)

    ADV_PARAMS="--alpha 0 --beta 0.8 --class-pair --reproduce-model-zoo"

    BASH_COMMAND_LIST.append((return_command(FILE_NAME, LOG_NAME, PREFIX_NAME, IND, ADV_PARAMS), RESULT_NAME))
    
    # beta = 0.7
    PREFIX_NAME="thickness_adv_200_beta_07"

    LOG_NAME=PREFIX_NAME
    
    RESULT_NAME=return_results(RESULT_DIR, PREFIX_NAME, IND)

    ADV_PARAMS="--alpha 0 --beta 0.7 --class-pair --reproduce-model-zoo"

    BASH_COMMAND_LIST.append((return_command(FILE_NAME, LOG_NAME, PREFIX_NAME, IND, ADV_PARAMS), RESULT_NAME))
    
    # beta = 0.6
    PREFIX_NAME="thickness_adv_200_beta_06"

    LOG_NAME=PREFIX_NAME
    
    RESULT_NAME=return_results(RESULT_DIR, PREFIX_NAME, IND)

    ADV_PARAMS="--alpha 0 --beta 0.6 --class-pair --reproduce-model-zoo"

    BASH_COMMAND_LIST.append((return_command(FILE_NAME, LOG_NAME, PREFIX_NAME, IND, ADV_PARAMS), RESULT_NAME))
    """

# Create new threads
dispatch_thread = DispatchThread(2, "Thread-2", 4, BASH_COMMAND_LIST[:])

# Start new Threads
dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")