""" configurations for this project

author baiyu
"""
from datetime import datetime

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

FREE_STAGES = 2

FC_CHANNEL = [1024]

EPOCH = 200


#time of we run the script
TIME_NOW = datetime.now().isoformat()

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epochYER
SAVE_EPOCH = 20








