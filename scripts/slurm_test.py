import tensorflow as tf
import logging
import os

from utils import *


set_path('saman')
log_dir = './logs/test_logs'
logging.basicConfig(filename=os.path.join(log_dir, 'test.log'),
                    filemode='a',
                    format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger('slurm_log ')

logger.info('hello from remote')
logger.info(f'Built with Cuda - {tf.test.is_built_with_cuda()}')
# logging.info(f'GPU count - {len(tf.config.list_physical_devices("GPU"))}')
