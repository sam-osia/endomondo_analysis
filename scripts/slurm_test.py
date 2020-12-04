import tensorflow as tf
import logging
import os
from pathlib import Path
from utils import *


set_path('saman')
Path('./logs/test_logs').mkdir(parents=True, exist_ok=True)

log_dir = './logs/test_logs'
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger('slurm_log ')

logger.info('hello from remote')
logger.info(f'Version: {tf.__version__}')
logger.info(f'Built with Cuda - {tf.test.is_built_with_cuda()}')
logging.info(f'GPU count - {len(tf.config.list_physical_devices("GPU"))}')

