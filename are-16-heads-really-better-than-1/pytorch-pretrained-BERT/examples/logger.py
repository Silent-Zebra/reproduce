import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
