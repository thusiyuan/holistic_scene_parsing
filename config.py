"""
Created on Aug 26, 2017

@author: Siyuan Huang

configuration of the project

"""

import errno
import logging
import os


class Paths(object):
    def __init__(self, dataset='SUNRGBD'):
        """
        Configuration of data paths.
        """
        self.dataset = dataset
        self.metadata_root = 'data'
        if self.dataset == 'SUNRGBD':
            self.data_root = '/home/siyuan/Documents/Dataset/SUNRGBD_ALL'
            self.clean_data_root = os.path.join(self.data_root, 'data_clean')
            self.proposal_root = os.path.join(self.metadata_root, 'sunrgbdproposals')


def set_logger(name='learner.log'):
    if not os.path.exists(os.path.dirname(name)):
        try:
            os.makedirs(os.path.dirname(name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    logger = logging.getLogger(name)
    file_handler = logging.FileHandler(name, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                                "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger
