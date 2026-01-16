import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import cv2
from pathlib import Path

module_logger = logging.getLogger(__name__)

def get_data_path():
    return Path(__file__).parent / "data"

def gamma_correct(img, gamma=2.2):
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if img.dtype == np.float32 or img.dtype == np.float64:
        return img**(1/gamma)

    # Use a LUT for fixed precision
    if img.dtype == np.uint8:
        M = 255
    elif img.dtype == np.uint16:
        M = 65535
    
    x = np.arange(M + 1)
    y = ((x.astype(np.float64) / M)**(1/gamma)*M).astype(img.dtype)

    img_gc = np.take(y, img)
    return img_gc


def list_of_dicts_to_dict(lod):
    D = {}
    for d in lod:
        D.update(d)
    return D

class CircularBuffer:
    def __init__(self, buffer_size, dtype):
        self.buf = np.zeros(buffer_size, dtype=dtype)
        self.head = 0
        self.queue_full = False
        self.length = buffer_size
    
    def push(self, x):
        self.buf[self.head] = x

        if self.head == self.buf.size - 1:
            self.queue_full = True

        self.head = (self.head + 1) % self.buf.size
    
    def push_batch(self, x: np.ndarray):
        if self.head + x.shape[0] <= self.buf.size:
            self.buf[self.head:self.head + x.shape[0]] = x
            self.head += x.shape[0]
        else:
            end_space = self.length - self.head
            self.buf[self.head:] = x[:end_space]
            self.buf[:x.shape[0] - end_space] = x[end_space:]
            self.head = (self.head + x.shape[0]) % self.buf.size
            self.queue_full = True
    
    def unravel(self, pad_to_full_size=False):
        if not self.queue_full:
            if not pad_to_full_size:
                return self.buf[:self.head]
            else:
                return np.concatenate((self.buf[:self.head], np.zeros(self.buf.size - self.head, dtype=self.buf.dtype)))
        else:
            return np.roll(self.buf, -self.head)