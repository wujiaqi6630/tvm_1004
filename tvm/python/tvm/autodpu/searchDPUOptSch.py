# wjq 2020/10/04
# Search a opt cfg for current workload(TODO: wjq)
from tvm import DPU_path as Dp

import sys
sys.path.append(Dp.addDPUOptSchToLibPath)
from  addDPUOptSchToLib import *

import numpy

#from .addDPUOptSchToLib import addDPUOptSchToLib

def searchDPUOptSch(workload):
    workload1 = 'inputStr'
    cfg = 'schedule'
    addDPUOptSchToLib(workload1, cfg)
    return cfg
