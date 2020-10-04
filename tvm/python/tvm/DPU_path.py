# wjq 2020/10/04
# dpu lib path

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
DPUAutoTuningLibraryPath = current_dir + r'/autodpu/DPUAutoTuningibrary.json'
searchDPUOptSchPath = current_dir + r'/autodpu'
addDPUOptSchToLibPath = current_dir + r'/autodpu'
randomArrayPath = current_dir + r'/autodpu/randomData.txt'