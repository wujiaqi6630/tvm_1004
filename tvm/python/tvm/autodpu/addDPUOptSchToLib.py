# wjq 2020/10/04
# Initial opt lib and add a new cfg into LIB
import json
import random
import optDictLib

from tvm import DPU_path as Dp

def transferIntoStr(workload):
    str1 = ''
    if workload:
        if isinstance(workload, tuple):
            for key in workload:
                if isinstance(key, str):
                    str1 = str1 + key + r'##'
                elif isinstance(key, tuple):
                    for k in key:
                        if isinstance(k, int):
                            str1 = str1 + str(k) + r'&'
                        elif isinstance(k, str):
                            str1 = str1 + k
                        else:
                            raise ValueError("DPUError: Illegal format.")
                    str1 = str1 + r'##'
                elif isinstance(key, int):
                    str1 = str1 + str(key) + r'##'
                elif isinstance(key, float):
                    floatKey = float('%0.6f'%key)
                    str1 = str1 + str(floatKey) + r'##'
                else:
                    raise ValueError("DPUError: Illegal format.")
        elif isinstance(workload, str):
            str1 = workload
        else:
            raise ValueError("DPUError: Illegal format.")
        
    return str1

def addDPUOptSchToLib(workload, cfg):
    r"""
    cfg : list, len=3
    cfg[0] : int
        On behalf of pragma SIMD.
        Use number to represents the layer-th of the loop body.
        Such as : 1 means 'bn' layer loop, 2 means 'oc' layer loop,...,7 means 'kw' layer loop.
    cfg[1] : list 
        On behalf of pragma loop_splite.
        For example, [2,3,-1,4,-4,1] : 
            First number '2' represents the layer-th of the loop body.
            Second number '3' represents the nums of splite segments of current layer loop.
            Thrid number '-1' means blockIdx_z; '-2' means blockIdx_y; '-3' means blockIdx_x.
                         '-4' means threadIdx_z; '-5' means threadIdx_y; '-6' means threadIdx_x.
            Fourth number '4' means the val of threadIdx_xx.
            Fifth number '-4' threadIdx_z
            Sixth number '1' means the val of local loop in current loop.
    cfg[2] : list
        On behalf of pragma unroll.
        The val of the list represents the layer-th of the loop body.
    """

    def preTreatment():
        # data_layout="NCHW",
        # kernel_layout="OIHW",
        AlexNetTuningParamsDict = {}
        for k in optDictLib.preDict:
            kStr = transferIntoStr(k)
            AlexNetTuningParamsDict[kStr] = optDictLib.preDict[k]

        with open(Dp.DPUAutoTuningLibraryPath, 'w') as f:
            json.dump(AlexNetTuningParamsDict,f)
        f.close()

    preTreatment()
    retStr = transferIntoStr(workload)
    # notice : if a new tuning params need to be added into Lib, 
    #          must load first, and then write into
    #retDict = {retStr:cfg}
    DPUAutoTuningData = {}
    with open(Dp.DPUAutoTuningLibraryPath, 'r') as f:
        result = f.read()
        if len(result) > 10:
            f.seek(0)
            DPUAutoTuningData = json.load(f)
            DPUAutoTuningData[retStr] = cfg
    f.close()
    with open(Dp.DPUAutoTuningLibraryPath, 'w') as f:
        json.dump(DPUAutoTuningData,f)
    f.close()

