import numpy as np
import random
import copy

class Utils:
    def __init__(self):
        pass
    
    def decimalToBinaryRec(num,ret,idx):
        if num > 1:
            Utils.decimalToBinaryRec(num // 2, ret, idx-1)
        ret[idx] = num % 2
    
    def decimalToBinary(num):
        ret = [0,0,0,0,0,0,0,0]
        Utils.decimalToBinaryRec(num,ret,len(ret)-1)
        return ret

    def binaryToDecimal(num):
        aux = copy.deepcopy(num)
        aux.reverse()
        res = 0
        for idx, n in enumerate(aux):
            res += n * (2**idx)
        return res
    
    #[0,1,2,3,4,5,6,7]
    #pm = 0.20 * (i*0.10)
    
    def bitMutation(bit, pm):
        if pm > random.random():
            if bit == 0:
                bit = 1
            else:
                bit = 0
        return bit
            