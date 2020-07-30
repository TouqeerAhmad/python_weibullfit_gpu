import numpy as np
import torch
from pynvml import *

from backend_weibull import fit

def determine_splits(inputTensor, tailSize, isSorted = 0):
    
    dtype_bytes = 8 # since float64
    # split chunks according to available GPU memory
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    gpu_free_mem = info.free / (1024 * 1024) # amount of free memeory in MB
    print(gpu_free_mem)
    
    height, width = inputTensor.shape[0], inputTensor.shape[1]
    if (isSorted): 
      # memory to hold sorted tensor
      size_in = height * tailSize * dtype_bytes
    else:
      # memory to hold input + sorted tensor 
      size_in = height * width * dtype_bytes + height * tailSize * dtype_bytes
    
    size_intermediate = height * 3 * dtype_bytes + height * 2 * dtype_bytes + height * tailSize * dtype_bytes  
    size_out = height * 5 * dtype_bytes
    total_mem = (size_in + size_intermediate + size_out) / (1024 * 1024) # amount in MB
    print(total_mem)
    
    if total_mem < (gpu_free_mem * 0.7): #no chunks if GPU mem is enough
        split = 1
    else:
        split = round((total_mem) / (gpu_free_mem * 0.7))  
    return split
    

def weibullFitting(dataTensor, tailSize, sign, isSorted = 0):
  
  N = dataTensor.shape[0]
  dtype = dataTensor.dtype
  deviceName = dataTensor.device
  
  if (isSorted):
    sortedTensor = dataTensor
  else:
    if sign == -1:
      dataTensor = -dataTensor
    sortedTensor = torch.topk(dataTensor, tailSize, dim=1, largest=True, sorted=True).values
  
  smallScoreTensor = sortedTensor[:,tailSize-1].unsqueeze(1)
  signTensor = sign * torch.ones((N,1), dtype=dtype, device=deviceName)
  trnaslateAmoutTensor = torch.ones((N,1), dtype=dtype, device=deviceName)
  processedTensor = sortedTensor + trnaslateAmoutTensor - smallScoreTensor
  wbFits = fit(processedTensor)
  
  print ("signTensor",wbFits.dtype,signTensor.dtype,smallScoreTensor.dtype)
  resultTensor = torch.cat([wbFits[:,1].unsqueeze(1),wbFits[:,0].unsqueeze(1),signTensor,trnaslateAmoutTensor,smallScoreTensor], dim=1)

  return resultTensor

def FitLow(data, tailSize, isSorted = 0):
  return weibullFitting(data, tailSize, -1, isSorted)
  
def FitHigh(data, tailSize, isSorted = 0):
  return weibullFitting(data, tailSize, 1, isSorted)
  

