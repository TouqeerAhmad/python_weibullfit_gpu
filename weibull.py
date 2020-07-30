import numpy as np
import torch
from backend_weibull import fit

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
  
  resultTensor = torch.cat([wbFits[:,1].unsqueeze(1),wbFits[:,0].unsqueeze(1),signTensor,trnaslateAmoutTensor,smallScoreTensor], dim=1)

  return resultTensor

def FitLow(data, tailSize, isSorted = 0):
  return weibullFitting(data, tailSize, -1, isSorted)
  
def FitHigh(data, tailSize, isSorted = 0):
  return weibullFitting(data, tailSize, 1, isSorted)
