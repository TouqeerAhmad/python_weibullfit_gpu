import numpy as np
import torch
from weibull.backend_pytorch import fit
import timeit


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
  

def test_weibullFit():
  fileName = 'sample_data/weibulls_example_protocol2.npy'
  data = np.load(fileName, allow_pickle=True)
  
  numInstances = 10
  dataSize = 40000
  tailSize = 2500
  distance_multiplier = 0.5
  
  data1 = np.zeros(shape=(numInstances,dataSize), dtype=np.float64)
  weibull_fits_libmr = np.zeros(shape=(numInstances,5), dtype=np.float64)
  
  for k in range(numInstances):
    data1[k,:] = distance_multiplier * data[k][0]
    weibull_fits_libmr[k,:] = data[k][1]
  
  dataTensor = torch.from_numpy(data1)
  dataTensor = dataTensor.cuda()
  
  result = FitLow(dataTensor, tailSize, 0)
  
  print(result)
  
  return
  
if __name__ == '__main__':
  test_weibullFit()
  
