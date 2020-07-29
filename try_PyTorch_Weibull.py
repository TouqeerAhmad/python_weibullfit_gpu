import numpy as np
import torch
#import weibull
from weibull.backend_pytorch import fit
import timeit

def main():
  
  start = timeit.default_timer()
  
  data1 = np.zeros(shape=(10,2500), dtype=np.float64)
  for k in range(10):
    fileName1 = '/home/tahmad/work/stand_alone_libMr/libmr_gpu/sample_data/FV_' + str(k) + '_tail_processed.bin'
    data1[k] = np.fromfile(fileName1, dtype='float64')
  
  stop = timeit.default_timer()
  print('Time: ', stop - start)  

  
  start = timeit.default_timer()
  for count in range(1000):
    for k in range(10):
      #print('Weibull fit through https://github.com/mlosch/python-weibullfit -- testing with numpy version for now')
      A = weibull.fit(data1[k])
      #print('Weibull Parameters for instance = ' + str(k))
      #print(A)  
  
  stop = timeit.default_timer()
  print('Time: ', stop - start)  



def main2():
  
  data1 = np.zeros(shape=(10,2500), dtype=np.float64)
  
  
  iter = 0
  k = 0
  for iter in range(10):
    #fileName1 = '/home/tahmad/work/stand_alone_libMr/libmr_gpu/sample_data/FV_' + str(k) + '_tail_processed.bin'
    fileName1 = '/home/tahmad/work/stand_alone_libMr/libmr_gpu/sample_data/FV_' + str(k) + '_tail.bin'
    data1[iter] = np.fromfile(fileName1, dtype='float64')
    k += 1
    if k == 10:
      k = 0
    
  
  print(data1.shape)
  data = torch.from_numpy(data1)
  
  start = timeit.default_timer()
  for k in range(1):
    A = fit(data)
    print(A)
  stop = timeit.default_timer()
  print('Time: ', stop - start)



def verify_pre_processing():
  print('verify_pre_processing()')
  fileName = '/home/tahmad/work/stand_alone_libMr/data_from_Steve/weibulls_example_protocol2.npy'
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
  
  # Pre-processing -- same as weibull.c
  sign = -1.0 * np.ones(shape=numInstances, dtype=np.float64)
  data2 =  np.zeros(shape=(numInstances,tailSize), dtype=np.float64)
  
  translate_amount = 1.0 * np.ones(shape=numInstances, dtype=np.float64)
  small_score = np.zeros(shape=numInstances, dtype=np.float64) 
  
  for k in range(numInstances):
    data2[k,:] = (np.sort(sign[k] * data1[k,:])[::-1])[:tailSize]
    small_score[k] = data2[k,tailSize-1]
    data2[k,:] = data2[k,:] + translate_amount[k] - small_score[k]
  
  
  
  print(' ')
  print('Weibull Fits as reported by libmr -- saved by Steve')  
  print(weibull_fits_libmr)
  
  
  dataTensor = torch.from_numpy(data2)
  
  print(' ')
  print('Weibull Fits as Torch Tensor')
  start = timeit.default_timer()
  A = fit(dataTensor)
  stop = timeit.default_timer()
  print(A)
  print('Time: ', stop - start)
  
  
  print(' ')
  print('Reporting Weibull Fits in libmr format -- with other parameters e.g. samll_score')
  
  weibull_fits_PyTorch = np.zeros(shape=(numInstances,5), dtype=np.float64)
  tempTensor = A.cpu().numpy()
  
  weibull_fits_PyTorch[:,0] = tempTensor[:,1]
  weibull_fits_PyTorch[:,1] = tempTensor[:,0]
  weibull_fits_PyTorch[:,2] = sign
  weibull_fits_PyTorch[:,3] = translate_amount
  weibull_fits_PyTorch[:,4] = small_score

  print(weibull_fits_PyTorch)


def tryingTorchSort():
  print('verify_pre_processing()')
  fileName = '/home/tahmad/work/stand_alone_libMr/data_from_Steve/weibulls_example_protocol2.npy'
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
  
  # Pre-processing -- same as weibull.c
  sign = -1.0 * np.ones(shape=numInstances, dtype=np.float64)
  data2a =  np.zeros(shape=(numInstances,dataSize), dtype=np.float64)
  data2b =  np.zeros(shape=(numInstances,tailSize), dtype=np.float64)
  data2 =  np.zeros(shape=(numInstances,tailSize), dtype=np.float64)
  
  translate_amount = 1.0 * np.ones(shape=(numInstances,1), dtype=np.float64)
  small_score = np.zeros(shape=(numInstances,1), dtype=np.float64) 
  
  for k in range(numInstances):
    data2a[k,:] = sign[k] * data1[k,:]
    data2b[k,:] = (np.sort(sign[k] * data1[k,:])[::-1])[:tailSize]
    small_score[k] = data2b[k,tailSize-1]
    data2[k,:] = data2b[k,:] + translate_amount[k] - small_score[k]
  
  print(data2b[0,:10])
  print(data2[0,:10])
  
  # PyTorch
  
  dataTensor = torch.from_numpy(data2a)
  sortedTensor = torch.topk(dataTensor, tailSize, dim=1, largest=True, sorted=True).values
  smallScoreTensor = sortedTensor[:,tailSize-1].unsqueeze(1)
  trnaslateAmoutTensor = torch.from_numpy(translate_amount)
  processedTensor = sortedTensor + trnaslateAmoutTensor - smallScoreTensor 
  
  print(processedTensor[0,:10])
  
  A = fit(processedTensor)
  print(A)
  
  resultTensor = torch.cat([A[:,1].cpu().unsqueeze(1),A[:,0].cpu().unsqueeze(1),trnaslateAmoutTensor,smallScoreTensor], dim=1)
  print(resultTensor)

if __name__ == '__main__':
  #main()
  #main2()
  #verify_pre_processing()
  tryingTorchSort()
  
