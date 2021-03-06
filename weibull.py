import numpy as np
import torch
import os, sys
from pynvml import *

class weibull:
    def __init__(self, saved_model=None):
        if saved_model:
            self.wbFits = torch.zeros(saved_model['Scale'].shape[0],2)
            self.wbFits[:, 1] = saved_model['Scale']
            self.wbFits[:, 0] = saved_model['Shape']
            self.smallScoreTensor = saved_model['smallScoreTensor']
            self.sign = saved_model['signTensor']
        return


    def return_all_parameters(self):
        return dict(Scale = self.wbFits[:, 1],
                    Shape = self.wbFits[:, 0],
                    smallScoreTensor = self.smallScoreTensor,
                    signTensor = self.sign)


    def FitLow(self,data, tailSize, isSorted=False):
        self.sign = -1
        self._determine_splits(data, tailSize, isSorted)
        if self.splits == 1:
            return self._weibullFitting(data, tailSize, isSorted)
        else:
            return self._weibullFilltingInBatches(data, tailSize, isSorted)
            

    def FitHigh(self, data, tailSize, isSorted=False):
        self.sign = 1
        self._determine_splits(data, tailSize, isSorted)
        if self.splits == 1:
            return self._weibullFitting(data, tailSize, isSorted)
        else:
            return self._weibullFilltingInBatches(data, tailSize, isSorted)
        

    def wscore(self, distances):
        """
        This function can calculate scores from various weibulls for a given set of distances
        :param distances: a 2-D tensor
        :return:
        """
        self.deviceName = distances.device
        scale_tensor = self.wbFits[:,1]
        shape_tensor = self.wbFits[:, 0]
        if self.sign == -1:
            distances = -distances
        if len(distances.shape)==1:
            distances = distances.repeat(shape_tensor.shape[0],1)
        distances = torch.transpose(distances + 1 - self.smallScoreTensor.to(self.deviceName),1,0)
        weibulls = torch.distributions.weibull.Weibull(scale_tensor.to(self.deviceName),shape_tensor.to(self.deviceName))
        return torch.transpose(weibulls.cdf(distances),1,0)


    def _weibullFitting(self, dataTensor, tailSize, isSorted=False):
        self.deviceName = dataTensor.device
        if isSorted:
            sortedTensor = dataTensor
        else:
            if self.sign == -1:
                dataTensor = -dataTensor
            sortedTensor = torch.topk(dataTensor, tailSize, dim=1, largest=True, sorted=True).values

        self.smallScoreTensor = sortedTensor[:, tailSize - 1].unsqueeze(1)
        processedTensor = sortedTensor + 1 - self.smallScoreTensor
        # Returned in the format [Shape,Scale]
        if self.splits == 1:
            self.wbFits = self._fit(processedTensor)
        else:
            return self._fit(processedTensor)


    def _weibullFilltingInBatches(self, dataTensor, tailSize, isSorted = False):
        N =  dataTensor.shape[0]
        dtype = dataTensor.dtype
        batchSize = int(np.ceil(N / self.splits))
        resultTensor = torch.zeros(size=(N,2), dtype=dtype)
        
        for batchIter in range(int(self.splits-1)):
          startIndex = batchIter*batchSize
          endIndex = startIndex + batchSize
          data_batch = dataTensor[startIndex:endIndex,:].cuda()
          result_batch = self._weibullFitting(data_batch, tailSize, isSorted)
          resultTensor[startIndex:endIndex,:] = result_batch.cpu()
          
        # process the left-over
        startIndex = (self.splits-1)*batchSize
        endIndex = N
          
        data_batch = dataTensor[startIndex:endIndex,:].cuda()
        result_batch = self._weibullFitting(data_batch, tailSize, isSorted)
        resultTensor[startIndex:endIndex,:] = result_batch.cpu()
    
        self.wbFits = resultTensor   


    def _fit(self, data, iters=100, eps=1e-6):
        """
        Fits multiple 2-parameter Weibull distributions to the given data using maximum-likelihood estimation.
        :param data: 2d-tensor of samples. Each value must satisfy x > 0.
        :param iters: Maximum number of iterations
        :param eps: Stopping criterion. Fit is stopped ff the change within two iterations is smaller than eps.
        :return: tensor with first column Shape, and second Scale these can be (NaN, NaN) if a fit is impossible.
            Impossible fits may be due to 0-values in data.
        """
        k = torch.ones(data.shape[0]).double().to(self.deviceName)
        k_t_1 = k.clone()
        ln_x = torch.log(data)
        computed_params = torch.zeros(data.shape[0],2).double().to(self.deviceName)
        not_completed = torch.ones(data.shape[0], dtype=torch.bool).to(self.deviceName)
        for t in range(iters):
            if torch.all(torch.logical_not(not_completed)):
                break
            # Partial derivative df/dk
            x_k = data ** torch.transpose(k.repeat(data.shape[1],1),0,1)
            x_k_ln_x = x_k * ln_x
            fg = torch.sum(x_k,dim=1)
            del x_k
            ff = torch.sum(x_k_ln_x,dim=1)
            ff_prime = torch.sum(x_k_ln_x * ln_x,dim=1)
            del x_k_ln_x
            ff_by_fg = ff/fg
            del ff
            f = ff_by_fg - torch.mean(ln_x,dim=1) - (1.0 / k)
            f_prime = (ff_prime / fg - (ff_by_fg**2)) + (1. / (k * k))
            del ff_prime, fg
            # Newton-Raphson method k = k - f(k;x)/f'(k;x)
            k -= f / f_prime
            computed_params[not_completed*torch.isnan(f),:] = torch.tensor([float('nan'),float('nan')]).double().to(self.deviceName)
            not_completed[abs(k - k_t_1) < eps] = False
            computed_params[torch.logical_not(not_completed),0] = k[torch.logical_not(not_completed)]
            lam = torch.mean(data ** torch.transpose(k.repeat(data.shape[1],1),0,1),dim=1) ** (1.0 / k)
            # Lambda (scale) can be calculated directly
            computed_params[torch.logical_not(not_completed), 1] = lam[torch.logical_not(not_completed)]
            k_t_1 = k.clone()
        return computed_params  # Shape (SC), Scale (FE)
        
    
    def _determine_splits(self, inputTensor, tailSize, isSorted = 0):
        dtype_bytes = 8 # since float64
        # split chunks according to available GPU memory
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        gpu_free_mem = info.free / (1024 * 1024) # amount of free memeory in MB
        #print(gpu_free_mem)
        
        height, width = inputTensor.shape[0], inputTensor.shape[1]
        size_estimate = height * width * dtype_bytes * 5
        total_mem = (size_estimate) / (1024 * 1024) # amount in MB
        #print(total_mem)
        
        if total_mem < (gpu_free_mem * 0.7): #no chunks if GPU mem is enough
            split = 1
        else:
            split = round((total_mem) / (gpu_free_mem * 0.7))  
        
        self.splits = split
        
        #print('self.splits = ', self.splits)
