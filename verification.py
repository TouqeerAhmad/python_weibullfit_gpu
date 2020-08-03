import numpy as np
import torch
import weibull
import libmr
from timeit import default_timer as timer

def load_data(fileName, numInstances, dataSize, tailSize, distance_multiplier):
    # loading and processing data    
    data = np.load(fileName, allow_pickle=True)
    
    data1 = np.zeros(shape=(numInstances, dataSize), dtype=np.float64)
    weibull_fits_libmr = np.zeros(shape=(numInstances, 5), dtype=np.float64)

    for k in range(numInstances):
        data1[k, :] = distance_multiplier * data[k][0]
        weibull_fits_libmr[k, :] = data[k][1]

    
    # libmr based weibull fitting
    """
    start = timer()
    for k in range(numInstances):
        mr = libmr.MR()
        mr.fit_low(data1[k,:], tailSize)
        param = mr.get_params()
    end = timer()
    print('Time using libmr weibull fitting:')
    print(end - start)
    """
    
    dataTensor = torch.from_numpy(data1)
    dataTensor = dataTensor.cuda()
    
    return dataTensor, weibull_fits_libmr
    


def call_weibullFit(dataTensor, tailSize):

    # actual call to the weibull fit
    weibullObj = weibull.weibull()
    weibullObj.FitLow(dataTensor, tailSize, 0)
    result = weibullObj.return_all_parameters()
    
    return result

def printCompareAllFits(weibull_fits_libmr, result, numInstances):
    scale_numpy = result["Scale"].cpu().numpy()
    shape_numpy = result["Shape"].cpu().numpy()
    
    # printing libmr and new fit estimates for one-to-one comparison
    for k in range(numInstances):
        print(k)
        print(weibull_fits_libmr[k,0],weibull_fits_libmr[k,1],scale_numpy[k],shape_numpy[k])
        print(np.isclose(weibull_fits_libmr[k,0],scale_numpy[k]))
        print(np.isclose(weibull_fits_libmr[k,1],shape_numpy[k]))
   
    return

def test_weibullFit():
    
    fileName = '/home/tahmad/work/stand_alone_libMr/python_weibullfit_gpu/sample_data/weibulls_example_protocol2.npy'
    numInstances = 10
    dataSize = 40000
    tailSize = 2500
    distance_multiplier = 0.5
    
    """
    #fileName = '/home/tahmad/work/stand_alone_libMr/python_weibullfit_gpu/sample_data/umd_example.npy'
    fileName = '/scratch/tahmad/files/umd_example.npy'
    numInstances = 4000
    dataSize = 474666
    tailSize = 33998
    distance_multiplier = 0.55
    """
    
    distanceTensor, weibull_fits_libmr = load_data(fileName, numInstances, dataSize, tailSize, distance_multiplier)
    
    start = timer()
    result = call_weibullFit(distanceTensor, tailSize)
    end = timer()
    print('Time using new weibull fitting:')
    print(end - start)
    
    print(weibull_fits_libmr)
    print(result["Scale"])
    print(result["Shape"])
    
    #printCompareAllFits(weibull_fits_libmr, result, numInstances)
    
    # Testing the wscore function
    #new_WeibullObj = weibull.weibull(result)
    #print(new_WeibullObj.wscore(distanceTensor))
    #print(new_WeibullObj.wscore(distanceTensor).shape)
    
    return


def test_determine_splits():
    numInstances = 10000
    dataSize = 450000
    tailSize = 2500
    
    dataTensor = torch.rand((numInstances, dataSize)).type(torch.DoubleTensor)*100	
    #dataTensor = dataTensor.cuda()	
    weibullObj = weibull.weibull()
    weibullObj._determine_splits(dataTensor, tailSize, 0)	
    
    print(weibullObj.splits)
    
    weibullObj.FitLow(dataTensor, tailSize, 0)
    result = weibullObj.return_all_parameters()
    print(result["Shape"].shape)
    print(result["Shape"])

if __name__ == '__main__':
    test_weibullFit()
    #test_determine_splits()	
