import numpy as np
import torch
import weibull

def test_weibullFit():
    fileName = '/home/tahmad/work/stand_alone_libMr/python_weibullfit_gpu/sample_data/weibulls_example_protocol2.npy'
    data = np.load(fileName, allow_pickle=True)

    numInstances = 10
    dataSize = 40000
    tailSize = 2500
    distance_multiplier = 0.5

    data1 = np.zeros(shape=(numInstances, dataSize), dtype=np.float64)
    weibull_fits_libmr = np.zeros(shape=(numInstances, 5), dtype=np.float64)

    for k in range(numInstances):
        data1[k, :] = distance_multiplier * data[k][0]
        weibull_fits_libmr[k, :] = data[k][1]

    dataTensor = torch.from_numpy(data1)
    dataTensor = dataTensor.cuda()

    weibullObj = weibull.weibull()
    weibullObj.FitLow(dataTensor, tailSize, 0)
    result = weibullObj.return_all_parameters()

    print(result["Scale"])
    print(result["Shape"])

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
    #test_weibullFit()
    test_determine_splits()	
