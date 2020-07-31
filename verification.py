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
    dataTensor = torch.rand(size=(1000,40000), dtype=torch.float64)	
    dataTensor = dataTensor.cuda()	
    weibullObj = weibull.weibull()
    
    weibullObj.determine_splits(dataTensor, 2500, 0)	
    print(weibullObj.splits)	


if __name__ == '__main__':
    #test_weibullFit()
    test_determine_splits()	
