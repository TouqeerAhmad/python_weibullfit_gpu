import torch
import weibull

def test_weibullFit():
    numInstances = 10
    dataSize = 40000
    tailSize = 2500

    dummy_data = torch.rand((numInstances, dataSize)).type(torch.DoubleTensor)*100
    weibullObj = weibull.weibull()
    weibullObj.FitLow(dummy_data.cuda(), tailSize, 0)
    result = weibullObj.return_all_parameters()
    print(result)

    dummy_test_data = torch.rand(dataSize).type(torch.DoubleTensor).cuda()
    print(weibullObj.wscore(dummy_test_data))
    print(weibullObj.wscore(dummy_test_data).shape)
    return

if __name__ == '__main__':
    test_weibullFit()