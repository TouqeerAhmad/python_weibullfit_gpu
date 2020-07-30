import torch
import weibull
def test_weibullFit():
    numInstances = 10
    dataSize = 40000
    tailSize = 2500

    dummy_data = torch.rand((numInstances, dataSize)).type(torch.DoubleTensor)
    result = weibull.FitLow(dummy_data.cuda(), tailSize, 0)
    print(result)

    return


if __name__ == '__main__':
    test_weibullFit()

