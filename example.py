import torch
import weibull

def test_weibullFit():
    numInstances = 10
    dataSize = 40000
    tailSize = 2500

    dummy_data = torch.rand((numInstances, dataSize)).type(torch.DoubleTensor)*100
    weibullObj = weibull.weibull()
    weibullObj.FitLow(dummy_data.cuda(), tailSize, isSorted=False)
    result = weibullObj.return_all_parameters()
    print(result)

    del weibullObj
    new_WeibullObj = weibull.weibull(result)
    dummy_test_data = torch.rand(dataSize).type(torch.DoubleTensor).cuda()
    print(new_WeibullObj.wscore(dummy_test_data))
    print(new_WeibullObj.wscore(dummy_test_data).shape)
    return

def verify_wscore():
    import libmr
    numInstances = 10
    dataSize = 40000
    tailSize = 2500
    dummy_training_data = torch.rand((numInstances, dataSize)).type(torch.DoubleTensor)
    dummy_test_data = torch.rand((numInstances, dataSize)).type(torch.DoubleTensor)

    models={}
    probs=[]
    for i in range(numInstances):
        models[i] = libmr.MR()
        models[i].fit_low(dummy_training_data[i,:].numpy(), tailSize)
        k_l=[]
        for l in dummy_test_data[i,:].numpy().tolist():
            k_l.append(models[i].w_score(l))
        probs.append(k_l)

    rt = torch.tensor(probs)
    print(rt)

    weibullObj = weibull.weibull()
    weibullObj.FitLow(dummy_training_data.cuda(), tailSize, isSorted=False)
    print(weibullObj.wscore(dummy_test_data))
    print(weibullObj.wscore(dummy_test_data).shape)

if __name__ == '__main__':
    test_weibullFit()
    # verify_wscore()
