import torch

def fit(x, iters=100, eps=1e-6, use_cuda=True):
    """
    Fits a 2-parameter Weibull distribution to the given data using maximum-likelihood estimation.
    :param x: 1d-ndarray of samples from an (unknown) distribution. Each value must satisfy x > 0.
    :param iters: Maximum number of iterations
    :param eps: Stopping criterion. Fit is stopped ff the change within two iterations is smaller than eps.
    :param use_cuda: Use gpu
    :return: tensor with first column Shape, and second Scale these can be (NaN, NaN) if a fit is impossible.
        Impossible fits may be due to 0-values in x.
    """
    if use_cuda:
        dtype = torch.cuda.DoubleTensor
    else:
        dtype = torch.DoubleTensor
    k = torch.ones(x.shape[0]).type(dtype)
    k_t_1 = k.clone()
    xvar = x.type(dtype)
    ln_x = torch.log(xvar)
    computed_params = torch.zeros(xvar.shape[0],2).type(dtype)
    not_completed = torch.ones(xvar.shape[0], dtype=torch.bool).cuda()
    for t in range(iters):
        if torch.all(torch.logical_not(not_completed)):
            break
        # Partial derivative df/dk
        x_k = xvar ** torch.transpose(k.repeat(xvar.shape[1],1),0,1)
        x_k_ln_x = x_k * ln_x
        ff = torch.sum(x_k_ln_x,dim=1)
        fg = torch.sum(x_k,dim=1)
        f1 = torch.mean(ln_x,dim=1)
        f = ff/fg - f1 - (1.0 / k)
        ff_prime = torch.sum(x_k_ln_x * ln_x,dim=1)
        fg_prime = ff
        f_prime = (ff_prime / fg - (ff / fg * fg_prime / fg)) + (1. / (k * k))
        # Newton-Raphson method k = k - f(k;x)/f'(k;x)
        k -= f / f_prime
        computed_params[not_completed*torch.isnan(f),:] = torch.tensor([float('nan'),float('nan')]).type(dtype)
        not_completed[abs(k - k_t_1) < eps] = False
        computed_params[torch.logical_not(not_completed),0] = k[torch.logical_not(not_completed)]
        lam = torch.mean(xvar ** torch.transpose(k.repeat(xvar.shape[1],1),0,1),dim=1) ** (1.0 / k)
        # Lambda (scale) can be calculated directly
        computed_params[torch.logical_not(not_completed), 1] = lam[torch.logical_not(not_completed)]
        k_t_1 = k.clone()
    return computed_params  # Shape (SC), Scale (FE)

