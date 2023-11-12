import torch

def IGW(fhat, gamma):
    from math import sqrt
    fhatahat, ahat = fhat.max(dim=1)
    A = fhat.shape[1]
    gamma *= sqrt(A)
    p = 1 / (A +  gamma * (fhatahat.unsqueeze(1) - fhat))
    sump = p.sum(dim=1)
    p[range(p.shape[0]), ahat] += torch.clamp(1 - sump, min=0, max=None)
    return torch.multinomial(p, num_samples=1).squeeze(1), ahat

def SamplingIGW(A, P, gamma):
    exploreind, _ = IGW(P, gamma)
    explore = [ ind for _, ind in zip(A, exploreind) ]
    return explore
