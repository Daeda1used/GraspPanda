"""Exact nested state checks for resumable native training adapters."""


def same_state(actual, expected):
    import torch
    if isinstance(expected, torch.Tensor):
        return (isinstance(actual,torch.Tensor) and actual.shape == expected.shape
                and actual.dtype == expected.dtype
                and torch.equal(actual.detach().cpu(),expected.detach().cpu()))
    if isinstance(expected,dict):
        return (isinstance(actual,dict) and actual.keys() == expected.keys()
                and all(same_state(actual[key],value) for key,value in expected.items()))
    if isinstance(expected,(list,tuple)):
        return (isinstance(actual,(list,tuple)) and len(actual) == len(expected)
                and all(same_state(a,b) for a,b in zip(actual,expected)))
    return actual == expected
