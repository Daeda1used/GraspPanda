"""Exact nested state checks for resumable native training adapters."""


def capture_rng_state():
    """A weights-only-loadable snapshot of Python, NumPy and Torch generators."""
    import random
    import numpy as np
    import torch
    state = np.random.get_state()
    return {'python':random.getstate(), 'numpy':[state[0],state[1].tolist(),*state[2:]],
            'torch':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state_all()}


def restore_rng_state(state):
    import random
    import numpy as np
    import torch
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value[0],np.asarray(value[1],dtype=np.uint32),*value[2:]))
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def write_result(out, result):
    """Publish only a complete, finite result for the UI and experiment reader."""
    import json
    temp = out/'result.json.tmp'
    temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    temp.replace(out/'result.json')


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
