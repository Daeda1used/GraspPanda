"""Read the author's NumPy fusion dictionary without executing pickle callables."""
from dataclasses import dataclass
from pathlib import Path
import pickletools
import numpy as np


@dataclass
class Array:
    value: object = None


@dataclass
class Dtype:
    name: str
    order: str = '='


def read_fusion(path):
    path=Path(path)
    if path.suffix=='.npz':
        with np.load(path,allow_pickle=False) as data:result={key:data[key] for key in ('xyz','normal','color')}
    else:
        with path.open('rb') as stream:
            version=np.lib.format.read_magic(stream)
            reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}.get(version)
            if reader is None:raise ValueError('Use numeric fusion NPZ for this NumPy format')
            shape,order,dtype=reader(stream)
            if shape!=() or dtype.kind!='O':raise ValueError('Expected the author scalar fusion dictionary or numeric NPZ')
            payload=stream.read()
        stack=[];memo={};mark=object();stopped=False
        allowed={('numpy.core.multiarray','_reconstruct'),('numpy._core.multiarray','_reconstruct'),('numpy','ndarray'),('numpy','dtype')}
        def marked():
            i=len(stack)-1
            while i>=0 and stack[i] is not mark:i-=1
            if i<0:raise ValueError('Invalid fusion payload stack')
            values=stack[i+1:];del stack[i:];return values
        try:
            for op,arg,pos in pickletools.genops(payload):
                name=op.name
                if name in ('PROTO','FRAME'):continue
                if name=='MARK':stack.append(mark)
                elif name in ('BINPUT','LONG_BINPUT','PUT'):memo[arg]=stack[-1]
                elif name=='MEMOIZE':memo[len(memo)]=stack[-1]
                elif name in ('BINGET','LONG_BINGET','GET'):stack.append(memo[arg])
                elif name in ('GLOBAL','STACK_GLOBAL'):
                    pair=tuple(arg.split(' ')) if name=='GLOBAL' else (stack[-2],stack[-1])
                    if name=='STACK_GLOBAL':del stack[-2:]
                    if pair not in allowed:raise ValueError('Executable or unsupported object in fusion payload')
                    stack.append(pair)
                elif name in ('NONE','NEWTRUE','NEWFALSE'):stack.append({'NONE':None,'NEWTRUE':True,'NEWFALSE':False}[name])
                elif name in ('BININT','BININT1','BININT2','INT','LONG','LONG1','LONG4','BINUNICODE','SHORT_BINUNICODE','UNICODE','BINBYTES','SHORT_BINBYTES','BINBYTES8'):stack.append(arg)
                elif name=='EMPTY_TUPLE':stack.append(())
                elif name=='EMPTY_LIST':stack.append([])
                elif name=='EMPTY_DICT':stack.append({})
                elif name=='TUPLE':stack.append(tuple(marked()))
                elif name in ('TUPLE1','TUPLE2','TUPLE3'):
                    n=int(name[-1]);values=tuple(stack[-n:]);del stack[-n:];stack.append(values)
                elif name=='APPEND':
                    value=stack.pop();stack[-1].append(value)
                elif name=='APPENDS':
                    values=marked();stack[-1].extend(values)
                elif name=='SETITEM':
                    value=stack.pop();key=stack.pop();stack[-1][key]=value
                elif name=='SETITEMS':
                    values=marked()
                    if len(values)%2:raise ValueError('Invalid fusion dictionary')
                    for key,value in zip(values[::2],values[1::2]):stack[-1][key]=value
                elif name=='REDUCE':
                    args=stack.pop();function=stack.pop()
                    if function in (('numpy.core.multiarray','_reconstruct'),('numpy._core.multiarray','_reconstruct')) and args==(('numpy','ndarray'),(0,),b'b'):
                        stack.append(Array())
                    elif function==('numpy','dtype') and isinstance(args,tuple) and len(args)==3 and args[0] in ('O4','O8','f4','f8') and args[1:] == (False,True):
                        stack.append(Dtype(args[0]))
                    else:raise ValueError('Unsupported construction in fusion payload')
                elif name=='BUILD':
                    state=stack.pop();obj=stack[-1]
                    if isinstance(obj,Dtype):
                        if not isinstance(state,tuple) or len(state)!=8 or state[0]!=3 or state[1] not in ('<','>','=','|') or state[2:5]!=(None,None,None) or state[5:7]!=(-1,-1):raise ValueError('Unsupported fusion dtype')
                        obj.order=state[1]
                    elif isinstance(obj,Array):
                        if not isinstance(state,tuple) or len(state)!=5 or state[0]!=1:raise ValueError('Unsupported fusion array state')
                        _,shape,dtype,order,data=state
                        if not isinstance(dtype,Dtype) or type(order) is not bool:raise ValueError('Invalid fusion array metadata')
                        if dtype.name.startswith('O'):
                            if shape!=() or not isinstance(data,list) or len(data)!=1 or not isinstance(data[0],dict):raise ValueError('Only the outer fusion dictionary may contain objects')
                            obj.value=data[0]
                        else:
                            dt=np.dtype(dtype.name).newbyteorder(dtype.order)
                            if not isinstance(shape,tuple) or len(shape)!=2 or shape[1]!=3 or type(shape[0]) is not int or shape[0]<1 or not isinstance(data,bytes) or len(data)!=shape[0]*3*dt.itemsize:raise ValueError('Invalid numeric fusion array')
                            obj.value=np.frombuffer(data,dtype=dt).reshape(shape,order='F' if order else 'C').copy()
                    else:raise ValueError('Unsupported state target in fusion payload')
                elif name=='STOP':
                    if len(stack)!=1 or not isinstance(stack[0],Array) or not isinstance(stack[0].value,dict) or pos+1!=len(payload):raise ValueError('Invalid fusion dictionary root')
                    stopped=True;break
                else:raise ValueError('Unsupported fusion opcode: '+name)
            if not stopped:raise ValueError('Incomplete fusion payload')
            outer=stack[0].value
            if set(outer)!={'xyz','normal','color'} or not all(isinstance(v,Array) for v in outer.values()):raise ValueError('Expected xyz, normal and color arrays')
            result={k:v.value for k,v in outer.items()}
        except (KeyError,IndexError,TypeError,AttributeError) as error:
            raise ValueError('Malformed fusion payload') from error
    if any(not isinstance(v,np.ndarray) or v.dtype.kind!='f' or v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() for v in result.values()):raise ValueError('Fusion inputs must be finite floating-point [N,3] arrays')
    if len({v.shape for v in result.values()})!=1:raise ValueError('Fusion XYZ, normals and colors must share point rows')
    return result


def install_dataset_reader(dataset):
    """Use the restricted reader inside native data methods without global patches."""
    from types import SimpleNamespace,FunctionType,MethodType
    dataset.pcdpath=[str(Path(p).with_suffix('.npz')) if Path(p).with_suffix('.npz').is_file() else p for p in dataset.pcdpath]
    def load(path,*args,**kwargs):
        if Path(path).name in ('points.npy','points.npz'):
            return SimpleNamespace(item=lambda:read_fusion(path))
        if kwargs.get('allow_pickle',False):raise ValueError('Object loading is restricted to the fusion dictionary')
        return np.load(path,*args,**kwargs)
    proxy=SimpleNamespace(**{**vars(np),'load':load})
    for name in ('get_data','get_data_label'):
        method=getattr(dataset,name).__func__
        namespace={**method.__globals__,'np':proxy}
        setattr(dataset,name,MethodType(FunctionType(method.__code__,namespace,method.__name__,method.__defaults__,method.__closure__),dataset))
