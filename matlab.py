"""Small MATLAB indexing layer; equation algebra lives in *_equations.py.

Only real scalar/vector/matrix operations used by the pinned sources are
implemented. Equations retain one-based indices to allow side-by-side study.
"""
import copy
import numpy as np


def ms(start=None, stop=None, step=1):
    return slice(None if start is None else int(start)-1, None if stop is None else int(stop), int(step))


def mr(start, stop, step=1):
    return range(int(start), int(stop)+1, int(step))


class M:
    __array_priority__ = 1000

    def __init__(self, values=None):
        source = np.zeros((0,1)) if values is None else np.asarray(values)
        if source.ndim < 2:
            source = source.reshape(-1,1)
        self._storage = np.zeros(max(256,source.size),dtype=np.result_type(source.dtype,float))
        self._storage[:source.size] = source.ravel(order='F')
        self.a = self._storage[:source.size].reshape(source.shape,order='F')

    def copy(self):
        return M(self.a)

    def __deepcopy__(self, memo):
        result = self.copy(); memo[id(self)] = result
        return result

    def vector(self):
        return self.a.ravel(order='F').copy()

    def __bool__(self):
        return bool(self.a.size and np.all(self.a))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = tuple(k if isinstance(k, slice) else int(k)-1 for k in key)
            out = self.a[key]
        else:
            key = key if isinstance(key, slice) else int(key)-1
            out = self.a.ravel(order='F')[key]
        return M(out) if isinstance(out, np.ndarray) else out

    def __setitem__(self, key, value):
        value = value.a if isinstance(value, M) else value
        if np.iscomplexobj(value) and not np.iscomplexobj(self._storage):
            shape = self.a.shape
            self._storage = self._storage.astype(complex)
            self.a = self._storage[:self.a.size].reshape(shape,order='F')
        if isinstance(key, tuple):
            key = tuple(k if isinstance(k, slice) else int(k)-1 for k in key)
            shape = tuple(max(self.a.shape[j], (k.stop or self.a.shape[j]) if isinstance(k,slice) else k+1) for j,k in enumerate(key))
            if shape != self.a.shape:
                new = np.zeros(shape,order='F',dtype=self._storage.dtype); new[:self.a.shape[0], :self.a.shape[1]] = self.a
                self._storage = new.ravel(order='F'); self.a = new
            target = self.a[key]
            self.a[key] = np.asarray(value).reshape(np.shape(target), order='F') if isinstance(target,np.ndarray) else value
        else:
            k = key if isinstance(key, slice) else int(key)-1
            end = (k.stop or self.a.size) if isinstance(k,slice) else k+1
            if end > self._storage.size:
                new = np.zeros(max(end,self._storage.size*2),dtype=self._storage.dtype)
                new[:self.a.size] = self.a.ravel(order='F'); self._storage = new
            if end > self.a.size:
                self.a = self._storage[:end].reshape(-1,1)
            flat = self.a.ravel(order='F')
            target = flat[k]
            flat[k] = np.asarray(value).reshape(np.shape(target), order='F') if isinstance(target,np.ndarray) else value

    def __mul__(self, x): return M(self.a * (x.a if isinstance(x,M) else x))
    __rmul__ = __mul__
    def __truediv__(self, x): return M(self.a / (x.a if isinstance(x,M) else x))
    def __add__(self, x): return M(self.a + (x.a if isinstance(x,M) else x))
    __radd__ = __add__
    def __sub__(self, x): return M(self.a - (x.a if isinstance(x,M) else x))
    def __rsub__(self, x): return M(x-self.a)
    def __eq__(self, x): return bool(self.a.size and np.all(self.a == x))
    def __lt__(self, x): return bool(self.a.size and np.all(self.a < x))
    def __le__(self, x): return bool(self.a.size and np.all(self.a <= x))
    def __gt__(self, x): return bool(self.a.size and np.all(self.a > x))
    def __ge__(self, x): return bool(self.a.size and np.all(self.a >= x))


def zeros(rows, cols=None):
    return M(np.zeros((int(rows), int(rows if cols is None else cols))))


def ones(rows, cols=None):
    return M(np.ones((int(rows), int(rows if cols is None else cols))))


def copy_value(value):
    return value.copy() if isinstance(value, M) else value


def odeset(*args):
    return dict(zip(args[::2], args[1::2]))


class Context:
    """MATLAB global workspace is isolated per model instance."""
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        value = M()
        setattr(self, name, value)
        return value

    def clone(self):
        return copy.deepcopy(self)
