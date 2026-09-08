"""Bounded parallel C++ compilation for the single large CGAL extension."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import runpy
import sys
import setuptools  # activate its distutils shim before resolving compiler class
from distutils.ccompiler import CCompiler

ROOT = Path(__file__).resolve().parents[2]
original = CCompiler.compile


def parallel_compile(self, *args, **kwargs):
    compile_one = self._compile
    futures = []
    with ThreadPoolExecutor(max_workers=int(os.environ.get('MAX_JOBS','4'))) as pool:
        def submit(*a, **k):
            # Explicit local development shortcut; release builds compile all
            # units, because system/header changes need more than timestamps.
            if os.environ.get('GRASPPANDA_INCREMENTAL_GEOMETRY')=='1' and os.path.exists(a[0]) and os.path.getmtime(a[0])>os.path.getmtime(a[1]):
                return
            futures.append(pool.submit(compile_one,*a,**k))
        self._compile = submit
        try:
            objects = original(self,*args,**kwargs)
            for future in futures:
                future.result()
            return objects
        finally:
            self._compile = compile_one


CCompiler.compile = parallel_compile
os.chdir(ROOT/'environments/build/scikit-geometry')
sys.argv = ['setup.py','bdist_wheel','--dist-dir',str(ROOT/'environments/wheels')]
runpy.run_path('setup.py',run_name='__main__')
