from __future__ import annotations
import math as math
import numpy
from numpy import AxisError
from numpy import ComplexWarning
from numpy import DataSource
from numpy import ModuleDeprecationWarning
from numpy import RankWarning
from numpy import TooHardError
from numpy import VisibleDeprecationWarning
from numpy.__config__ import show as show_config
from numpy import _get_promotion_state
from numpy import _no_nep50_warning
from numpy import _set_promotion_state
from numpy import all
from numpy import allclose
from numpy import alltrue
from numpy import amax
from numpy import amin
from numpy import angle
from numpy import any
from numpy import append
from numpy import apply_along_axis
from numpy import apply_over_axes
from numpy import arange
from numpy import argmax
from numpy import argmin
from numpy import argpartition
from numpy import argsort
from numpy import argwhere
from numpy import around
from numpy import array
from numpy import array2string
from numpy import array_equal
from numpy import array_equiv
from numpy import array_repr
from numpy import array_split
from numpy import array_str
from numpy import asanyarray
from numpy import asarray
from numpy import asarray_chkfinite
from numpy import ascontiguousarray
from numpy import asfarray
from numpy import asfortranarray
from numpy import asmatrix as mat
from numpy import asmatrix
from numpy import atleast_1d
from numpy import atleast_2d
from numpy import atleast_3d
from numpy import average
from numpy import bartlett
from numpy import base_repr
from numpy import binary_repr
from numpy import bincount
from numpy import blackman
from numpy import block
from numpy import bmat
from numpy import bool_
from numpy import broadcast
from numpy import broadcast_arrays
from numpy import broadcast_shapes
from numpy import broadcast_to
from numpy import busday_count
from numpy import busday_offset
from numpy import busdaycalendar
from numpy import bytes_
from numpy import bytes_ as string_
from numpy import can_cast
from numpy import character
from numpy import chararray
from numpy import choose
from numpy import clip
from numpy import column_stack
from numpy import common_type
from numpy import complex128 as cfloat
from numpy import complex128 as cdouble
from numpy import complex128
from numpy import complex128 as complex_
from numpy import complex256
from numpy import complex256 as clongdouble
from numpy import complex256 as clongfloat
from numpy import complex256 as longcomplex
from numpy import complex64 as singlecomplex
from numpy import complex64
from numpy import complex64 as csingle
from numpy import complexfloating
from numpy import compress
from numpy import concatenate
from numpy import convolve
from numpy import copy
from numpy import copyto
from numpy.core._multiarray_umath import _add_newdoc_ufunc as add_newdoc_ufunc
from numpy.core._multiarray_umath import _add_newdoc_ufunc
from numpy.core._multiarray_umath import add_docstring
from numpy.core._multiarray_umath import compare_chararrays
from numpy.core._multiarray_umath import fastCopyAndTranspose
from numpy.core.arrayprint import set_string_function
from numpy.core import defchararray as char
from numpy.core.function_base import add_newdoc
import numpy.core.numerictypes
from numpy.core import records as rec
from numpy import corrcoef
from numpy import correlate
from numpy import count_nonzero
from numpy import cov
from numpy import cross
from numpy import ctypeslib
from numpy import cumprod
from numpy import cumproduct
from numpy import cumsum
from numpy import datetime64
from numpy import datetime_as_string
from numpy import datetime_data
from numpy import delete
from numpy import diag
from numpy import diag_indices
from numpy import diag_indices_from
from numpy import diagflat
from numpy import diagonal
from numpy import diff
from numpy import digitize
from numpy import dot
from numpy import dsplit
from numpy import dstack
from numpy import dtype
from numpy import ediff1d
from numpy import einsum
from numpy import einsum_path
from numpy import empty
from numpy import empty_like
from numpy import errstate
from numpy import expand_dims
from numpy import extract
from numpy import eye
from numpy import fft
from numpy import fill_diagonal
from numpy import find_common_type
from numpy import finfo
from numpy import fix
from numpy import flatiter
from numpy import flatnonzero
from numpy import flexible
from numpy import flip
from numpy import fliplr
from numpy import flipud
from numpy import float128 as longdouble
from numpy import float128
from numpy import float128 as longfloat
from numpy import float16
from numpy import float16 as half
from numpy import float32
from numpy import float32 as single
from numpy import float64 as double
from numpy import float64 as float_
from numpy import float64
from numpy import floating
from numpy import format_float_positional
from numpy import format_float_scientific
from numpy import format_parser
from numpy import from_dlpack
from numpy import frombuffer
from numpy import fromfile
from numpy import fromfunction
from numpy import fromiter
from numpy import frompyfunc
from numpy import fromregex
from numpy import fromstring
from numpy import full
from numpy import full_like
from numpy import generic
from numpy import genfromtxt
from numpy import geomspace
from numpy import get_printoptions
from numpy import getbufsize
from numpy import geterr
from numpy import geterrcall
from numpy import geterrobj
from numpy import gradient
from numpy import hamming
from numpy import hanning
from numpy import histogram
from numpy import histogram2d
from numpy import histogram_bin_edges
from numpy import histogramdd
from numpy import hsplit
from numpy import hstack
from numpy import i0
from numpy import identity
from numpy import iinfo
from numpy import imag
from numpy import in1d
from numpy import indices
from numpy import inexact
from numpy import info
from numpy import inner
from numpy import insert
from numpy import int16
from numpy import int16 as short
from numpy import int32
from numpy import int32 as intc
from numpy import int64 as int_
from numpy import int64
from numpy import int64 as intp
from numpy import int8
from numpy import int8 as byte
from numpy import integer
from numpy import interp
from numpy import intersect1d
from numpy import is_busday
from numpy import isclose
from numpy import iscomplex
from numpy import iscomplexobj
from numpy import isfortran
from numpy import isin
from numpy import isneginf
from numpy import isposinf
from numpy import isreal
from numpy import isrealobj
from numpy import isscalar
from numpy import issctype
from numpy import issubclass_
from numpy import issubdtype
from numpy import issubsctype
from numpy import iterable
from numpy import ix_
from numpy import kaiser
from numpy import kron
from numpy import lexsort
from numpy.lib.function_base import disp
import numpy.lib.index_tricks
from numpy.lib.npyio import recfromcsv
from numpy.lib.npyio import recfromtxt
from numpy.lib import scimath as emath
from numpy.lib.shape_base import get_array_wrap
from numpy.lib.utils import byte_bounds
from numpy.lib.utils import deprecate
from numpy.lib.utils import deprecate_with_doc
from numpy.lib.utils import get_include
from numpy.lib.utils import safe_eval
from numpy.lib.utils import show_runtime
from numpy.lib.utils import who
from numpy import linalg
from numpy import linspace
from numpy import load
from numpy import loadtxt
from numpy import logspace
from numpy import longlong
from numpy import lookfor
from numpy import ma
from numpy import mask_indices
from numpy import matrix
from numpy import maximum_sctype
from numpy import may_share_memory
from numpy import mean
from numpy import median
from numpy import memmap
from numpy import meshgrid
from numpy import min_scalar_type
from numpy import mintypecode
from numpy import moveaxis
from numpy import msort
from numpy import nan_to_num
from numpy import nanargmax
from numpy import nanargmin
from numpy import nancumprod
from numpy import nancumsum
from numpy import nanmax
from numpy import nanmean
from numpy import nanmedian
from numpy import nanmin
from numpy import nanpercentile
from numpy import nanprod
from numpy import nanquantile
from numpy import nanstd
from numpy import nansum
from numpy import nanvar
from numpy import ndarray
from numpy import ndenumerate
from numpy import ndim
from numpy import ndindex
from numpy import nditer
from numpy import nested_iters
from numpy import nonzero
from numpy import number
from numpy import obj2sctype
from numpy import object_
from numpy import ones
from numpy import ones_like
from numpy import outer
from numpy import packbits
from numpy import pad
from numpy import partition
from numpy import percentile
from numpy import piecewise
from numpy import place
from numpy import poly
from numpy import poly1d
from numpy import polyadd
from numpy import polyder
from numpy import polydiv
from numpy import polyfit
from numpy import polyint
from numpy import polymul
from numpy import polysub
from numpy import polyval
from numpy import printoptions
from numpy import prod
from numpy import product
from numpy import promote_types
from numpy import ptp
from numpy import put
from numpy import put_along_axis
from numpy import putmask
from numpy import quantile
from numpy import random
from numpy import ravel
from numpy import ravel_multi_index
from numpy import real
from numpy import real_if_close
from numpy import recarray
from numpy import record
from numpy import repeat
from numpy import require
from numpy import reshape
from numpy import resize
from numpy import result_type
from numpy import roll
from numpy import rollaxis
from numpy import roots
from numpy import rot90
from numpy import round_
from numpy import save
from numpy import savetxt
from numpy import savez
from numpy import savez_compressed
from numpy import sctype2char
from numpy import searchsorted
from numpy import select
from numpy import set_numeric_ops
from numpy import set_printoptions
from numpy import setbufsize
from numpy import setdiff1d
from numpy import seterr
from numpy import seterrcall
from numpy import seterrobj
from numpy import setxor1d
from numpy import shape
from numpy import shares_memory
from numpy import signedinteger
from numpy import sinc
from numpy import size
from numpy import sometrue
from numpy import sort
from numpy import sort_complex
from numpy import source
from numpy import split
from numpy import squeeze
from numpy import stack
from numpy import std
from numpy import str_
from numpy import str_ as unicode_
from numpy import sum
from numpy import swapaxes
from numpy import take
from numpy import take_along_axis
from numpy import tensordot
from numpy import tile
from numpy import timedelta64
from numpy import trace
from numpy import transpose
from numpy import trapz
from numpy import tri
from numpy import tril
from numpy import tril_indices
from numpy import tril_indices_from
from numpy import trim_zeros
from numpy import triu
from numpy import triu_indices
from numpy import triu_indices_from
from numpy import typename
from numpy import ufunc
from numpy import uint16
from numpy import uint16 as ushort
from numpy import uint32
from numpy import uint32 as uintc
from numpy import uint64 as uint
from numpy import uint64 as uintp
from numpy import uint64
from numpy import uint8 as ubyte
from numpy import uint8
from numpy import ulonglong
from numpy import union1d
from numpy import unique
from numpy import unpackbits
from numpy import unravel_index
from numpy import unsignedinteger
from numpy import unwrap
from numpy import vander
from numpy import var
from numpy import vdot
from numpy import vectorize
from numpy import void
from numpy import vsplit
from numpy import vstack
from numpy import vstack as row_stack
from numpy import where
from numpy import zeros
from numpy import zeros_like
from rdkit.sping import pid
import typing
__all__ = ['ALLOW_THREADS', 'AxisError', 'BUFSIZE', 'CLIP', 'ComplexWarning', 'DataSource', 'DrawSpiral', 'ERR_CALL', 'ERR_DEFAULT', 'ERR_IGNORE', 'ERR_LOG', 'ERR_PRINT', 'ERR_RAISE', 'ERR_WARN', 'FLOATING_POINT_SUPPORT', 'FPE_DIVIDEBYZERO', 'FPE_INVALID', 'FPE_OVERFLOW', 'FPE_UNDERFLOW', 'False_', 'Inf', 'Infinity', 'MAXDIMS', 'MAY_SHARE_BOUNDS', 'MAY_SHARE_EXACT', 'ModuleDeprecationWarning', 'NAN', 'NINF', 'NZERO', 'NaN', 'PINF', 'PZERO', 'RAISE', 'RankWarning', 'SHIFT_DIVIDEBYZERO', 'SHIFT_INVALID', 'SHIFT_OVERFLOW', 'SHIFT_UNDERFLOW', 'ScalarType', 'TooHardError', 'True_', 'UFUNC_BUFSIZE_DEFAULT', 'UFUNC_PYVALS_NAME', 'VisibleDeprecationWarning', 'WRAP', 'absolute', 'add', 'add_docstring', 'add_newdoc', 'add_newdoc_ufunc', 'all', 'allclose', 'alltrue', 'amax', 'amin', 'angle', 'any', 'append', 'apply_along_axis', 'apply_over_axes', 'arange', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh', 'argmax', 'argmin', 'argpartition', 'argsort', 'argwhere', 'around', 'array', 'array2string', 'array_equal', 'array_equiv', 'array_repr', 'array_split', 'array_str', 'asanyarray', 'asarray', 'asarray_chkfinite', 'ascontiguousarray', 'asfarray', 'asfortranarray', 'asmatrix', 'atleast_1d', 'atleast_2d', 'atleast_3d', 'average', 'bartlett', 'base_repr', 'binary_repr', 'bincount', 'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'blackman', 'block', 'bmat', 'bool_', 'broadcast', 'broadcast_arrays', 'broadcast_shapes', 'broadcast_to', 'busday_count', 'busday_offset', 'busdaycalendar', 'byte', 'byte_bounds', 'bytes_', 'c_', 'can_cast', 'cast', 'cbrt', 'cdouble', 'ceil', 'cfloat', 'char', 'character', 'chararray', 'choose', 'clip', 'clongdouble', 'clongfloat', 'column_stack', 'common_type', 'compare_chararrays', 'complex128', 'complex256', 'complex64', 'complex_', 'complexfloating', 'compress', 'concatenate', 'conj', 'conjugate', 'convolve', 'copy', 'copysign', 'copyto', 'corrcoef', 'correlate', 'cos', 'cosh', 'count_nonzero', 'cov', 'cross', 'csingle', 'ctypeslib', 'cumprod', 'cumproduct', 'cumsum', 'datetime64', 'datetime_as_string', 'datetime_data', 'deg2rad', 'degrees', 'delete', 'deprecate', 'deprecate_with_doc', 'diag', 'diag_indices', 'diag_indices_from', 'diagflat', 'diagonal', 'diff', 'digitize', 'disp', 'divide', 'divmod', 'dot', 'double', 'dsplit', 'dstack', 'dtype', 'e', 'ediff1d', 'einsum', 'einsum_path', 'emath', 'empty', 'empty_like', 'equal', 'errstate', 'euler_gamma', 'exp', 'exp2', 'expand_dims', 'expm1', 'extract', 'eye', 'fabs', 'fastCopyAndTranspose', 'fft', 'fill_diagonal', 'find_common_type', 'finfo', 'fix', 'flatiter', 'flatnonzero', 'flexible', 'flip', 'fliplr', 'flipud', 'float128', 'float16', 'float32', 'float64', 'float_', 'float_power', 'floating', 'floor', 'floor_divide', 'fmax', 'fmin', 'fmod', 'format_float_positional', 'format_float_scientific', 'format_parser', 'frexp', 'from_dlpack', 'frombuffer', 'fromfile', 'fromfunction', 'fromiter', 'frompyfunc', 'fromregex', 'fromstring', 'full', 'full_like', 'gcd', 'generic', 'genfromtxt', 'geomspace', 'get_array_wrap', 'get_include', 'get_printoptions', 'getbufsize', 'geterr', 'geterrcall', 'geterrobj', 'gradient', 'greater', 'greater_equal', 'half', 'hamming', 'hanning', 'heaviside', 'histogram', 'histogram2d', 'histogram_bin_edges', 'histogramdd', 'hsplit', 'hstack', 'hypot', 'i0', 'identity', 'iinfo', 'imag', 'in1d', 'index_exp', 'indices', 'inexact', 'inf', 'info', 'infty', 'inner', 'insert', 'int16', 'int32', 'int64', 'int8', 'int_', 'intc', 'integer', 'interp', 'intersect1d', 'intp', 'invert', 'is_busday', 'isclose', 'iscomplex', 'iscomplexobj', 'isfinite', 'isfortran', 'isin', 'isinf', 'isnan', 'isnat', 'isneginf', 'isposinf', 'isreal', 'isrealobj', 'isscalar', 'issctype', 'issubclass_', 'issubdtype', 'issubsctype', 'iterable', 'ix_', 'kaiser', 'kron', 'lcm', 'ldexp', 'left_shift', 'less', 'less_equal', 'lexsort', 'linalg', 'linspace', 'little_endian', 'load', 'loadtxt', 'log', 'log10', 'log1p', 'log2', 'logaddexp', 'logaddexp2', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'logspace', 'longcomplex', 'longdouble', 'longfloat', 'longlong', 'lookfor', 'ma', 'mask_indices', 'mat', 'math', 'matmul', 'matrix', 'maximum', 'maximum_sctype', 'may_share_memory', 'mean', 'median', 'memmap', 'meshgrid', 'mgrid', 'min_scalar_type', 'minimum', 'mintypecode', 'mod', 'modf', 'moveaxis', 'msort', 'multiply', 'nan', 'nan_to_num', 'nanargmax', 'nanargmin', 'nancumprod', 'nancumsum', 'nanmax', 'nanmean', 'nanmedian', 'nanmin', 'nanpercentile', 'nanprod', 'nanquantile', 'nanstd', 'nansum', 'nanvar', 'nbytes', 'ndarray', 'ndenumerate', 'ndim', 'ndindex', 'nditer', 'negative', 'nested_iters', 'newaxis', 'nextafter', 'nonzero', 'not_equal', 'number', 'obj2sctype', 'object_', 'ogrid', 'ones', 'ones_like', 'outer', 'packbits', 'pad', 'partition', 'percentile', 'pi', 'pid', 'piecewise', 'place', 'poly', 'poly1d', 'polyadd', 'polyder', 'polydiv', 'polyfit', 'polyint', 'polymul', 'polysub', 'polyval', 'positive', 'power', 'printoptions', 'prod', 'product', 'promote_types', 'ptp', 'put', 'put_along_axis', 'putmask', 'quantile', 'r_', 'rad2deg', 'radians', 'random', 'ravel', 'ravel_multi_index', 'real', 'real_if_close', 'rec', 'recarray', 'recfromcsv', 'recfromtxt', 'reciprocal', 'record', 'remainder', 'repeat', 'require', 'reshape', 'resize', 'result_type', 'right_shift', 'rint', 'roll', 'rollaxis', 'roots', 'rot90', 'round_', 'row_stack', 's_', 'safe_eval', 'save', 'savetxt', 'savez', 'savez_compressed', 'sctype2char', 'sctypeDict', 'sctypes', 'searchsorted', 'select', 'set_numeric_ops', 'set_printoptions', 'set_string_function', 'setbufsize', 'setdiff1d', 'seterr', 'seterrcall', 'seterrobj', 'setxor1d', 'shape', 'shares_memory', 'short', 'show_config', 'show_runtime', 'sign', 'signbit', 'signedinteger', 'sin', 'sinc', 'single', 'singlecomplex', 'sinh', 'size', 'sometrue', 'sort', 'sort_complex', 'source', 'spacing', 'split', 'sqrt', 'square', 'squeeze', 'stack', 'std', 'str_', 'string_', 'subtract', 'sum', 'swapaxes', 'take', 'take_along_axis', 'tan', 'tanh', 'tensordot', 'tile', 'timedelta64', 'trace', 'tracemalloc_domain', 'transpose', 'trapz', 'tri', 'tril', 'tril_indices', 'tril_indices_from', 'trim_zeros', 'triu', 'triu_indices', 'triu_indices_from', 'true_divide', 'trunc', 'typecodes', 'typename', 'ubyte', 'ufunc', 'uint', 'uint16', 'uint32', 'uint64', 'uint8', 'uintc', 'uintp', 'ulonglong', 'unicode_', 'union1d', 'unique', 'unpackbits', 'unravel_index', 'unsignedinteger', 'unwrap', 'ushort', 'vander', 'var', 'vdot', 'vectorize', 'void', 'vsplit', 'vstack', 'where', 'who', 'zeros', 'zeros_like']
def DrawSpiral(canvas, startColor, endColor, startRadius, endRadius, nLoops, degsPerSlice = 70, degsPerStep = 1, startAngle = 0, centerPos = None, dir = 1):
    ...
ALLOW_THREADS: int = 1
BUFSIZE: int = 8192
CLIP: int = 0
ERR_CALL: int = 3
ERR_DEFAULT: int = 521
ERR_IGNORE: int = 0
ERR_LOG: int = 5
ERR_PRINT: int = 4
ERR_RAISE: int = 2
ERR_WARN: int = 1
FLOATING_POINT_SUPPORT: int = 1
FPE_DIVIDEBYZERO: int = 1
FPE_INVALID: int = 8
FPE_OVERFLOW: int = 2
FPE_UNDERFLOW: int = 4
False_: numpy.bool_  # value = False
Inf: float  # value = inf
Infinity: float  # value = inf
MAXDIMS: int = 32
MAY_SHARE_BOUNDS: int = 0
MAY_SHARE_EXACT: int = -1
NAN: float  # value = nan
NINF: float  # value = -inf
NZERO: float = -0.0
NaN: float  # value = nan
PINF: float  # value = inf
PZERO: float = 0.0
RAISE: int = 2
SHIFT_DIVIDEBYZERO: int = 0
SHIFT_INVALID: int = 9
SHIFT_OVERFLOW: int = 3
SHIFT_UNDERFLOW: int = 6
ScalarType: tuple = (int, float, complex, bool, bytes, str, memoryview, numpy.bool_, numpy.complex64, numpy.complex128, numpy.complex256, numpy.float16, numpy.float32, numpy.float64, numpy.float128, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.longlong, numpy.datetime64, numpy.timedelta64, numpy.object_, numpy.bytes_, numpy.str_, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.ulonglong, numpy.void)
True_: numpy.bool_  # value = True
UFUNC_BUFSIZE_DEFAULT: int = 8192
UFUNC_PYVALS_NAME: str = 'UFUNC_PYVALS'
WRAP: int = 1
_UFUNC_API: typing.Any  # value = <capsule object>
__version__: str = '1.24.4'
absolute: numpy.ufunc  # value = <ufunc 'absolute'>
add: numpy.ufunc  # value = <ufunc 'add'>
arccos: numpy.ufunc  # value = <ufunc 'arccos'>
arccosh: numpy.ufunc  # value = <ufunc 'arccosh'>
arcsin: numpy.ufunc  # value = <ufunc 'arcsin'>
arcsinh: numpy.ufunc  # value = <ufunc 'arcsinh'>
arctan: numpy.ufunc  # value = <ufunc 'arctan'>
arctan2: numpy.ufunc  # value = <ufunc 'arctan2'>
arctanh: numpy.ufunc  # value = <ufunc 'arctanh'>
bitwise_and: numpy.ufunc  # value = <ufunc 'bitwise_and'>
bitwise_not: numpy.ufunc  # value = <ufunc 'invert'>
bitwise_or: numpy.ufunc  # value = <ufunc 'bitwise_or'>
bitwise_xor: numpy.ufunc  # value = <ufunc 'bitwise_xor'>
c_: numpy.lib.index_tricks.CClass  # value = <numpy.lib.index_tricks.CClass object>
cast: numpy.core.numerictypes._typedict  # value = {<class 'numpy.str_'>: <function <lambda> at 0x7fd8a39e3820>, <class 'numpy.int64'>: <function <lambda> at 0x7fd8a39e3940>, <class 'numpy.uint64'>: <function <lambda> at 0x7fd8a39e39d0>, <class 'numpy.float128'>: <function <lambda> at 0x7fd8a39e3a60>, <class 'numpy.complex256'>: <function <lambda> at 0x7fd8a39e3af0>, <class 'numpy.bool_'>: <function <lambda> at 0x7fd8a39e3b80>, <class 'numpy.void'>: <function <lambda> at 0x7fd8a39e3c10>, <class 'numpy.longlong'>: <function <lambda> at 0x7fd8a39e3ca0>, <class 'numpy.ulonglong'>: <function <lambda> at 0x7fd8a39e3d30>, <class 'numpy.datetime64'>: <function <lambda> at 0x7fd8a39e3dc0>, <class 'numpy.int8'>: <function <lambda> at 0x7fd8a39e3e50>, <class 'numpy.uint8'>: <function <lambda> at 0x7fd8a39e3ee0>, <class 'numpy.float16'>: <function <lambda> at 0x7fd8a39e3f70>, <class 'numpy.timedelta64'>: <function <lambda> at 0x7fd8a39e6040>, <class 'numpy.object_'>: <function <lambda> at 0x7fd8a39e60d0>, <class 'numpy.int16'>: <function <lambda> at 0x7fd8a39e6160>, <class 'numpy.uint16'>: <function <lambda> at 0x7fd8a39e61f0>, <class 'numpy.float32'>: <function <lambda> at 0x7fd8a39e6280>, <class 'numpy.complex64'>: <function <lambda> at 0x7fd8a39e6310>, <class 'numpy.bytes_'>: <function <lambda> at 0x7fd8a39e63a0>, <class 'numpy.int32'>: <function <lambda> at 0x7fd8a39e6430>, <class 'numpy.uint32'>: <function <lambda> at 0x7fd8a39e64c0>, <class 'numpy.float64'>: <function <lambda> at 0x7fd8a39e6550>, <class 'numpy.complex128'>: <function <lambda> at 0x7fd8a39e65e0>}
cbrt: numpy.ufunc  # value = <ufunc 'cbrt'>
ceil: numpy.ufunc  # value = <ufunc 'ceil'>
conj: numpy.ufunc  # value = <ufunc 'conjugate'>
conjugate: numpy.ufunc  # value = <ufunc 'conjugate'>
copysign: numpy.ufunc  # value = <ufunc 'copysign'>
cos: numpy.ufunc  # value = <ufunc 'cos'>
cosh: numpy.ufunc  # value = <ufunc 'cosh'>
deg2rad: numpy.ufunc  # value = <ufunc 'deg2rad'>
degrees: numpy.ufunc  # value = <ufunc 'degrees'>
divide: numpy.ufunc  # value = <ufunc 'divide'>
divmod: numpy.ufunc  # value = <ufunc 'divmod'>
e: float = 2.718281828459045
equal: numpy.ufunc  # value = <ufunc 'equal'>
euler_gamma: float = 0.5772156649015329
exp: numpy.ufunc  # value = <ufunc 'exp'>
exp2: numpy.ufunc  # value = <ufunc 'exp2'>
expm1: numpy.ufunc  # value = <ufunc 'expm1'>
fabs: numpy.ufunc  # value = <ufunc 'fabs'>
float_power: numpy.ufunc  # value = <ufunc 'float_power'>
floor: numpy.ufunc  # value = <ufunc 'floor'>
floor_divide: numpy.ufunc  # value = <ufunc 'floor_divide'>
fmax: numpy.ufunc  # value = <ufunc 'fmax'>
fmin: numpy.ufunc  # value = <ufunc 'fmin'>
fmod: numpy.ufunc  # value = <ufunc 'fmod'>
frexp: numpy.ufunc  # value = <ufunc 'frexp'>
gcd: numpy.ufunc  # value = <ufunc 'gcd'>
greater: numpy.ufunc  # value = <ufunc 'greater'>
greater_equal: numpy.ufunc  # value = <ufunc 'greater_equal'>
heaviside: numpy.ufunc  # value = <ufunc 'heaviside'>
hypot: numpy.ufunc  # value = <ufunc 'hypot'>
index_exp: numpy.lib.index_tricks.IndexExpression  # value = <numpy.lib.index_tricks.IndexExpression object>
inf: float  # value = inf
infty: float  # value = inf
invert: numpy.ufunc  # value = <ufunc 'invert'>
isfinite: numpy.ufunc  # value = <ufunc 'isfinite'>
isinf: numpy.ufunc  # value = <ufunc 'isinf'>
isnan: numpy.ufunc  # value = <ufunc 'isnan'>
isnat: numpy.ufunc  # value = <ufunc 'isnat'>
lcm: numpy.ufunc  # value = <ufunc 'lcm'>
ldexp: numpy.ufunc  # value = <ufunc 'ldexp'>
left_shift: numpy.ufunc  # value = <ufunc 'left_shift'>
less: numpy.ufunc  # value = <ufunc 'less'>
less_equal: numpy.ufunc  # value = <ufunc 'less_equal'>
little_endian: bool = True
log: numpy.ufunc  # value = <ufunc 'log'>
log10: numpy.ufunc  # value = <ufunc 'log10'>
log1p: numpy.ufunc  # value = <ufunc 'log1p'>
log2: numpy.ufunc  # value = <ufunc 'log2'>
logaddexp: numpy.ufunc  # value = <ufunc 'logaddexp'>
logaddexp2: numpy.ufunc  # value = <ufunc 'logaddexp2'>
logical_and: numpy.ufunc  # value = <ufunc 'logical_and'>
logical_not: numpy.ufunc  # value = <ufunc 'logical_not'>
logical_or: numpy.ufunc  # value = <ufunc 'logical_or'>
logical_xor: numpy.ufunc  # value = <ufunc 'logical_xor'>
matmul: numpy.ufunc  # value = <ufunc 'matmul'>
maximum: numpy.ufunc  # value = <ufunc 'maximum'>
mgrid: numpy.lib.index_tricks.MGridClass  # value = <numpy.lib.index_tricks.MGridClass object>
minimum: numpy.ufunc  # value = <ufunc 'minimum'>
mod: numpy.ufunc  # value = <ufunc 'remainder'>
modf: numpy.ufunc  # value = <ufunc 'modf'>
multiply: numpy.ufunc  # value = <ufunc 'multiply'>
nan: float  # value = nan
nbytes: numpy.core.numerictypes._typedict  # value = {<class 'numpy.bool_'>: 1, <class 'numpy.int8'>: 1, <class 'numpy.uint8'>: 1, <class 'numpy.int16'>: 2, <class 'numpy.uint16'>: 2, <class 'numpy.int32'>: 4, <class 'numpy.uint32'>: 4, <class 'numpy.int64'>: 8, <class 'numpy.uint64'>: 8, <class 'numpy.longlong'>: 8, <class 'numpy.ulonglong'>: 8, <class 'numpy.float16'>: 2, <class 'numpy.float32'>: 4, <class 'numpy.float64'>: 8, <class 'numpy.float128'>: 16, <class 'numpy.complex64'>: 8, <class 'numpy.complex128'>: 16, <class 'numpy.complex256'>: 32, <class 'numpy.object_'>: 8, <class 'numpy.bytes_'>: 0, <class 'numpy.str_'>: 0, <class 'numpy.void'>: 0, <class 'numpy.datetime64'>: 8, <class 'numpy.timedelta64'>: 8}
negative: numpy.ufunc  # value = <ufunc 'negative'>
newaxis = None
nextafter: numpy.ufunc  # value = <ufunc 'nextafter'>
not_equal: numpy.ufunc  # value = <ufunc 'not_equal'>
ogrid: numpy.lib.index_tricks.OGridClass  # value = <numpy.lib.index_tricks.OGridClass object>
pi: float = 3.141592653589793
positive: numpy.ufunc  # value = <ufunc 'positive'>
power: numpy.ufunc  # value = <ufunc 'power'>
r_: numpy.lib.index_tricks.RClass  # value = <numpy.lib.index_tricks.RClass object>
rad2deg: numpy.ufunc  # value = <ufunc 'rad2deg'>
radians: numpy.ufunc  # value = <ufunc 'radians'>
reciprocal: numpy.ufunc  # value = <ufunc 'reciprocal'>
remainder: numpy.ufunc  # value = <ufunc 'remainder'>
right_shift: numpy.ufunc  # value = <ufunc 'right_shift'>
rint: numpy.ufunc  # value = <ufunc 'rint'>
s_: numpy.lib.index_tricks.IndexExpression  # value = <numpy.lib.index_tricks.IndexExpression object>
sctypeDict: dict = {'?': numpy.bool_, 0: numpy.bool_, 'byte': numpy.int8, 'b': numpy.int8, 1: numpy.int8, 'ubyte': numpy.uint8, 'B': numpy.uint8, 2: numpy.uint8, 'short': numpy.int16, 'h': numpy.int16, 3: numpy.int16, 'ushort': numpy.uint16, 'H': numpy.uint16, 4: numpy.uint16, 'i': numpy.int32, 5: numpy.int32, 'uint': numpy.uint64, 'I': numpy.uint32, 6: numpy.uint32, 'intp': numpy.int64, 'p': numpy.int64, 7: numpy.int64, 'uintp': numpy.uint64, 'P': numpy.uint64, 8: numpy.uint64, 'long': numpy.int64, 'l': numpy.int64, 'ulong': numpy.uint64, 'L': numpy.uint64, 'longlong': numpy.longlong, 'q': numpy.longlong, 9: numpy.longlong, 'ulonglong': numpy.ulonglong, 'Q': numpy.ulonglong, 10: numpy.ulonglong, 'half': numpy.float16, 'e': numpy.float16, 23: numpy.float16, 'f': numpy.float32, 11: numpy.float32, 'double': numpy.float64, 'd': numpy.float64, 12: numpy.float64, 'longdouble': numpy.float128, 'g': numpy.float128, 13: numpy.float128, 'cfloat': numpy.complex128, 'F': numpy.complex64, 14: numpy.complex64, 'cdouble': numpy.complex128, 'D': numpy.complex128, 15: numpy.complex128, 'clongdouble': numpy.complex256, 'G': numpy.complex256, 16: numpy.complex256, 'O': numpy.object_, 17: numpy.object_, 'S': numpy.bytes_, 18: numpy.bytes_, 'unicode': numpy.str_, 'U': numpy.str_, 19: numpy.str_, 'void': numpy.void, 'V': numpy.void, 20: numpy.void, 'M': numpy.datetime64, 21: numpy.datetime64, 'm': numpy.timedelta64, 22: numpy.timedelta64, 'b1': numpy.bool_, 'bool8': numpy.bool_, 'i8': numpy.int64, 'int64': numpy.int64, 'u8': numpy.uint64, 'uint64': numpy.uint64, 'f2': numpy.float16, 'float16': numpy.float16, 'f4': numpy.float32, 'float32': numpy.float32, 'f8': numpy.float64, 'float64': numpy.float64, 'f16': numpy.float128, 'float128': numpy.float128, 'c8': numpy.complex64, 'complex64': numpy.complex64, 'c16': numpy.complex128, 'complex128': numpy.complex128, 'c32': numpy.complex256, 'complex256': numpy.complex256, 'object0': numpy.object_, 'bytes0': numpy.bytes_, 'str0': numpy.str_, 'void0': numpy.void, 'M8': numpy.datetime64, 'datetime64': numpy.datetime64, 'm8': numpy.timedelta64, 'timedelta64': numpy.timedelta64, 'int32': numpy.int32, 'i4': numpy.int32, 'uint32': numpy.uint32, 'u4': numpy.uint32, 'int16': numpy.int16, 'i2': numpy.int16, 'uint16': numpy.uint16, 'u2': numpy.uint16, 'int8': numpy.int8, 'i1': numpy.int8, 'uint8': numpy.uint8, 'u1': numpy.uint8, 'complex_': numpy.complex128, 'single': numpy.float32, 'csingle': numpy.complex64, 'singlecomplex': numpy.complex64, 'float_': numpy.float64, 'intc': numpy.int32, 'uintc': numpy.uint32, 'int_': numpy.int64, 'longfloat': numpy.float128, 'clongfloat': numpy.complex256, 'longcomplex': numpy.complex256, 'bool_': numpy.bool_, 'bytes_': numpy.bytes_, 'string_': numpy.bytes_, 'str_': numpy.str_, 'unicode_': numpy.str_, 'object_': numpy.object_, 'int': numpy.int64, 'float': numpy.float64, 'complex': numpy.complex128, 'bool': numpy.bool_, 'object': numpy.object_, 'str': numpy.str_, 'bytes': numpy.bytes_, 'a': numpy.bytes_, 'int0': numpy.int64, 'uint0': numpy.uint64}
sctypes: dict = {'int': [numpy.int8, numpy.int16, numpy.int32, numpy.int64], 'uint': [numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64], 'float': [numpy.float16, numpy.float32, numpy.float64, numpy.float128], 'complex': [numpy.complex64, numpy.complex128, numpy.complex256], 'others': [bool, object, bytes, str, numpy.void]}
sign: numpy.ufunc  # value = <ufunc 'sign'>
signbit: numpy.ufunc  # value = <ufunc 'signbit'>
sin: numpy.ufunc  # value = <ufunc 'sin'>
sinh: numpy.ufunc  # value = <ufunc 'sinh'>
spacing: numpy.ufunc  # value = <ufunc 'spacing'>
sqrt: numpy.ufunc  # value = <ufunc 'sqrt'>
square: numpy.ufunc  # value = <ufunc 'square'>
subtract: numpy.ufunc  # value = <ufunc 'subtract'>
tan: numpy.ufunc  # value = <ufunc 'tan'>
tanh: numpy.ufunc  # value = <ufunc 'tanh'>
tracemalloc_domain: int = 389047
true_divide: numpy.ufunc  # value = <ufunc 'divide'>
trunc: numpy.ufunc  # value = <ufunc 'trunc'>
typecodes: dict = {'Character': 'c', 'Integer': 'bhilqp', 'UnsignedInteger': 'BHILQP', 'Float': 'efdg', 'Complex': 'FDG', 'AllInteger': 'bBhHiIlLqQpP', 'AllFloat': 'efdgFDG', 'Datetime': 'Mm', 'All': '?bhilqpBHILQPefdgFDGSUVOMm'}
