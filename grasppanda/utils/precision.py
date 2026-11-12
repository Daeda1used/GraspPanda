"""Whether CUDA convolutions and matrix products may round float32 to TF32.

PyTorch has two interfaces for it. The first is two booleans,
``torch.backends.cudnn.allow_tf32`` and ``torch.backends.cuda.matmul.allow_tf32``, with
``torch.set_float32_matmul_precision`` beside them. PyTorch 2.9 added a precision per
backend and operator (``torch.backends.cudnn.conv.fp32_precision``, ...), which an operator
inherits from its backend and from ``torch.backends.fp32_precision`` while its own says
``"none"``. The two are kept side by side: a boolean writes its operators too, an operator
does not write the boolean, and PyTorch refuses to read a boolean that disagrees with its
operators.

A process that hosts GraspPanda (a notebook, another program) may have used either. This
module therefore reads what is in effect through the operators where they exist, and writes
through the booleans first, which keeps both interfaces readable, and through the operators
where a boolean cannot say it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any, NamedTuple

import torch

__all__ = ["Tf32", "set_tf32", "tf32", "tf32_restored"]


class Tf32(NamedTuple):
    """What is in effect: may convolutions, may matrix products round to TF32."""

    convolutions: bool
    matmul: bool


def _operators() -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """The switches per operator of PyTorch 2.9 and newer, ``((conv, rnn), (matmul,))``;
    two empty tuples on an older PyTorch."""
    try:
        cudnn = (torch.backends.cudnn.conv, torch.backends.cudnn.rnn)
        matmul = (torch.backends.cuda.matmul,)
        for operator in (*cudnn, *matmul):
            operator.fp32_precision  # noqa: B018 (the attribute an older PyTorch lacks)
    except AttributeError:
        return (), ()
    return cudnn, matmul


def _cudnn_boolean(allowed: bool) -> None:
    torch.backends.cudnn.allow_tf32 = allowed


def _matmul_boolean(allowed: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = allowed


def _read(switch: Callable[[], Any]) -> Any:
    """A value of the first interface, or ``None`` where PyTorch refuses to read it."""
    try:
        return switch()
    except RuntimeError:
        return None


def tf32() -> Tf32:
    """The precision this process computes in now."""
    cudnn, matmul = _operators()
    if not cudnn:
        return Tf32(
            bool(torch.backends.cudnn.allow_tf32), bool(torch.backends.cuda.matmul.allow_tf32)
        )
    return Tf32(cudnn[0].fp32_precision == "tf32", matmul[0].fp32_precision == "tf32")


def _set(boolean: Callable[[bool], None], operators: tuple[Any, ...], allowed: bool) -> None:
    with contextlib.suppress(RuntimeError):  # a state the boolean cannot express
        boolean(allowed)
    for operator in operators:
        # What is in effect, the operator's own or inherited: for float32 the boolean
        # writes "none", which a backend set to TF32 through the operators would overrule.
        if (operator.fp32_precision == "tf32") is not allowed:
            operator.fp32_precision = "tf32" if allowed else "ieee"


def set_tf32(*, convolutions: bool | None = None, matmul: bool | None = None) -> None:
    """Allow or forbid TF32 for cuDNN (convolutions and recurrent layers) and for matrix
    products; ``None`` leaves a switch as it is."""
    cudnn, product = _operators()
    if convolutions is not None:
        _set(_cudnn_boolean, cudnn, convolutions)
    if matmul is not None:
        _set(_matmul_boolean, product, matmul)


@contextlib.contextmanager
def tf32_restored() -> Iterator[None]:
    """Leave the precision switches as they were found, readable through both interfaces
    wherever they were.

    Per operator, what was in effect is put back. Whether the operator had inherited it or
    had it set cannot be read back, so it inherits again where that gives the same. The
    name of the matmul precision is put back as well as its boolean: ``medium`` is a third
    value behind it.
    """
    cudnn, product = _operators()
    if not cudnn:
        convolutions = torch.backends.cudnn.allow_tf32
        precision = torch.get_float32_matmul_precision()
        try:
            yield
        finally:
            torch.backends.cudnn.allow_tf32 = convolutions
            torch.set_float32_matmul_precision(precision)
        return
    booleans = (
        (_cudnn_boolean, _read(lambda: torch.backends.cudnn.allow_tf32)),
        (_matmul_boolean, _read(lambda: torch.backends.cuda.matmul.allow_tf32)),
    )
    name = _read(torch.get_float32_matmul_precision)
    found = [(operator, operator.fp32_precision) for operator in (*cudnn, *product)]
    try:
        yield
    finally:
        for boolean, allowed in booleans:
            if allowed is not None:
                with contextlib.suppress(RuntimeError):
                    boolean(bool(allowed))
        if name is not None and _read(torch.get_float32_matmul_precision) != name:
            torch.set_float32_matmul_precision(name)
        for operator, precision in found:
            if operator.fp32_precision != precision:
                operator.fp32_precision = "none"
            if operator.fp32_precision != precision:
                operator.fp32_precision = precision
