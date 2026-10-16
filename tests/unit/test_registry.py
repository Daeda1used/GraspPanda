"""Registry behaviour: registration rules, lookup messages, building with injection."""

from __future__ import annotations

import itertools
import sys

import pytest

from grasppanda import registry as registry_module
from grasppanda.registry import (
    BACKBONES,
    HEADS,
    INJECTED,
    BuildError,
    Registry,
    RegistryError,
    all_registries,
    registry_for,
)
from grasppanda.utils.imports import MissingDependencyError, require

from .conftest import TBackbone

_counter = itertools.count()


@pytest.fixture(autouse=True)
def _forget_throwaway_registries():
    """Every registry is recorded in a table of the process; the ones a test makes are
    taken out of it again (with the base classes they declared), so that a later test
    sees only the toolbox's own slots. Bases that the toolbox's registries learn while a
    test imports their modules stay."""
    before = set(registry_module._REGISTRIES)
    yield
    made = [r for name, r in registry_module._REGISTRIES.items() if name not in before]
    for registry in made:
        del registry_module._REGISTRIES[registry.name]
    for cls, owner in list(registry_module._BASES.items()):
        if any(owner is registry for registry in made):
            del registry_module._BASES[cls]


def fresh(**kwargs) -> Registry:
    return Registry(f"TEST_{next(_counter)}", **kwargs)


def test_a_throwaway_registry_is_forgotten_after_its_test():
    """Without the fixture above, the catalogue test fails when it runs after this file."""
    names = [registry.name for registry in all_registries().values()]
    assert not [name for name in names if name.startswith("TEST_")]
    fresh()
    assert any(name.startswith("TEST_") for name in all_registries())


def test_register_and_get():
    registry = fresh()

    @registry.register()
    class Alpha:
        def __init__(self, width: int = 1) -> None:
            self.width = width

    assert registry.get("Alpha") is Alpha
    assert "Alpha" in registry and registry.names() == ["Alpha"] and len(registry) == 1


def test_duplicate_names_raise_unless_forced():
    registry = fresh()

    class First:
        def __init__(self) -> None: ...

    class Second:
        def __init__(self) -> None: ...

    registry.add(First, "thing")
    with pytest.raises(RegistryError, match="already registered"):
        registry.add(Second, "thing")
    registry.add(Second, "thing", force=True)
    assert registry.get("thing") is Second


def test_var_kwargs_are_rejected():
    registry = fresh()
    with pytest.raises(TypeError, match=r"accepts \*\*kwargs"):

        @registry.register()
        class Loose:
            def __init__(self, **kwargs) -> None: ...


def test_base_class_is_enforced():
    with pytest.raises(TypeError, match="must subclass Backbone"):

        @BACKBONES.register()
        class NotABackbone:
            def __init__(self) -> None: ...


def test_lookup_failure_lists_close_matches_and_other_registries():
    with pytest.raises(RegistryError) as error:
        BACKBONES.get("TBackbon")
    assert "Did you mean 'TBackbone'" in str(error.value)
    with pytest.raises(RegistryError, match="registered in BACKBONES, which owns a different slot"):
        HEADS.get("TBackbone")


def test_registry_for_follows_inheritance():
    assert registry_for(TBackbone) is BACKBONES
    assert registry_for(int) is None


def test_build_injects_only_what_is_missing_and_never_overrides():
    registry = fresh()

    @registry.register()
    class Child:
        def __init__(self, in_channels: int = INJECTED, scale: float = 1.0) -> None:
            self.in_channels, self.scale = in_channels, scale

    assert registry.build({"type": "Child"}, in_channels=8, unused=1).in_channels == 8
    assert registry.build({"type": "Child", "in_channels": 8}, in_channels=8).in_channels == 8
    with pytest.raises(BuildError, match="in the config, but the parent component provides 8"):
        registry.build({"type": "Child", "in_channels": 4}, in_channels=8)
    with pytest.raises(BuildError, match="needs in_channels"):
        registry.build({"type": "Child"})
    with pytest.raises(BuildError, match="needs a 'type' key"):
        registry.build({"scale": 2.0})

    built = Child(in_channels=2)
    assert registry.build(built) is built


def test_modules_below_the_location_are_imported_lazily(tmp_path, monkeypatch):
    name = f"TEST_{next(_counter)}"
    package = tmp_path / f"pkg_{name.lower()}"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "good.py").write_text(
        f"from grasppanda.registry import all_registries\n\n"
        f"@all_registries()['{name}'].register()\nclass Lazy:\n    def __init__(self) -> None: ...\n"
    )
    (package / "needs_extra.py").write_text("import a_module_that_does_not_exist\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = Registry(name, location=package.name)
    assert f"{package.name}.good" not in sys.modules
    assert registry.names() == ["Lazy"]
    with pytest.raises(RegistryError, match="1 module\\(s\\) failed to import"):
        registry.get("Missing")


def test_missing_optional_dependency_says_how_to_install_it():
    with pytest.raises(MissingDependencyError) as error:
        require("a_module_that_does_not_exist", extra="timm", purpose="Uni3D")
    assert str(error.value) == (
        "Uni3D needs 'a_module_that_does_not_exist', which is not installed. Install it from "
        'the repository root with: pip install -e ".[timm]"'  # this test runs in a checkout
    )
    assert require("math", extra="x", purpose="y").pi > 3


def test_an_installed_wheel_is_told_the_wheel_command(monkeypatch):
    from grasppanda.utils import imports

    monkeypatch.setattr(imports, "_checkout", lambda: False)
    assert imports.install_hint("timm") == 'Install it with: pip install "grasppanda[timm]"'
    with pytest.raises(MissingDependencyError, match=r'pip install "grasppanda\[timm\]"$'):
        require("a_module_that_does_not_exist", extra="timm", purpose="Uni3D")


@pytest.mark.parametrize(
    ("cuda", "build"),
    [("11.8", "spconv-cu118"), ("12.4", "spconv-cu124"), (None, "spconv-cpu"), ("12.8", None)],
)
def test_spconv_hints_name_the_build_of_the_installed_pytorch(monkeypatch, cuda, build):
    from grasppanda.utils import imports

    monkeypatch.setattr(imports, "_checkout", lambda: False)
    monkeypatch.setattr(imports, "_torch_cuda", lambda: cuda)
    everything = "spconv-cu118 | spconv-cu121 | spconv-cu124 | spconv-cu126 | spconv-cpu"
    for extra in ("spconv", "spconv-cu118"):  # a build name stands for its library
        hint = imports.install_hint(extra)
        if build is not None:
            others = everything.replace(f"{build} | ", "").replace(f" | {build}", "")
            assert imports.installable(extra, this_machine=True) == build
            assert hint == (
                f'Install it with: pip install "grasppanda[{build}]". That is the build for your '
                f"PyTorch ({f'CUDA {cuda}' if cuda else 'CPU only'}); the others: {others}."
            )
        else:  # no build for this CUDA version: the choice, with the version to match
            assert imports.installable(extra, this_machine=True) == everything
            assert hint == (
                'Install it with: pip install "grasppanda[<build>]". Choose <build> to match '
                f"your PyTorch (CUDA 12.8): {everything}."
            )
    assert imports.installable("spconv") == everything  # the catalogue lists every build


def test_a_component_without_a_constructor_takes_no_arguments():
    registry = fresh()

    @registry.register()
    class Plain:
        pass

    assert isinstance(registry.build({"type": "Plain"}), Plain)
    with pytest.raises(BuildError, match="cannot build Plain"):
        registry.build({"type": "Plain", "anything": 1})
