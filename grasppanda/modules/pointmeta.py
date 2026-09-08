"""Native PointMetaBase features in an isolated OpenPoints Python namespace."""
import ast
from functools import lru_cache
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types

from torch import nn

from .pointnext import PointNeXtBackbone


_PREFIX = '_grasppanda_pointmeta'


class _SourceLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        tree = ast.parse(self.get_data(self.path), filename=self.path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module == 'openpoints' or node.module.startswith('openpoints.'):
                    node.module = _PREFIX + node.module[len('openpoints'):]
        return compile(tree, self.path, 'exec')


class _SourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root):
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX + '.'):
            return None
        relative = fullname[len(_PREFIX)+1:].replace('.', '/')
        module = self.root/(relative + '.py')
        package = self.root/relative/'__init__.py'
        file = package if package.is_file() else module
        if not file.is_file():
            return None
        return importlib.util.spec_from_file_location(fullname, file,
            loader=_SourceLoader(fullname, str(file)),
            submodule_search_locations=[str(file.parent)] if file == package else None)


@lru_cache(maxsize=1)
def native_module():
    from ..config import ROOT
    root = ROOT/'environments/sources/cv/pointmetabase/openpoints'
    if not (root/'models/backbone/pointmetabase.py').is_file():
        raise ValueError('PointMetaBase source is missing; run ./panda install')
    try:
        import _grasppanda_openpoints_cuda as extension
    except ImportError as error:
        raise ValueError('Shared OpenPoints operators are missing; run ./panda install') from error
    # Preserve the author's Python layers without importing unrelated datasets
    # or replacing another component's global openpoints package.
    for relative in ('', 'utils', 'models', 'models.backbone', 'cpp', 'cpp.pointnet2_batch'):
        name = _PREFIX + ('.' + relative if relative else '')
        package = types.ModuleType(name)
        package.__path__ = [str(root/relative.replace('.', '/'))]
        package.__package__ = name
        if relative.startswith('cpp'):
            package.pointnet2_cuda = extension
        sys.modules[name] = package
    sys.meta_path.insert(0, _SourceFinder(root))
    encoder = importlib.import_module(_PREFIX + '.models.backbone.pointmetabase')
    decoder = importlib.import_module(_PREFIX + '.models.backbone.pointnext')
    return encoder, decoder


class PointMetaBackbone(PointNeXtBackbone):
    family = 'PointMetaBase'

    def __init__(self, width=32, blocks=(1,3,5,3,3), nsample=32, radius=.05,
                 radius_scaling=2., expansion=1, normalize_dp=True,
                 local_reduction='max', activation='relu', use_res=True,
                 sa_layers=1, sa_use_res=False, decoder_layers=2):
        nn.Module.__init__(self)
        from easydict import EasyDict
        if blocks[0] != 1:
            raise ValueError('PointMetaBase requires a single stem block in blocks[0]')
        native, decoder = native_module()
        self.encoder = native.PointMetaBaseEncoder(in_channels=3, width=width,
            blocks=list(blocks), strides=[1,4,4,4,4], nsample=nsample,
            radius=radius, radius_scaling=radius_scaling, expansion=expansion,
            sa_layers=sa_layers, sa_use_res=sa_use_res, use_res=use_res,
            group_args=EasyDict(NAME='ballquery', normalize_dp=normalize_dp),
            conv_args={}, aggr_args={'feature_type':'dp_fj','reduction':local_reduction},
            norm_args={'norm':'bn'}, act_args={'act':activation})
        self.decoder = decoder.PointNextDecoder(self.encoder.channel_list.copy(),
                                               decoder_layers=decoder_layers)
        self.projection = nn.Conv1d(self.decoder.out_channels, 256, 1)
