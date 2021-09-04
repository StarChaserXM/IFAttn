from .asm import AsmFeature
from .cfg import CfgFeature
from .cg import CgFeature
from .data import DataFeature
from .functype import TypeFeature


class FeatureManager:
    def __init__(self):
        self.all_features = [
            AsmFeature,
            CfgFeature,
            CgFeature,
            DataFeature,
            TypeFeature,
        ]
        pass

    def get_all(self, f):
        result = {}
        for feature in self.all_features:
            result.update(feature.get(f))
        return result
