import imp
from .backbones import *  # noqa: F401,F403
from .builder import (build_backbone, build_model, build_head, build_loss, build_memory)
from .byol import BYOL
from .heads import *
from .classification import Classification
from .deepcluster import DeepCluster
from .odc import ODC
from .necks import *
from .npid import NPID
from .memories import *
from .moco import MOCO
from .registry import (BACKBONES, MODELS, NECKS, MEMORIES, HEADS, LOSSES)
from .rotation_pred import RotationPred
from .relative_loc import RelativeLoc
from .simclr import SimCLR
from .densecl import DenseCL
from .densespatialcl import DenseSpatialCL
from .densespatial2hcl import DenseSpatial2hCL
from .densetopkcl import DenseTopkCL
from .spatial2hcl import Spatial2hCL
from .spatialcl import SpatialCL
from .spatialcl2 import SpatialCL2, SpatialCL3, SpatialCL4, SpatialCL5, SpatialCL6, SpatialCL7, SpatialCL8 , SpatialCL9 , SpatialCL10
from .spatialmoco   import SpatialMoco
from .spaticalclmcrop import SpatialCLMcrop
from .spatialclocrop import SpatialCLOcrop
from .npidmcrop import NPIDMcrop
from .npidocrop import NpidOcrop
from .spatialcljigsaw import SpatialCLJigsaw
from .pirl import PIRL