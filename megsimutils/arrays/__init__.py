"""Classes modelling MEG sensor arrays"""

from .base import ConstraintPenalty, SensorArray, noise_max, noise_mean
from .BarbuteArray import BarbuteArray, GoldenRatioError
from .BarbuteArraySL import BarbuteArraySL
from .FixedBarbuteArraySL import FixedBarbuteArraySL
from .BarbuteArrayML import BarbuteArrayML
from .BarbuteArraySLGrid import BarbuteArraySLGrid
