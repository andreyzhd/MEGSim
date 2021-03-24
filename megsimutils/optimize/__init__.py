"""Classes modelling MEG sensor arrays"""

from .base import ConstraintPenalty, SensorArray
from .BarbuteArray import BarbuteArray, GoldenRatioError
from .BarbuteArraySL import BarbuteArraySL
from .FixedBarbuteArraySL import FixedBarbuteArraySL
from .BarbuteArrayML import BarbuteArrayML
