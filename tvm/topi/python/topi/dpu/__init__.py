# wjq 2020/10/04
"""dpu specific declaration and schedules."""

from __future__ import absolute_import as _abs

from .conv2d import schedule_conv2d
from .injective import *
from .bitserial_conv2d import schedule_bitserial_conv2d

from .pooling import *
from .bias_add import *
from .relu import *
from .lrn import *
from .dropout import *
from .softmax import *
from .dense import *
