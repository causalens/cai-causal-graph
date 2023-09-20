"""
Copyright 2023 Impulse Innovations Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import inspect
from enum import Enum


class BaseDefinition(str, Enum):
    """Base Definitions in the style of StrEnums."""

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"'{self.value}'"

    @classmethod
    def values(cls):
        """Returns all values defined in the class. Excludes values starting with __."""
        return [x[1].value for x in inspect.getmembers(cls) if not (inspect.isroutine(x[1]) or x[0].startswith('__'))]

    @classmethod
    def has_value(cls, value):
        return value in cls.values()


class EdgeActivation(BaseDefinition):
    ABSOLUTE = 'absolute'
    CONSTANT = 'constant'
    COS = 'cos'
    HARD_TANH = 'hard_tanh'
    IDENTITY = 'identity'
    LINEAR = 'linear'
    BIAS = 'bias'
    POSITIVE_LINEAR = 'positive_linear'
    NEGATIVE_LINEAR = 'negative_linear'
    LOG_SIGMOID = 'log_sigmoid'
    MONOTONIC = 'monotonic'
    MONOTONIC_DECREASING = 'monotonic_decreasing'
    MONOTONIC_INCREASING = 'monotonic_increasing'
    NEGATIVE = 'negative'
    PIECEWISE_LINEAR = 'piecewise_linear'
    PIECEWISE_LINEAR_INCREASING = 'piecewise_linear_increasing'
    PIECEWISE_LINEAR_DECREASING = 'piecewise_linear_decreasing'
    PIECEWISE_LINEAR_MONOTONIC = 'piecewise_linear_monotonic'
    PIECEWISE_LINEAR_MLP = 'piecewise_linear_mlp'
    POLYNOMIAL = 'polynomial'
    POWER = 'power'
    PWL_WITH_SKIP = 'piecewise_linear_with_skip_connection'
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    SIN = 'sin'
    SQUARE = 'square'
    TAN = 'tan'
    TANH = 'tanh'
    BOOST = 'boost'
    BOOST_OVER_TIME = 'boost_over_time'
    FORCE = 'force'
    DISCONTINUITY = 'discontinuity'
    CONVEX = 'convex'
    CONCAVE = 'concave'
    # Pyro activations
    GP = 'gp'
    GP_POSITIVE = 'gp_positive'
    GP_NEGATIVE = 'gp_negative'
    GP_MONOTONIC_INCREASING = 'gp_monotonic_increasing'
    GP_MONOTONIC_DECREASING = 'gp_monotonic_decreasing'
    GP_CONVEX = 'gp_convex'
    GP_CONCAVE = 'gp_concave'


class NodeAggregation(BaseDefinition):
    MAX = 'max'
    MIN = 'min'
    PRODUCT = 'product'
    SUM = 'sum'
    SUM_WITH_BIAS = 'sum_with_bias'
    IDENTITY = 'identity'
    SOFTMAX = 'softmax'
    SOFTMIN = 'softmin'
    MLP = 'mlp'
    INTERACTION = 'interaction'
