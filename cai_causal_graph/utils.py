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
import re
from itertools import tee
from typing import Dict, Iterable, List, Tuple, TypeVar

from cai_causal_graph.type_definitions import HasIdentifier, NodeLike


def get_variable_name_and_lag(node_name: NodeLike) -> Tuple[str, int]:
    """
    Extract the variable name and time series lag from a node name.

    A lag or future lag is indicated by the string 'lag(n=Y)' or 'future(n=Y)' where Y is the lag. The lag can be any
    integer. If these strings appear multiple times in a node name, then a `ValueError` is raised.

    It is assumed the variable name and the lag or future lag are separated by a space. Anything prior to the space is
    considered the variable name.

    Example:
        'X lag(n=2)' -> 'X', -2 if lagged in the past,
        'X future(n=2)' -> 'X', 2 if lagged in the future.

    :param node_name: The name of the node.
    :return: A tuple with the variable name and lag.
    """
    # get the string name if the node is a Node object
    if isinstance(node_name, HasIdentifier):
        node_name = node_name.identifier

    if not isinstance(node_name, str):
        raise TypeError(f'Expected node name to be a string, got type {type(node_name)}.')

    is_match = re.match(r'^(.+?)(?: lag\(n=(\d+)\))?(?: future\(n=(\d+)\))?$', node_name)

    lag_matches = re.findall(r'lag\(n=(\d+)\)', node_name)
    future_matches = re.findall(r'future\(n=(\d+)\)', node_name)
    num_matches = (len(lag_matches) if lag_matches is not None else 0) + (
        len(future_matches) if future_matches is not None else 0
    )

    if is_match:

        if num_matches > 1:
            raise ValueError(f'Invalid node name: {node_name}. Multiple lag or future lags detected.')

        variable_name = is_match.group(1)
        past_lag = is_match.group(2)
        future_lag = is_match.group(3)

        # # we need to check that the node name does not contain both a lag and future lag
        if past_lag:
            return variable_name, -int(past_lag)
        elif future_lag:
            return variable_name, int(future_lag)
        else:
            return variable_name, 0  # no lag information

    else:
        raise ValueError(f'Invalid node name: {node_name}. Expected format: "X", "X lag(n=2)", or "X future(n=2)".')


def get_name_with_lag(variable_or_node_name: str, lag: int) -> str:
    """
    Get the name of a lagged node.

    If the lag is 0, then the variable name is returned.
    If the lag is not 0, then the old lag is removed and
    variable name is appended with the lag.

    Example:
        'X', -2 -> 'X lag(n=2)', # lagged in the past
        'X', 2 -> 'X future(n=2)', # lagged in the future

    :param variable_or_node_name: The name of the variable or node.
    :param lag: The lag of the node.
    :return: The name of the lagged node.
    """
    # remove the old lag if it exists
    variable_name, old_lag = get_variable_name_and_lag(variable_or_node_name)

    assert isinstance(lag, int), f'Expected lag to be an integer, got type {type(lag)}.'
    if lag == 0:
        return variable_name
    elif lag > 0:
        return f'{variable_name} future(n={lag})'
    else:
        return f'{variable_name} lag(n={-lag})'


def extract_names_and_lags(
    node_names: List[NodeLike],
) -> Tuple[List[Dict[str, int]], int]:
    """
    Extract the names and lags from a list of node names.

    This is useful for converting a list of node names into a list of variable names and lags.

    Example:
    >>> ['X', 'Y', 'Z lag(n=2)'] -> [{'X': 0}, {'Y': 0}, {'Z': 2}], 2

    :param node_names: List of node names.
    :return: Tuple with the first element being a list of dictionaries with variable names and lags and the second
        element being the maximum lag.
    """
    names_and_lags = []
    max_lag: int = 0
    for node_name in node_names:
        variable_name, lag = get_variable_name_and_lag(node_name)
        if abs(lag) > abs(max_lag):
            max_lag = lag
        names_and_lags.append({variable_name: lag})

    return names_and_lags, max_lag


T = TypeVar('T')


def pairwise(iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
    """
    Equivalent to `itertools.pairwise`.

    This is used because `itertools.pairwise` method is not evailable for Python 3.8 and 3.9.
    """
    a, b = tee(iterable, 2)
    next(b, None)
    return zip(a, b)
