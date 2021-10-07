"""
    Calculate Baseline Shapley (BShap) attribution
    ( https://proceedings.mlr.press/v119/sundararajan20b.html )
"""


import itertools
import math
import typing

import keras
import tensorflow as tf

from integrated_certainty_gradients import collection_utilities

Element = typing.TypeVar('Element')


def power_set(
        elements: typing.Collection[Element]
        ) -> typing.Set[typing.FrozenSet[Element]]:
    """
        Given a collection of items, which should be unique, return all the
        possible subsets, including the empty set and the original collection.
    """
    result = set()
    for subset_size in range(len(elements) + 1):
        for subset in itertools.combinations(elements, subset_size):
            result.add(frozenset(subset))
    return result


def shapley_values(
        set_scores: typing.Dict[typing.FrozenSet[Element], float]
        ) -> typing.Dict[Element, float]:
    """
        Calculate the Shapley values of elements based on the scores of sets
        that contain them.

        Shapley values are the unique mathematically justified way of
        dividing an outcome score between elements, based on how much score
        would be achieved for each subset of the elements.

        :param set_scores: Must contain a score for every possible subset of
            a set of elements.
    """
    elements_count = round(math.log2(len(set_scores)))
    for subset in set_scores:
        if len(subset) == elements_count:
            elements = subset
            break
    result: typing.Dict[Element, float] = {}
    for element in elements:
        result[element] = 0
        other_elements = elements - frozenset([element])
        for excluded_set in power_set(other_elements):
            included_set = excluded_set.union(frozenset([element]))
            contribution = set_scores[included_set] - set_scores[excluded_set]
            prefactor = math.factorial(len(excluded_set)) * math.factorial(
                elements_count - len(excluded_set) - 1)
            result[element] += prefactor * contribution
        result[element] = result[element] / math.factorial(elements_count)
    return result


def bshap(
        score_function: typing.Callable, baseline: typing.List[Element],
        target: typing.List[Element]) -> typing.List[float]:
    """

        :param score_function: callable that takes a list of lists of elements
            (list of vectors) and returns a list of scores, one for each
            vector.
        :param baseline: baseline vector for attribution
        :param target: target vector for attribution
        :return: an attribution score for each vector component index
    """
    component_selections = []
    vectors = []
    for selection in power_set(range(len(baseline))):
        component_selections.append(selection)
        vector = baseline.copy()
        for index in selection:
            vector[index] = target[index]
        vectors.append(vector)
    scores = score_function(vectors)
    set_scores = {
        component_selections[index]: scores[index]
        for index in range(len(scores))}
    return collection_utilities.dict_to_list(shapley_values(set_scores))


def bshap_classifier(
        model: keras.Model, output_class: int,
        baseline: typing.List[tf.Tensor],
        target: typing.List[tf.Tensor]) -> typing.List[float]:
    """
        Perform BShap on the output of a Keras classifier

        :param model: A classifier model
        :param output_class: The output class to take probability of as the
            score.
        :param baseline: An input to the model as a list of tensors. Each
            list element will be attributed. The list will be converted to
            a tensor before being input to the model.
        :param target: An input to the model as a list of tensors. Each
            list element will be attributed. The list will be converted to
            a tensor before being input to the model.
        :return: A list of the scores for each component.
    """

    def adapted_model(list_of_lists):
        output = model(tf.convert_to_tensor(list_of_lists))
        return [
            probabilities[output_class].numpy() for probabilities in output]

    return bshap(adapted_model, baseline, target)
