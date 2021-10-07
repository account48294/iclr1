"""
    General convenience utilities for working with collections.
"""


import typing


Element = typing.TypeVar('Element')


def dict_to_list(
        dictionary: typing.Dict[int, Element]) -> typing.List[Element]:
    """
        Convert a number keyed dictionary (starting at 0) into a list of its
        values.
    """
    return [dictionary[index] for index in range(len(dictionary))]


Key = typing.TypeVar("Key")
Value = typing.TypeVar("Value")


def constant_dict(
        keys: typing.Iterable[Key], value: Value) -> typing.Dict[Key, Value]:
    """ Return a dict where all the keys have the same value """
    return {key: value for key in keys}
