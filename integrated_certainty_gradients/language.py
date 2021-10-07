"""
    Utilities relating to the Python language.
"""
import typing
from typing import List, Dict, Callable


def call_method(
        method_name: str, positional_arguments: List = [],
        keyword_arguments: Dict = {}) -> Callable:
    """
        Return a function that calls the given method on its argument, with
        the given arguments.
    """
    return lambda object: getattr(object, method_name)(
        *positional_arguments, **keyword_arguments)


def delegate_to_attribute(
        delegating_class: typing.Type, attribute_name: str,
        method_name: str) -> None:
    """
        Pass calls to a method of a class to an attribute of the class
        instance.

        For example, if the delegating class is MyClass, the attribute name
        is _my_attribute, the method name is my_method, and my_object is an
        instance of MyClass, then my_object.my_method will result in
        redirecting the call to my_object._my_attribute.my_method.

        :param delegating_class: The class whose instances should have their
            method calls diverted.
        :param attribute_name: The name of the attribute contained by instances
            of the delegating class which should receive the redirected method
            calls.
        :param method_name: The name of the method which has its calls
            redirected from instances of the delegating class (and also the
            name of the method on the attribute object which the call gets
            redirected to). All other methods of this delegating class are not
            affected.
    """
    def delegating_method(self, *positional_arguments, **keyword_arguments):
        return getattr(getattr(self, attribute_name), method_name)(
            *positional_arguments, **keyword_arguments)
    setattr(delegating_class, method_name, delegating_method)


def do_nothing(
        *arguments: typing.Any, **keyword_arguments: typing.Any) -> None:
    """ For use when a placeholder (no effect) callback is required. """
    return
