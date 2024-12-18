# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.descriptor import (
    Descriptor as google___protobuf___descriptor___Descriptor,
)

from google.protobuf.internal.containers import (
    RepeatedScalarFieldContainer as google___protobuf___internal___containers___RepeatedScalarFieldContainer,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from typing import (
    Iterable as typing___Iterable,
    Optional as typing___Optional,
)

from typing_extensions import (
    Literal as typing_extensions___Literal,
)


builtin___bool = bool
builtin___bytes = bytes
builtin___float = float
builtin___int = int


class AgentActionProto(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    vector_actions_deprecated = ... # type: google___protobuf___internal___containers___RepeatedScalarFieldContainer[builtin___float]
    value = ... # type: builtin___float
    continuous_actions = ... # type: google___protobuf___internal___containers___RepeatedScalarFieldContainer[builtin___float]
    discrete_actions = ... # type: google___protobuf___internal___containers___RepeatedScalarFieldContainer[builtin___int]

    def __init__(self,
        *,
        vector_actions_deprecated : typing___Optional[typing___Iterable[builtin___float]] = None,
        value : typing___Optional[builtin___float] = None,
        continuous_actions : typing___Optional[typing___Iterable[builtin___float]] = None,
        discrete_actions : typing___Optional[typing___Iterable[builtin___int]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: builtin___bytes) -> AgentActionProto: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"continuous_actions",u"discrete_actions",u"value",u"vector_actions_deprecated"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"continuous_actions",b"continuous_actions",u"discrete_actions",b"discrete_actions",u"value",b"value",u"vector_actions_deprecated",b"vector_actions_deprecated"]) -> None: ...
