# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from mlagents_envs.communicator_objects import unity_message_pb2 as mlagents__envs_dot_communicator__objects_dot_unity__message__pb2


class UnityToExternalProtoStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Exchange = channel.unary_unary(
                '/communicator_objects.UnityToExternalProto/Exchange',
                request_serializer=mlagents__envs_dot_communicator__objects_dot_unity__message__pb2.UnityMessageProto.SerializeToString,
                response_deserializer=mlagents__envs_dot_communicator__objects_dot_unity__message__pb2.UnityMessageProto.FromString,
                )


class UnityToExternalProtoServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Exchange(self, request, context):
        """Sends the academy parameters
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_UnityToExternalProtoServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Exchange': grpc.unary_unary_rpc_method_handler(
                    servicer.Exchange,
                    request_deserializer=mlagents__envs_dot_communicator__objects_dot_unity__message__pb2.UnityMessageProto.FromString,
                    response_serializer=mlagents__envs_dot_communicator__objects_dot_unity__message__pb2.UnityMessageProto.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'communicator_objects.UnityToExternalProto', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class UnityToExternalProto(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Exchange(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communicator_objects.UnityToExternalProto/Exchange',
            mlagents__envs_dot_communicator__objects_dot_unity__message__pb2.UnityMessageProto.SerializeToString,
            mlagents__envs_dot_communicator__objects_dot_unity__message__pb2.UnityMessageProto.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
