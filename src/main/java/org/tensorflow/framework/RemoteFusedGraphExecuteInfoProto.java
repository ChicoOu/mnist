// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: apis/core/framework/remote_fused_graph_execute_info.proto

package org.tensorflow.framework;

public final class RemoteFusedGraphExecuteInfoProto {
  private RemoteFusedGraphExecuteInfoProto() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tensorflow_RemoteFusedGraphExecuteInfo_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tensorflow_RemoteFusedGraphExecuteInfo_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tensorflow_RemoteFusedGraphExecuteInfo_TensorShapeTypeProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tensorflow_RemoteFusedGraphExecuteInfo_TensorShapeTypeProto_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n9apis/core/framework/remote_fused_graph" +
      "_execute_info.proto\022\ntensorflow\032\037apis/co" +
      "re/framework/graph.proto\032&apis/core/fram" +
      "ework/tensor_shape.proto\032\037apis/core/fram" +
      "ework/types.proto\"\202\004\n\033RemoteFusedGraphEx" +
      "ecuteInfo\022*\n\014remote_graph\030\001 \001(\0132\024.tensor" +
      "flow.GraphDef\022\035\n\025graph_input_node_name\030\002" +
      " \003(\t\022\036\n\026graph_output_node_name\030\003 \003(\t\022\025\n\r" +
      "executor_name\030\004 \001(\t\022&\n\036serialized_execut" +
      "or_parameters\030\005 \001(\014\022f\n default_graph_inp",
      "ut_tensor_shape\030\006 \003(\0132<.tensorflow.Remot" +
      "eFusedGraphExecuteInfo.TensorShapeTypePr" +
      "oto\022g\n!default_graph_output_tensor_shape" +
      "\030\007 \003(\0132<.tensorflow.RemoteFusedGraphExec" +
      "uteInfo.TensorShapeTypeProto\032h\n\024TensorSh" +
      "apeTypeProto\022#\n\005dtype\030\001 \001(\0162\024.tensorflow" +
      ".DataType\022+\n\005shape\030\002 \001(\0132\034.tensorflow.Te" +
      "nsorShapeProtoB\200\001\n\030org.tensorflow.framew" +
      "orkB RemoteFusedGraphExecuteInfoProtoP\001Z" +
      "=github.com/tensorflow/tensorflow/tensor",
      "flow/go/core/framework\370\001\001b\006proto3"
    };
    com.google.protobuf.Descriptors.FileDescriptor.InternalDescriptorAssigner assigner =
        new com.google.protobuf.Descriptors.FileDescriptor.    InternalDescriptorAssigner() {
          public com.google.protobuf.ExtensionRegistry assignDescriptors(
              com.google.protobuf.Descriptors.FileDescriptor root) {
            descriptor = root;
            return null;
          }
        };
    com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
          org.tensorflow.framework.GraphProtos.getDescriptor(),
          org.tensorflow.framework.TensorShapeProtos.getDescriptor(),
          org.tensorflow.framework.TypesProtos.getDescriptor(),
        }, assigner);
    internal_static_tensorflow_RemoteFusedGraphExecuteInfo_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_tensorflow_RemoteFusedGraphExecuteInfo_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tensorflow_RemoteFusedGraphExecuteInfo_descriptor,
        new java.lang.String[] { "RemoteGraph", "GraphInputNodeName", "GraphOutputNodeName", "ExecutorName", "SerializedExecutorParameters", "DefaultGraphInputTensorShape", "DefaultGraphOutputTensorShape", });
    internal_static_tensorflow_RemoteFusedGraphExecuteInfo_TensorShapeTypeProto_descriptor =
      internal_static_tensorflow_RemoteFusedGraphExecuteInfo_descriptor.getNestedTypes().get(0);
    internal_static_tensorflow_RemoteFusedGraphExecuteInfo_TensorShapeTypeProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tensorflow_RemoteFusedGraphExecuteInfo_TensorShapeTypeProto_descriptor,
        new java.lang.String[] { "Dtype", "Shape", });
    org.tensorflow.framework.GraphProtos.getDescriptor();
    org.tensorflow.framework.TensorShapeProtos.getDescriptor();
    org.tensorflow.framework.TypesProtos.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}