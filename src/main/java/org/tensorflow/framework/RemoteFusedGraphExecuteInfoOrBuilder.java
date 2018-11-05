// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: apis/core/framework/remote_fused_graph_execute_info.proto

package org.tensorflow.framework;

public interface RemoteFusedGraphExecuteInfoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.RemoteFusedGraphExecuteInfo)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Definition of remote graph
   * </pre>
   *
   * <code>optional .tensorflow.GraphDef remote_graph = 1;</code>
   */
  boolean hasRemoteGraph();
  /**
   * <pre>
   * Definition of remote graph
   * </pre>
   *
   * <code>optional .tensorflow.GraphDef remote_graph = 1;</code>
   */
  org.tensorflow.framework.GraphDef getRemoteGraph();
  /**
   * <pre>
   * Definition of remote graph
   * </pre>
   *
   * <code>optional .tensorflow.GraphDef remote_graph = 1;</code>
   */
  org.tensorflow.framework.GraphDefOrBuilder getRemoteGraphOrBuilder();

  /**
   * <pre>
   * Remote fused graph input node name
   * </pre>
   *
   * <code>repeated string graph_input_node_name = 2;</code>
   */
  java.util.List<java.lang.String>
      getGraphInputNodeNameList();
  /**
   * <pre>
   * Remote fused graph input node name
   * </pre>
   *
   * <code>repeated string graph_input_node_name = 2;</code>
   */
  int getGraphInputNodeNameCount();
  /**
   * <pre>
   * Remote fused graph input node name
   * </pre>
   *
   * <code>repeated string graph_input_node_name = 2;</code>
   */
  java.lang.String getGraphInputNodeName(int index);
  /**
   * <pre>
   * Remote fused graph input node name
   * </pre>
   *
   * <code>repeated string graph_input_node_name = 2;</code>
   */
  com.google.protobuf.ByteString
      getGraphInputNodeNameBytes(int index);

  /**
   * <pre>
   * Remote fused graph output node name
   * </pre>
   *
   * <code>repeated string graph_output_node_name = 3;</code>
   */
  java.util.List<java.lang.String>
      getGraphOutputNodeNameList();
  /**
   * <pre>
   * Remote fused graph output node name
   * </pre>
   *
   * <code>repeated string graph_output_node_name = 3;</code>
   */
  int getGraphOutputNodeNameCount();
  /**
   * <pre>
   * Remote fused graph output node name
   * </pre>
   *
   * <code>repeated string graph_output_node_name = 3;</code>
   */
  java.lang.String getGraphOutputNodeName(int index);
  /**
   * <pre>
   * Remote fused graph output node name
   * </pre>
   *
   * <code>repeated string graph_output_node_name = 3;</code>
   */
  com.google.protobuf.ByteString
      getGraphOutputNodeNameBytes(int index);

  /**
   * <pre>
   * Executor's name
   * </pre>
   *
   * <code>optional string executor_name = 4;</code>
   */
  java.lang.String getExecutorName();
  /**
   * <pre>
   * Executor's name
   * </pre>
   *
   * <code>optional string executor_name = 4;</code>
   */
  com.google.protobuf.ByteString
      getExecutorNameBytes();

  /**
   * <pre>
   * Optional: Parameters given to the executor
   * </pre>
   *
   * <code>optional bytes serialized_executor_parameters = 5;</code>
   */
  com.google.protobuf.ByteString getSerializedExecutorParameters();

  /**
   * <pre>
   * Optional: Default graph input tensor shape used to allocate memory
   * before executing op
   * </pre>
   *
   * <code>repeated .tensorflow.RemoteFusedGraphExecuteInfo.TensorShapeTypeProto default_graph_input_tensor_shape = 6;</code>
   */
  java.util.List<org.tensorflow.framework.RemoteFusedGraphExecuteInfo.TensorShapeTypeProto> 
      getDefaultGraphInputTensorShapeList();
  /**
   * <pre>
   * Optional: Default graph input tensor shape used to allocate memory
   * before executing op
   * </pre>
   *
   * <code>repeated .tensorflow.RemoteFusedGraphExecuteInfo.TensorShapeTypeProto default_graph_input_tensor_shape = 6;</code>
   */
  org.tensorflow.framework.RemoteFusedGraphExecuteInfo.TensorShapeTypeProto getDefaultGraphInputTensorShape(int index);
  /**
   * <pre>
   * Optional: Default graph input tensor shape used to allocate memory
   * before executing op
   * </pre>
   *
   * <code>repeated .tensorflow.RemoteFusedGraphExecuteInfo.TensorShapeTypeProto default_graph_input_tensor_shape = 6;</code>
   */
  int getDefaultGraphInputTensorShapeCount();
  /**
   * <pre>
   * Optional: Default graph input tensor shape used to allocate memory
   * before executing op
   * </pre>
   *
   * <code>repeated .tensorflow.RemoteFusedGraphExecuteInfo.TensorShapeTypeProto default_graph_input_tensor_shape = 6;</code>
   */
  java.util.List<? extends org.tensorflow.framework.RemoteFusedGraphExecuteInfo.TensorShapeTypeProtoOrBuilder> 
      getDefaultGraphInputTensorShapeOrBuilderList();
  /**
   * <pre>
   * Optional: Default graph input tensor shape used to allocate memory
   * before executing op
   * </pre>
   *
   * <code>repeated .tensorflow.RemoteFusedGraphExecuteInfo.TensorShapeTypeProto default_graph_input_tensor_shape = 6;</code>
   */
  org.tensorflow.framework.RemoteFusedGraphExecuteInfo.TensorShapeTypeProtoOrBuilder getDefaultGraphInputTensorShapeOrBuilder(
      int index);

  /**
   * <pre>
   * Optional: Default graph input tensor shape used to allocate memory
   * before executing op
   * TODO(satok): Remote output tensor shape once shape information is stored
   * in NodeDef
   * </pre>
   *
   * <code>repeated .tensorflow.RemoteFusedGraphExecuteInfo.TensorShapeTypeProto default_graph_output_tensor_shape = 7;</code>
   */
  java.util.List<org.tensorflow.framework.RemoteFusedGraphExecuteInfo.TensorShapeTypeProto> 
      getDefaultGraphOutputTensorShapeList();
  /**
   * <pre>
   * Optional: Default graph input tensor shape used to allocate memory
   * before executing op
   * TODO(satok): Remote output tensor shape once shape information is stored
   * in NodeDef
   * </pre>
   *
   * <code>repeated .tensorflow.RemoteFusedGraphExecuteInfo.TensorShapeTypeProto default_graph_output_tensor_shape = 7;</code>
   */
  org.tensorflow.framework.RemoteFusedGraphExecuteInfo.TensorShapeTypeProto getDefaultGraphOutputTensorShape(int index);
  /**
   * <pre>
   * Optional: Default graph input tensor shape used to allocate memory
   * before executing op
   * TODO(satok): Remote output tensor shape once shape information is stored
   * in NodeDef
   * </pre>
   *
   * <code>repeated .tensorflow.RemoteFusedGraphExecuteInfo.TensorShapeTypeProto default_graph_output_tensor_shape = 7;</code>
   */
  int getDefaultGraphOutputTensorShapeCount();
  /**
   * <pre>
   * Optional: Default graph input tensor shape used to allocate memory
   * before executing op
   * TODO(satok): Remote output tensor shape once shape information is stored
   * in NodeDef
   * </pre>
   *
   * <code>repeated .tensorflow.RemoteFusedGraphExecuteInfo.TensorShapeTypeProto default_graph_output_tensor_shape = 7;</code>
   */
  java.util.List<? extends org.tensorflow.framework.RemoteFusedGraphExecuteInfo.TensorShapeTypeProtoOrBuilder> 
      getDefaultGraphOutputTensorShapeOrBuilderList();
  /**
   * <pre>
   * Optional: Default graph input tensor shape used to allocate memory
   * before executing op
   * TODO(satok): Remote output tensor shape once shape information is stored
   * in NodeDef
   * </pre>
   *
   * <code>repeated .tensorflow.RemoteFusedGraphExecuteInfo.TensorShapeTypeProto default_graph_output_tensor_shape = 7;</code>
   */
  org.tensorflow.framework.RemoteFusedGraphExecuteInfo.TensorShapeTypeProtoOrBuilder getDefaultGraphOutputTensorShapeOrBuilder(
      int index);
}
