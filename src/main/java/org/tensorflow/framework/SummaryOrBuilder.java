// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: apis/core/framework/summary.proto

package org.tensorflow.framework;

public interface SummaryOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.Summary)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Set of values for the summary.
   * </pre>
   *
   * <code>repeated .tensorflow.Summary.Value value = 1;</code>
   */
  java.util.List<org.tensorflow.framework.Summary.Value> 
      getValueList();
  /**
   * <pre>
   * Set of values for the summary.
   * </pre>
   *
   * <code>repeated .tensorflow.Summary.Value value = 1;</code>
   */
  org.tensorflow.framework.Summary.Value getValue(int index);
  /**
   * <pre>
   * Set of values for the summary.
   * </pre>
   *
   * <code>repeated .tensorflow.Summary.Value value = 1;</code>
   */
  int getValueCount();
  /**
   * <pre>
   * Set of values for the summary.
   * </pre>
   *
   * <code>repeated .tensorflow.Summary.Value value = 1;</code>
   */
  java.util.List<? extends org.tensorflow.framework.Summary.ValueOrBuilder> 
      getValueOrBuilderList();
  /**
   * <pre>
   * Set of values for the summary.
   * </pre>
   *
   * <code>repeated .tensorflow.Summary.Value value = 1;</code>
   */
  org.tensorflow.framework.Summary.ValueOrBuilder getValueOrBuilder(
      int index);
}
