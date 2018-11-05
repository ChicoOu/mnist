// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: apis/core/framework/allocation_description.proto

package org.tensorflow.framework;

public interface AllocationDescriptionOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.AllocationDescription)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Total number of bytes requested
   * </pre>
   *
   * <code>optional int64 requested_bytes = 1;</code>
   */
  long getRequestedBytes();

  /**
   * <pre>
   * Total number of bytes allocated if known
   * </pre>
   *
   * <code>optional int64 allocated_bytes = 2;</code>
   */
  long getAllocatedBytes();

  /**
   * <pre>
   * Name of the allocator used
   * </pre>
   *
   * <code>optional string allocator_name = 3;</code>
   */
  java.lang.String getAllocatorName();
  /**
   * <pre>
   * Name of the allocator used
   * </pre>
   *
   * <code>optional string allocator_name = 3;</code>
   */
  com.google.protobuf.ByteString
      getAllocatorNameBytes();

  /**
   * <pre>
   * Identifier of the allocated buffer if known
   * </pre>
   *
   * <code>optional int64 allocation_id = 4;</code>
   */
  long getAllocationId();

  /**
   * <pre>
   * Set if this tensor only has one remaining reference
   * </pre>
   *
   * <code>optional bool has_single_reference = 5;</code>
   */
  boolean getHasSingleReference();

  /**
   * <pre>
   * Address of the allocation.
   * </pre>
   *
   * <code>optional uint64 ptr = 6;</code>
   */
  long getPtr();
}
