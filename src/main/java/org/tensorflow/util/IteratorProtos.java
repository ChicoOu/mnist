// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: apis/core/framework/iterator.proto

package org.tensorflow.util;

public final class IteratorProtos {
  private IteratorProtos() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tensorflow_IteratorStateMetadata_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tensorflow_IteratorStateMetadata_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\"apis/core/framework/iterator.proto\022\nte" +
      "nsorflow\"6\n\025IteratorStateMetadata\022\017\n\007ver" +
      "sion\030\001 \001(\t\022\014\n\004keys\030\002 \003(\tBi\n\023org.tensorfl" +
      "ow.utilB\016IteratorProtosP\001Z=github.com/te" +
      "nsorflow/tensorflow/tensorflow/go/core/f" +
      "ramework\370\001\001b\006proto3"
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
        }, assigner);
    internal_static_tensorflow_IteratorStateMetadata_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_tensorflow_IteratorStateMetadata_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tensorflow_IteratorStateMetadata_descriptor,
        new java.lang.String[] { "Version", "Keys", });
  }

  // @@protoc_insertion_point(outer_class_scope)
}