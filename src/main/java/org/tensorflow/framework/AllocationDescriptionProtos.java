// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: apis/core/framework/allocation_description.proto

package org.tensorflow.framework;

public final class AllocationDescriptionProtos {
  private AllocationDescriptionProtos() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tensorflow_AllocationDescription_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tensorflow_AllocationDescription_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n0apis/core/framework/allocation_descrip" +
      "tion.proto\022\ntensorflow\"\243\001\n\025AllocationDes" +
      "cription\022\027\n\017requested_bytes\030\001 \001(\003\022\027\n\017all" +
      "ocated_bytes\030\002 \001(\003\022\026\n\016allocator_name\030\003 \001" +
      "(\t\022\025\n\rallocation_id\030\004 \001(\003\022\034\n\024has_single_" +
      "reference\030\005 \001(\010\022\013\n\003ptr\030\006 \001(\004B{\n\030org.tens" +
      "orflow.frameworkB\033AllocationDescriptionP" +
      "rotosP\001Z=github.com/tensorflow/tensorflo" +
      "w/tensorflow/go/core/framework\370\001\001b\006proto" +
      "3"
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
    internal_static_tensorflow_AllocationDescription_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_tensorflow_AllocationDescription_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tensorflow_AllocationDescription_descriptor,
        new java.lang.String[] { "RequestedBytes", "AllocatedBytes", "AllocatorName", "AllocationId", "HasSingleReference", "Ptr", });
  }

  // @@protoc_insertion_point(outer_class_scope)
}
