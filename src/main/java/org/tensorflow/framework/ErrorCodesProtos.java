// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: apis/core/lib/core/error_codes.proto

package org.tensorflow.framework;

public final class ErrorCodesProtos {
  private ErrorCodesProtos() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n$apis/core/lib/core/error_codes.proto\022\020" +
      "tensorflow.error*\204\003\n\004Code\022\006\n\002OK\020\000\022\r\n\tCAN" +
      "CELLED\020\001\022\013\n\007UNKNOWN\020\002\022\024\n\020INVALID_ARGUMEN" +
      "T\020\003\022\025\n\021DEADLINE_EXCEEDED\020\004\022\r\n\tNOT_FOUND\020" +
      "\005\022\022\n\016ALREADY_EXISTS\020\006\022\025\n\021PERMISSION_DENI" +
      "ED\020\007\022\023\n\017UNAUTHENTICATED\020\020\022\026\n\022RESOURCE_EX" +
      "HAUSTED\020\010\022\027\n\023FAILED_PRECONDITION\020\t\022\013\n\007AB" +
      "ORTED\020\n\022\020\n\014OUT_OF_RANGE\020\013\022\021\n\rUNIMPLEMENT" +
      "ED\020\014\022\014\n\010INTERNAL\020\r\022\017\n\013UNAVAILABLE\020\016\022\r\n\tD" +
      "ATA_LOSS\020\017\022K\nGDO_NOT_USE_RESERVED_FOR_FU",
      "TURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INS" +
      "TEAD_\020\024Bo\n\030org.tensorflow.frameworkB\020Err" +
      "orCodesProtosP\001Z<github.com/tensorflow/t" +
      "ensorflow/tensorflow/go/core/lib/core\370\001\001" +
      "b\006proto3"
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
  }

  // @@protoc_insertion_point(outer_class_scope)
}
