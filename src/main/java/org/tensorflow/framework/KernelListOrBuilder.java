// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: apis/core/framework/kernel_def.proto

package org.tensorflow.framework;

public interface KernelListOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.KernelList)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>repeated .tensorflow.KernelDef kernel = 1;</code>
   */
  java.util.List<org.tensorflow.framework.KernelDef> 
      getKernelList();
  /**
   * <code>repeated .tensorflow.KernelDef kernel = 1;</code>
   */
  org.tensorflow.framework.KernelDef getKernel(int index);
  /**
   * <code>repeated .tensorflow.KernelDef kernel = 1;</code>
   */
  int getKernelCount();
  /**
   * <code>repeated .tensorflow.KernelDef kernel = 1;</code>
   */
  java.util.List<? extends org.tensorflow.framework.KernelDefOrBuilder> 
      getKernelOrBuilderList();
  /**
   * <code>repeated .tensorflow.KernelDef kernel = 1;</code>
   */
  org.tensorflow.framework.KernelDefOrBuilder getKernelOrBuilder(
      int index);
}
