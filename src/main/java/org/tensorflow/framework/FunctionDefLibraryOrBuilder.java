// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: apis/core/framework/function.proto

package org.tensorflow.framework;

public interface FunctionDefLibraryOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.FunctionDefLibrary)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>repeated .tensorflow.FunctionDef function = 1;</code>
   */
  java.util.List<org.tensorflow.framework.FunctionDef> 
      getFunctionList();
  /**
   * <code>repeated .tensorflow.FunctionDef function = 1;</code>
   */
  org.tensorflow.framework.FunctionDef getFunction(int index);
  /**
   * <code>repeated .tensorflow.FunctionDef function = 1;</code>
   */
  int getFunctionCount();
  /**
   * <code>repeated .tensorflow.FunctionDef function = 1;</code>
   */
  java.util.List<? extends org.tensorflow.framework.FunctionDefOrBuilder> 
      getFunctionOrBuilderList();
  /**
   * <code>repeated .tensorflow.FunctionDef function = 1;</code>
   */
  org.tensorflow.framework.FunctionDefOrBuilder getFunctionOrBuilder(
      int index);

  /**
   * <code>repeated .tensorflow.GradientDef gradient = 2;</code>
   */
  java.util.List<org.tensorflow.framework.GradientDef> 
      getGradientList();
  /**
   * <code>repeated .tensorflow.GradientDef gradient = 2;</code>
   */
  org.tensorflow.framework.GradientDef getGradient(int index);
  /**
   * <code>repeated .tensorflow.GradientDef gradient = 2;</code>
   */
  int getGradientCount();
  /**
   * <code>repeated .tensorflow.GradientDef gradient = 2;</code>
   */
  java.util.List<? extends org.tensorflow.framework.GradientDefOrBuilder> 
      getGradientOrBuilderList();
  /**
   * <code>repeated .tensorflow.GradientDef gradient = 2;</code>
   */
  org.tensorflow.framework.GradientDefOrBuilder getGradientOrBuilder(
      int index);
}
