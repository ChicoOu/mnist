// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: apis/core/framework/function.proto

package org.tensorflow.framework;

public interface FunctionDefOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.FunctionDef)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * The definition of the function's name, arguments, return values,
   * attrs etc.
   * </pre>
   *
   * <code>optional .tensorflow.OpDef signature = 1;</code>
   */
  boolean hasSignature();
  /**
   * <pre>
   * The definition of the function's name, arguments, return values,
   * attrs etc.
   * </pre>
   *
   * <code>optional .tensorflow.OpDef signature = 1;</code>
   */
  org.tensorflow.framework.OpDef getSignature();
  /**
   * <pre>
   * The definition of the function's name, arguments, return values,
   * attrs etc.
   * </pre>
   *
   * <code>optional .tensorflow.OpDef signature = 1;</code>
   */
  org.tensorflow.framework.OpDefOrBuilder getSignatureOrBuilder();

  /**
   * <pre>
   * Attributes specific to this function definition.
   * </pre>
   *
   * <code>map&lt;string, .tensorflow.AttrValue&gt; attr = 5;</code>
   */
  int getAttrCount();
  /**
   * <pre>
   * Attributes specific to this function definition.
   * </pre>
   *
   * <code>map&lt;string, .tensorflow.AttrValue&gt; attr = 5;</code>
   */
  boolean containsAttr(
      java.lang.String key);
  /**
   * Use {@link #getAttrMap()} instead.
   */
  @java.lang.Deprecated
  java.util.Map<java.lang.String, org.tensorflow.framework.AttrValue>
  getAttr();
  /**
   * <pre>
   * Attributes specific to this function definition.
   * </pre>
   *
   * <code>map&lt;string, .tensorflow.AttrValue&gt; attr = 5;</code>
   */
  java.util.Map<java.lang.String, org.tensorflow.framework.AttrValue>
  getAttrMap();
  /**
   * <pre>
   * Attributes specific to this function definition.
   * </pre>
   *
   * <code>map&lt;string, .tensorflow.AttrValue&gt; attr = 5;</code>
   */

  org.tensorflow.framework.AttrValue getAttrOrDefault(
      java.lang.String key,
      org.tensorflow.framework.AttrValue defaultValue);
  /**
   * <pre>
   * Attributes specific to this function definition.
   * </pre>
   *
   * <code>map&lt;string, .tensorflow.AttrValue&gt; attr = 5;</code>
   */

  org.tensorflow.framework.AttrValue getAttrOrThrow(
      java.lang.String key);

  /**
   * <pre>
   * By convention, "op" in node_def is resolved by consulting with a
   * user-defined library first. If not resolved, "func" is assumed to
   * be a builtin op.
   * </pre>
   *
   * <code>repeated .tensorflow.NodeDef node_def = 3;</code>
   */
  java.util.List<org.tensorflow.framework.NodeDef> 
      getNodeDefList();
  /**
   * <pre>
   * By convention, "op" in node_def is resolved by consulting with a
   * user-defined library first. If not resolved, "func" is assumed to
   * be a builtin op.
   * </pre>
   *
   * <code>repeated .tensorflow.NodeDef node_def = 3;</code>
   */
  org.tensorflow.framework.NodeDef getNodeDef(int index);
  /**
   * <pre>
   * By convention, "op" in node_def is resolved by consulting with a
   * user-defined library first. If not resolved, "func" is assumed to
   * be a builtin op.
   * </pre>
   *
   * <code>repeated .tensorflow.NodeDef node_def = 3;</code>
   */
  int getNodeDefCount();
  /**
   * <pre>
   * By convention, "op" in node_def is resolved by consulting with a
   * user-defined library first. If not resolved, "func" is assumed to
   * be a builtin op.
   * </pre>
   *
   * <code>repeated .tensorflow.NodeDef node_def = 3;</code>
   */
  java.util.List<? extends org.tensorflow.framework.NodeDefOrBuilder> 
      getNodeDefOrBuilderList();
  /**
   * <pre>
   * By convention, "op" in node_def is resolved by consulting with a
   * user-defined library first. If not resolved, "func" is assumed to
   * be a builtin op.
   * </pre>
   *
   * <code>repeated .tensorflow.NodeDef node_def = 3;</code>
   */
  org.tensorflow.framework.NodeDefOrBuilder getNodeDefOrBuilder(
      int index);

  /**
   * <pre>
   * A mapping from the output arg names from `signature` to the
   * outputs from `node_def` that should be returned by the function.
   * </pre>
   *
   * <code>map&lt;string, string&gt; ret = 4;</code>
   */
  int getRetCount();
  /**
   * <pre>
   * A mapping from the output arg names from `signature` to the
   * outputs from `node_def` that should be returned by the function.
   * </pre>
   *
   * <code>map&lt;string, string&gt; ret = 4;</code>
   */
  boolean containsRet(
      java.lang.String key);
  /**
   * Use {@link #getRetMap()} instead.
   */
  @java.lang.Deprecated
  java.util.Map<java.lang.String, java.lang.String>
  getRet();
  /**
   * <pre>
   * A mapping from the output arg names from `signature` to the
   * outputs from `node_def` that should be returned by the function.
   * </pre>
   *
   * <code>map&lt;string, string&gt; ret = 4;</code>
   */
  java.util.Map<java.lang.String, java.lang.String>
  getRetMap();
  /**
   * <pre>
   * A mapping from the output arg names from `signature` to the
   * outputs from `node_def` that should be returned by the function.
   * </pre>
   *
   * <code>map&lt;string, string&gt; ret = 4;</code>
   */

  java.lang.String getRetOrDefault(
      java.lang.String key,
      java.lang.String defaultValue);
  /**
   * <pre>
   * A mapping from the output arg names from `signature` to the
   * outputs from `node_def` that should be returned by the function.
   * </pre>
   *
   * <code>map&lt;string, string&gt; ret = 4;</code>
   */

  java.lang.String getRetOrThrow(
      java.lang.String key);
}
