// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: apis/example/feature.proto

package org.tensorflow.example;

public interface FeaturesOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.Features)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Map from feature name to feature.
   * </pre>
   *
   * <code>map&lt;string, .tensorflow.Feature&gt; feature = 1;</code>
   */
  int getFeatureCount();
  /**
   * <pre>
   * Map from feature name to feature.
   * </pre>
   *
   * <code>map&lt;string, .tensorflow.Feature&gt; feature = 1;</code>
   */
  boolean containsFeature(
      java.lang.String key);
  /**
   * Use {@link #getFeatureMap()} instead.
   */
  @java.lang.Deprecated
  java.util.Map<java.lang.String, org.tensorflow.example.Feature>
  getFeature();
  /**
   * <pre>
   * Map from feature name to feature.
   * </pre>
   *
   * <code>map&lt;string, .tensorflow.Feature&gt; feature = 1;</code>
   */
  java.util.Map<java.lang.String, org.tensorflow.example.Feature>
  getFeatureMap();
  /**
   * <pre>
   * Map from feature name to feature.
   * </pre>
   *
   * <code>map&lt;string, .tensorflow.Feature&gt; feature = 1;</code>
   */

  org.tensorflow.example.Feature getFeatureOrDefault(
      java.lang.String key,
      org.tensorflow.example.Feature defaultValue);
  /**
   * <pre>
   * Map from feature name to feature.
   * </pre>
   *
   * <code>map&lt;string, .tensorflow.Feature&gt; feature = 1;</code>
   */

  org.tensorflow.example.Feature getFeatureOrThrow(
      java.lang.String key);
}
