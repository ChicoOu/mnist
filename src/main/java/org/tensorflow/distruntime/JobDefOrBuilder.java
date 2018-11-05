// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: apis/core/protobuf/cluster.proto

package org.tensorflow.distruntime;

public interface JobDefOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.JobDef)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * The name of this job.
   * </pre>
   *
   * <code>optional string name = 1;</code>
   */
  java.lang.String getName();
  /**
   * <pre>
   * The name of this job.
   * </pre>
   *
   * <code>optional string name = 1;</code>
   */
  com.google.protobuf.ByteString
      getNameBytes();

  /**
   * <pre>
   * Mapping from task ID to "hostname:port" string.
   * If the `name` field contains "worker", and the `tasks` map contains a
   * mapping from 7 to "example.org:2222", then the device prefix
   * "/job:worker/task:7" will be assigned to "example.org:2222".
   * </pre>
   *
   * <code>map&lt;int32, string&gt; tasks = 2;</code>
   */
  int getTasksCount();
  /**
   * <pre>
   * Mapping from task ID to "hostname:port" string.
   * If the `name` field contains "worker", and the `tasks` map contains a
   * mapping from 7 to "example.org:2222", then the device prefix
   * "/job:worker/task:7" will be assigned to "example.org:2222".
   * </pre>
   *
   * <code>map&lt;int32, string&gt; tasks = 2;</code>
   */
  boolean containsTasks(
      int key);
  /**
   * Use {@link #getTasksMap()} instead.
   */
  @java.lang.Deprecated
  java.util.Map<java.lang.Integer, java.lang.String>
  getTasks();
  /**
   * <pre>
   * Mapping from task ID to "hostname:port" string.
   * If the `name` field contains "worker", and the `tasks` map contains a
   * mapping from 7 to "example.org:2222", then the device prefix
   * "/job:worker/task:7" will be assigned to "example.org:2222".
   * </pre>
   *
   * <code>map&lt;int32, string&gt; tasks = 2;</code>
   */
  java.util.Map<java.lang.Integer, java.lang.String>
  getTasksMap();
  /**
   * <pre>
   * Mapping from task ID to "hostname:port" string.
   * If the `name` field contains "worker", and the `tasks` map contains a
   * mapping from 7 to "example.org:2222", then the device prefix
   * "/job:worker/task:7" will be assigned to "example.org:2222".
   * </pre>
   *
   * <code>map&lt;int32, string&gt; tasks = 2;</code>
   */

  java.lang.String getTasksOrDefault(
      int key,
      java.lang.String defaultValue);
  /**
   * <pre>
   * Mapping from task ID to "hostname:port" string.
   * If the `name` field contains "worker", and the `tasks` map contains a
   * mapping from 7 to "example.org:2222", then the device prefix
   * "/job:worker/task:7" will be assigned to "example.org:2222".
   * </pre>
   *
   * <code>map&lt;int32, string&gt; tasks = 2;</code>
   */

  java.lang.String getTasksOrThrow(
      int key);
}
