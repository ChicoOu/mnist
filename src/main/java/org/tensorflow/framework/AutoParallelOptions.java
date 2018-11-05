// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: apis/core/protobuf/rewriter_config.proto

package org.tensorflow.framework;

/**
 * Protobuf type {@code tensorflow.AutoParallelOptions}
 */
public  final class AutoParallelOptions extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tensorflow.AutoParallelOptions)
    AutoParallelOptionsOrBuilder {
  // Use AutoParallelOptions.newBuilder() to construct.
  private AutoParallelOptions(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private AutoParallelOptions() {
    enable_ = false;
    numReplicas_ = 0;
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return com.google.protobuf.UnknownFieldSet.getDefaultInstance();
  }
  private AutoParallelOptions(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    this();
    int mutable_bitField0_ = 0;
    try {
      boolean done = false;
      while (!done) {
        int tag = input.readTag();
        switch (tag) {
          case 0:
            done = true;
            break;
          default: {
            if (!input.skipField(tag)) {
              done = true;
            }
            break;
          }
          case 8: {

            enable_ = input.readBool();
            break;
          }
          case 16: {

            numReplicas_ = input.readInt32();
            break;
          }
        }
      }
    } catch (com.google.protobuf.InvalidProtocolBufferException e) {
      throw e.setUnfinishedMessage(this);
    } catch (java.io.IOException e) {
      throw new com.google.protobuf.InvalidProtocolBufferException(
          e).setUnfinishedMessage(this);
    } finally {
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tensorflow.framework.RewriterConfigProtos.internal_static_tensorflow_AutoParallelOptions_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tensorflow.framework.RewriterConfigProtos.internal_static_tensorflow_AutoParallelOptions_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tensorflow.framework.AutoParallelOptions.class, org.tensorflow.framework.AutoParallelOptions.Builder.class);
  }

  public static final int ENABLE_FIELD_NUMBER = 1;
  private boolean enable_;
  /**
   * <code>optional bool enable = 1;</code>
   */
  public boolean getEnable() {
    return enable_;
  }

  public static final int NUM_REPLICAS_FIELD_NUMBER = 2;
  private int numReplicas_;
  /**
   * <code>optional int32 num_replicas = 2;</code>
   */
  public int getNumReplicas() {
    return numReplicas_;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (enable_ != false) {
      output.writeBool(1, enable_);
    }
    if (numReplicas_ != 0) {
      output.writeInt32(2, numReplicas_);
    }
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (enable_ != false) {
      size += com.google.protobuf.CodedOutputStream
        .computeBoolSize(1, enable_);
    }
    if (numReplicas_ != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(2, numReplicas_);
    }
    memoizedSize = size;
    return size;
  }

  private static final long serialVersionUID = 0L;
  @java.lang.Override
  public boolean equals(final java.lang.Object obj) {
    if (obj == this) {
     return true;
    }
    if (!(obj instanceof org.tensorflow.framework.AutoParallelOptions)) {
      return super.equals(obj);
    }
    org.tensorflow.framework.AutoParallelOptions other = (org.tensorflow.framework.AutoParallelOptions) obj;

    boolean result = true;
    result = result && (getEnable()
        == other.getEnable());
    result = result && (getNumReplicas()
        == other.getNumReplicas());
    return result;
  }

  @java.lang.Override
  public int hashCode() {
    if (memoizedHashCode != 0) {
      return memoizedHashCode;
    }
    int hash = 41;
    hash = (19 * hash) + getDescriptorForType().hashCode();
    hash = (37 * hash) + ENABLE_FIELD_NUMBER;
    hash = (53 * hash) + com.google.protobuf.Internal.hashBoolean(
        getEnable());
    hash = (37 * hash) + NUM_REPLICAS_FIELD_NUMBER;
    hash = (53 * hash) + getNumReplicas();
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tensorflow.framework.AutoParallelOptions parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tensorflow.framework.AutoParallelOptions parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tensorflow.framework.AutoParallelOptions parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tensorflow.framework.AutoParallelOptions parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tensorflow.framework.AutoParallelOptions parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tensorflow.framework.AutoParallelOptions parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tensorflow.framework.AutoParallelOptions parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.tensorflow.framework.AutoParallelOptions parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tensorflow.framework.AutoParallelOptions parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tensorflow.framework.AutoParallelOptions parseFrom(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  public Builder newBuilderForType() { return newBuilder(); }
  public static Builder newBuilder() {
    return DEFAULT_INSTANCE.toBuilder();
  }
  public static Builder newBuilder(org.tensorflow.framework.AutoParallelOptions prototype) {
    return DEFAULT_INSTANCE.toBuilder().mergeFrom(prototype);
  }
  public Builder toBuilder() {
    return this == DEFAULT_INSTANCE
        ? new Builder() : new Builder().mergeFrom(this);
  }

  @java.lang.Override
  protected Builder newBuilderForType(
      com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
    Builder builder = new Builder(parent);
    return builder;
  }
  /**
   * Protobuf type {@code tensorflow.AutoParallelOptions}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tensorflow.AutoParallelOptions)
      org.tensorflow.framework.AutoParallelOptionsOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tensorflow.framework.RewriterConfigProtos.internal_static_tensorflow_AutoParallelOptions_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tensorflow.framework.RewriterConfigProtos.internal_static_tensorflow_AutoParallelOptions_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tensorflow.framework.AutoParallelOptions.class, org.tensorflow.framework.AutoParallelOptions.Builder.class);
    }

    // Construct using org.tensorflow.framework.AutoParallelOptions.newBuilder()
    private Builder() {
      maybeForceBuilderInitialization();
    }

    private Builder(
        com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
      super(parent);
      maybeForceBuilderInitialization();
    }
    private void maybeForceBuilderInitialization() {
      if (com.google.protobuf.GeneratedMessageV3
              .alwaysUseFieldBuilders) {
      }
    }
    public Builder clear() {
      super.clear();
      enable_ = false;

      numReplicas_ = 0;

      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tensorflow.framework.RewriterConfigProtos.internal_static_tensorflow_AutoParallelOptions_descriptor;
    }

    public org.tensorflow.framework.AutoParallelOptions getDefaultInstanceForType() {
      return org.tensorflow.framework.AutoParallelOptions.getDefaultInstance();
    }

    public org.tensorflow.framework.AutoParallelOptions build() {
      org.tensorflow.framework.AutoParallelOptions result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.tensorflow.framework.AutoParallelOptions buildPartial() {
      org.tensorflow.framework.AutoParallelOptions result = new org.tensorflow.framework.AutoParallelOptions(this);
      result.enable_ = enable_;
      result.numReplicas_ = numReplicas_;
      onBuilt();
      return result;
    }

    public Builder clone() {
      return (Builder) super.clone();
    }
    public Builder setField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        Object value) {
      return (Builder) super.setField(field, value);
    }
    public Builder clearField(
        com.google.protobuf.Descriptors.FieldDescriptor field) {
      return (Builder) super.clearField(field);
    }
    public Builder clearOneof(
        com.google.protobuf.Descriptors.OneofDescriptor oneof) {
      return (Builder) super.clearOneof(oneof);
    }
    public Builder setRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        int index, Object value) {
      return (Builder) super.setRepeatedField(field, index, value);
    }
    public Builder addRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        Object value) {
      return (Builder) super.addRepeatedField(field, value);
    }
    public Builder mergeFrom(com.google.protobuf.Message other) {
      if (other instanceof org.tensorflow.framework.AutoParallelOptions) {
        return mergeFrom((org.tensorflow.framework.AutoParallelOptions)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tensorflow.framework.AutoParallelOptions other) {
      if (other == org.tensorflow.framework.AutoParallelOptions.getDefaultInstance()) return this;
      if (other.getEnable() != false) {
        setEnable(other.getEnable());
      }
      if (other.getNumReplicas() != 0) {
        setNumReplicas(other.getNumReplicas());
      }
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.tensorflow.framework.AutoParallelOptions parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.tensorflow.framework.AutoParallelOptions) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }

    private boolean enable_ ;
    /**
     * <code>optional bool enable = 1;</code>
     */
    public boolean getEnable() {
      return enable_;
    }
    /**
     * <code>optional bool enable = 1;</code>
     */
    public Builder setEnable(boolean value) {
      
      enable_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional bool enable = 1;</code>
     */
    public Builder clearEnable() {
      
      enable_ = false;
      onChanged();
      return this;
    }

    private int numReplicas_ ;
    /**
     * <code>optional int32 num_replicas = 2;</code>
     */
    public int getNumReplicas() {
      return numReplicas_;
    }
    /**
     * <code>optional int32 num_replicas = 2;</code>
     */
    public Builder setNumReplicas(int value) {
      
      numReplicas_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional int32 num_replicas = 2;</code>
     */
    public Builder clearNumReplicas() {
      
      numReplicas_ = 0;
      onChanged();
      return this;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return this;
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return this;
    }


    // @@protoc_insertion_point(builder_scope:tensorflow.AutoParallelOptions)
  }

  // @@protoc_insertion_point(class_scope:tensorflow.AutoParallelOptions)
  private static final org.tensorflow.framework.AutoParallelOptions DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tensorflow.framework.AutoParallelOptions();
  }

  public static org.tensorflow.framework.AutoParallelOptions getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<AutoParallelOptions>
      PARSER = new com.google.protobuf.AbstractParser<AutoParallelOptions>() {
    public AutoParallelOptions parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new AutoParallelOptions(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<AutoParallelOptions> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<AutoParallelOptions> getParserForType() {
    return PARSER;
  }

  public org.tensorflow.framework.AutoParallelOptions getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

