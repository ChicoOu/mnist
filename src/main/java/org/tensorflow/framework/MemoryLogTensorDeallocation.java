// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: apis/core/framework/log_memory.proto

package org.tensorflow.framework;

/**
 * Protobuf type {@code tensorflow.MemoryLogTensorDeallocation}
 */
public  final class MemoryLogTensorDeallocation extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tensorflow.MemoryLogTensorDeallocation)
    MemoryLogTensorDeallocationOrBuilder {
  // Use MemoryLogTensorDeallocation.newBuilder() to construct.
  private MemoryLogTensorDeallocation(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private MemoryLogTensorDeallocation() {
    allocationId_ = 0L;
    allocatorName_ = "";
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return com.google.protobuf.UnknownFieldSet.getDefaultInstance();
  }
  private MemoryLogTensorDeallocation(
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

            allocationId_ = input.readInt64();
            break;
          }
          case 18: {
            java.lang.String s = input.readStringRequireUtf8();

            allocatorName_ = s;
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
    return org.tensorflow.framework.LogMemoryProtos.internal_static_tensorflow_MemoryLogTensorDeallocation_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tensorflow.framework.LogMemoryProtos.internal_static_tensorflow_MemoryLogTensorDeallocation_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tensorflow.framework.MemoryLogTensorDeallocation.class, org.tensorflow.framework.MemoryLogTensorDeallocation.Builder.class);
  }

  public static final int ALLOCATION_ID_FIELD_NUMBER = 1;
  private long allocationId_;
  /**
   * <pre>
   * Id of the tensor buffer being deallocated, used to match to a
   * corresponding allocation.
   * </pre>
   *
   * <code>optional int64 allocation_id = 1;</code>
   */
  public long getAllocationId() {
    return allocationId_;
  }

  public static final int ALLOCATOR_NAME_FIELD_NUMBER = 2;
  private volatile java.lang.Object allocatorName_;
  /**
   * <pre>
   * Name of the allocator used.
   * </pre>
   *
   * <code>optional string allocator_name = 2;</code>
   */
  public java.lang.String getAllocatorName() {
    java.lang.Object ref = allocatorName_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      allocatorName_ = s;
      return s;
    }
  }
  /**
   * <pre>
   * Name of the allocator used.
   * </pre>
   *
   * <code>optional string allocator_name = 2;</code>
   */
  public com.google.protobuf.ByteString
      getAllocatorNameBytes() {
    java.lang.Object ref = allocatorName_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      allocatorName_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
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
    if (allocationId_ != 0L) {
      output.writeInt64(1, allocationId_);
    }
    if (!getAllocatorNameBytes().isEmpty()) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 2, allocatorName_);
    }
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (allocationId_ != 0L) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt64Size(1, allocationId_);
    }
    if (!getAllocatorNameBytes().isEmpty()) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(2, allocatorName_);
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
    if (!(obj instanceof org.tensorflow.framework.MemoryLogTensorDeallocation)) {
      return super.equals(obj);
    }
    org.tensorflow.framework.MemoryLogTensorDeallocation other = (org.tensorflow.framework.MemoryLogTensorDeallocation) obj;

    boolean result = true;
    result = result && (getAllocationId()
        == other.getAllocationId());
    result = result && getAllocatorName()
        .equals(other.getAllocatorName());
    return result;
  }

  @java.lang.Override
  public int hashCode() {
    if (memoizedHashCode != 0) {
      return memoizedHashCode;
    }
    int hash = 41;
    hash = (19 * hash) + getDescriptorForType().hashCode();
    hash = (37 * hash) + ALLOCATION_ID_FIELD_NUMBER;
    hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
        getAllocationId());
    hash = (37 * hash) + ALLOCATOR_NAME_FIELD_NUMBER;
    hash = (53 * hash) + getAllocatorName().hashCode();
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tensorflow.framework.MemoryLogTensorDeallocation parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tensorflow.framework.MemoryLogTensorDeallocation parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tensorflow.framework.MemoryLogTensorDeallocation parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tensorflow.framework.MemoryLogTensorDeallocation parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tensorflow.framework.MemoryLogTensorDeallocation parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tensorflow.framework.MemoryLogTensorDeallocation parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tensorflow.framework.MemoryLogTensorDeallocation parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.tensorflow.framework.MemoryLogTensorDeallocation parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tensorflow.framework.MemoryLogTensorDeallocation parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tensorflow.framework.MemoryLogTensorDeallocation parseFrom(
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
  public static Builder newBuilder(org.tensorflow.framework.MemoryLogTensorDeallocation prototype) {
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
   * Protobuf type {@code tensorflow.MemoryLogTensorDeallocation}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tensorflow.MemoryLogTensorDeallocation)
      org.tensorflow.framework.MemoryLogTensorDeallocationOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tensorflow.framework.LogMemoryProtos.internal_static_tensorflow_MemoryLogTensorDeallocation_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tensorflow.framework.LogMemoryProtos.internal_static_tensorflow_MemoryLogTensorDeallocation_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tensorflow.framework.MemoryLogTensorDeallocation.class, org.tensorflow.framework.MemoryLogTensorDeallocation.Builder.class);
    }

    // Construct using org.tensorflow.framework.MemoryLogTensorDeallocation.newBuilder()
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
      allocationId_ = 0L;

      allocatorName_ = "";

      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tensorflow.framework.LogMemoryProtos.internal_static_tensorflow_MemoryLogTensorDeallocation_descriptor;
    }

    public org.tensorflow.framework.MemoryLogTensorDeallocation getDefaultInstanceForType() {
      return org.tensorflow.framework.MemoryLogTensorDeallocation.getDefaultInstance();
    }

    public org.tensorflow.framework.MemoryLogTensorDeallocation build() {
      org.tensorflow.framework.MemoryLogTensorDeallocation result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.tensorflow.framework.MemoryLogTensorDeallocation buildPartial() {
      org.tensorflow.framework.MemoryLogTensorDeallocation result = new org.tensorflow.framework.MemoryLogTensorDeallocation(this);
      result.allocationId_ = allocationId_;
      result.allocatorName_ = allocatorName_;
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
      if (other instanceof org.tensorflow.framework.MemoryLogTensorDeallocation) {
        return mergeFrom((org.tensorflow.framework.MemoryLogTensorDeallocation)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tensorflow.framework.MemoryLogTensorDeallocation other) {
      if (other == org.tensorflow.framework.MemoryLogTensorDeallocation.getDefaultInstance()) return this;
      if (other.getAllocationId() != 0L) {
        setAllocationId(other.getAllocationId());
      }
      if (!other.getAllocatorName().isEmpty()) {
        allocatorName_ = other.allocatorName_;
        onChanged();
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
      org.tensorflow.framework.MemoryLogTensorDeallocation parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.tensorflow.framework.MemoryLogTensorDeallocation) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }

    private long allocationId_ ;
    /**
     * <pre>
     * Id of the tensor buffer being deallocated, used to match to a
     * corresponding allocation.
     * </pre>
     *
     * <code>optional int64 allocation_id = 1;</code>
     */
    public long getAllocationId() {
      return allocationId_;
    }
    /**
     * <pre>
     * Id of the tensor buffer being deallocated, used to match to a
     * corresponding allocation.
     * </pre>
     *
     * <code>optional int64 allocation_id = 1;</code>
     */
    public Builder setAllocationId(long value) {
      
      allocationId_ = value;
      onChanged();
      return this;
    }
    /**
     * <pre>
     * Id of the tensor buffer being deallocated, used to match to a
     * corresponding allocation.
     * </pre>
     *
     * <code>optional int64 allocation_id = 1;</code>
     */
    public Builder clearAllocationId() {
      
      allocationId_ = 0L;
      onChanged();
      return this;
    }

    private java.lang.Object allocatorName_ = "";
    /**
     * <pre>
     * Name of the allocator used.
     * </pre>
     *
     * <code>optional string allocator_name = 2;</code>
     */
    public java.lang.String getAllocatorName() {
      java.lang.Object ref = allocatorName_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        allocatorName_ = s;
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <pre>
     * Name of the allocator used.
     * </pre>
     *
     * <code>optional string allocator_name = 2;</code>
     */
    public com.google.protobuf.ByteString
        getAllocatorNameBytes() {
      java.lang.Object ref = allocatorName_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        allocatorName_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <pre>
     * Name of the allocator used.
     * </pre>
     *
     * <code>optional string allocator_name = 2;</code>
     */
    public Builder setAllocatorName(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  
      allocatorName_ = value;
      onChanged();
      return this;
    }
    /**
     * <pre>
     * Name of the allocator used.
     * </pre>
     *
     * <code>optional string allocator_name = 2;</code>
     */
    public Builder clearAllocatorName() {
      
      allocatorName_ = getDefaultInstance().getAllocatorName();
      onChanged();
      return this;
    }
    /**
     * <pre>
     * Name of the allocator used.
     * </pre>
     *
     * <code>optional string allocator_name = 2;</code>
     */
    public Builder setAllocatorNameBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  checkByteStringIsUtf8(value);
      
      allocatorName_ = value;
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


    // @@protoc_insertion_point(builder_scope:tensorflow.MemoryLogTensorDeallocation)
  }

  // @@protoc_insertion_point(class_scope:tensorflow.MemoryLogTensorDeallocation)
  private static final org.tensorflow.framework.MemoryLogTensorDeallocation DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tensorflow.framework.MemoryLogTensorDeallocation();
  }

  public static org.tensorflow.framework.MemoryLogTensorDeallocation getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<MemoryLogTensorDeallocation>
      PARSER = new com.google.protobuf.AbstractParser<MemoryLogTensorDeallocation>() {
    public MemoryLogTensorDeallocation parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new MemoryLogTensorDeallocation(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<MemoryLogTensorDeallocation> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<MemoryLogTensorDeallocation> getParserForType() {
    return PARSER;
  }

  public org.tensorflow.framework.MemoryLogTensorDeallocation getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

