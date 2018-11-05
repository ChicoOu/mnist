// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: apis/core/framework/device_attributes.proto

package org.tensorflow.framework;

/**
 * Protobuf type {@code tensorflow.DeviceAttributes}
 */
public  final class DeviceAttributes extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tensorflow.DeviceAttributes)
    DeviceAttributesOrBuilder {
  // Use DeviceAttributes.newBuilder() to construct.
  private DeviceAttributes(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private DeviceAttributes() {
    name_ = "";
    deviceType_ = "";
    memoryLimit_ = 0L;
    incarnation_ = 0L;
    physicalDeviceDesc_ = "";
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return com.google.protobuf.UnknownFieldSet.getDefaultInstance();
  }
  private DeviceAttributes(
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
          case 10: {
            java.lang.String s = input.readStringRequireUtf8();

            name_ = s;
            break;
          }
          case 18: {
            java.lang.String s = input.readStringRequireUtf8();

            deviceType_ = s;
            break;
          }
          case 32: {

            memoryLimit_ = input.readInt64();
            break;
          }
          case 42: {
            org.tensorflow.framework.DeviceLocality.Builder subBuilder = null;
            if (locality_ != null) {
              subBuilder = locality_.toBuilder();
            }
            locality_ = input.readMessage(org.tensorflow.framework.DeviceLocality.parser(), extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom(locality_);
              locality_ = subBuilder.buildPartial();
            }

            break;
          }
          case 49: {

            incarnation_ = input.readFixed64();
            break;
          }
          case 58: {
            java.lang.String s = input.readStringRequireUtf8();

            physicalDeviceDesc_ = s;
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
    return org.tensorflow.framework.DeviceAttributesProtos.internal_static_tensorflow_DeviceAttributes_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tensorflow.framework.DeviceAttributesProtos.internal_static_tensorflow_DeviceAttributes_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tensorflow.framework.DeviceAttributes.class, org.tensorflow.framework.DeviceAttributes.Builder.class);
  }

  public static final int NAME_FIELD_NUMBER = 1;
  private volatile java.lang.Object name_;
  /**
   * <pre>
   * Fully specified name of the device within a cluster.
   * </pre>
   *
   * <code>optional string name = 1;</code>
   */
  public java.lang.String getName() {
    java.lang.Object ref = name_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      name_ = s;
      return s;
    }
  }
  /**
   * <pre>
   * Fully specified name of the device within a cluster.
   * </pre>
   *
   * <code>optional string name = 1;</code>
   */
  public com.google.protobuf.ByteString
      getNameBytes() {
    java.lang.Object ref = name_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      name_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int DEVICE_TYPE_FIELD_NUMBER = 2;
  private volatile java.lang.Object deviceType_;
  /**
   * <pre>
   * String representation of device_type.
   * </pre>
   *
   * <code>optional string device_type = 2;</code>
   */
  public java.lang.String getDeviceType() {
    java.lang.Object ref = deviceType_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      deviceType_ = s;
      return s;
    }
  }
  /**
   * <pre>
   * String representation of device_type.
   * </pre>
   *
   * <code>optional string device_type = 2;</code>
   */
  public com.google.protobuf.ByteString
      getDeviceTypeBytes() {
    java.lang.Object ref = deviceType_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      deviceType_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int MEMORY_LIMIT_FIELD_NUMBER = 4;
  private long memoryLimit_;
  /**
   * <pre>
   * Memory capacity of device in bytes.
   * </pre>
   *
   * <code>optional int64 memory_limit = 4;</code>
   */
  public long getMemoryLimit() {
    return memoryLimit_;
  }

  public static final int LOCALITY_FIELD_NUMBER = 5;
  private org.tensorflow.framework.DeviceLocality locality_;
  /**
   * <pre>
   * Platform-specific data about device that may be useful
   * for supporting efficient data transfers.
   * </pre>
   *
   * <code>optional .tensorflow.DeviceLocality locality = 5;</code>
   */
  public boolean hasLocality() {
    return locality_ != null;
  }
  /**
   * <pre>
   * Platform-specific data about device that may be useful
   * for supporting efficient data transfers.
   * </pre>
   *
   * <code>optional .tensorflow.DeviceLocality locality = 5;</code>
   */
  public org.tensorflow.framework.DeviceLocality getLocality() {
    return locality_ == null ? org.tensorflow.framework.DeviceLocality.getDefaultInstance() : locality_;
  }
  /**
   * <pre>
   * Platform-specific data about device that may be useful
   * for supporting efficient data transfers.
   * </pre>
   *
   * <code>optional .tensorflow.DeviceLocality locality = 5;</code>
   */
  public org.tensorflow.framework.DeviceLocalityOrBuilder getLocalityOrBuilder() {
    return getLocality();
  }

  public static final int INCARNATION_FIELD_NUMBER = 6;
  private long incarnation_;
  /**
   * <pre>
   * A device is assigned a global unique number each time it is
   * initialized. "incarnation" should never be 0.
   * </pre>
   *
   * <code>optional fixed64 incarnation = 6;</code>
   */
  public long getIncarnation() {
    return incarnation_;
  }

  public static final int PHYSICAL_DEVICE_DESC_FIELD_NUMBER = 7;
  private volatile java.lang.Object physicalDeviceDesc_;
  /**
   * <pre>
   * String representation of the physical device that this device maps to.
   * </pre>
   *
   * <code>optional string physical_device_desc = 7;</code>
   */
  public java.lang.String getPhysicalDeviceDesc() {
    java.lang.Object ref = physicalDeviceDesc_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      physicalDeviceDesc_ = s;
      return s;
    }
  }
  /**
   * <pre>
   * String representation of the physical device that this device maps to.
   * </pre>
   *
   * <code>optional string physical_device_desc = 7;</code>
   */
  public com.google.protobuf.ByteString
      getPhysicalDeviceDescBytes() {
    java.lang.Object ref = physicalDeviceDesc_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      physicalDeviceDesc_ = b;
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
    if (!getNameBytes().isEmpty()) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 1, name_);
    }
    if (!getDeviceTypeBytes().isEmpty()) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 2, deviceType_);
    }
    if (memoryLimit_ != 0L) {
      output.writeInt64(4, memoryLimit_);
    }
    if (locality_ != null) {
      output.writeMessage(5, getLocality());
    }
    if (incarnation_ != 0L) {
      output.writeFixed64(6, incarnation_);
    }
    if (!getPhysicalDeviceDescBytes().isEmpty()) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 7, physicalDeviceDesc_);
    }
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (!getNameBytes().isEmpty()) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(1, name_);
    }
    if (!getDeviceTypeBytes().isEmpty()) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(2, deviceType_);
    }
    if (memoryLimit_ != 0L) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt64Size(4, memoryLimit_);
    }
    if (locality_ != null) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(5, getLocality());
    }
    if (incarnation_ != 0L) {
      size += com.google.protobuf.CodedOutputStream
        .computeFixed64Size(6, incarnation_);
    }
    if (!getPhysicalDeviceDescBytes().isEmpty()) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(7, physicalDeviceDesc_);
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
    if (!(obj instanceof org.tensorflow.framework.DeviceAttributes)) {
      return super.equals(obj);
    }
    org.tensorflow.framework.DeviceAttributes other = (org.tensorflow.framework.DeviceAttributes) obj;

    boolean result = true;
    result = result && getName()
        .equals(other.getName());
    result = result && getDeviceType()
        .equals(other.getDeviceType());
    result = result && (getMemoryLimit()
        == other.getMemoryLimit());
    result = result && (hasLocality() == other.hasLocality());
    if (hasLocality()) {
      result = result && getLocality()
          .equals(other.getLocality());
    }
    result = result && (getIncarnation()
        == other.getIncarnation());
    result = result && getPhysicalDeviceDesc()
        .equals(other.getPhysicalDeviceDesc());
    return result;
  }

  @java.lang.Override
  public int hashCode() {
    if (memoizedHashCode != 0) {
      return memoizedHashCode;
    }
    int hash = 41;
    hash = (19 * hash) + getDescriptorForType().hashCode();
    hash = (37 * hash) + NAME_FIELD_NUMBER;
    hash = (53 * hash) + getName().hashCode();
    hash = (37 * hash) + DEVICE_TYPE_FIELD_NUMBER;
    hash = (53 * hash) + getDeviceType().hashCode();
    hash = (37 * hash) + MEMORY_LIMIT_FIELD_NUMBER;
    hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
        getMemoryLimit());
    if (hasLocality()) {
      hash = (37 * hash) + LOCALITY_FIELD_NUMBER;
      hash = (53 * hash) + getLocality().hashCode();
    }
    hash = (37 * hash) + INCARNATION_FIELD_NUMBER;
    hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
        getIncarnation());
    hash = (37 * hash) + PHYSICAL_DEVICE_DESC_FIELD_NUMBER;
    hash = (53 * hash) + getPhysicalDeviceDesc().hashCode();
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tensorflow.framework.DeviceAttributes parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tensorflow.framework.DeviceAttributes parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tensorflow.framework.DeviceAttributes parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tensorflow.framework.DeviceAttributes parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tensorflow.framework.DeviceAttributes parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tensorflow.framework.DeviceAttributes parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tensorflow.framework.DeviceAttributes parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.tensorflow.framework.DeviceAttributes parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tensorflow.framework.DeviceAttributes parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tensorflow.framework.DeviceAttributes parseFrom(
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
  public static Builder newBuilder(org.tensorflow.framework.DeviceAttributes prototype) {
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
   * Protobuf type {@code tensorflow.DeviceAttributes}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tensorflow.DeviceAttributes)
      org.tensorflow.framework.DeviceAttributesOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tensorflow.framework.DeviceAttributesProtos.internal_static_tensorflow_DeviceAttributes_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tensorflow.framework.DeviceAttributesProtos.internal_static_tensorflow_DeviceAttributes_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tensorflow.framework.DeviceAttributes.class, org.tensorflow.framework.DeviceAttributes.Builder.class);
    }

    // Construct using org.tensorflow.framework.DeviceAttributes.newBuilder()
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
      name_ = "";

      deviceType_ = "";

      memoryLimit_ = 0L;

      if (localityBuilder_ == null) {
        locality_ = null;
      } else {
        locality_ = null;
        localityBuilder_ = null;
      }
      incarnation_ = 0L;

      physicalDeviceDesc_ = "";

      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tensorflow.framework.DeviceAttributesProtos.internal_static_tensorflow_DeviceAttributes_descriptor;
    }

    public org.tensorflow.framework.DeviceAttributes getDefaultInstanceForType() {
      return org.tensorflow.framework.DeviceAttributes.getDefaultInstance();
    }

    public org.tensorflow.framework.DeviceAttributes build() {
      org.tensorflow.framework.DeviceAttributes result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.tensorflow.framework.DeviceAttributes buildPartial() {
      org.tensorflow.framework.DeviceAttributes result = new org.tensorflow.framework.DeviceAttributes(this);
      result.name_ = name_;
      result.deviceType_ = deviceType_;
      result.memoryLimit_ = memoryLimit_;
      if (localityBuilder_ == null) {
        result.locality_ = locality_;
      } else {
        result.locality_ = localityBuilder_.build();
      }
      result.incarnation_ = incarnation_;
      result.physicalDeviceDesc_ = physicalDeviceDesc_;
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
      if (other instanceof org.tensorflow.framework.DeviceAttributes) {
        return mergeFrom((org.tensorflow.framework.DeviceAttributes)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tensorflow.framework.DeviceAttributes other) {
      if (other == org.tensorflow.framework.DeviceAttributes.getDefaultInstance()) return this;
      if (!other.getName().isEmpty()) {
        name_ = other.name_;
        onChanged();
      }
      if (!other.getDeviceType().isEmpty()) {
        deviceType_ = other.deviceType_;
        onChanged();
      }
      if (other.getMemoryLimit() != 0L) {
        setMemoryLimit(other.getMemoryLimit());
      }
      if (other.hasLocality()) {
        mergeLocality(other.getLocality());
      }
      if (other.getIncarnation() != 0L) {
        setIncarnation(other.getIncarnation());
      }
      if (!other.getPhysicalDeviceDesc().isEmpty()) {
        physicalDeviceDesc_ = other.physicalDeviceDesc_;
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
      org.tensorflow.framework.DeviceAttributes parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.tensorflow.framework.DeviceAttributes) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }

    private java.lang.Object name_ = "";
    /**
     * <pre>
     * Fully specified name of the device within a cluster.
     * </pre>
     *
     * <code>optional string name = 1;</code>
     */
    public java.lang.String getName() {
      java.lang.Object ref = name_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        name_ = s;
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <pre>
     * Fully specified name of the device within a cluster.
     * </pre>
     *
     * <code>optional string name = 1;</code>
     */
    public com.google.protobuf.ByteString
        getNameBytes() {
      java.lang.Object ref = name_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        name_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <pre>
     * Fully specified name of the device within a cluster.
     * </pre>
     *
     * <code>optional string name = 1;</code>
     */
    public Builder setName(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  
      name_ = value;
      onChanged();
      return this;
    }
    /**
     * <pre>
     * Fully specified name of the device within a cluster.
     * </pre>
     *
     * <code>optional string name = 1;</code>
     */
    public Builder clearName() {
      
      name_ = getDefaultInstance().getName();
      onChanged();
      return this;
    }
    /**
     * <pre>
     * Fully specified name of the device within a cluster.
     * </pre>
     *
     * <code>optional string name = 1;</code>
     */
    public Builder setNameBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  checkByteStringIsUtf8(value);
      
      name_ = value;
      onChanged();
      return this;
    }

    private java.lang.Object deviceType_ = "";
    /**
     * <pre>
     * String representation of device_type.
     * </pre>
     *
     * <code>optional string device_type = 2;</code>
     */
    public java.lang.String getDeviceType() {
      java.lang.Object ref = deviceType_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        deviceType_ = s;
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <pre>
     * String representation of device_type.
     * </pre>
     *
     * <code>optional string device_type = 2;</code>
     */
    public com.google.protobuf.ByteString
        getDeviceTypeBytes() {
      java.lang.Object ref = deviceType_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        deviceType_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <pre>
     * String representation of device_type.
     * </pre>
     *
     * <code>optional string device_type = 2;</code>
     */
    public Builder setDeviceType(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  
      deviceType_ = value;
      onChanged();
      return this;
    }
    /**
     * <pre>
     * String representation of device_type.
     * </pre>
     *
     * <code>optional string device_type = 2;</code>
     */
    public Builder clearDeviceType() {
      
      deviceType_ = getDefaultInstance().getDeviceType();
      onChanged();
      return this;
    }
    /**
     * <pre>
     * String representation of device_type.
     * </pre>
     *
     * <code>optional string device_type = 2;</code>
     */
    public Builder setDeviceTypeBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  checkByteStringIsUtf8(value);
      
      deviceType_ = value;
      onChanged();
      return this;
    }

    private long memoryLimit_ ;
    /**
     * <pre>
     * Memory capacity of device in bytes.
     * </pre>
     *
     * <code>optional int64 memory_limit = 4;</code>
     */
    public long getMemoryLimit() {
      return memoryLimit_;
    }
    /**
     * <pre>
     * Memory capacity of device in bytes.
     * </pre>
     *
     * <code>optional int64 memory_limit = 4;</code>
     */
    public Builder setMemoryLimit(long value) {
      
      memoryLimit_ = value;
      onChanged();
      return this;
    }
    /**
     * <pre>
     * Memory capacity of device in bytes.
     * </pre>
     *
     * <code>optional int64 memory_limit = 4;</code>
     */
    public Builder clearMemoryLimit() {
      
      memoryLimit_ = 0L;
      onChanged();
      return this;
    }

    private org.tensorflow.framework.DeviceLocality locality_ = null;
    private com.google.protobuf.SingleFieldBuilderV3<
        org.tensorflow.framework.DeviceLocality, org.tensorflow.framework.DeviceLocality.Builder, org.tensorflow.framework.DeviceLocalityOrBuilder> localityBuilder_;
    /**
     * <pre>
     * Platform-specific data about device that may be useful
     * for supporting efficient data transfers.
     * </pre>
     *
     * <code>optional .tensorflow.DeviceLocality locality = 5;</code>
     */
    public boolean hasLocality() {
      return localityBuilder_ != null || locality_ != null;
    }
    /**
     * <pre>
     * Platform-specific data about device that may be useful
     * for supporting efficient data transfers.
     * </pre>
     *
     * <code>optional .tensorflow.DeviceLocality locality = 5;</code>
     */
    public org.tensorflow.framework.DeviceLocality getLocality() {
      if (localityBuilder_ == null) {
        return locality_ == null ? org.tensorflow.framework.DeviceLocality.getDefaultInstance() : locality_;
      } else {
        return localityBuilder_.getMessage();
      }
    }
    /**
     * <pre>
     * Platform-specific data about device that may be useful
     * for supporting efficient data transfers.
     * </pre>
     *
     * <code>optional .tensorflow.DeviceLocality locality = 5;</code>
     */
    public Builder setLocality(org.tensorflow.framework.DeviceLocality value) {
      if (localityBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        locality_ = value;
        onChanged();
      } else {
        localityBuilder_.setMessage(value);
      }

      return this;
    }
    /**
     * <pre>
     * Platform-specific data about device that may be useful
     * for supporting efficient data transfers.
     * </pre>
     *
     * <code>optional .tensorflow.DeviceLocality locality = 5;</code>
     */
    public Builder setLocality(
        org.tensorflow.framework.DeviceLocality.Builder builderForValue) {
      if (localityBuilder_ == null) {
        locality_ = builderForValue.build();
        onChanged();
      } else {
        localityBuilder_.setMessage(builderForValue.build());
      }

      return this;
    }
    /**
     * <pre>
     * Platform-specific data about device that may be useful
     * for supporting efficient data transfers.
     * </pre>
     *
     * <code>optional .tensorflow.DeviceLocality locality = 5;</code>
     */
    public Builder mergeLocality(org.tensorflow.framework.DeviceLocality value) {
      if (localityBuilder_ == null) {
        if (locality_ != null) {
          locality_ =
            org.tensorflow.framework.DeviceLocality.newBuilder(locality_).mergeFrom(value).buildPartial();
        } else {
          locality_ = value;
        }
        onChanged();
      } else {
        localityBuilder_.mergeFrom(value);
      }

      return this;
    }
    /**
     * <pre>
     * Platform-specific data about device that may be useful
     * for supporting efficient data transfers.
     * </pre>
     *
     * <code>optional .tensorflow.DeviceLocality locality = 5;</code>
     */
    public Builder clearLocality() {
      if (localityBuilder_ == null) {
        locality_ = null;
        onChanged();
      } else {
        locality_ = null;
        localityBuilder_ = null;
      }

      return this;
    }
    /**
     * <pre>
     * Platform-specific data about device that may be useful
     * for supporting efficient data transfers.
     * </pre>
     *
     * <code>optional .tensorflow.DeviceLocality locality = 5;</code>
     */
    public org.tensorflow.framework.DeviceLocality.Builder getLocalityBuilder() {
      
      onChanged();
      return getLocalityFieldBuilder().getBuilder();
    }
    /**
     * <pre>
     * Platform-specific data about device that may be useful
     * for supporting efficient data transfers.
     * </pre>
     *
     * <code>optional .tensorflow.DeviceLocality locality = 5;</code>
     */
    public org.tensorflow.framework.DeviceLocalityOrBuilder getLocalityOrBuilder() {
      if (localityBuilder_ != null) {
        return localityBuilder_.getMessageOrBuilder();
      } else {
        return locality_ == null ?
            org.tensorflow.framework.DeviceLocality.getDefaultInstance() : locality_;
      }
    }
    /**
     * <pre>
     * Platform-specific data about device that may be useful
     * for supporting efficient data transfers.
     * </pre>
     *
     * <code>optional .tensorflow.DeviceLocality locality = 5;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.tensorflow.framework.DeviceLocality, org.tensorflow.framework.DeviceLocality.Builder, org.tensorflow.framework.DeviceLocalityOrBuilder> 
        getLocalityFieldBuilder() {
      if (localityBuilder_ == null) {
        localityBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.tensorflow.framework.DeviceLocality, org.tensorflow.framework.DeviceLocality.Builder, org.tensorflow.framework.DeviceLocalityOrBuilder>(
                getLocality(),
                getParentForChildren(),
                isClean());
        locality_ = null;
      }
      return localityBuilder_;
    }

    private long incarnation_ ;
    /**
     * <pre>
     * A device is assigned a global unique number each time it is
     * initialized. "incarnation" should never be 0.
     * </pre>
     *
     * <code>optional fixed64 incarnation = 6;</code>
     */
    public long getIncarnation() {
      return incarnation_;
    }
    /**
     * <pre>
     * A device is assigned a global unique number each time it is
     * initialized. "incarnation" should never be 0.
     * </pre>
     *
     * <code>optional fixed64 incarnation = 6;</code>
     */
    public Builder setIncarnation(long value) {
      
      incarnation_ = value;
      onChanged();
      return this;
    }
    /**
     * <pre>
     * A device is assigned a global unique number each time it is
     * initialized. "incarnation" should never be 0.
     * </pre>
     *
     * <code>optional fixed64 incarnation = 6;</code>
     */
    public Builder clearIncarnation() {
      
      incarnation_ = 0L;
      onChanged();
      return this;
    }

    private java.lang.Object physicalDeviceDesc_ = "";
    /**
     * <pre>
     * String representation of the physical device that this device maps to.
     * </pre>
     *
     * <code>optional string physical_device_desc = 7;</code>
     */
    public java.lang.String getPhysicalDeviceDesc() {
      java.lang.Object ref = physicalDeviceDesc_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        physicalDeviceDesc_ = s;
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <pre>
     * String representation of the physical device that this device maps to.
     * </pre>
     *
     * <code>optional string physical_device_desc = 7;</code>
     */
    public com.google.protobuf.ByteString
        getPhysicalDeviceDescBytes() {
      java.lang.Object ref = physicalDeviceDesc_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        physicalDeviceDesc_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <pre>
     * String representation of the physical device that this device maps to.
     * </pre>
     *
     * <code>optional string physical_device_desc = 7;</code>
     */
    public Builder setPhysicalDeviceDesc(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  
      physicalDeviceDesc_ = value;
      onChanged();
      return this;
    }
    /**
     * <pre>
     * String representation of the physical device that this device maps to.
     * </pre>
     *
     * <code>optional string physical_device_desc = 7;</code>
     */
    public Builder clearPhysicalDeviceDesc() {
      
      physicalDeviceDesc_ = getDefaultInstance().getPhysicalDeviceDesc();
      onChanged();
      return this;
    }
    /**
     * <pre>
     * String representation of the physical device that this device maps to.
     * </pre>
     *
     * <code>optional string physical_device_desc = 7;</code>
     */
    public Builder setPhysicalDeviceDescBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  checkByteStringIsUtf8(value);
      
      physicalDeviceDesc_ = value;
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


    // @@protoc_insertion_point(builder_scope:tensorflow.DeviceAttributes)
  }

  // @@protoc_insertion_point(class_scope:tensorflow.DeviceAttributes)
  private static final org.tensorflow.framework.DeviceAttributes DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tensorflow.framework.DeviceAttributes();
  }

  public static org.tensorflow.framework.DeviceAttributes getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<DeviceAttributes>
      PARSER = new com.google.protobuf.AbstractParser<DeviceAttributes>() {
    public DeviceAttributes parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new DeviceAttributes(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<DeviceAttributes> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<DeviceAttributes> getParserForType() {
    return PARSER;
  }

  public org.tensorflow.framework.DeviceAttributes getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

