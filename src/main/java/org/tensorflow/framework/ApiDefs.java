// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: apis/core/framework/api_def.proto

package org.tensorflow.framework;

/**
 * Protobuf type {@code tensorflow.ApiDefs}
 */
public  final class ApiDefs extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tensorflow.ApiDefs)
    ApiDefsOrBuilder {
  // Use ApiDefs.newBuilder() to construct.
  private ApiDefs(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private ApiDefs() {
    op_ = java.util.Collections.emptyList();
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return com.google.protobuf.UnknownFieldSet.getDefaultInstance();
  }
  private ApiDefs(
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
            if (!((mutable_bitField0_ & 0x00000001) == 0x00000001)) {
              op_ = new java.util.ArrayList<org.tensorflow.framework.ApiDef>();
              mutable_bitField0_ |= 0x00000001;
            }
            op_.add(
                input.readMessage(org.tensorflow.framework.ApiDef.parser(), extensionRegistry));
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
      if (((mutable_bitField0_ & 0x00000001) == 0x00000001)) {
        op_ = java.util.Collections.unmodifiableList(op_);
      }
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tensorflow.framework.ApiDefProtos.internal_static_tensorflow_ApiDefs_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tensorflow.framework.ApiDefProtos.internal_static_tensorflow_ApiDefs_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tensorflow.framework.ApiDefs.class, org.tensorflow.framework.ApiDefs.Builder.class);
  }

  public static final int OP_FIELD_NUMBER = 1;
  private java.util.List<org.tensorflow.framework.ApiDef> op_;
  /**
   * <code>repeated .tensorflow.ApiDef op = 1;</code>
   */
  public java.util.List<org.tensorflow.framework.ApiDef> getOpList() {
    return op_;
  }
  /**
   * <code>repeated .tensorflow.ApiDef op = 1;</code>
   */
  public java.util.List<? extends org.tensorflow.framework.ApiDefOrBuilder> 
      getOpOrBuilderList() {
    return op_;
  }
  /**
   * <code>repeated .tensorflow.ApiDef op = 1;</code>
   */
  public int getOpCount() {
    return op_.size();
  }
  /**
   * <code>repeated .tensorflow.ApiDef op = 1;</code>
   */
  public org.tensorflow.framework.ApiDef getOp(int index) {
    return op_.get(index);
  }
  /**
   * <code>repeated .tensorflow.ApiDef op = 1;</code>
   */
  public org.tensorflow.framework.ApiDefOrBuilder getOpOrBuilder(
      int index) {
    return op_.get(index);
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
    for (int i = 0; i < op_.size(); i++) {
      output.writeMessage(1, op_.get(i));
    }
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    for (int i = 0; i < op_.size(); i++) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(1, op_.get(i));
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
    if (!(obj instanceof org.tensorflow.framework.ApiDefs)) {
      return super.equals(obj);
    }
    org.tensorflow.framework.ApiDefs other = (org.tensorflow.framework.ApiDefs) obj;

    boolean result = true;
    result = result && getOpList()
        .equals(other.getOpList());
    return result;
  }

  @java.lang.Override
  public int hashCode() {
    if (memoizedHashCode != 0) {
      return memoizedHashCode;
    }
    int hash = 41;
    hash = (19 * hash) + getDescriptorForType().hashCode();
    if (getOpCount() > 0) {
      hash = (37 * hash) + OP_FIELD_NUMBER;
      hash = (53 * hash) + getOpList().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tensorflow.framework.ApiDefs parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tensorflow.framework.ApiDefs parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tensorflow.framework.ApiDefs parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tensorflow.framework.ApiDefs parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tensorflow.framework.ApiDefs parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tensorflow.framework.ApiDefs parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tensorflow.framework.ApiDefs parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.tensorflow.framework.ApiDefs parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tensorflow.framework.ApiDefs parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tensorflow.framework.ApiDefs parseFrom(
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
  public static Builder newBuilder(org.tensorflow.framework.ApiDefs prototype) {
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
   * Protobuf type {@code tensorflow.ApiDefs}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tensorflow.ApiDefs)
      org.tensorflow.framework.ApiDefsOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tensorflow.framework.ApiDefProtos.internal_static_tensorflow_ApiDefs_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tensorflow.framework.ApiDefProtos.internal_static_tensorflow_ApiDefs_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tensorflow.framework.ApiDefs.class, org.tensorflow.framework.ApiDefs.Builder.class);
    }

    // Construct using org.tensorflow.framework.ApiDefs.newBuilder()
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
        getOpFieldBuilder();
      }
    }
    public Builder clear() {
      super.clear();
      if (opBuilder_ == null) {
        op_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000001);
      } else {
        opBuilder_.clear();
      }
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tensorflow.framework.ApiDefProtos.internal_static_tensorflow_ApiDefs_descriptor;
    }

    public org.tensorflow.framework.ApiDefs getDefaultInstanceForType() {
      return org.tensorflow.framework.ApiDefs.getDefaultInstance();
    }

    public org.tensorflow.framework.ApiDefs build() {
      org.tensorflow.framework.ApiDefs result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.tensorflow.framework.ApiDefs buildPartial() {
      org.tensorflow.framework.ApiDefs result = new org.tensorflow.framework.ApiDefs(this);
      int from_bitField0_ = bitField0_;
      if (opBuilder_ == null) {
        if (((bitField0_ & 0x00000001) == 0x00000001)) {
          op_ = java.util.Collections.unmodifiableList(op_);
          bitField0_ = (bitField0_ & ~0x00000001);
        }
        result.op_ = op_;
      } else {
        result.op_ = opBuilder_.build();
      }
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
      if (other instanceof org.tensorflow.framework.ApiDefs) {
        return mergeFrom((org.tensorflow.framework.ApiDefs)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tensorflow.framework.ApiDefs other) {
      if (other == org.tensorflow.framework.ApiDefs.getDefaultInstance()) return this;
      if (opBuilder_ == null) {
        if (!other.op_.isEmpty()) {
          if (op_.isEmpty()) {
            op_ = other.op_;
            bitField0_ = (bitField0_ & ~0x00000001);
          } else {
            ensureOpIsMutable();
            op_.addAll(other.op_);
          }
          onChanged();
        }
      } else {
        if (!other.op_.isEmpty()) {
          if (opBuilder_.isEmpty()) {
            opBuilder_.dispose();
            opBuilder_ = null;
            op_ = other.op_;
            bitField0_ = (bitField0_ & ~0x00000001);
            opBuilder_ = 
              com.google.protobuf.GeneratedMessageV3.alwaysUseFieldBuilders ?
                 getOpFieldBuilder() : null;
          } else {
            opBuilder_.addAllMessages(other.op_);
          }
        }
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
      org.tensorflow.framework.ApiDefs parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.tensorflow.framework.ApiDefs) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private java.util.List<org.tensorflow.framework.ApiDef> op_ =
      java.util.Collections.emptyList();
    private void ensureOpIsMutable() {
      if (!((bitField0_ & 0x00000001) == 0x00000001)) {
        op_ = new java.util.ArrayList<org.tensorflow.framework.ApiDef>(op_);
        bitField0_ |= 0x00000001;
       }
    }

    private com.google.protobuf.RepeatedFieldBuilderV3<
        org.tensorflow.framework.ApiDef, org.tensorflow.framework.ApiDef.Builder, org.tensorflow.framework.ApiDefOrBuilder> opBuilder_;

    /**
     * <code>repeated .tensorflow.ApiDef op = 1;</code>
     */
    public java.util.List<org.tensorflow.framework.ApiDef> getOpList() {
      if (opBuilder_ == null) {
        return java.util.Collections.unmodifiableList(op_);
      } else {
        return opBuilder_.getMessageList();
      }
    }
    /**
     * <code>repeated .tensorflow.ApiDef op = 1;</code>
     */
    public int getOpCount() {
      if (opBuilder_ == null) {
        return op_.size();
      } else {
        return opBuilder_.getCount();
      }
    }
    /**
     * <code>repeated .tensorflow.ApiDef op = 1;</code>
     */
    public org.tensorflow.framework.ApiDef getOp(int index) {
      if (opBuilder_ == null) {
        return op_.get(index);
      } else {
        return opBuilder_.getMessage(index);
      }
    }
    /**
     * <code>repeated .tensorflow.ApiDef op = 1;</code>
     */
    public Builder setOp(
        int index, org.tensorflow.framework.ApiDef value) {
      if (opBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureOpIsMutable();
        op_.set(index, value);
        onChanged();
      } else {
        opBuilder_.setMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .tensorflow.ApiDef op = 1;</code>
     */
    public Builder setOp(
        int index, org.tensorflow.framework.ApiDef.Builder builderForValue) {
      if (opBuilder_ == null) {
        ensureOpIsMutable();
        op_.set(index, builderForValue.build());
        onChanged();
      } else {
        opBuilder_.setMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .tensorflow.ApiDef op = 1;</code>
     */
    public Builder addOp(org.tensorflow.framework.ApiDef value) {
      if (opBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureOpIsMutable();
        op_.add(value);
        onChanged();
      } else {
        opBuilder_.addMessage(value);
      }
      return this;
    }
    /**
     * <code>repeated .tensorflow.ApiDef op = 1;</code>
     */
    public Builder addOp(
        int index, org.tensorflow.framework.ApiDef value) {
      if (opBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureOpIsMutable();
        op_.add(index, value);
        onChanged();
      } else {
        opBuilder_.addMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .tensorflow.ApiDef op = 1;</code>
     */
    public Builder addOp(
        org.tensorflow.framework.ApiDef.Builder builderForValue) {
      if (opBuilder_ == null) {
        ensureOpIsMutable();
        op_.add(builderForValue.build());
        onChanged();
      } else {
        opBuilder_.addMessage(builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .tensorflow.ApiDef op = 1;</code>
     */
    public Builder addOp(
        int index, org.tensorflow.framework.ApiDef.Builder builderForValue) {
      if (opBuilder_ == null) {
        ensureOpIsMutable();
        op_.add(index, builderForValue.build());
        onChanged();
      } else {
        opBuilder_.addMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .tensorflow.ApiDef op = 1;</code>
     */
    public Builder addAllOp(
        java.lang.Iterable<? extends org.tensorflow.framework.ApiDef> values) {
      if (opBuilder_ == null) {
        ensureOpIsMutable();
        com.google.protobuf.AbstractMessageLite.Builder.addAll(
            values, op_);
        onChanged();
      } else {
        opBuilder_.addAllMessages(values);
      }
      return this;
    }
    /**
     * <code>repeated .tensorflow.ApiDef op = 1;</code>
     */
    public Builder clearOp() {
      if (opBuilder_ == null) {
        op_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000001);
        onChanged();
      } else {
        opBuilder_.clear();
      }
      return this;
    }
    /**
     * <code>repeated .tensorflow.ApiDef op = 1;</code>
     */
    public Builder removeOp(int index) {
      if (opBuilder_ == null) {
        ensureOpIsMutable();
        op_.remove(index);
        onChanged();
      } else {
        opBuilder_.remove(index);
      }
      return this;
    }
    /**
     * <code>repeated .tensorflow.ApiDef op = 1;</code>
     */
    public org.tensorflow.framework.ApiDef.Builder getOpBuilder(
        int index) {
      return getOpFieldBuilder().getBuilder(index);
    }
    /**
     * <code>repeated .tensorflow.ApiDef op = 1;</code>
     */
    public org.tensorflow.framework.ApiDefOrBuilder getOpOrBuilder(
        int index) {
      if (opBuilder_ == null) {
        return op_.get(index);  } else {
        return opBuilder_.getMessageOrBuilder(index);
      }
    }
    /**
     * <code>repeated .tensorflow.ApiDef op = 1;</code>
     */
    public java.util.List<? extends org.tensorflow.framework.ApiDefOrBuilder> 
         getOpOrBuilderList() {
      if (opBuilder_ != null) {
        return opBuilder_.getMessageOrBuilderList();
      } else {
        return java.util.Collections.unmodifiableList(op_);
      }
    }
    /**
     * <code>repeated .tensorflow.ApiDef op = 1;</code>
     */
    public org.tensorflow.framework.ApiDef.Builder addOpBuilder() {
      return getOpFieldBuilder().addBuilder(
          org.tensorflow.framework.ApiDef.getDefaultInstance());
    }
    /**
     * <code>repeated .tensorflow.ApiDef op = 1;</code>
     */
    public org.tensorflow.framework.ApiDef.Builder addOpBuilder(
        int index) {
      return getOpFieldBuilder().addBuilder(
          index, org.tensorflow.framework.ApiDef.getDefaultInstance());
    }
    /**
     * <code>repeated .tensorflow.ApiDef op = 1;</code>
     */
    public java.util.List<org.tensorflow.framework.ApiDef.Builder> 
         getOpBuilderList() {
      return getOpFieldBuilder().getBuilderList();
    }
    private com.google.protobuf.RepeatedFieldBuilderV3<
        org.tensorflow.framework.ApiDef, org.tensorflow.framework.ApiDef.Builder, org.tensorflow.framework.ApiDefOrBuilder> 
        getOpFieldBuilder() {
      if (opBuilder_ == null) {
        opBuilder_ = new com.google.protobuf.RepeatedFieldBuilderV3<
            org.tensorflow.framework.ApiDef, org.tensorflow.framework.ApiDef.Builder, org.tensorflow.framework.ApiDefOrBuilder>(
                op_,
                ((bitField0_ & 0x00000001) == 0x00000001),
                getParentForChildren(),
                isClean());
        op_ = null;
      }
      return opBuilder_;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return this;
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return this;
    }


    // @@protoc_insertion_point(builder_scope:tensorflow.ApiDefs)
  }

  // @@protoc_insertion_point(class_scope:tensorflow.ApiDefs)
  private static final org.tensorflow.framework.ApiDefs DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tensorflow.framework.ApiDefs();
  }

  public static org.tensorflow.framework.ApiDefs getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<ApiDefs>
      PARSER = new com.google.protobuf.AbstractParser<ApiDefs>() {
    public ApiDefs parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new ApiDefs(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<ApiDefs> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<ApiDefs> getParserForType() {
    return PARSER;
  }

  public org.tensorflow.framework.ApiDefs getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
