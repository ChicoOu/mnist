// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: apis/core/protobuf/debug.proto

package org.tensorflow.framework;

/**
 * Protobuf type {@code tensorflow.DebuggedSourceFile}
 */
public  final class DebuggedSourceFile extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tensorflow.DebuggedSourceFile)
    DebuggedSourceFileOrBuilder {
  // Use DebuggedSourceFile.newBuilder() to construct.
  private DebuggedSourceFile(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private DebuggedSourceFile() {
    host_ = "";
    filePath_ = "";
    lastModified_ = 0L;
    bytes_ = 0L;
    lines_ = com.google.protobuf.LazyStringArrayList.EMPTY;
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return com.google.protobuf.UnknownFieldSet.getDefaultInstance();
  }
  private DebuggedSourceFile(
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

            host_ = s;
            break;
          }
          case 18: {
            java.lang.String s = input.readStringRequireUtf8();

            filePath_ = s;
            break;
          }
          case 24: {

            lastModified_ = input.readInt64();
            break;
          }
          case 32: {

            bytes_ = input.readInt64();
            break;
          }
          case 42: {
            java.lang.String s = input.readStringRequireUtf8();
            if (!((mutable_bitField0_ & 0x00000010) == 0x00000010)) {
              lines_ = new com.google.protobuf.LazyStringArrayList();
              mutable_bitField0_ |= 0x00000010;
            }
            lines_.add(s);
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
      if (((mutable_bitField0_ & 0x00000010) == 0x00000010)) {
        lines_ = lines_.getUnmodifiableView();
      }
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tensorflow.framework.DebugProtos.internal_static_tensorflow_DebuggedSourceFile_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tensorflow.framework.DebugProtos.internal_static_tensorflow_DebuggedSourceFile_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tensorflow.framework.DebuggedSourceFile.class, org.tensorflow.framework.DebuggedSourceFile.Builder.class);
  }

  private int bitField0_;
  public static final int HOST_FIELD_NUMBER = 1;
  private volatile java.lang.Object host_;
  /**
   * <pre>
   * The host name on which a source code file is located.
   * </pre>
   *
   * <code>optional string host = 1;</code>
   */
  public java.lang.String getHost() {
    java.lang.Object ref = host_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      host_ = s;
      return s;
    }
  }
  /**
   * <pre>
   * The host name on which a source code file is located.
   * </pre>
   *
   * <code>optional string host = 1;</code>
   */
  public com.google.protobuf.ByteString
      getHostBytes() {
    java.lang.Object ref = host_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      host_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int FILE_PATH_FIELD_NUMBER = 2;
  private volatile java.lang.Object filePath_;
  /**
   * <pre>
   * Path to the source code file.
   * </pre>
   *
   * <code>optional string file_path = 2;</code>
   */
  public java.lang.String getFilePath() {
    java.lang.Object ref = filePath_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      filePath_ = s;
      return s;
    }
  }
  /**
   * <pre>
   * Path to the source code file.
   * </pre>
   *
   * <code>optional string file_path = 2;</code>
   */
  public com.google.protobuf.ByteString
      getFilePathBytes() {
    java.lang.Object ref = filePath_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      filePath_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int LAST_MODIFIED_FIELD_NUMBER = 3;
  private long lastModified_;
  /**
   * <pre>
   * The timestamp at which the source code file is last modified.
   * </pre>
   *
   * <code>optional int64 last_modified = 3;</code>
   */
  public long getLastModified() {
    return lastModified_;
  }

  public static final int BYTES_FIELD_NUMBER = 4;
  private long bytes_;
  /**
   * <pre>
   * Byte size of the file.
   * </pre>
   *
   * <code>optional int64 bytes = 4;</code>
   */
  public long getBytes() {
    return bytes_;
  }

  public static final int LINES_FIELD_NUMBER = 5;
  private com.google.protobuf.LazyStringList lines_;
  /**
   * <pre>
   * Line-by-line content of the source code file.
   * </pre>
   *
   * <code>repeated string lines = 5;</code>
   */
  public com.google.protobuf.ProtocolStringList
      getLinesList() {
    return lines_;
  }
  /**
   * <pre>
   * Line-by-line content of the source code file.
   * </pre>
   *
   * <code>repeated string lines = 5;</code>
   */
  public int getLinesCount() {
    return lines_.size();
  }
  /**
   * <pre>
   * Line-by-line content of the source code file.
   * </pre>
   *
   * <code>repeated string lines = 5;</code>
   */
  public java.lang.String getLines(int index) {
    return lines_.get(index);
  }
  /**
   * <pre>
   * Line-by-line content of the source code file.
   * </pre>
   *
   * <code>repeated string lines = 5;</code>
   */
  public com.google.protobuf.ByteString
      getLinesBytes(int index) {
    return lines_.getByteString(index);
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
    if (!getHostBytes().isEmpty()) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 1, host_);
    }
    if (!getFilePathBytes().isEmpty()) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 2, filePath_);
    }
    if (lastModified_ != 0L) {
      output.writeInt64(3, lastModified_);
    }
    if (bytes_ != 0L) {
      output.writeInt64(4, bytes_);
    }
    for (int i = 0; i < lines_.size(); i++) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 5, lines_.getRaw(i));
    }
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (!getHostBytes().isEmpty()) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(1, host_);
    }
    if (!getFilePathBytes().isEmpty()) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(2, filePath_);
    }
    if (lastModified_ != 0L) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt64Size(3, lastModified_);
    }
    if (bytes_ != 0L) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt64Size(4, bytes_);
    }
    {
      int dataSize = 0;
      for (int i = 0; i < lines_.size(); i++) {
        dataSize += computeStringSizeNoTag(lines_.getRaw(i));
      }
      size += dataSize;
      size += 1 * getLinesList().size();
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
    if (!(obj instanceof org.tensorflow.framework.DebuggedSourceFile)) {
      return super.equals(obj);
    }
    org.tensorflow.framework.DebuggedSourceFile other = (org.tensorflow.framework.DebuggedSourceFile) obj;

    boolean result = true;
    result = result && getHost()
        .equals(other.getHost());
    result = result && getFilePath()
        .equals(other.getFilePath());
    result = result && (getLastModified()
        == other.getLastModified());
    result = result && (getBytes()
        == other.getBytes());
    result = result && getLinesList()
        .equals(other.getLinesList());
    return result;
  }

  @java.lang.Override
  public int hashCode() {
    if (memoizedHashCode != 0) {
      return memoizedHashCode;
    }
    int hash = 41;
    hash = (19 * hash) + getDescriptorForType().hashCode();
    hash = (37 * hash) + HOST_FIELD_NUMBER;
    hash = (53 * hash) + getHost().hashCode();
    hash = (37 * hash) + FILE_PATH_FIELD_NUMBER;
    hash = (53 * hash) + getFilePath().hashCode();
    hash = (37 * hash) + LAST_MODIFIED_FIELD_NUMBER;
    hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
        getLastModified());
    hash = (37 * hash) + BYTES_FIELD_NUMBER;
    hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
        getBytes());
    if (getLinesCount() > 0) {
      hash = (37 * hash) + LINES_FIELD_NUMBER;
      hash = (53 * hash) + getLinesList().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tensorflow.framework.DebuggedSourceFile parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tensorflow.framework.DebuggedSourceFile parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tensorflow.framework.DebuggedSourceFile parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tensorflow.framework.DebuggedSourceFile parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tensorflow.framework.DebuggedSourceFile parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tensorflow.framework.DebuggedSourceFile parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tensorflow.framework.DebuggedSourceFile parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.tensorflow.framework.DebuggedSourceFile parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tensorflow.framework.DebuggedSourceFile parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tensorflow.framework.DebuggedSourceFile parseFrom(
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
  public static Builder newBuilder(org.tensorflow.framework.DebuggedSourceFile prototype) {
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
   * Protobuf type {@code tensorflow.DebuggedSourceFile}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tensorflow.DebuggedSourceFile)
      org.tensorflow.framework.DebuggedSourceFileOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tensorflow.framework.DebugProtos.internal_static_tensorflow_DebuggedSourceFile_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tensorflow.framework.DebugProtos.internal_static_tensorflow_DebuggedSourceFile_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tensorflow.framework.DebuggedSourceFile.class, org.tensorflow.framework.DebuggedSourceFile.Builder.class);
    }

    // Construct using org.tensorflow.framework.DebuggedSourceFile.newBuilder()
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
      host_ = "";

      filePath_ = "";

      lastModified_ = 0L;

      bytes_ = 0L;

      lines_ = com.google.protobuf.LazyStringArrayList.EMPTY;
      bitField0_ = (bitField0_ & ~0x00000010);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tensorflow.framework.DebugProtos.internal_static_tensorflow_DebuggedSourceFile_descriptor;
    }

    public org.tensorflow.framework.DebuggedSourceFile getDefaultInstanceForType() {
      return org.tensorflow.framework.DebuggedSourceFile.getDefaultInstance();
    }

    public org.tensorflow.framework.DebuggedSourceFile build() {
      org.tensorflow.framework.DebuggedSourceFile result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.tensorflow.framework.DebuggedSourceFile buildPartial() {
      org.tensorflow.framework.DebuggedSourceFile result = new org.tensorflow.framework.DebuggedSourceFile(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      result.host_ = host_;
      result.filePath_ = filePath_;
      result.lastModified_ = lastModified_;
      result.bytes_ = bytes_;
      if (((bitField0_ & 0x00000010) == 0x00000010)) {
        lines_ = lines_.getUnmodifiableView();
        bitField0_ = (bitField0_ & ~0x00000010);
      }
      result.lines_ = lines_;
      result.bitField0_ = to_bitField0_;
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
      if (other instanceof org.tensorflow.framework.DebuggedSourceFile) {
        return mergeFrom((org.tensorflow.framework.DebuggedSourceFile)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tensorflow.framework.DebuggedSourceFile other) {
      if (other == org.tensorflow.framework.DebuggedSourceFile.getDefaultInstance()) return this;
      if (!other.getHost().isEmpty()) {
        host_ = other.host_;
        onChanged();
      }
      if (!other.getFilePath().isEmpty()) {
        filePath_ = other.filePath_;
        onChanged();
      }
      if (other.getLastModified() != 0L) {
        setLastModified(other.getLastModified());
      }
      if (other.getBytes() != 0L) {
        setBytes(other.getBytes());
      }
      if (!other.lines_.isEmpty()) {
        if (lines_.isEmpty()) {
          lines_ = other.lines_;
          bitField0_ = (bitField0_ & ~0x00000010);
        } else {
          ensureLinesIsMutable();
          lines_.addAll(other.lines_);
        }
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
      org.tensorflow.framework.DebuggedSourceFile parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.tensorflow.framework.DebuggedSourceFile) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private java.lang.Object host_ = "";
    /**
     * <pre>
     * The host name on which a source code file is located.
     * </pre>
     *
     * <code>optional string host = 1;</code>
     */
    public java.lang.String getHost() {
      java.lang.Object ref = host_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        host_ = s;
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <pre>
     * The host name on which a source code file is located.
     * </pre>
     *
     * <code>optional string host = 1;</code>
     */
    public com.google.protobuf.ByteString
        getHostBytes() {
      java.lang.Object ref = host_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        host_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <pre>
     * The host name on which a source code file is located.
     * </pre>
     *
     * <code>optional string host = 1;</code>
     */
    public Builder setHost(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  
      host_ = value;
      onChanged();
      return this;
    }
    /**
     * <pre>
     * The host name on which a source code file is located.
     * </pre>
     *
     * <code>optional string host = 1;</code>
     */
    public Builder clearHost() {
      
      host_ = getDefaultInstance().getHost();
      onChanged();
      return this;
    }
    /**
     * <pre>
     * The host name on which a source code file is located.
     * </pre>
     *
     * <code>optional string host = 1;</code>
     */
    public Builder setHostBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  checkByteStringIsUtf8(value);
      
      host_ = value;
      onChanged();
      return this;
    }

    private java.lang.Object filePath_ = "";
    /**
     * <pre>
     * Path to the source code file.
     * </pre>
     *
     * <code>optional string file_path = 2;</code>
     */
    public java.lang.String getFilePath() {
      java.lang.Object ref = filePath_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        filePath_ = s;
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <pre>
     * Path to the source code file.
     * </pre>
     *
     * <code>optional string file_path = 2;</code>
     */
    public com.google.protobuf.ByteString
        getFilePathBytes() {
      java.lang.Object ref = filePath_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        filePath_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <pre>
     * Path to the source code file.
     * </pre>
     *
     * <code>optional string file_path = 2;</code>
     */
    public Builder setFilePath(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  
      filePath_ = value;
      onChanged();
      return this;
    }
    /**
     * <pre>
     * Path to the source code file.
     * </pre>
     *
     * <code>optional string file_path = 2;</code>
     */
    public Builder clearFilePath() {
      
      filePath_ = getDefaultInstance().getFilePath();
      onChanged();
      return this;
    }
    /**
     * <pre>
     * Path to the source code file.
     * </pre>
     *
     * <code>optional string file_path = 2;</code>
     */
    public Builder setFilePathBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  checkByteStringIsUtf8(value);
      
      filePath_ = value;
      onChanged();
      return this;
    }

    private long lastModified_ ;
    /**
     * <pre>
     * The timestamp at which the source code file is last modified.
     * </pre>
     *
     * <code>optional int64 last_modified = 3;</code>
     */
    public long getLastModified() {
      return lastModified_;
    }
    /**
     * <pre>
     * The timestamp at which the source code file is last modified.
     * </pre>
     *
     * <code>optional int64 last_modified = 3;</code>
     */
    public Builder setLastModified(long value) {
      
      lastModified_ = value;
      onChanged();
      return this;
    }
    /**
     * <pre>
     * The timestamp at which the source code file is last modified.
     * </pre>
     *
     * <code>optional int64 last_modified = 3;</code>
     */
    public Builder clearLastModified() {
      
      lastModified_ = 0L;
      onChanged();
      return this;
    }

    private long bytes_ ;
    /**
     * <pre>
     * Byte size of the file.
     * </pre>
     *
     * <code>optional int64 bytes = 4;</code>
     */
    public long getBytes() {
      return bytes_;
    }
    /**
     * <pre>
     * Byte size of the file.
     * </pre>
     *
     * <code>optional int64 bytes = 4;</code>
     */
    public Builder setBytes(long value) {
      
      bytes_ = value;
      onChanged();
      return this;
    }
    /**
     * <pre>
     * Byte size of the file.
     * </pre>
     *
     * <code>optional int64 bytes = 4;</code>
     */
    public Builder clearBytes() {
      
      bytes_ = 0L;
      onChanged();
      return this;
    }

    private com.google.protobuf.LazyStringList lines_ = com.google.protobuf.LazyStringArrayList.EMPTY;
    private void ensureLinesIsMutable() {
      if (!((bitField0_ & 0x00000010) == 0x00000010)) {
        lines_ = new com.google.protobuf.LazyStringArrayList(lines_);
        bitField0_ |= 0x00000010;
       }
    }
    /**
     * <pre>
     * Line-by-line content of the source code file.
     * </pre>
     *
     * <code>repeated string lines = 5;</code>
     */
    public com.google.protobuf.ProtocolStringList
        getLinesList() {
      return lines_.getUnmodifiableView();
    }
    /**
     * <pre>
     * Line-by-line content of the source code file.
     * </pre>
     *
     * <code>repeated string lines = 5;</code>
     */
    public int getLinesCount() {
      return lines_.size();
    }
    /**
     * <pre>
     * Line-by-line content of the source code file.
     * </pre>
     *
     * <code>repeated string lines = 5;</code>
     */
    public java.lang.String getLines(int index) {
      return lines_.get(index);
    }
    /**
     * <pre>
     * Line-by-line content of the source code file.
     * </pre>
     *
     * <code>repeated string lines = 5;</code>
     */
    public com.google.protobuf.ByteString
        getLinesBytes(int index) {
      return lines_.getByteString(index);
    }
    /**
     * <pre>
     * Line-by-line content of the source code file.
     * </pre>
     *
     * <code>repeated string lines = 5;</code>
     */
    public Builder setLines(
        int index, java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureLinesIsMutable();
      lines_.set(index, value);
      onChanged();
      return this;
    }
    /**
     * <pre>
     * Line-by-line content of the source code file.
     * </pre>
     *
     * <code>repeated string lines = 5;</code>
     */
    public Builder addLines(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureLinesIsMutable();
      lines_.add(value);
      onChanged();
      return this;
    }
    /**
     * <pre>
     * Line-by-line content of the source code file.
     * </pre>
     *
     * <code>repeated string lines = 5;</code>
     */
    public Builder addAllLines(
        java.lang.Iterable<java.lang.String> values) {
      ensureLinesIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, lines_);
      onChanged();
      return this;
    }
    /**
     * <pre>
     * Line-by-line content of the source code file.
     * </pre>
     *
     * <code>repeated string lines = 5;</code>
     */
    public Builder clearLines() {
      lines_ = com.google.protobuf.LazyStringArrayList.EMPTY;
      bitField0_ = (bitField0_ & ~0x00000010);
      onChanged();
      return this;
    }
    /**
     * <pre>
     * Line-by-line content of the source code file.
     * </pre>
     *
     * <code>repeated string lines = 5;</code>
     */
    public Builder addLinesBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  checkByteStringIsUtf8(value);
      ensureLinesIsMutable();
      lines_.add(value);
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


    // @@protoc_insertion_point(builder_scope:tensorflow.DebuggedSourceFile)
  }

  // @@protoc_insertion_point(class_scope:tensorflow.DebuggedSourceFile)
  private static final org.tensorflow.framework.DebuggedSourceFile DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tensorflow.framework.DebuggedSourceFile();
  }

  public static org.tensorflow.framework.DebuggedSourceFile getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<DebuggedSourceFile>
      PARSER = new com.google.protobuf.AbstractParser<DebuggedSourceFile>() {
    public DebuggedSourceFile parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new DebuggedSourceFile(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<DebuggedSourceFile> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<DebuggedSourceFile> getParserForType() {
    return PARSER;
  }

  public org.tensorflow.framework.DebuggedSourceFile getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

