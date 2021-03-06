// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: core/logging.proto

package tensorflow.serving;

public final class Logging {
  private Logging() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  public interface LogMetadataOrBuilder extends
      // @@protoc_insertion_point(interface_extends:tensorflow.serving.LogMetadata)
      com.google.protobuf.MessageOrBuilder {

    /**
     * <code>optional .tensorflow.serving.ModelSpec model_spec = 1;</code>
     */
    boolean hasModelSpec();
    /**
     * <code>optional .tensorflow.serving.ModelSpec model_spec = 1;</code>
     */
    tensorflow.serving.Model.ModelSpec getModelSpec();
    /**
     * <code>optional .tensorflow.serving.ModelSpec model_spec = 1;</code>
     */
    tensorflow.serving.Model.ModelSpecOrBuilder getModelSpecOrBuilder();

    /**
     * <pre>
     * TODO(b/33279154): Add more metadata as mentioned in the bug.
     * </pre>
     *
     * <code>optional .tensorflow.serving.SamplingConfig sampling_config = 2;</code>
     */
    boolean hasSamplingConfig();
    /**
     * <pre>
     * TODO(b/33279154): Add more metadata as mentioned in the bug.
     * </pre>
     *
     * <code>optional .tensorflow.serving.SamplingConfig sampling_config = 2;</code>
     */
    tensorflow.serving.LoggingConfigOuterClass.SamplingConfig getSamplingConfig();
    /**
     * <pre>
     * TODO(b/33279154): Add more metadata as mentioned in the bug.
     * </pre>
     *
     * <code>optional .tensorflow.serving.SamplingConfig sampling_config = 2;</code>
     */
    tensorflow.serving.LoggingConfigOuterClass.SamplingConfigOrBuilder getSamplingConfigOrBuilder();
  }
  /**
   * <pre>
   * Metadata logged along with the request logs.
   * </pre>
   *
   * Protobuf type {@code tensorflow.serving.LogMetadata}
   */
  public  static final class LogMetadata extends
      com.google.protobuf.GeneratedMessageV3 implements
      // @@protoc_insertion_point(message_implements:tensorflow.serving.LogMetadata)
      LogMetadataOrBuilder {
    // Use LogMetadata.newBuilder() to construct.
    private LogMetadata(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
      super(builder);
    }
    private LogMetadata() {
    }

    @java.lang.Override
    public final com.google.protobuf.UnknownFieldSet
    getUnknownFields() {
      return com.google.protobuf.UnknownFieldSet.getDefaultInstance();
    }
    private LogMetadata(
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
              tensorflow.serving.Model.ModelSpec.Builder subBuilder = null;
              if (modelSpec_ != null) {
                subBuilder = modelSpec_.toBuilder();
              }
              modelSpec_ = input.readMessage(tensorflow.serving.Model.ModelSpec.parser(), extensionRegistry);
              if (subBuilder != null) {
                subBuilder.mergeFrom(modelSpec_);
                modelSpec_ = subBuilder.buildPartial();
              }

              break;
            }
            case 18: {
              tensorflow.serving.LoggingConfigOuterClass.SamplingConfig.Builder subBuilder = null;
              if (samplingConfig_ != null) {
                subBuilder = samplingConfig_.toBuilder();
              }
              samplingConfig_ = input.readMessage(tensorflow.serving.LoggingConfigOuterClass.SamplingConfig.parser(), extensionRegistry);
              if (subBuilder != null) {
                subBuilder.mergeFrom(samplingConfig_);
                samplingConfig_ = subBuilder.buildPartial();
              }

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
      return tensorflow.serving.Logging.internal_static_tensorflow_serving_LogMetadata_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return tensorflow.serving.Logging.internal_static_tensorflow_serving_LogMetadata_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              tensorflow.serving.Logging.LogMetadata.class, tensorflow.serving.Logging.LogMetadata.Builder.class);
    }

    public static final int MODEL_SPEC_FIELD_NUMBER = 1;
    private tensorflow.serving.Model.ModelSpec modelSpec_;
    /**
     * <code>optional .tensorflow.serving.ModelSpec model_spec = 1;</code>
     */
    public boolean hasModelSpec() {
      return modelSpec_ != null;
    }
    /**
     * <code>optional .tensorflow.serving.ModelSpec model_spec = 1;</code>
     */
    public tensorflow.serving.Model.ModelSpec getModelSpec() {
      return modelSpec_ == null ? tensorflow.serving.Model.ModelSpec.getDefaultInstance() : modelSpec_;
    }
    /**
     * <code>optional .tensorflow.serving.ModelSpec model_spec = 1;</code>
     */
    public tensorflow.serving.Model.ModelSpecOrBuilder getModelSpecOrBuilder() {
      return getModelSpec();
    }

    public static final int SAMPLING_CONFIG_FIELD_NUMBER = 2;
    private tensorflow.serving.LoggingConfigOuterClass.SamplingConfig samplingConfig_;
    /**
     * <pre>
     * TODO(b/33279154): Add more metadata as mentioned in the bug.
     * </pre>
     *
     * <code>optional .tensorflow.serving.SamplingConfig sampling_config = 2;</code>
     */
    public boolean hasSamplingConfig() {
      return samplingConfig_ != null;
    }
    /**
     * <pre>
     * TODO(b/33279154): Add more metadata as mentioned in the bug.
     * </pre>
     *
     * <code>optional .tensorflow.serving.SamplingConfig sampling_config = 2;</code>
     */
    public tensorflow.serving.LoggingConfigOuterClass.SamplingConfig getSamplingConfig() {
      return samplingConfig_ == null ? tensorflow.serving.LoggingConfigOuterClass.SamplingConfig.getDefaultInstance() : samplingConfig_;
    }
    /**
     * <pre>
     * TODO(b/33279154): Add more metadata as mentioned in the bug.
     * </pre>
     *
     * <code>optional .tensorflow.serving.SamplingConfig sampling_config = 2;</code>
     */
    public tensorflow.serving.LoggingConfigOuterClass.SamplingConfigOrBuilder getSamplingConfigOrBuilder() {
      return getSamplingConfig();
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
      if (modelSpec_ != null) {
        output.writeMessage(1, getModelSpec());
      }
      if (samplingConfig_ != null) {
        output.writeMessage(2, getSamplingConfig());
      }
    }

    public int getSerializedSize() {
      int size = memoizedSize;
      if (size != -1) return size;

      size = 0;
      if (modelSpec_ != null) {
        size += com.google.protobuf.CodedOutputStream
          .computeMessageSize(1, getModelSpec());
      }
      if (samplingConfig_ != null) {
        size += com.google.protobuf.CodedOutputStream
          .computeMessageSize(2, getSamplingConfig());
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
      if (!(obj instanceof tensorflow.serving.Logging.LogMetadata)) {
        return super.equals(obj);
      }
      tensorflow.serving.Logging.LogMetadata other = (tensorflow.serving.Logging.LogMetadata) obj;

      boolean result = true;
      result = result && (hasModelSpec() == other.hasModelSpec());
      if (hasModelSpec()) {
        result = result && getModelSpec()
            .equals(other.getModelSpec());
      }
      result = result && (hasSamplingConfig() == other.hasSamplingConfig());
      if (hasSamplingConfig()) {
        result = result && getSamplingConfig()
            .equals(other.getSamplingConfig());
      }
      return result;
    }

    @java.lang.Override
    public int hashCode() {
      if (memoizedHashCode != 0) {
        return memoizedHashCode;
      }
      int hash = 41;
      hash = (19 * hash) + getDescriptorForType().hashCode();
      if (hasModelSpec()) {
        hash = (37 * hash) + MODEL_SPEC_FIELD_NUMBER;
        hash = (53 * hash) + getModelSpec().hashCode();
      }
      if (hasSamplingConfig()) {
        hash = (37 * hash) + SAMPLING_CONFIG_FIELD_NUMBER;
        hash = (53 * hash) + getSamplingConfig().hashCode();
      }
      hash = (29 * hash) + unknownFields.hashCode();
      memoizedHashCode = hash;
      return hash;
    }

    public static tensorflow.serving.Logging.LogMetadata parseFrom(
        com.google.protobuf.ByteString data)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data);
    }
    public static tensorflow.serving.Logging.LogMetadata parseFrom(
        com.google.protobuf.ByteString data,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data, extensionRegistry);
    }
    public static tensorflow.serving.Logging.LogMetadata parseFrom(byte[] data)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data);
    }
    public static tensorflow.serving.Logging.LogMetadata parseFrom(
        byte[] data,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data, extensionRegistry);
    }
    public static tensorflow.serving.Logging.LogMetadata parseFrom(java.io.InputStream input)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseWithIOException(PARSER, input);
    }
    public static tensorflow.serving.Logging.LogMetadata parseFrom(
        java.io.InputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseWithIOException(PARSER, input, extensionRegistry);
    }
    public static tensorflow.serving.Logging.LogMetadata parseDelimitedFrom(java.io.InputStream input)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseDelimitedWithIOException(PARSER, input);
    }
    public static tensorflow.serving.Logging.LogMetadata parseDelimitedFrom(
        java.io.InputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
    }
    public static tensorflow.serving.Logging.LogMetadata parseFrom(
        com.google.protobuf.CodedInputStream input)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseWithIOException(PARSER, input);
    }
    public static tensorflow.serving.Logging.LogMetadata parseFrom(
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
    public static Builder newBuilder(tensorflow.serving.Logging.LogMetadata prototype) {
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
     * <pre>
     * Metadata logged along with the request logs.
     * </pre>
     *
     * Protobuf type {@code tensorflow.serving.LogMetadata}
     */
    public static final class Builder extends
        com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
        // @@protoc_insertion_point(builder_implements:tensorflow.serving.LogMetadata)
        tensorflow.serving.Logging.LogMetadataOrBuilder {
      public static final com.google.protobuf.Descriptors.Descriptor
          getDescriptor() {
        return tensorflow.serving.Logging.internal_static_tensorflow_serving_LogMetadata_descriptor;
      }

      protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
          internalGetFieldAccessorTable() {
        return tensorflow.serving.Logging.internal_static_tensorflow_serving_LogMetadata_fieldAccessorTable
            .ensureFieldAccessorsInitialized(
                tensorflow.serving.Logging.LogMetadata.class, tensorflow.serving.Logging.LogMetadata.Builder.class);
      }

      // Construct using tensorflow.serving.Logging.LogMetadata.newBuilder()
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
        if (modelSpecBuilder_ == null) {
          modelSpec_ = null;
        } else {
          modelSpec_ = null;
          modelSpecBuilder_ = null;
        }
        if (samplingConfigBuilder_ == null) {
          samplingConfig_ = null;
        } else {
          samplingConfig_ = null;
          samplingConfigBuilder_ = null;
        }
        return this;
      }

      public com.google.protobuf.Descriptors.Descriptor
          getDescriptorForType() {
        return tensorflow.serving.Logging.internal_static_tensorflow_serving_LogMetadata_descriptor;
      }

      public tensorflow.serving.Logging.LogMetadata getDefaultInstanceForType() {
        return tensorflow.serving.Logging.LogMetadata.getDefaultInstance();
      }

      public tensorflow.serving.Logging.LogMetadata build() {
        tensorflow.serving.Logging.LogMetadata result = buildPartial();
        if (!result.isInitialized()) {
          throw newUninitializedMessageException(result);
        }
        return result;
      }

      public tensorflow.serving.Logging.LogMetadata buildPartial() {
        tensorflow.serving.Logging.LogMetadata result = new tensorflow.serving.Logging.LogMetadata(this);
        if (modelSpecBuilder_ == null) {
          result.modelSpec_ = modelSpec_;
        } else {
          result.modelSpec_ = modelSpecBuilder_.build();
        }
        if (samplingConfigBuilder_ == null) {
          result.samplingConfig_ = samplingConfig_;
        } else {
          result.samplingConfig_ = samplingConfigBuilder_.build();
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
        if (other instanceof tensorflow.serving.Logging.LogMetadata) {
          return mergeFrom((tensorflow.serving.Logging.LogMetadata)other);
        } else {
          super.mergeFrom(other);
          return this;
        }
      }

      public Builder mergeFrom(tensorflow.serving.Logging.LogMetadata other) {
        if (other == tensorflow.serving.Logging.LogMetadata.getDefaultInstance()) return this;
        if (other.hasModelSpec()) {
          mergeModelSpec(other.getModelSpec());
        }
        if (other.hasSamplingConfig()) {
          mergeSamplingConfig(other.getSamplingConfig());
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
        tensorflow.serving.Logging.LogMetadata parsedMessage = null;
        try {
          parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
        } catch (com.google.protobuf.InvalidProtocolBufferException e) {
          parsedMessage = (tensorflow.serving.Logging.LogMetadata) e.getUnfinishedMessage();
          throw e.unwrapIOException();
        } finally {
          if (parsedMessage != null) {
            mergeFrom(parsedMessage);
          }
        }
        return this;
      }

      private tensorflow.serving.Model.ModelSpec modelSpec_ = null;
      private com.google.protobuf.SingleFieldBuilderV3<
          tensorflow.serving.Model.ModelSpec, tensorflow.serving.Model.ModelSpec.Builder, tensorflow.serving.Model.ModelSpecOrBuilder> modelSpecBuilder_;
      /**
       * <code>optional .tensorflow.serving.ModelSpec model_spec = 1;</code>
       */
      public boolean hasModelSpec() {
        return modelSpecBuilder_ != null || modelSpec_ != null;
      }
      /**
       * <code>optional .tensorflow.serving.ModelSpec model_spec = 1;</code>
       */
      public tensorflow.serving.Model.ModelSpec getModelSpec() {
        if (modelSpecBuilder_ == null) {
          return modelSpec_ == null ? tensorflow.serving.Model.ModelSpec.getDefaultInstance() : modelSpec_;
        } else {
          return modelSpecBuilder_.getMessage();
        }
      }
      /**
       * <code>optional .tensorflow.serving.ModelSpec model_spec = 1;</code>
       */
      public Builder setModelSpec(tensorflow.serving.Model.ModelSpec value) {
        if (modelSpecBuilder_ == null) {
          if (value == null) {
            throw new NullPointerException();
          }
          modelSpec_ = value;
          onChanged();
        } else {
          modelSpecBuilder_.setMessage(value);
        }

        return this;
      }
      /**
       * <code>optional .tensorflow.serving.ModelSpec model_spec = 1;</code>
       */
      public Builder setModelSpec(
          tensorflow.serving.Model.ModelSpec.Builder builderForValue) {
        if (modelSpecBuilder_ == null) {
          modelSpec_ = builderForValue.build();
          onChanged();
        } else {
          modelSpecBuilder_.setMessage(builderForValue.build());
        }

        return this;
      }
      /**
       * <code>optional .tensorflow.serving.ModelSpec model_spec = 1;</code>
       */
      public Builder mergeModelSpec(tensorflow.serving.Model.ModelSpec value) {
        if (modelSpecBuilder_ == null) {
          if (modelSpec_ != null) {
            modelSpec_ =
              tensorflow.serving.Model.ModelSpec.newBuilder(modelSpec_).mergeFrom(value).buildPartial();
          } else {
            modelSpec_ = value;
          }
          onChanged();
        } else {
          modelSpecBuilder_.mergeFrom(value);
        }

        return this;
      }
      /**
       * <code>optional .tensorflow.serving.ModelSpec model_spec = 1;</code>
       */
      public Builder clearModelSpec() {
        if (modelSpecBuilder_ == null) {
          modelSpec_ = null;
          onChanged();
        } else {
          modelSpec_ = null;
          modelSpecBuilder_ = null;
        }

        return this;
      }
      /**
       * <code>optional .tensorflow.serving.ModelSpec model_spec = 1;</code>
       */
      public tensorflow.serving.Model.ModelSpec.Builder getModelSpecBuilder() {
        
        onChanged();
        return getModelSpecFieldBuilder().getBuilder();
      }
      /**
       * <code>optional .tensorflow.serving.ModelSpec model_spec = 1;</code>
       */
      public tensorflow.serving.Model.ModelSpecOrBuilder getModelSpecOrBuilder() {
        if (modelSpecBuilder_ != null) {
          return modelSpecBuilder_.getMessageOrBuilder();
        } else {
          return modelSpec_ == null ?
              tensorflow.serving.Model.ModelSpec.getDefaultInstance() : modelSpec_;
        }
      }
      /**
       * <code>optional .tensorflow.serving.ModelSpec model_spec = 1;</code>
       */
      private com.google.protobuf.SingleFieldBuilderV3<
          tensorflow.serving.Model.ModelSpec, tensorflow.serving.Model.ModelSpec.Builder, tensorflow.serving.Model.ModelSpecOrBuilder> 
          getModelSpecFieldBuilder() {
        if (modelSpecBuilder_ == null) {
          modelSpecBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
              tensorflow.serving.Model.ModelSpec, tensorflow.serving.Model.ModelSpec.Builder, tensorflow.serving.Model.ModelSpecOrBuilder>(
                  getModelSpec(),
                  getParentForChildren(),
                  isClean());
          modelSpec_ = null;
        }
        return modelSpecBuilder_;
      }

      private tensorflow.serving.LoggingConfigOuterClass.SamplingConfig samplingConfig_ = null;
      private com.google.protobuf.SingleFieldBuilderV3<
          tensorflow.serving.LoggingConfigOuterClass.SamplingConfig, tensorflow.serving.LoggingConfigOuterClass.SamplingConfig.Builder, tensorflow.serving.LoggingConfigOuterClass.SamplingConfigOrBuilder> samplingConfigBuilder_;
      /**
       * <pre>
       * TODO(b/33279154): Add more metadata as mentioned in the bug.
       * </pre>
       *
       * <code>optional .tensorflow.serving.SamplingConfig sampling_config = 2;</code>
       */
      public boolean hasSamplingConfig() {
        return samplingConfigBuilder_ != null || samplingConfig_ != null;
      }
      /**
       * <pre>
       * TODO(b/33279154): Add more metadata as mentioned in the bug.
       * </pre>
       *
       * <code>optional .tensorflow.serving.SamplingConfig sampling_config = 2;</code>
       */
      public tensorflow.serving.LoggingConfigOuterClass.SamplingConfig getSamplingConfig() {
        if (samplingConfigBuilder_ == null) {
          return samplingConfig_ == null ? tensorflow.serving.LoggingConfigOuterClass.SamplingConfig.getDefaultInstance() : samplingConfig_;
        } else {
          return samplingConfigBuilder_.getMessage();
        }
      }
      /**
       * <pre>
       * TODO(b/33279154): Add more metadata as mentioned in the bug.
       * </pre>
       *
       * <code>optional .tensorflow.serving.SamplingConfig sampling_config = 2;</code>
       */
      public Builder setSamplingConfig(tensorflow.serving.LoggingConfigOuterClass.SamplingConfig value) {
        if (samplingConfigBuilder_ == null) {
          if (value == null) {
            throw new NullPointerException();
          }
          samplingConfig_ = value;
          onChanged();
        } else {
          samplingConfigBuilder_.setMessage(value);
        }

        return this;
      }
      /**
       * <pre>
       * TODO(b/33279154): Add more metadata as mentioned in the bug.
       * </pre>
       *
       * <code>optional .tensorflow.serving.SamplingConfig sampling_config = 2;</code>
       */
      public Builder setSamplingConfig(
          tensorflow.serving.LoggingConfigOuterClass.SamplingConfig.Builder builderForValue) {
        if (samplingConfigBuilder_ == null) {
          samplingConfig_ = builderForValue.build();
          onChanged();
        } else {
          samplingConfigBuilder_.setMessage(builderForValue.build());
        }

        return this;
      }
      /**
       * <pre>
       * TODO(b/33279154): Add more metadata as mentioned in the bug.
       * </pre>
       *
       * <code>optional .tensorflow.serving.SamplingConfig sampling_config = 2;</code>
       */
      public Builder mergeSamplingConfig(tensorflow.serving.LoggingConfigOuterClass.SamplingConfig value) {
        if (samplingConfigBuilder_ == null) {
          if (samplingConfig_ != null) {
            samplingConfig_ =
              tensorflow.serving.LoggingConfigOuterClass.SamplingConfig.newBuilder(samplingConfig_).mergeFrom(value).buildPartial();
          } else {
            samplingConfig_ = value;
          }
          onChanged();
        } else {
          samplingConfigBuilder_.mergeFrom(value);
        }

        return this;
      }
      /**
       * <pre>
       * TODO(b/33279154): Add more metadata as mentioned in the bug.
       * </pre>
       *
       * <code>optional .tensorflow.serving.SamplingConfig sampling_config = 2;</code>
       */
      public Builder clearSamplingConfig() {
        if (samplingConfigBuilder_ == null) {
          samplingConfig_ = null;
          onChanged();
        } else {
          samplingConfig_ = null;
          samplingConfigBuilder_ = null;
        }

        return this;
      }
      /**
       * <pre>
       * TODO(b/33279154): Add more metadata as mentioned in the bug.
       * </pre>
       *
       * <code>optional .tensorflow.serving.SamplingConfig sampling_config = 2;</code>
       */
      public tensorflow.serving.LoggingConfigOuterClass.SamplingConfig.Builder getSamplingConfigBuilder() {
        
        onChanged();
        return getSamplingConfigFieldBuilder().getBuilder();
      }
      /**
       * <pre>
       * TODO(b/33279154): Add more metadata as mentioned in the bug.
       * </pre>
       *
       * <code>optional .tensorflow.serving.SamplingConfig sampling_config = 2;</code>
       */
      public tensorflow.serving.LoggingConfigOuterClass.SamplingConfigOrBuilder getSamplingConfigOrBuilder() {
        if (samplingConfigBuilder_ != null) {
          return samplingConfigBuilder_.getMessageOrBuilder();
        } else {
          return samplingConfig_ == null ?
              tensorflow.serving.LoggingConfigOuterClass.SamplingConfig.getDefaultInstance() : samplingConfig_;
        }
      }
      /**
       * <pre>
       * TODO(b/33279154): Add more metadata as mentioned in the bug.
       * </pre>
       *
       * <code>optional .tensorflow.serving.SamplingConfig sampling_config = 2;</code>
       */
      private com.google.protobuf.SingleFieldBuilderV3<
          tensorflow.serving.LoggingConfigOuterClass.SamplingConfig, tensorflow.serving.LoggingConfigOuterClass.SamplingConfig.Builder, tensorflow.serving.LoggingConfigOuterClass.SamplingConfigOrBuilder> 
          getSamplingConfigFieldBuilder() {
        if (samplingConfigBuilder_ == null) {
          samplingConfigBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
              tensorflow.serving.LoggingConfigOuterClass.SamplingConfig, tensorflow.serving.LoggingConfigOuterClass.SamplingConfig.Builder, tensorflow.serving.LoggingConfigOuterClass.SamplingConfigOrBuilder>(
                  getSamplingConfig(),
                  getParentForChildren(),
                  isClean());
          samplingConfig_ = null;
        }
        return samplingConfigBuilder_;
      }
      public final Builder setUnknownFields(
          final com.google.protobuf.UnknownFieldSet unknownFields) {
        return this;
      }

      public final Builder mergeUnknownFields(
          final com.google.protobuf.UnknownFieldSet unknownFields) {
        return this;
      }


      // @@protoc_insertion_point(builder_scope:tensorflow.serving.LogMetadata)
    }

    // @@protoc_insertion_point(class_scope:tensorflow.serving.LogMetadata)
    private static final tensorflow.serving.Logging.LogMetadata DEFAULT_INSTANCE;
    static {
      DEFAULT_INSTANCE = new tensorflow.serving.Logging.LogMetadata();
    }

    public static tensorflow.serving.Logging.LogMetadata getDefaultInstance() {
      return DEFAULT_INSTANCE;
    }

    private static final com.google.protobuf.Parser<LogMetadata>
        PARSER = new com.google.protobuf.AbstractParser<LogMetadata>() {
      public LogMetadata parsePartialFrom(
          com.google.protobuf.CodedInputStream input,
          com.google.protobuf.ExtensionRegistryLite extensionRegistry)
          throws com.google.protobuf.InvalidProtocolBufferException {
          return new LogMetadata(input, extensionRegistry);
      }
    };

    public static com.google.protobuf.Parser<LogMetadata> parser() {
      return PARSER;
    }

    @java.lang.Override
    public com.google.protobuf.Parser<LogMetadata> getParserForType() {
      return PARSER;
    }

    public tensorflow.serving.Logging.LogMetadata getDefaultInstanceForType() {
      return DEFAULT_INSTANCE;
    }

  }

  private static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tensorflow_serving_LogMetadata_descriptor;
  private static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tensorflow_serving_LogMetadata_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\022core/logging.proto\022\022tensorflow.serving" +
      "\032\020apis/model.proto\032\033config/logging_confi" +
      "g.proto\"}\n\013LogMetadata\0221\n\nmodel_spec\030\001 \001" +
      "(\0132\035.tensorflow.serving.ModelSpec\022;\n\017sam" +
      "pling_config\030\002 \001(\0132\".tensorflow.serving." +
      "SamplingConfigB\003\370\001\001b\006proto3"
    };
    com.google.protobuf.Descriptors.FileDescriptor.InternalDescriptorAssigner assigner =
        new com.google.protobuf.Descriptors.FileDescriptor.    InternalDescriptorAssigner() {
          public com.google.protobuf.ExtensionRegistry assignDescriptors(
              com.google.protobuf.Descriptors.FileDescriptor root) {
            descriptor = root;
            return null;
          }
        };
    com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
          tensorflow.serving.Model.getDescriptor(),
          tensorflow.serving.LoggingConfigOuterClass.getDescriptor(),
        }, assigner);
    internal_static_tensorflow_serving_LogMetadata_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_tensorflow_serving_LogMetadata_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tensorflow_serving_LogMetadata_descriptor,
        new java.lang.String[] { "ModelSpec", "SamplingConfig", });
    tensorflow.serving.Model.getDescriptor();
    tensorflow.serving.LoggingConfigOuterClass.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}
