// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: apis/prediction_service.proto

package tensorflow.serving;

public final class PredictionServiceOuterClass {
  private PredictionServiceOuterClass() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\035apis/prediction_service.proto\022\022tensorf" +
      "low.serving\032\031apis/classification.proto\032\035" +
      "apis/get_model_metadata.proto\032\024apis/infe" +
      "rence.proto\032\022apis/predict.proto\032\025apis/re" +
      "gression.proto2\374\003\n\021PredictionService\022a\n\010" +
      "Classify\022).tensorflow.serving.Classifica" +
      "tionRequest\032*.tensorflow.serving.Classif" +
      "icationResponse\022X\n\007Regress\022%.tensorflow." +
      "serving.RegressionRequest\032&.tensorflow.s" +
      "erving.RegressionResponse\022R\n\007Predict\022\".t",
      "ensorflow.serving.PredictRequest\032#.tenso" +
      "rflow.serving.PredictResponse\022g\n\016MultiIn" +
      "ference\022).tensorflow.serving.MultiInfere" +
      "nceRequest\032*.tensorflow.serving.MultiInf" +
      "erenceResponse\022m\n\020GetModelMetadata\022+.ten" +
      "sorflow.serving.GetModelMetadataRequest\032" +
      ",.tensorflow.serving.GetModelMetadataRes" +
      "ponseB\003\370\001\001b\006proto3"
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
          tensorflow.serving.Classification.getDescriptor(),
          tensorflow.serving.GetModelMetadata.getDescriptor(),
          tensorflow.serving.Inference.getDescriptor(),
          tensorflow.serving.Predict.getDescriptor(),
          tensorflow.serving.RegressionOuterClass.getDescriptor(),
        }, assigner);
    tensorflow.serving.Classification.getDescriptor();
    tensorflow.serving.GetModelMetadata.getDescriptor();
    tensorflow.serving.Inference.getDescriptor();
    tensorflow.serving.Predict.getDescriptor();
    tensorflow.serving.RegressionOuterClass.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}
