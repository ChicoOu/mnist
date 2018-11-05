package com.zq.controller;

import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;

import com.google.protobuf.Int64Value;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

@RestController
@RequestMapping("/mnist")
public class MnistController {
	
	@Value("${serving.ip}")
	private String ip;
	
	@Value("${serving.port}")
	private int port;
	
	@ResponseBody
	@RequestMapping("predict")
	public Map<String, Object> predict() throws Exception{
		ManagedChannel channel = ManagedChannelBuilder.forAddress(ip, port).usePlaintext().build();
        PredictionServiceGrpc.PredictionServiceBlockingStub blockingStub = PredictionServiceGrpc.newBlockingStub(channel);
        Model.ModelSpec modelSpec = Model.ModelSpec.newBuilder().setName("mnist").setVersion(Int64Value.newBuilder().setValue(1l).build()).setSignatureName("mnist_pred").build();
        Predict.PredictRequest.Builder predictRequestBuilder = Predict.PredictRequest.newBuilder();
        predictRequestBuilder.setModelSpec(modelSpec);
        
        List<Float> floatList = readImage("./train_17.bmp");
        //设置入参,访问默认是最新版本，如果需要特定版本可以使用tensorProtoBuilder.setVersionNumber方法
        TensorProto.Builder tensorProtoBuilder = TensorProto.newBuilder();
        tensorProtoBuilder.setDtype(DataType.DT_FLOAT);
        TensorShapeProto.Builder tensorShapeBuilder = TensorShapeProto.newBuilder();
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(1));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(28));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(28));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(1));
        tensorProtoBuilder.setTensorShape(tensorShapeBuilder.build());
        tensorProtoBuilder.addAllFloatVal(floatList);
        predictRequestBuilder.putInputs("image", tensorProtoBuilder.build());

        Predict.PredictRequest request = predictRequestBuilder.build();
        Predict.PredictResponse predictResponse = blockingStub.predict(request);
        Map<String, TensorProto> tensorProtoMap = predictResponse.getOutputsMap();
        TensorProto proto = tensorProtoMap.get("result");
        List<Float> probs = proto.getFloatValList();
        float maxProb = 0;
        int index = 0;
        int result = 0;
        for( Float prob : probs ) {
        	if( prob > maxProb ) {
        		maxProb = prob;
        		result = index;
        	}
        	index++;
        }
        
        Map<String, Object> t = new HashMap<String, Object>();
        t.put("result", result);
		return t;
	}
	
	private List<Float> readImage(String filePath) throws Exception{
		//读取文件，强制修改图片大小，设置输出文件格式bmp(模型定义时输入数据是无编码的)
		BufferedImage im = ImageIO.read(new File(filePath));
		//转换图片到图片数组，匹配输入数据类型为Float
		Raster raster = im.getData();
		List<Float> floatList = new ArrayList<>();
		float [] temp = new float[raster.getWidth() * raster.getHeight() * raster.getNumBands()];
		float [] pixels  = raster.getPixels(0,0,raster.getWidth(),raster.getHeight(),temp);
		for(float pixel: pixels) {
		    floatList.add(pixel);
		}
		
		return floatList;
	}
}
