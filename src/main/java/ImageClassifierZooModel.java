import org.bytedeco.javacpp.Loader;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
//import org.deeplearning4j.nn.modelimport.keras.trainedmodels.Utils.ImageNetLabels;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.util.imagenet.ImageNetLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

public class ImageClassifierZooModel {
    public static void main(String[] args) throws IOException {

        File testFile = new File("/testImages/audi.jpg");
        if(!testFile.exists()){
            throw new FileNotFoundException("File not found" +testFile.getAbsolutePath());
        }

//        try {
//            Loader.load(jnicuda.class);
//        } catch (UnsatisfiedLinkError e) {
//            String path = Loader.cacheResource(<module>.class, "windows-x86_64/jni<module>.dll").getPath();
//            new ProcessBuilder("c:/path/to/depends.exe", path).start().waitFor();
//        }

        // set up model
        ZooModel zooModel = new VGG16();
        ComputationGraph  vgg16 = (ComputationGraph)
                zooModel.initPretrained(PretrainedType.IMAGENET);

        // set up input and feedforward
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        ClassLoader classloader = Thread.currentThread().getContextClassLoader();
        INDArray image = loader.asMatrix(testFile);
        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);
        INDArray[] output = vgg16.output(false, image);

        // check output labels of result
        String decodedLabels = new ImageNetLabels().decodePredictions(output[0]);
        System.out.println(decodedLabels);
    }
}
