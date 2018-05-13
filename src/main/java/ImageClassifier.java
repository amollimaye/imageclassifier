import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.util.imagenet.ImageNetLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import java.io.File;
import java.io.IOException;

//import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;

//import static org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels.VGG16;

public class ImageClassifier {

    public static void main(String[] args) throws IOException {
        // https://github.com/tomthetrainer/KerasWorkshop/releases

        File savedNetwork = new File("C:\\Projects\\ML\\vgg16.zip");
//        System.out.println(savedNetwork.getAbsolutePath());
        ComputationGraph vgg16 = ModelSerializer.restoreComputationGraph(savedNetwork);

        File fileToTest = new File("C:\\testImages\\audi.jpg");

        // define native image loaders
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        INDArray image = loader.asMatrix(fileToTest);

        // Scale image in same manner as network was trained on
        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);
        INDArray[] output = vgg16.output(false,image);

//        String predictions = TrainedModels.VGG16.decodePredictions(output[0]);
        String predictions = new ImageNetLabels().decodePredictions(output[0]);
        System.out.println(predictions);
    }
}
