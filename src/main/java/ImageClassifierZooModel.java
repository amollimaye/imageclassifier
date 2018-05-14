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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.*;

public class ImageClassifierZooModel {

    public static Map<String,List<String>> labels;

    public static List predictionLabels;

    static int maxNumberOfFilesToScan = 20; //Number of files to scan

    static boolean detailedLabels = false; // If set to true, this will print top 5 predictions for each image
    //Else it will gather the top predictions of each image and group the images by label

    static String folderName = "C://cars_test"; //Folder to scan for files

    public static void main(String[] args) throws IOException {

//        File testFile = new File("/testImages/audi.jpg");
//        if(!testFile.exists()){
//            throw new FileNotFoundException("File not found" +testFile.getAbsolutePath());
//        }

        // set up model
        ZooModel zooModel = new VGG16();
        ComputationGraph  vgg16 = (ComputationGraph)
                zooModel.initPretrained(PretrainedType.IMAGENET);
        initLabels();
        File folder = new File(folderName);
        if(folder.listFiles().length<1){
            System.out.println("No files to scan !");
            System.exit(1);
        }
        int count = 0;
        for(File file:folder.listFiles()){
            if(file.isDirectory()){
                continue;
            }
            try {
                labelImage(vgg16,file);
            } catch (IOException e) {
                System.out.println(e.getLocalizedMessage());
            }
            count++;
            if(count >=maxNumberOfFilesToScan){
                 break;
            }
        }
        if(!detailedLabels) {
            printLabelledFiles();
        }
    }

    public static void printLabelledFiles(){
        for(String label:labels.keySet()){
            System.out.println("---- "+label+"---"+labels.get(label).size()+" files");
            for(String fileName:labels.get(label)){
                System.out.println(fileName);
            }
            System.out.println();
        }
    }

    public static void addLabel(String label,String fileName){
        if(labels==null){
            labels = new HashMap<String, List<String>>();
        }
        List<String> images = labels.get(label);
        if(images==null){
            images = new ArrayList<String>();
        }
        images.add(fileName);
        labels.put(label,images);
    }

    public static void labelImage(ComputationGraph  vgg16,File testFile) throws IOException {
        System.out.println("Scanning file "+testFile.getAbsolutePath());
        // set up input and feedforward
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        ClassLoader classloader = Thread.currentThread().getContextClassLoader();
        INDArray image = loader.asMatrix(testFile);
        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);
        INDArray[] output = vgg16.output(false, image);

        // check output labels of result
        if(detailedLabels){
            String decodedLabels =
                new ImageNetLabels().
                decodePredictions(output[0]);
            System.out.println(decodedLabels);
            return;
        }
        String decodedLabels = decodePredictions(output[0]);
        System.out.println(decodedLabels);
        addLabel(decodedLabels,testFile.getName());
    }

    public static String decodePredictions(INDArray predictions) {
        String predictionDescription = "";
        int[] top5 = new int[1];
        float[] top5Prob = new float[1];
        int i = 0;

        for(int batch = 0; batch < predictions.size(0); ++batch) {
//            predictionDescription = predictionDescription + "Predictions for batch ";
//            if (predictions.size(0) > 1) {
//                predictionDescription = predictionDescription + String.valueOf(batch);
//            }

//            predictionDescription = predictionDescription + " :";

            for(INDArray currentBatch = predictions.getRow(batch).dup(); i < 1; ++i) {
                top5[i] = Nd4j.argMax(currentBatch, new int[]{1}).getInt(new int[]{0, 0});
                top5Prob[i] = currentBatch.getFloat(batch, top5[i]);
                currentBatch.putScalar(0, top5[i], 0.0D);
                predictionDescription = (String)predictionLabels.get(top5[i]);
            }
        }

        return predictionDescription;
    }

    public static void initLabels() {

        if (predictionLabels == null) {
            try {
                HashMap<String, ArrayList<String>> jsonMap = (HashMap)(new ObjectMapper()).readValue(new URL("http://blob.deeplearning4j.org/utils/imagenet_class_index.json"), HashMap.class);
                predictionLabels = new ArrayList(jsonMap.size());

                for(int i = 0; i < jsonMap.size(); ++i) {
                    predictionLabels.add(((ArrayList)jsonMap.get(String.valueOf(i))).get(1));
                }
            } catch (IOException var2) {
                var2.printStackTrace();
            }
        }
    }
}
