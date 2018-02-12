/**
 * Example from: https://github.com/eugenp/tutorials/tree/master/deeplearning4j
 */

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

public class IrisClassifier {

    // parameters of the Iris dataset
    private static final int nClasses = 3;
    private static final int nFeatures = 4;
    private static final int nInstances = 150;
    
    // hyper-parameters for training
    private static final int seed = 123;
    private static final double learningRate = 0.01;
    private static final int nEpochs = 5;

    // network architecture
    private static final int nHiddenNodes_1 = 3;  //จำนวนhidden node
    private static final int nHiddenNodes_2 = 3;

    public static void main(String[] args) throws IOException, InterruptedException {

        DataSet allData, trainingData, testData;

        // load data from resource file into a DataSet object
        try (RecordReader recordReader = new CSVRecordReader(0,',')) {
            recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));

            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 
                                                                       nInstances, /*read all instances*/ 
                                                                       nFeatures,  /*index of class label*/
                                                                       nClasses);
            allData = iterator.next();
        }

        // pre-process the data
        allData.shuffle(seed);
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(allData);
        normalizer.transform(allData);

        // split the data into "training" and "validation" sets
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
        trainingData = testAndTrain.getTrain();
        testData = testAndTrain.getTest();

        // design neural network architecture
        MultiLayerConfiguration netConfig = new NeuralNetConfiguration.Builder()
            .seed(seed) //จุดที่จะให้เริ่มshuffle
            .iterations(1000)
            .learningRate(learningRate)
            .list()
            .layer(0, new DenseLayer.Builder()  //layerที่ใช้hiddenNode   layer0,1คือจน.layerภายในNN
                            .nIn(nFeatures) //input
                            .nOut(nHiddenNodes_1) //outputออก
                            .activation(Activation.RELU) // algoที่ใช้
                            .build())
            .layer(1, new DenseLayer.Builder()
                            .nIn(nHiddenNodes_1)
                            .nOut(nHiddenNodes_2)
                            .activation(Activation.RELU)
                            .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD ) //layer output
                            .nIn(nHiddenNodes_2) //inputNode
                            .nOut(nClasses)
                            .activation(Activation.SOFTMAX)
                            .build())
            .pretrain(false).backprop(true).build();

        // initialize the neural net
        MultiLayerNetwork model = new MultiLayerNetwork(netConfig);
        model.init();

        // train!
        for (int i = 0; i < nEpochs; i++) {
            System.out.println("Training epoch " + (i+1) + "/" + nEpochs);
            model.fit(trainingData);
        }

        // Evaluate
        System.out.println("Evaluate model...");

        INDArray output = model.output(testData.getFeatureMatrix(),false);
        Evaluation eval = new Evaluation(nClasses);
        eval.eval(testData.getLabels(), output);
        System.out.println(eval.stats());

    }
}

