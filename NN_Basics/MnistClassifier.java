package co.kulwadee.int492.lect04;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.eval.Evaluation;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.apache.commons.io.FilenameUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class MnistClassifier {
    private static Logger log = LoggerFactory.getLogger(MnistImages.class);

    public static final String DATA_URL = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";

    public static final String DATA_PATH = FilenameUtils.concat(
            System.getProperty("user.home"), "dl4j_mnist");

    public static void main(String[] args) throws Exception {
        //
        // MNIST Image: 28 x 28 grayscale
        //
        int height = 28;
        int width = 28;
        int channels = 1;
        int rngseed = 123;
        Random randNumGen = new Random(rngseed);
        int batchSize = 100;
        int outputNum = 10;
        int numEpochs = 4;
        int numHidden = 64;

        // download the MNIST data and store it in ~/mnist_png/training
        MnistImages.downloadData();

        // Define the File Paths
        File trainData = new File(DATA_PATH + "/mnist_png/training");
        File testData = new File(DATA_PATH + "/mnist_png/testing");

        // Define the FileSplit
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        // Extract the parent path as the image label
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        // Construct and initialize the Image record reader
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(train);

        // Data set iterator
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize,1, outputNum);

        // Scale the image pixel from [0..255] to [0..1]
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        // Design neural net architecture
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngseed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.01)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(height * width)
                        .nOut(numHidden)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(numHidden)
                        .nOut(numHidden)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                .nIn(numHidden)
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .build())
                .pretrain(false).backprop(true)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        log.info("BUILD MODEL");
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        // The ScoreIterationListener
        // will log output to show how well the network is training
        model.setListeners(new ScoreIterationListener(10));

        log.info("TRAIN MODEL");
        for (int i = 0; i < numEpochs; i++) {
            model.fit(dataIter);
        }

        log.info("EVALUATE MODEL");
        recordReader.reset();
        recordReader.initialize(test);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader,
                                            batchSize, 1, outputNum);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);

        Evaluation eval = new Evaluation(outputNum);
        while (testIter.hasNext()) {
            DataSet next = testIter.next();
            INDArray output = model.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), output);
        }

        log.info(eval.stats());

    }

}
