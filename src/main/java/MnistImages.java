import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Random;

public class MnistImages {
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
        int batchSize = 1;
        int outputNum = 10;

        // download the MNIST data and store it in ~/mnist_png/training
        downloadData();

        // Define the File Paths
        File trainData = new File(DATA_PATH + "/mnist_png/training");

        // Define the FileSplit
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        // Extract the parent path as the image label
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        // Construct and initialize the Image record reader
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(train);
        recordReader.setListeners(new LogRecordListener());

        // Data set iterator
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize,1, outputNum);

        // Scale the image pixel from [0..255] to [0..1]
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);


        for (int i = 1; i < 2; i++) {
            DataSet ds = dataIter.next();
            log.info(ds.toString());
            log.info(dataIter.getLabels().toString());
        }

    }


    protected static void downloadData() throws Exception {
        // Create directory if required
        File directory = new File(DATA_PATH);
        if (!directory.exists())
            directory.mkdir();

        // Download file:
        String archizePath = DATA_PATH + "/mnist_png.tar.gz";
        File archiveFile = new File(archizePath);
        String extractedPath = DATA_PATH + "mnist_png";
        File extractedFile = new File(extractedPath);

        if (!archiveFile.exists()) {
            log.info("Starting data download (15MB)...");
            getMnistPNG();
            //Extract tar.gz file to output directory
            DataUtilities.extractTarGz(archizePath, DATA_PATH);
        } else {
            //Assume if archive (.tar.gz) exists, then data has already been extracted
            log.info("Data (.tar.gz file) already exists at {}", archiveFile.getAbsolutePath());
            if (!extractedFile.exists()) {
                //Extract tar.gz file to output directory
                DataUtilities.extractTarGz(archizePath, DATA_PATH);
            } else {
                log.info("Data (extracted) already exists at {}", extractedFile.getAbsolutePath());
            }
        }
    }

    public static void getMnistPNG() throws IOException {
        String tmpDirStr = System.getProperty("java.io.tmpdir");
        String archizePath = DATA_PATH + "/mnist_png.tar.gz";

        if (tmpDirStr == null) {
            throw new IOException("System property 'java.io.tmpdir' does specify a tmp dir");
        }

        File f = new File(archizePath);
        if (!f.exists()) {
            DataUtilities.downloadFile(DATA_URL, archizePath);
            log.info("Data downloaded to ", archizePath);
        } else {
            log.info("Using existing directory at ", f.getAbsolutePath());
        }
    }


}
