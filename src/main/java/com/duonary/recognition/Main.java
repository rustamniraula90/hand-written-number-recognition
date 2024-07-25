package com.duonary.recognition;

import com.duonary.recognition.network.MLPNetwork;
import com.duonary.recognition.network.MLPNetworkOptimized;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class Main {

    private static final Logger LOG = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) throws IOException {
        LOG.info("Preparing dataset...");
        Image[] dataset = prepareDataset("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
//        MLPNetwork network = MLPNetwork.builder(2)
//                .addInputLayer(dataset[0].getRows() * dataset[0].getColumns())
//                .addOutputLayer(10)
//                .addHiddenLayer(128)
//                .addHiddenLayer(64)
//                .setLearningRate(0.01)
//                .setBatchSize(32)
//                .setEpoch(20)
//                .build();

        MLPNetworkOptimized networkOptimized = MLPNetworkOptimized.builder(2)
                .addInputLayer(dataset[0].getRows() * dataset[0].getColumns())
                .addOutputLayer(10)
                .addHiddenLayer(128)
                .addHiddenLayer(64)
                .setLearningRate(0.001)
                .setBatchSize(10)
                .setEpoch(20)
                .build();

        LOG.info("Training network...");
        double[][] x = new double[dataset.length][];
        double[][] y = new double[dataset.length][];
        for (int i = 0; i < dataset.length; i++) {
            x[i] = transformInput(dataset[i]);
            y[i] = transformLabel(dataset[i]);
        }
        networkOptimized.train(x, y);
//        network.train(x, y);

        LOG.info("Testing network...");
        dataset = prepareDataset("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        int correct = 0;
        int incorrect = 0;
        Map<Integer, Integer> predictionMap = new HashMap<>();
        for (Image image : dataset) {
            double[] input = transformInput(image);
            double[] output = networkOptimized.predict(input);
            int predicted = 0;
            double max = output[0];
            for (int j = 1; j < output.length; j++) {
                if (output[j] > max) {
                    max = output[j];
                    predicted = j;
                }
            }
            if (predictionMap.containsKey(predicted)) {
                predictionMap.put(predicted, predictionMap.get(predicted) + 1);
            } else {
                predictionMap.put(predicted, 1);
            }
            if (predicted == image.label()) {
                correct++;
            } else {
                incorrect++;
            }
        }

        LOG.info("Correct: {}", correct);
        LOG.info("Incorrect: {}", incorrect);
        LOG.info("Accuracy: {}", (double) correct / (correct + incorrect));
        LOG.info("Prediction: {}", predictionMap);

    }

    private static double[] transformInput(Image image) {
        double[] input = new double[image.getRows() * image.getColumns()];
        for (int r = 0; r < image.getRows(); r++) {
            for (int c = 0; c < image.getColumns(); c++) {
                input[r * image.getColumns() + c] = (image.getPixel(r, c) / 254.0);
            }
        }
        return input;
    }

    private static double[] transformLabel(Image image) {
        double[] output = new double[10];
        output[image.label()] = 1.0;
        return output;
    }

    private static Image[] prepareDataset(String dataFileName, String dataLabelName) throws IOException {
        MNISTReader reader = new MNISTReader();
        int[][][] images = reader.readImage("data/" + dataFileName);
        byte[] labels = reader.readLabel("data/" + dataLabelName);
        int dataSize = images.length;
        Image[] dataset = new Image[dataSize];
        for (int i = 0; i < dataSize; i++) {
            dataset[i] = new Image(images[i], labels[i]);
        }
        return dataset;
    }
}