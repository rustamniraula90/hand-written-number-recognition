package com.duonary.recognition.network;

import com.duonary.recognition.util.MathUtil;

import java.util.Arrays;

import static com.duonary.recognition.util.MathUtil.*;

public class MLPNetworkOptimized {
    private final double[][][] weight;
    private final double[][] bias;
    private final double learningRate;
    private final int batchSize;
    private final int epoch;
    private double[][][] inputs;
    private double[][][] outputs;
    private double[][] expectedOutputs;
    private int correct;
    private int predicted;
    private int total;
    private int index;

    public MLPNetworkOptimized(double[][][] weight, double[][] bias, double learningRate, int batchSize, int epoch) {
        this.weight = weight;
        this.bias = bias;
        this.learningRate = learningRate;
        this.batchSize = batchSize;
        this.epoch = epoch;
        this.index = 0;
        this.correct = 0;
        this.predicted = 0;
        this.total = 0;
        this.inputs = new double[batchSize][weight.length][];
        this.outputs = new double[batchSize][weight.length][];
        this.expectedOutputs = new double[batchSize][];
    }

    public static MLPNetworkOptimized.Builder builder(int hiddenLayerCount) {
        return new MLPNetworkOptimized.Builder(hiddenLayerCount);
    }

    public static class Builder {
        private final int[] hiddenLayers;
        private int inputLayerSize;
        private int outputLayerSize;
        private int index;
        private double learningRate;
        private int batchSize;
        private int epoch;

        public Builder(int hiddenLayerCount) {
            this.hiddenLayers = new int[hiddenLayerCount];
            this.index = 0;
        }

        public Builder addInputLayer(int size) {
            this.inputLayerSize = size;
            return this;
        }

        public Builder addOutputLayer(int size) {
            this.outputLayerSize = size;
            return this;
        }

        public Builder addHiddenLayer(int size) {
            if (this.index >= this.hiddenLayers.length) {
                throw new IllegalStateException("Cannot add more hidden layers");
            }
            this.hiddenLayers[this.index] = size;
            this.index++;
            return this;
        }

        public Builder setLearningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder setBatchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder setEpoch(int epoch) {
            this.epoch = epoch;
            return this;
        }

        public MLPNetworkOptimized build() {
            if (index != hiddenLayers.length)
                throw new IllegalStateException("Not all hidden layers have been added");

            double[][][] weights = new double[hiddenLayers.length + 1][][];
            weights[0] = new double[inputLayerSize][hiddenLayers[0]];
            for (int i = 0; i < hiddenLayers.length; i++) {
                int row = hiddenLayers[i];
                int col = (i == hiddenLayers.length - 1) ? outputLayerSize : hiddenLayers[i + 1];
                weights[i + 1] = new double[row][col];
            }

            double[][] bias = new double[hiddenLayers.length + 1][];

            for (int i = 0; i < hiddenLayers.length + 1; i++)
                if (i == hiddenLayers.length) bias[i] = new double[outputLayerSize];
                else bias[i] = new double[hiddenLayers[i]];

            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++)
                    for (int k = 0; k < weights[i][j].length; k++)
                        weights[i][j][k] = MathUtil.xavier(inputLayerSize, outputLayerSize);
            }

            for (int i = 0; i < bias.length; i++) {
                for (int j = 0; j < bias[i].length; j++)
                    bias[i][j] = MathUtil.xavier(inputLayerSize, outputLayerSize);
            }

            return new MLPNetworkOptimized(weights, bias, learningRate, batchSize, epoch);
        }
    }

    public void train(double[][] inputs, double[][] labels) {
        for (int i = 0; i < epoch; i++) {
            for (int j = 0; j < inputs.length; j++) {
                this.total = inputs.length;
                double[] output = forwardPass(inputs[j]);
                int currentPrediction = 0;
                int actual = 0;
                double max = output[0];
                for (int k = 1; k < output.length; k++) {
                    if (output[k] > max) {
                        max = output[k];
                        currentPrediction = k;
                    }
                    if (labels[j][k] == 1.0) {
                        actual = k;
                    }
                }
                if (currentPrediction == actual) {
                    this.correct++;
                }
                this.predicted++;
                this.expectedOutputs[index] = labels[j];
                index++;
                backwardPass(i);
            }
            this.correct = 0;
            this.predicted = 0;
            System.out.println();
        }
    }

    private double[] forwardPass(double[] input) {
        for (int i = 0; i < weight.length; i++) {
            this.inputs[index][i] = input;
            double[][] in = new double[1][input.length];
            in[0] = input;
            double[][] b = new double[1][bias[i].length];
            b[0] = bias[i];
            input = transpose(add(multiply(transpose(weight[i]), transpose(in)), transpose(b)))[0];
            if (i == weight.length - 1) {
                input = softmax(input);
            } else {
                for (int j = 0; j < input.length; j++) {
                    input[j] = MathUtil.relu(input[j]);
                }
            }
            this.outputs[index][i] = input;
        }
        return input;
    }

    private void backwardPass(int epoch) {
        if (index == batchSize) {
            index = 0;
            double[][] finalOutput = new double[batchSize][];
            for (int i = 0; i < this.outputs.length; i++) {
                finalOutput[i] = this.outputs[i][this.outputs[i].length - 1];
            }
            double loss = MathUtil.crossEntropy(finalOutput, this.expectedOutputs);
            System.out.printf("\rEpoch: %d\tLoss: %f\tTrained: %d/%d\tCorrect: %d/%d\tAccuracy: %f", epoch, loss, predicted, total, correct, total, (double) correct / predicted);
            tuneParameters();
            this.inputs = new double[batchSize][weight.length][];
            this.outputs = new double[batchSize][weight.length][];
            this.expectedOutputs = new double[batchSize][];
        }
    }

    private void tuneParameters() {
        int outputLayerSize = weight[weight.length - 1][0].length;
        double[][] delta = new double[batchSize][outputLayerSize];

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < outputLayerSize; j++) {
                delta[i][j] = outputs[i][outputs[i].length - 1][j] - expectedOutputs[i][j];
            }
        }

        for (int i = weight.length - 1; i >= 0; i--) {
            for (int j = 0; j < weight[i].length; j++) {
                for (int k = 0; k < weight[i][j].length; k++) {
                    double gradientSum = 0.0;
                    for (int l = 0; l < batchSize; l++) {
                        gradientSum += delta[l][k] * inputs[l][i][j];
                    }
                    weight[i][j][k] -= learningRate * (gradientSum / batchSize);
                }
            }
            for (int j = 0; j < bias[i].length; j++) {
                double biasGradient = 0.0;
                for (int k = 0; k < batchSize; k++) {
                    biasGradient += delta[k][j];
                }
                bias[i][j] -= learningRate * (biasGradient / batchSize);
            }
            if (i > 0) {
                double[][] newDelta = new double[batchSize][weight[i - 1][0].length];
                for (int j = 0; j < batchSize; j++) {
                    for (int k = 0; k < weight[i].length; k++) {
                        double sum = 0.0;
                        for (int l = 0; l < weight[i][j].length; l++) { //TODO: think about j
                            sum += delta[j][l] * weight[i][k][l];
                        }
                        newDelta[j][k] = sum * MathUtil.reluDerivative(outputs[j][i-1][k]);
                    }
                }
                delta = newDelta;
            }
        }
    }

    public double[] predict(double[] input) {
        return forwardPass(input);
    }
}
