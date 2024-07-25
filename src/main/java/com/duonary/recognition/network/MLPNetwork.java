package com.duonary.recognition.network;

import com.duonary.recognition.util.MathUtil;

import static com.duonary.recognition.util.MathUtil.*;

public class MLPNetwork {

    private final Layer[] layers;
    private final double learningRate;
    private final int batchSize;
    private final int epoch;
    private int index;
    private double[][] inputs;
    private double[][] outputs;
    private double[][] expectedOutputs;
    private int correct;
    private int predicted;
    private int total;

    private MLPNetwork(Layer[] layers, double learningRate, int batchSize, int epoch) {
        this.layers = layers;
        this.learningRate = learningRate;
        this.batchSize = batchSize;
        this.epoch = epoch;
        this.index = 0;
        this.inputs = new double[batchSize][];
        this.outputs = new double[batchSize][];
        this.expectedOutputs = new double[batchSize][];
    }

    private static class Layer {
        Neuron[] neurons;

        public Layer(int layerSize, int previousLayerSize, int inputSize, int outPutSize) {
            this.neurons = new Neuron[layerSize];
            for (int i = 0; i < layerSize; i++) {
                this.neurons[i] = new Neuron(previousLayerSize, inputSize, outPutSize);
            }
        }
    }

    private static class Neuron {
        double[] weights;
        double bias;
        double input;
        double output;

        public Neuron(int previousLayerSize, int inSize, int outSize) {
            this.weights = new double[previousLayerSize];
            for (int i = 0; i < this.weights.length; i++) {
                this.weights[i] = MathUtil.xavier(inSize, outSize);
            }
            this.bias = 0.0;
        }
    }

    public static Builder builder(int hiddenLayerCount) {
        return new Builder(hiddenLayerCount);
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

        public MLPNetwork build() {
            if (index != hiddenLayers.length) {
                throw new IllegalStateException("Not all hidden layers have been added");
            }

            Layer[] layers = new Layer[hiddenLayers.length + 1];
            for (int i = 0; i < layers.length; i++) {
                int layerSize = i == (hiddenLayers.length) ? outputLayerSize : hiddenLayers[i];
                int previousLayerSize = i == 0 ? inputLayerSize : hiddenLayers[i - 1];
                layers[i] = new Layer(layerSize, previousLayerSize, inputLayerSize, outputLayerSize);
            }

            return new MLPNetwork(layers, learningRate, batchSize, epoch);
        }
    }

    private double[] forwardPass(double[] input) {
        double[] activations = input;
        for (Layer layer : layers) {
            double[] nextActivations = new double[layer.neurons.length];
            for (int i = 0; i < layer.neurons.length; i++) {
                Neuron neuron = layer.neurons[i];
                double z = 0.0;
                for (int j = 0; j < neuron.weights.length; j++) {
                    z += neuron.weights[j] * activations[j];
                }
                z += neuron.bias;
                neuron.input = z;
                neuron.output = relu(z);
                nextActivations[i] = neuron.output;
            }
            activations = nextActivations;
        }
        return softmax(activations);
    }

    private void backwardPass(int epoch) {
        if (index == batchSize) {
            index = 0;
            double loss = MathUtil.crossEntropy(this.outputs, this.expectedOutputs);
            System.out.printf("\rEpoch: %d\tLoss: %f\tTrained: %d/%d\tCorrect: %d/%d\tAccuracy: %f", epoch, loss, predicted, total, correct, total, (double) correct / predicted);
            tuneParameters();
            this.inputs = new double[batchSize][];
            this.outputs = new double[batchSize][];
            this.expectedOutputs = new double[batchSize][];
        }
    }

    private void tuneParameters() {
        int outputLayerSize = layers[layers.length - 1].neurons.length;
        double[][] delta = new double[batchSize][];

        // Backpropagation for output layer
        for (int i = 0; i < batchSize; i++) {
            delta[i] = new double[outputLayerSize];
            for (int j = 0; j < outputLayerSize; j++) {
                // Compute delta for the output layer
                delta[i][j] = outputs[i][j] - expectedOutputs[i][j];
            }
        }

        // Backpropagation through layers
        for (int layerIndex = layers.length - 1; layerIndex >= 0; layerIndex--) {
            Layer layer = layers[layerIndex];
            Layer prevLayer = layerIndex > 0 ? layers[layerIndex - 1] : null;

            for (int i = 0; i < layer.neurons.length; i++) {
                Neuron neuron = layer.neurons[i];

                // Update weights and biases
                for (int j = 0; j < neuron.weights.length; j++) {
                    double gradientSum = 0.0;
                    for (int k = 0; k < batchSize; k++) {
                        double input = layerIndex == 0 ? inputs[k][j] : prevLayer.neurons[j].output;
                        gradientSum += delta[k][i] * input;
                    }
                    neuron.weights[j] -= learningRate * gradientSum / batchSize;
                }

                double biasGradientSum = 0.0;
                for (int k = 0; k < batchSize; k++) {
                    biasGradientSum += delta[k][i];
                }
                neuron.bias -= learningRate * biasGradientSum / batchSize;
            }

            // Compute delta for the previous layer (if not input layer)
            if (layerIndex > 0) {
                double[][] newDelta = new double[batchSize][prevLayer.neurons.length];
                for (int i = 0; i < batchSize; i++) {
                    for (int j = 0; j < prevLayer.neurons.length; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < layer.neurons.length; k++) {
                            sum += delta[i][k] * layer.neurons[k].weights[j];
                        }
                        newDelta[i][j] = sum * reluDerivative(prevLayer.neurons[j].output);
                    }
                }
                delta = newDelta;
            }
        }
    }

    public void train(double[][] input, double[][] expectedOutput) {
        this.total = input.length;
        for (int i = 0; i < epoch; i++) {
            for (int j = 0; j < input.length; j++) {
                double[] output = forwardPass(input[j]);
                int currentPrediction = 0;
                int actual = 0;
                double max = output[0];
                for (int k = 1; k < output.length; k++) {
                    if (output[k] > max) {
                        max = output[k];
                        currentPrediction = k;
                    }
                    if (expectedOutput[j][k] == 1.0) {
                        actual = k;
                    }
                }
                if (currentPrediction == actual) {
                    this.correct++;
                }
                this.predicted++;

                this.inputs[index] = input[j];
                this.outputs[index] = output;
                this.expectedOutputs[index] = expectedOutput[j];
                index++;
                backwardPass(i);
            }
            this.correct = 0;
            this.predicted = 0;
            System.out.println();
        }
    }

    public double[] predict(double[] input) {
        return forwardPass(input);
    }
}
