package com.duonary.recognition.util;

import java.util.Random;

public class MathUtil {
    private MathUtil() {
    }

    private static final Random random = new Random();

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }

    public static double relu(double x) {
        return Math.clamp(x, 0,6 );
    }

    public static double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    public static double xavier(int in, int out) {
        return random.nextGaussian() * Math.sqrt(2.0 / (in + out));
    }

    public static double[] softmax(double[] x) {
        double sum = 0;
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = Math.exp(x[i]);
            sum += result[i];
        }
        if (sum == 0) {
            return result;
        }
        for (int i = 0; i < x.length; i++) {
            result[i] /= sum;
        }
        return result;
    }

    public static double crossEntropy(double[][] actual, double[][] expected) {
        double loss = 0;
        for (int i = 0; i < actual.length; i++) {
            for (int j = 0; j < actual[i].length; j++) {
                loss += expected[i][j] * Math.log(actual[i][j]);
            }
        }
        return -loss;
    }

    public static double[][] multiply(double[][] a, double[][] b) {
        double[][] result = new double[a.length][b[0].length];

        for (int i = 0; i < a.length; ++i) {
            for (int j = 0; j < b[0].length; ++j) {
                for (int k = 0; k < a[0].length; ++k) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return result;
    }

    public static double[][] transpose(double[][] a) {
        int r = a.length;
        int c = a[0].length;
        double[][] t = new double[c][r];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                t[j][i] = a[i][j];
            }
        }
        return t;
    }

    public static double[][] add(double[][] a, double[][] b) {
        double[][] s = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                s[i][j] = a[i][j] + b[i][j];
            }
        }
        return s;
    }
}
