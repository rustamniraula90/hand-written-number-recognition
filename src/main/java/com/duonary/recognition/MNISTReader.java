package com.duonary.recognition;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;

public class MNISTReader {

    private static final Logger LOG = LoggerFactory.getLogger(MNISTReader.class);

    public int[][][] readImage(String filePath) throws IOException {
        long start = System.currentTimeMillis();
        try (FileInputStream fis = new FileInputStream(filePath)) {
            // Read and verify the magic number
            int magicNumber = readInt(fis);
            if (magicNumber != 2051) { // 2051 is the magic number for the MNIST image files
                throw new IOException("Invalid magic number: " + magicNumber);
            }

            // Read the number of images
            int numberOfImages = readInt(fis);
            LOG.info("Number of images: {}", numberOfImages);

            // Read the number of rows
            int rows = readInt(fis);
            LOG.info("Number of rows: {}", rows);

            // Read the number of columns
            int columns = readInt(fis);
            LOG.info("Number of columns: {}", columns);

            // Read the image data
            int[][][] image = new int[numberOfImages][rows][columns];
            for (int i = 0; i < numberOfImages; i++) {
                System.out.printf("\rReading dataset: %d/%d", i + 1, numberOfImages);
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c++) {
                        image[i][r][c] = fis.read(); // Read the pixel value (0-255)
                    }
                }
            }
            System.out.println("\nData read complete in " + (System.currentTimeMillis() - start) + " ms");
            return image;
        }
    }

    public byte[] readLabel(String filePath) throws IOException {
        try (FileInputStream fis = new FileInputStream(filePath)) {
            // Read and verify the magic number
            int magicNumber = readInt(fis);
            if (magicNumber != 2049) { // 2049 is the magic number for the MNIST label files
                throw new IOException("Invalid magic number: " + magicNumber);
            }

            // Read the number of labels
            int numberOfLabels = readInt(fis);
            LOG.info("Number of labels: {}", numberOfLabels);

            // Read the labels
            byte[] labels = new byte[numberOfLabels];
            int read = fis.read(labels);
            if (read != numberOfLabels) {
                throw new IOException("Could not read all labels");
            }
            return labels;

        }
    }

    private static int readInt(FileInputStream fis) throws IOException {
        return (fis.read() << 24) | (fis.read() << 16) | (fis.read() << 8) | fis.read();
    }

}
