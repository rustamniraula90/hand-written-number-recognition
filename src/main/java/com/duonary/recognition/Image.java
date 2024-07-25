package com.duonary.recognition;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public record Image(int[][] pixels, int label) {

    public int getRows() {
        return pixels.length;
    }

    public int getColumns() {
        return pixels[0].length;
    }

    public int getPixel(int row, int column) {
        return pixels[row][column];
    }

    public void saveImage(String outputFile) throws IOException {
        int rows = pixels.length;
        int columns = pixels[0].length;
        BufferedImage bufferedImage = new BufferedImage(columns, rows, BufferedImage.TYPE_BYTE_GRAY);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                int pixelValue = pixels[r][c];
                int rgb = (pixelValue << 16) | (pixelValue << 8) | pixelValue; // Convert to grayscale
                bufferedImage.setRGB(c, r, rgb);
            }
        }
        File outputfile = new File(outputFile);
        ImageIO.write(bufferedImage, "png", outputfile);
        System.out.println("Saved image: " + outputfile.getPath());
    }
}
