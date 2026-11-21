package rawnet.docs.flows.main.cnn.tasks.styleTransform;

import rawnet.code.net.TensorShape;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.model.Model;
import rawnet.docs.flows.main.cnn.tasks.imageDetect.ImageDatasetGenerator;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Классический (Gatys 2015) Style Transfer: оптимизирует <b>пиксели</b>
 * {@code output}, веса CNN заморожены.
 */
public final class SlowStyleTransfer {

    private SlowStyleTransfer() {}

    public static BufferedImage transfer(Model cnn,
                                         BufferedImage content,
                                         BufferedImage style,
                                         int contentLayer,
                                         int[] styleLayers,
                                         float alpha, float beta,
                                         int iterations, float lr) throws IOException {
        if (content.getWidth() != style.getWidth() || content.getHeight() != style.getHeight())
            throw new IllegalArgumentException(
                    "SlowStyleTransfer: content и style должны быть одного размера");

        final int S = content.getWidth();
        final int H = content.getHeight();

        List<Layer> layers = Model.layers(cnn);
        int maxLayer = Math.max(contentLayer, max(styleLayers));
        if (maxLayer >= layers.size())
            throw new IllegalArgumentException(
                    "SlowStyleTransfer: maxLayer=" + maxLayer + " >= layers=" + layers.size());

        float[] contentFlat = ImageDatasetGenerator.toFlatRgb(content);
        float[] styleFlat   = ImageDatasetGenerator.toFlatRgb(style);

        float[] contentFeatures = Model.forwardUpTo(cnn, contentFlat, contentLayer).clone();

        float[][] styleGram = new float[styleLayers.length][];
        TensorShape[] styleShapes = new TensorShape[styleLayers.length];
        for (int k = 0; k < styleLayers.length; k++) {
            float[] f = Model.forwardUpTo(cnn, styleFlat, styleLayers[k]);
            styleShapes[k] = layers.get(styleLayers[k]).outputShape();
            styleGram[k] = StyleOps.gram(f, styleShapes[k]);
        }

        float[] output = contentFlat.clone();
        TensorShape contentShape = layers.get(contentLayer).outputShape();

        for (int it = 0; it < iterations; it++) {
            Model.forwardUpTo(cnn, output, maxLayer);

            float[] fContent = Model.forwardUpTo(cnn, output, contentLayer);
            float lContent = StyleOps.contentLoss(fContent, contentFeatures);

            float lStyle = 0f;
            List<float[]> styleGrads = new ArrayList<>(styleLayers.length);
            for (int k = 0; k < styleLayers.length; k++) {
                float[] fStyle = Model.forwardUpTo(cnn, output, styleLayers[k]);
                TensorShape sh = styleShapes[k];
                float[] G = StyleOps.gram(fStyle, sh);
                lStyle += StyleOps.gramLoss(G, styleGram[k]);
                float[] gradStyle = new float[fStyle.length];
                StyleOps.gramLossGrad(fStyle, G, styleGram[k], sh, gradStyle);
                styleGrads.add(gradStyle);
            }

            float[] gradPixels = new float[output.length];
            {
                float[] dContent = new float[fContent.length];
                StyleOps.contentLossGrad(fContent, contentFeatures, dContent);
                float[] g = Model.backwardFrom(cnn, contentLayer, dContent);
                for (int i = 0; i < g.length; i++) gradPixels[i] += alpha * g[i];
            }
            for (int k = 0; k < styleLayers.length; k++) {
                float[] g = Model.backwardFrom(cnn, styleLayers[k], styleGrads.get(k));
                float w = beta / styleLayers.length;
                for (int i = 0; i < g.length; i++) gradPixels[i] += w * g[i];
            }

            for (int i = 0; i < output.length; i++)
                output[i] -= lr * gradPixels[i];
            for (int i = 0; i < output.length; i++)
                output[i] = Math.max(0f, Math.min(1f, output[i]));

            if (it == 0 || (it + 1) % 50 == 0) {
                System.out.printf("  it %3d/%d   content=%.4f  style=%.4f%n",
                        it + 1, iterations, lContent, lStyle);
            }
        }

        return ImageDatasetGenerator.fromFlatRgb(output, S, H);
    }

    private static int max(int[] a) {
        int m = Integer.MIN_VALUE;
        for (int v : a) if (v > m) m = v;
        return m;
    }
}