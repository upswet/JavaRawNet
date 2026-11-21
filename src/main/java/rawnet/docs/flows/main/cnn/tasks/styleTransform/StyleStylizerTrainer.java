package rawnet.docs.flows.main.cnn.tasks.styleTransform;

import rawnet.code.net.TensorShape;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.model.Model;
import rawnet.code.net.model.sequence.Sequence;
import rawnet.code.net.core.optimizator.AdamW;
import rawnet.code.net.core.optimizator.Optimizer;
import rawnet.docs.flows.main.cnn.tasks.imageDetect.ImageDatasetGenerator;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Обучение «быстрого» Style Transfer: сеть-стилизатор {@code S} учится
 * преобразовывать любой вход {@code x} в «x в стиле style».
 *
 * <p>Loss:
 * <pre>
 * L(S) = α · ||features(S(x)) − features(x)||²
 *      + β(it) · ||Gram(features(S(x))) − Gram(features(style))||²
 * </pre>
 *
 * <p><b>Адаптивный β.</b> В начале обучения {@code β(it)} в 4 раза больше
 * {@code β} — чтобы {@code S} <b>отошёл</b> от тождества (иначе
 * {@code content=0} — локальный минимум, S «ленится»). К концу
 * {@code β(it) → β} — S <b>стабилизирует</b> content.
 *
 * <p>Одна итерация:
 * <ol>
 *   <li>{@code y = S.forward(x, true)};</li>
 *   <li>{@code yClipped = clip(y, 0, 1)};</li>
 *   <li>{@link Model#forwardUpTo}(cnn, yClipped, maxLayer);</li>
 *   <li>loss + градиенты по features(yClipped);</li>
 *   <li>{@link Model#backwardFrom}(cnn, ..., dFeatures) → {@code dL/dyClipped};</li>
 *   <li>{@code S.backward(dL/dy)}, {@code S.update(opt, 1)}.</li>
 * </ol>
 *
 * <p>Датасет не нужен: {@code x} генерируется на лету ({@link ImageDatasetGenerator}).
 */
public final class StyleStylizerTrainer {

    private StyleStylizerTrainer() {}

    /**
     * Обучает {@code S} на одну стилевую картинку.
     *
     * @param stylizer      сеть-стилизатор (см. {@link StyleStylizer#build(int)})
     * @param cnn           обученная CNN для фич
     * @param style         картинка-стиль (одна)
     * @param gen           генератор content-картинок
     * @param contentLayer  индекс слоя для content loss в CNN
     * @param styleLayers   индексы слоёв для style loss в CNN
     * @param alpha         вес content
     * @param beta          базовый вес style (адаптивный: 4β → β)
     * @param iterations    число итераций
     * @param lr            learning rate
     * @return обученный стилизатор ({@code S} обновлён на месте)
     */
    public static Sequence train(Sequence stylizer,
                                 Model cnn,
                                 BufferedImage style,
                                 ImageDatasetGenerator gen,
                                 int contentLayer,
                                 int[] styleLayers,
                                 float alpha, float beta,
                                 int iterations, float lr) throws IOException {
        List<Layer> cnnLayers = Model.layers(cnn);
        int maxLayer = Math.max(contentLayer, max(styleLayers));
        if (maxLayer >= cnnLayers.size())
            throw new IllegalArgumentException(
                    "StyleStylizerTrainer: maxLayer=" + maxLayer
                            + " >= cnn layers=" + cnnLayers.size());

        int S = style.getWidth();
        if (S != style.getHeight())
            throw new IllegalArgumentException("StyleStylizerTrainer: style должен быть квадратным");
        if (stylizer.inputShape().size() != 3 * S * S)
            throw new IllegalArgumentException(
                    "StyleStylizerTrainer: stylizer.inputShape=" + stylizer.inputShape()
                            + " не совпадает с 3*" + S + "*" + S);

        // ── 1. Style-фичи и Gram (один раз) ──────────────────────────
        float[] styleFlat = ImageDatasetGenerator.toFlatRgb(style);
        float[][] styleGram = new float[styleLayers.length][];
        TensorShape[] styleShapes = new TensorShape[styleLayers.length];
        for (int k = 0; k < styleLayers.length; k++) {
            float[] f = Model.forwardUpTo(cnn, styleFlat, styleLayers[k]);
            styleShapes[k] = cnnLayers.get(styleLayers[k]).outputShape();
            styleGram[k] = StyleOps.gram(f, styleShapes[k]);
        }

        // ── 2. Оптимизатор ───────────────────────────────────────────
        Optimizer opt = new AdamW(lr);
        TensorShape contentShape = cnnLayers.get(contentLayer).outputShape();

        // ── 3. Цикл ──────────────────────────────────────────────────
        for (int it = 0; it < iterations; it++) {
            // 3a. Случайный input.
            BufferedImage xImg = gen.generateOne(ImageDatasetGenerator.Task.GRID).image();
            float[] x = ImageDatasetGenerator.toFlatRgb(xImg);

            // 3b. Forward стилизатора → y.
            stylizer.zeroGrad();
            float[] y = stylizer.forward(x, true);

            // 3c. Клип y (не мутируем y — Layer кэширует, работаем с копией).
            float[] yClipped = new float[y.length];
            for (int i = 0; i < y.length; i++)
                yClipped[i] = Math.max(0f, Math.min(1f, y[i]));

            // 3d. Forward CNN до maxLayer (кэш активаций).
            Model.forwardUpTo(cnn, yClipped, maxLayer);

            // 3e. Content loss + features.
            float[] fContent = Model.forwardUpTo(cnn, yClipped, contentLayer);
            float[] contentFeatures = Model.forwardUpTo(cnn, x, contentLayer).clone();
            float lContent = StyleOps.contentLoss(fContent, contentFeatures);

            // 3f. Style loss + градиенты.
            //     β адаптивный: в начале в 4 раза больше, к концу → beta.
            float progress = (float) it / Math.max(1, iterations - 1);   // 0..1
            float betaIt = beta * (1f + 3f * (1f - progress));          // 4β → β

            float lStyle = 0f;
            List<float[]> styleGrads = new ArrayList<>(styleLayers.length);
            for (int k = 0; k < styleLayers.length; k++) {
                float[] fStyle = Model.forwardUpTo(cnn, yClipped, styleLayers[k]);
                TensorShape sh = styleShapes[k];
                float[] G = StyleOps.gram(fStyle, sh);
                lStyle += StyleOps.gramLoss(G, styleGram[k]);
                float[] g = new float[fStyle.length];
                StyleOps.gramLossGrad(fStyle, G, styleGram[k], sh, g);
                styleGrads.add(g);
            }

            // 3g. dL/dyClipped от CNN: content + style.
            float[] gradYPixels = new float[yClipped.length];

            // content
            {
                float[] dContent = new float[fContent.length];
                StyleOps.contentLossGrad(fContent, contentFeatures, dContent);
                float[] g = Model.backwardFrom(cnn, contentLayer, dContent);
                for (int i = 0; i < g.length; i++) gradYPixels[i] += alpha * g[i];
            }

            // style (используем betaIt, а не beta)
            for (int k = 0; k < styleLayers.length; k++) {
                float[] g = Model.backwardFrom(cnn, styleLayers[k], styleGrads.get(k));
                float w = betaIt / styleLayers.length;
                for (int i = 0; i < g.length; i++) gradYPixels[i] += w * g[i];
            }

            // 3h. Клип derivative: dL/dyClipped → dL/dy.
            //     Там, где y вне [0,1], derivative клипа = 0.
            float[] gradY = new float[y.length];
            for (int i = 0; i < y.length; i++) {
                boolean inRange = y[i] >= 0f && y[i] <= 1f;
                gradY[i] = inRange ? gradYPixels[i] : 0f;
            }

            // 3i. Backward через стилизатор и шаг.
            stylizer.backward(gradY);
            stylizer.update(opt, 1);

            // 3j. Лог (раз в 50 итераций + первая).
            if (it == 0 || (it + 1) % 50 == 0) {
                System.out.printf("  it %4d/%d   content=%.4f  style=%.4f  β=%.1f%n",
                        it + 1, iterations, lContent, lStyle, betaIt);
            }
        }

        return stylizer;
    }

    private static int max(int[] a) {
        int m = Integer.MIN_VALUE;
        for (int v : a) if (v > m) m = v;
        return m;
    }
}