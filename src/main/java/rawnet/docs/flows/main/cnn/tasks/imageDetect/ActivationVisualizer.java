package rawnet.docs.flows.main.cnn.tasks.imageDetect;

import lombok.extern.slf4j.Slf4j;
import rawnet.code.net.TensorShape;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.model.Model;
import rawnet.code.net.model.functional.Functional;
import rawnet.code.net.model.sequence.Sequence;
import rawnet.docs.flows.main.mlp.tasks.Mnist;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Визуализатор активаций (карт признаков) для конкретного входа.
 *
 * <p>Прогоняет вход через модель <b>слой за слоем</b>, сохраняя выход
 * каждого слоя, и рисует его как сетку картинок (для CNN-форм) или
 * как полосу (для flat-форм).
 *
 * <p>Для одного примера {@code X} генерирует каталог {@code step-NN.png}:
 * <ul>
 *   <li>{@code step-00.png} — вход (RGB);</li>
 *   <li>{@code step-01.png} — выход первого слоя (обычно Conv, 16 карт);</li>
 *   <li>{@code step-02.png} — выход второго (LeakyReLU, те же 16 карт);</li>
 *   <li>... и так далее.</li>
 * </ul>
 *
 * <p>Плюс общий «склеенный» PNG {@code all.png}: сетка всех слоёв подряд,
 * чтобы одним взглядом увидеть всю иерархию.
 *
 * <h3>Что это даёт</h3>
 * <ul>
 *   <li>первый Conv-слой: карты показывают, какие края/углы сработали
 *       на конкретной картинке (например, вертикальный край цифры 7);</li>
 *   <li>второй: видно, как «край» превращается в «угол/часть»;</li>
 *   <li>голова (flatten/dense): видно, какие нейроны «горят» —
 *       и почему сеть ошиблась/угадала.</li>
 * </ul>
 *
 * <h3>Ограничения</h3>
 * <ul>
 *   <li>работает только для {@link Sequence} и {@link Functional};</li>
 *   <li>если модель содержит {@code ModelLayer}, вложенные слои
 *       визуализируются как один шаг (внутренний граф не раскрывается);</li>
 *   <li>{@code forward} для каждого слоя выполняется последовательно,
 *       поэтому работает для любой модели без хуков.</li>
 * </ul>
 *
 * <h3>Использование</h3>
 * <pre>{@code
 * Sequence model = load...;
 * float[] input = ...;                     // S*S*3 в [0,1], channel-major
 * ActivationVisualizer.visualize(
 *         model, input, 128,
 *         new File("out/activations/grid-sample-00"));
 * }</pre>
 */
@Slf4j
public final class ActivationVisualizer {

    /** Размер одной карты на картинке (пикселей). */
    private static final int CELL = 64;
    /** Отступ между картами. */
    private static final int GAP = 4;
    /** Отступ от края. */
    private static final int MARGIN = 8;
    /** Высота подписи под картой. */
    private static final int LABEL_H = 14;
    /** Сколько карт максимум рисовать в одном слое. */
    private static final int MAX_MAPS = 64;

    private ActivationVisualizer() {}

    // ==================================================================
    // ============================ API =================================
    // ==================================================================

    /**
     * Визуализирует активации всех слоёв для одного входа.
     *
     * @param model      модель (Sequence или Functional)
     * @param input      входной вектор длины {@code model.inputShape().size()}
     * @param imageSize  сторона исходного изображения (для шага 0)
     * @param outDir     каталог для PNG (создаётся, если нет)
     * @throws IOException при ошибке записи
     */
    public static void visualize(Model model,
                                 float[] input,
                                 int imageSize,
                                 File outDir) throws IOException {
        if (model == null) throw new IllegalArgumentException("model == null");
        if (input == null) throw new IllegalArgumentException("input == null");
        if (input.length != model.inputShape().size())
            throw new IllegalArgumentException(
                    "ActivationVisualizer: ожидалось " + model.inputShape().size()
                            + " входных значений, получено " + input.length);
        if (!outDir.exists() && !outDir.mkdirs())
            throw new IOException("Не удалось создать каталог: " + outDir);

        // Шаг 0: сам вход. Для RGB-входа (3*S*S) рисуем исходную картинку.
        BufferedImage step0 = renderInput(input, imageSize, model.inputShape().size());
        ImageIO.write(step0, "png", new File(outDir, "step-00-input.png"));

        List<StepRecord> records = new ArrayList<>();
        records.add(new StepRecord("input", step0));

        // Шаги 1..N: активации каждого слоя.
        float[] current = input;
        int step = 1;
        for (Layer layer : layerSequence(model)) {
            // Порядок: у Layer.forward второй аргумент — training. Используем false,
            // чтобы dropout был тождественным, и мы видели «чистые» активации.
            current = layer.forward(current, false);

            BufferedImage img = renderActivation(current, layer.outputShape(),
                    layer.inputShape(), layer.getClass().getSimpleName(), step);
            File out = new File(outDir, String.format("step-%02d-%s.png", step, shortName(layer)));
            ImageIO.write(img, "png", out);
            records.add(new StepRecord(shortName(layer), img));

            System.out.printf("  [step-%02d] %-28s  %s → %s%n",
                    step, layer.getClass().getSimpleName(),
                    layer.inputShape(), layer.outputShape());
            step++;
        }

        // Склеенный «всё подряд» (для быстрого просмотра).
        BufferedImage all = stackVertically(records);
        ImageIO.write(all, "png", new File(outDir, "all.png"));

        System.out.printf("  готово: %s (%d слоёв)%n", outDir.getAbsolutePath(), records.size() - 1);
    }

    // ==================================================================
    // ================== ПОЛУЧЕНИЕ ЛИНЕЙНОГО СПИСКА СЛОЁВ ==============
    // ==================================================================

    /**
     * Возвращает слои модели в порядке forward.
     *
     * <p>Для {@link Sequence}: {@code layersForPrint()}.
     * Для {@link Functional}: топологический порядок {@code debugTopo()},
     * только узлы с не-{@code null} {@code layer}.
     *
     * @throws IllegalStateException для других реализаций {@link Model}
     */
    private static List<Layer> layerSequence(Model model) {
        List<Layer> layers = new ArrayList<>();
        if (model instanceof Sequence seq) {
            for (Layer l : seq.layersForPrint()) layers.add(l);
            return layers;
        }
        if (model instanceof Functional fn) {
            for (Functional.Node n : fn.debugTopo()) {
                if (n.layer != null) layers.add(n.layer);
            }
            return layers;
        }
        throw new IllegalStateException(
                "ActivationVisualizer: поддерживаются только Sequence и Functional, получено "
                        + model.getClass().getName());
    }

    /** Короткое имя слоя для подписи в логе/имени файла. */
    private static String shortName(Layer l) {
        String n = l.getClass().getSimpleName();
        // ConvLayer → conv, ActivationLayer → activation, ...
        if (n.endsWith("Layer")) n = n.substring(0, n.length() - "Layer".length());
        return n.toLowerCase();
    }

    /** Пара «имя — картинка» для склейки. */
    private record StepRecord(String name, BufferedImage image) {}

    // ==================================================================
    // ======================== РЕНДЕР ШАГА 0 (ВХОД) ====================
    // ==================================================================

    /**
     * Вход: если это RGB {@code 3*S*S} — рисуем как картинку. Иначе —
     * как полосу чисел (например, для MNIST {@code 1*S*S} — серая карта).
     */
    private static BufferedImage renderInput(float[] input, int size, int expected) {
        if (expected == 3 * size * size) {
            return rgbFromFlat(input, size, size);
        }
        // Иначе — карта в оттенках серого: одно значение на пиксель.
        return grayFromFlat(input, size, size);
    }

    /** {@code flat(3*S*S)} → BufferedImage. */
    private static BufferedImage rgbFromFlat(float[] flat, int w, int h) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        int n = w * h;
        for (int i = 0; i < n; i++) {
            int r = clamp255(flat[i]         * 255f);
            int g = clamp255(flat[n + i]     * 255f);
            int b = clamp255(flat[2 * n + i] * 255f);
            img.setRGB(i % w, i / w, (r << 16) | (g << 8) | b);
        }
        return img;
    }

    /** {@code flat(S*S)} → BufferedImage в оттенках серого. */
    private static BufferedImage grayFromFlat(float[] flat, int w, int h) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        int n = w * h;
        for (int i = 0; i < n; i++) {
            int v = clamp255(flat[i] * 255f);
            img.setRGB(i % w, i / w, (v << 16) | (v << 8) | v);
        }
        return img;
    }

    // ==================================================================
    // ===================== РЕНДЕР АКТИВАЦИЙ СЛОЯ ======================
    // ==================================================================

    /**
     * Рисует выход одного слоя.
     *
     * <p>Для CNN-формы {@code cnn(C,H,W)} — сетка из {@code C} карт,
     * каждая {@code H×W}, с подписью {@code c0..cC-1}. Для flat-формы —
     * одна полоса (обрезанная до {@link #MAX_MAPS}×{@link #MAX_MAPS}).
     */
    private static BufferedImage renderActivation(float[] act,
                                                  TensorShape outShape,
                                                  TensorShape inShape,
                                                  String layerName,
                                                  int step) {
        if (outShape.isCnn()) {
            return renderCnnActivation(act, outShape, layerName, step);
        } else {
            return renderFlatActivation(act, outShape, layerName, step);
        }
    }

    /** Активации CNN-слоя: сетка карт {@code C} × {@code H×W}. */
    private static BufferedImage renderCnnActivation(float[] act,
                                                     TensorShape shape,
                                                     String layerName,
                                                     int step) {
        final int C = shape.channels();
        final int H = shape.height();
        final int W = shape.width();

        // Общая нормировка по всему слою — чтобы «яркое» в одной карте
        // было сравнимо с «ярким» в другой.
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (float v : act) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        float range = Math.max(max - min, 1e-8f);

        // Сетка: columns × rows так, чтобы влезло C карт.
        int cols = (int) Math.ceil(Math.sqrt(C));
        int rows = (int) Math.ceil((double) C / cols);
        int mapW = Math.max(W, 4);
        int mapH = Math.max(H, 4);
        int cellW = mapW;
        int cellH = mapH + LABEL_H;

        int totalW = MARGIN * 2 + cols * (cellW + GAP) - GAP;
        int totalH = MARGIN * 2 + rows * (cellH + GAP) - GAP + 22;   // +22 под заголовок

        BufferedImage img = new BufferedImage(totalW, totalH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, totalW, totalH);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING,
                RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        // заголовок
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 12));
        g.setColor(Color.BLACK);
        g.drawString(String.format("step %d: %s  cnn(%d,%d,%d)  [min=%.3f  max=%.3f]",
                step, layerName, C, H, W, min, max), MARGIN, 16);

        // карты
        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 10));
        for (int c = 0; c < C && c < MAX_MAPS; c++) {
            int col = c % cols;
            int row = c / cols;
            int x0 = MARGIN + col * (cellW + GAP);
            int y0 = MARGIN + 22 + row * (cellH + GAP);

            BufferedImage map = renderMap(act, c, W, H, min, range, mapW, mapH);
            g.drawImage(map, x0, y0, null);

            // подпись
            g.setColor(new Color(60, 60, 60));
            String label = "c" + c;
            int tw = g.getFontMetrics().stringWidth(label);
            g.drawString(label, x0 + (mapW - tw) / 2, y0 + mapH + 11);

            // рамка
            g.setColor(Color.GRAY);
            g.drawRect(x0, y0, mapW - 1, mapH - 1);
        }
        if (C > MAX_MAPS) {
            g.setColor(new Color(180, 0, 0));
            g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 10));
            g.drawString("... " + (C - MAX_MAPS) + " карт не показано",
                    MARGIN, totalH - 6);
        }
        g.dispose();
        return img;
    }

    /** Одна карта признаков {@code c} → BufferedImage {@code w×h}. */
    private static BufferedImage renderMap(float[] act, int c, int W, int H,
                                           float min, float range, int dstW, int dstH) {
        BufferedImage img = new BufferedImage(dstW, dstH, BufferedImage.TYPE_INT_RGB);
        int base = c * H * W;
        for (int y = 0; y < dstH; y++) {
            for (int x = 0; x < dstW; x++) {
                int sx = Math.min(W - 1, x * W / dstW);
                int sy = Math.min(H - 1, y * H / dstH);
                float v = act[base + sy * W + sx];
                float t = (v - min) / range;        // [0,1]
                img.setRGB(x, y, signedColor(t * 2f - 1f).getRGB());
            }
        }
        return img;
    }

    /** Активации flat-слоя (Dense, Flatten, ...) — одна полоса. */
    private static BufferedImage renderFlatActivation(float[] act,
                                                      TensorShape shape,
                                                      String layerName,
                                                      int step) {
        final int N = act.length;
        final int shown = Math.min(N, 256);
        final int cols = 16;
        final int rows = (shown + cols - 1) / cols;
        final int cell = 16;
        final int gap = 1;

        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < shown; i++) {
            float v = act[i];
            if (v < min) min = v;
            if (v > max) max = v;
        }
        float range = Math.max(max - min, 1e-8f);

        int gridW = cols * (cell + gap) + gap;
        int gridH = rows * (cell + gap) + gap;
        int totalW = MARGIN * 2 + gridW;
        int totalH = MARGIN * 2 + gridH + 22;

        BufferedImage img = new BufferedImage(totalW, totalH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, totalW, totalH);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING,
                RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 12));
        g.setColor(Color.BLACK);
        g.drawString(String.format("step %d: %s  flat(%d)  [min=%.3f  max=%.3f  shown=%d]",
                step, layerName, N, min, max, shown), MARGIN, 16);

        for (int i = 0; i < shown; i++) {
            int col = i % cols;
            int row = i / cols;
            int x0 = MARGIN + gap + col * (cell + gap);
            int y0 = MARGIN + 22 + gap + row * (cell + gap);
            float t = (act[i] - min) / range;
            Color col1 = signedColor(t * 2f - 1f);
            g.setColor(col1);
            g.fillRect(x0, y0, cell, cell);
        }
        g.dispose();
        return img;
    }

    // ==================================================================
    // ========================== СКЛЕЙКА ===============================
    // ==================================================================

    /** Склеивает все шаги вертикально в одну картинку (с подписями). */
    private static BufferedImage stackVertically(List<StepRecord> records) {
        int totalW = 0, totalH = 0;
        for (StepRecord r : records) {
            totalW = Math.max(totalW, r.image().getWidth());
            totalH += r.image().getHeight() + 6;
        }
        BufferedImage out = new BufferedImage(totalW, totalH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = out.createGraphics();
        g.setColor(new Color(240, 240, 240));
        g.fillRect(0, 0, totalW, totalH);
        int y = 0;
        for (StepRecord r : records) {
            g.drawImage(r.image(), 0, y, null);
            y += r.image().getHeight() + 6;
        }
        g.dispose();
        return out;
    }

    // ==================================================================
    // ========================== УТИЛИТЫ ===============================
    // ==================================================================

    /** {@code -1} — синий, {@code 0} — белый, {@code +1} — красный. */
    private static Color signedColor(float v) {
        float t = Math.max(-1f, Math.min(1f, v));
        if (t >= 0) {
            int gb = (int) (255 * (1f - t));
            return new Color(255, gb, gb);
        } else {
            int rg = (int) (255 * (1f + t));
            return new Color(rg, rg, 255);
        }
    }

    private static int clamp255(float v) {
        int i = Math.round(v);
        return i < 0 ? 0 : Math.min(i, 255);
    }



    public static void main() throws IOException {
        log.info("=== ВИЗУАЛИЗАЦИЯ ВЕСОВ ===");

        ActivationVisualizer.visualize(Model.load(Path.of("model/mnist/mnist-cnn.bin")), Mnist.prepareDataForMnist(Path.of("d:\\Work\\Project\\0files\\mnist\\mnist_test.csv")).getFirst().getInputs(), 28,  new File("out/weights/mnist_feaches"));


        ImageDatasetGenerator gen = new ImageDatasetGenerator(128, /*maxObjects=*/4, /*minShape=*/10, /*maxShape=*/22, 42L, 0f);

        ActivationVisualizer.visualize(Model.load(Path.of("model/detect/detection-cnn.bin")), gen.generateOne(ImageDatasetGenerator.Task.DETECTION).inputs(), 128,  new File("out/weights/detect_feaches"));
        ActivationVisualizer.visualize(Model.load(Path.of("model/grid/grid-cnn.bin")), gen.generateOne(ImageDatasetGenerator.Task.GRID).inputs(), 128,  new File("out/weights/grid_feaches"));
        ActivationVisualizer.visualize(Model.load(Path.of("model/heatmap/heatmap-cnn.bin")), gen.generateOne(ImageDatasetGenerator.Task.HEATMAP).inputs(), 128,  new File("out/weights/heatmap_feaches"));
        ActivationVisualizer.visualize(Model.load(Path.of("model/seg/seg-cnn.bin")), gen.generateOne(ImageDatasetGenerator.Task.SEMSEG).inputs(), 128,  new File("out/weights/seg_feaches"));
    }
}