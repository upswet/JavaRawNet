package rawnet.docs.flows.main.cnn.tasks.imageDetect;

import lombok.extern.slf4j.Slf4j;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.cnn.ConvLayer;
import rawnet.code.net.model.Model;
import rawnet.code.net.model.functional.Functional;
import rawnet.code.net.model.sequence.Sequence;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Визуализатор весов свёрточных слоёв сохранённой модели.
 *
 * <p>Для каждого {@link ConvLayer} рисует сетку карт весов:
 * <ul>
 *   <li>по горизонтали — входные каналы ({@code ic});</li>
 *   <li>по вертикали — выходные каналы ({@code oc}), т.е. фильтры;</li>
 *   <li>в ячейке — ядро {@code W[oc][ic]} как картинка {@code k×k}.</li>
 * </ul>
 *
 * <p>Нормировка: для каждого {@code oc} независимо считаем
 * {@code min/max} по всем {@code (ic, kh, kw)} и растягиваем в {@code [0,1]}.
 * Так фильтры с малыми весами не «теряются» рядом с большими.
 *
 * <p>Цвет: отрицательные веса — синие, ноль — белый, положительные —
 * красные. Так видно «направление» фильтра (тёмное→светлое, край, blob).
 *
 * <p>Использование:
 * <pre>{@code
 * // по пути к сохранённой модели
 * ConvWeightVisualizer.visualize(
 *         Path.of("model/grid/grid-cnn.bin"),
 *         new File("out/weights/grid"));
 *
 * // по уже загруженной модели
 * ConvWeightVisualizer.visualize(sequenceModel, new File("out/weights/seq"));
 * }</pre>
 *
 * <p><b>Важно:</b> работает <b>только</b> с сохранённой/загруженной
 * моделью — обучение не запускает, датасет не нужен.
 *
 * <h3>Ограничения</h3>
 * <ul>
 *   <li>Показывает <b>только</b> {@link ConvLayer}. Веса {@code DenseLayer}
 *       тоже можно нарисовать, но они не пространственные — для Dense
 *       обычно смотрят норму строки/столбца, а не «картинку».</li>
 *   <li>Не показывает активации (что сеть «видит» на конкретной картинке) —
 *       для этого нужен forward с хуками, отдельная задача.</li>
 *   <li>Работает и с {@link Sequence}, и с {@link Functional}.</li>
 * </ul>
 */
@Slf4j
public final class ConvWeightVisualizer {

    /** Размер одной ячейки ядра на картинке (пикселей). */
    private static final int CELL = 40;
    /** Отступ между ячейками. */
    private static final int GAP = 2;
    /** Отступ от края картинки. */
    private static final int MARGIN = 8;
    /** Высота подписи под картой. */
    private static final int LABEL_H = 14;

    private ConvWeightVisualizer() {}

    // ==================================================================
    // ============================ API =================================
    // ==================================================================

    /**
     * Загружает модель и визуализирует <b>все</b> свёрточные слои
     * в {@code outDir}. Имена файлов: {@code conv-00.png}, {@code conv-01.png}, ...
     *
     * @param modelPath путь к сохранённой модели
     * @param outDir    каталог для PNG (создаётся, если нет)
     * @throws IOException              при ошибке чтения/записи
     * @throws IllegalArgumentException если файла нет
     * @throws IllegalStateException    если это не Sequence и не Functional
     */
    public static void visualize(Path modelPath, File outDir) throws IOException {
        if (!Files.exists(modelPath))
            throw new IllegalArgumentException("Модель не найдена: " + modelPath);

        Model loaded = Model.load(modelPath);
        visualize(loaded, outDir);
    }

    /**
     * Визуализирует все {@link ConvLayer} из уже загруженной модели.
     * Поддерживает {@link Sequence} и {@link Functional}.
     */
    public static void visualize(Model model, File outDir) throws IOException {
        if (model == null)
            throw new IllegalArgumentException("model == null");
        if (!outDir.exists() && !outDir.mkdirs())
            throw new IOException("Не удалось создать каталог: " + outDir);

        List<NamedLayer> convs = extractConvLayers(model);
        if (convs.isEmpty()) {
            System.out.println("  (свёрточных слоёв не найдено в " + model.getClass().getSimpleName() + ")");
            return;
        }

        int idx = 0;
        for (NamedLayer nl : convs) {
            BufferedImage img = renderConv(nl.layer());
            File out = new File(outDir, String.format("conv-%02d.png", idx));
            ImageIO.write(img, "png", out);
            System.out.printf("  [conv-%02d] %-30s  %s → %s,  %s%n",
                    idx, nl.name(), nl.layer().inputShape(), nl.layer().outputShape(), out.getName());
            idx++;
        }
        System.out.printf("  всего свёрточных слоёв: %d → %s%n", idx, outDir.getAbsolutePath());
    }

    // ==================================================================
    // ============== ИЗВЛЕЧЕНИЕ ConvLayer ИЗ МОДЕЛИ ====================
    // ==================================================================

    /** ConvLayer + имя узла (для лога). */
    private record NamedLayer(String name, ConvLayer layer) {}

    /**
     * Собирает все {@link ConvLayer} из модели.
     *
     * <p>{@link Sequence}: порядок — как в сети (через {@code layersForPrint()}).
     * <p>{@link Functional}: порядок — топологический (через {@code debugTopo()}),
     * ConvLayer'ы в нём идут как узлы графа.
     *
     * <p>Другие реализации {@link Model} не поддерживаются — бросит
     * {@link IllegalStateException}.
     */
    private static List<NamedLayer> extractConvLayers(Model model) {
        List<NamedLayer> out = new ArrayList<>();

        if (model instanceof Sequence seq) {
            Layer[] layers = seq.layersForPrint();   // публичный [DEBUG]-метод, отдаёт копию
            for (int i = 0; i < layers.length; i++) {
                if (layers[i] instanceof ConvLayer conv) {
                    out.add(new NamedLayer("#" + i, conv));
                }
            }
            return out;
        }

        if (model instanceof Functional fn) {
            for (Functional.Node n : fn.debugTopo()) {
                if (n.layer instanceof ConvLayer conv) {
                    out.add(new NamedLayer(n.name, conv));
                }
            }
            return out;
        }

        throw new IllegalStateException(
                "ConvWeightVisualizer: поддерживаются только Sequence и Functional, получено "
                        + model.getClass().getName());
    }

    // ==================================================================
    // ======================= РЕНДЕР ОДНОГО СЛОЯ =======================
    // ==================================================================

    /**
     * Рисует один {@link ConvLayer} как сетку карт весов.
     * Строки — выходные каналы (фильтры), столбцы — входные каналы.
     */
    private static BufferedImage renderConv(ConvLayer conv) {
        // 1. Забираем веса и размеры через рефлексию.
        //    Когда в ConvLayer появятся публичные геттеры weights()/outChannels()/
        //    inChannels()/kernelSize() — замените этот блок на прямые вызовы.
        final float[][][][] w = convWeights(conv);
        final int oc = w.length;
        final int ic = w[0].length;
        final int kH = w[0][0].length;
        final int kW = w[0][0][0].length;

        int cellW = CELL;
        int cellH = CELL + LABEL_H;
        int gridW = ic * (cellW + GAP) + GAP;
        int gridH = oc * (cellH + GAP) + GAP;

        int totalW = gridW + 2 * MARGIN;
        int totalH = gridH + 2 * MARGIN;

        BufferedImage img = new BufferedImage(totalW, totalH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, totalW, totalH);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING,
                RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 10));

        // ── ядра ──
        for (int o = 0; o < oc; o++) {
            // Нормировка по всем ic и по всему k×k для данного oc
            float min = Float.POSITIVE_INFINITY;
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < ic; c++) {
                for (int ky = 0; ky < kH; ky++) {
                    for (int kx = 0; kx < kW; kx++) {
                        float v = w[o][c][ky][kx];
                        if (v < min) min = v;
                        if (v > max) max = v;
                    }
                }
            }
            float range = Math.max(max - min, 1e-8f);

            for (int c = 0; c < ic; c++) {
                int x0 = MARGIN + GAP + c * (cellW + GAP);
                int y0 = MARGIN + GAP + o * (cellH + GAP);

                // ядро как картинка k×k, растянутое в CELL×CELL
                BufferedImage kernel = renderKernel(w, o, c, kH, kW, min, range, CELL);
                g.drawImage(kernel, x0, y0, null);

                // подпись под ядром
                g.setColor(new Color(60, 60, 60));
                String label = String.format("o%d·c%d", o, c);
                int tw = g.getFontMetrics().stringWidth(label);
                g.drawString(label, x0 + (CELL - tw) / 2, y0 + CELL + 11);
            }
        }

        // ── заголовок ──
        g.setColor(Color.BLACK);
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 12));
        g.drawString(String.format("Conv  oc=%d × ic=%d, k=%d×%d  →  %s",
                        oc, ic, kH, kW, conv.outputShape()),
                MARGIN, 14);

        g.dispose();
        return img;
    }

    /**
     * Рисует одно ядро {@code W[o][c]} размером {@code kH×kW}, растянутое
     * в {@code size×size}. Нормировка — по переданным {@code min/range}.
     */
    private static BufferedImage renderKernel(float[][][][] w, int o, int c,
                                              int kH, int kW,
                                              float min, float range, int size) {
        BufferedImage img = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
        int px = Math.max(1, size / Math.max(kH, kW));
        for (int ky = 0; ky < kH; ky++) {
            for (int kx = 0; kx < kW; kx++) {
                float v = w[o][c][ky][kx];
                float t = (v - min) / range;        // [0,1]
                Color col = signedColor(t * 2f - 1f); // [0,1] → [-1,1]
                for (int dy = 0; dy < px; dy++) {
                    for (int dx = 0; dx < px; dx++) {
                        int xx = kx * px + dx;
                        int yy = ky * px + dy;
                        if (xx < size && yy < size) img.setRGB(xx, yy, col.getRGB());
                    }
                }
            }
        }
        return img;
    }

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

    // ==================================================================
    // ========================== РЕФЛЕКСИЯ =============================
    // ==================================================================

    /**
     * Достаёт {@code weights} из {@link ConvLayer}.
     *
     * <p>Поле {@code weights} объявлено как {@code float[][][][]} с
     * package-private доступом. Когда в библиотеке появится публичный
     * геттер ({@code weights()}), этот метод можно заменить на
     * {@code conv.weights()}.
     *
     * <p>Рефлексия кешируется в {@link Field} — вызов дешёвый.
     */
    private static final class WeightsFieldHolder {
        static final Field FIELD = lookup();

        private static Field lookup() {
            try {
                Field f = ConvLayer.class.getDeclaredField("weights");
                f.setAccessible(true);
                return f;
            } catch (NoSuchFieldException e) {
                throw new IllegalStateException(
                        "ConvWeightVisualizer: в ConvLayer нет поля 'weights' — "
                                + "проверьте версию библиотеки", e);
            }
        }
    }

    private static float[][][][] convWeights(ConvLayer conv) {
        try {
            return (float[][][][]) WeightsFieldHolder.FIELD.get(conv);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(
                    "ConvWeightVisualizer: не удалось прочитать weights у "
                            + conv.getClass().getName(), e);
        }
    }

    public static void main() throws IOException {
        // в ImageDetectDemo после всех сценариев
        log.info("=== ВИЗУАЛИЗАЦИЯ ВЕСОВ ===");
        ConvWeightVisualizer.visualize(Path.of("model/mnist/mnist-cnn.bin"), new File("out/weights/mnist"));


        ConvWeightVisualizer.visualize(Path.of("model/detect/detection-cnn.bin"), new File("out/weights/detect"));
        ConvWeightVisualizer.visualize(Path.of("model/grid/grid-cnn.bin"),       new File("out/weights/grid"));
        ConvWeightVisualizer.visualize(Path.of("model/heatmap/heatmap-cnn.bin"), new File("out/weights/heatmap"));
        ConvWeightVisualizer.visualize(Path.of("model/seg/seg-cnn.bin"),         new File("out/weights/seg"));
    }
}