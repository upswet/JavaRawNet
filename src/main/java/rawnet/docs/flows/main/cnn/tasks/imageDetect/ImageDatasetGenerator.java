package rawnet.docs.flows.main.cnn.tasks.imageDetect;

import rawnet.code.dataset.SampleDataset;
import rawnet.code.dataset.Sample;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Генератор синтетического датасета геометрических фигур.
 *
 * <p>Один пример — квадратное изображение {@code size × size}, белый фон,
 * {@code 0..maxObjects} непересекающихся фигур случайного размера и цвета.
 *
 * <p>Поддерживаемые задачи ({@link Task}):
 * <ul>
 *   <li>{@link Task#DETECTION} — multi-label: {@code flat(4)} с 0/1;</li>
 *   <li>{@link Task#HEATMAP}   — CenterNet: гауссовы пики {@code C*S*S};</li>
 *   <li>{@link Task#GRID}      — YOLO: ячейки {@code G×G}, obj+box+class;</li>
 *   <li>{@link Task#SEMSEG}    — one-hot маска {@code 5*S*S}.</li>
 * </ul>
 *
 * <p>Помимо генерации картинок, класс содержит <b>построители целей</b>
 * ({@link #buildGridTargets}, {@link #buildHeatmapTargets},
 * {@link #buildSemsegTargets}) — чтобы демо не зависело от отдельных
 * классов {@code GridTargets}/{@code HeatmapTargets}.
 */
public final class ImageDatasetGenerator {

    /** Типы фигур. Индекс — целевой класс (0..3); фон = -1. */
    public enum ShapeType { CIRCLE, SQUARE, CROSS, TRIANGLE }

    /** Задача, под которую генерируются targets. */
    public enum Task { DETECTION, HEATMAP, GRID, SEMSEG }

    /** Один объект в пикселях: центр + полуразмеры ограничивающего прямоугольника. */
    public record ShapeInstance(
            ShapeType type,
            int cx, int cy,
            int halfW, int halfH,
            Color color
    ) {
        public int left()   { return cx - halfW; }
        public int right()  { return cx + halfW; }
        public int top()    { return cy - halfH; }
        public int bottom() { return cy + halfH; }
        public int width()  { return 2 * halfW; }
        public int height() { return 2 * halfH; }
        public Rectangle bounds() { return new Rectangle(left(), top(), width(), height()); }
        public int classIndex()   { return type.ordinal(); }
    }

    /** Полностью размеченный пример. */
    public record LabeledSample(
            BufferedImage image,
            List<ShapeInstance> objects,
            Task task,
            float[] targets,
            float[] inputs
    ) {}

    // ------------------------------------------------------------------ параметры

    public static final int NUM_CLASSES = ShapeType.values().length;

    private final float emptyBias;   // вероятность n=0 для DETECTION

    private final int size;
    private final int maxObjects;
    private final int minShape;
    private final int maxShape;
    private final int gridCells;
    private final float heatmapSigma;
    private final Random rnd;

    private static final Color[] PALETTE = {
            new Color(220,  30,  30), new Color( 30, 120, 220),
            new Color( 30, 170,  60), new Color(230, 160,  20),
            new Color(150,  40, 200), new Color( 20, 180, 180),
            new Color(200,  60, 150), new Color(120,  80,  40),
    };

    public ImageDatasetGenerator(int size, int maxObjects,
                                 int minShape, int maxShape, long seed, float emptyBias) {
        this(emptyBias, size, maxObjects, minShape, maxShape, /*gridCells=*/8, /*sigma=*/1.5f, seed);
    }

    public ImageDatasetGenerator(float emptyBias, int size, int maxObjects,
                                 int minShape, int maxShape,
                                 int gridCells, float heatmapSigma, long seed) {
        if (emptyBias < 0f || emptyBias > 1f)
            throw new IllegalArgumentException("emptyBias in [0,1]");
        this.emptyBias = emptyBias;

        if (size < 16)            throw new IllegalArgumentException("size >= 16");
        if (maxObjects < 0)       throw new IllegalArgumentException("maxObjects >= 0");
        if (minShape < 1)         throw new IllegalArgumentException("minShape >= 1");
        if (maxShape < minShape)  throw new IllegalArgumentException("maxShape >= minShape");
        if (2 * maxShape >= size) throw new IllegalArgumentException("2*maxShape < size");
        if (gridCells < 1)        throw new IllegalArgumentException("gridCells >= 1");
        if (heatmapSigma <= 0)    throw new IllegalArgumentException("heatmapSigma > 0");

        this.size = size;
        this.maxObjects = maxObjects;
        this.minShape = minShape;
        this.maxShape = maxShape;
        this.gridCells = gridCells;
        this.heatmapSigma = heatmapSigma;
        this.rnd = new Random(seed);
    }

    // ------------------------------------------------------------------ генерация

    /** Сгенерировать {@code count} примеров под задачу. */
    public List<LabeledSample> generate(int count, Task task) {
        List<LabeledSample> out = new ArrayList<>(count);
        for (int i = 0; i < count; i++) out.add(generateOne(task));
        return out;
    }

    /** Сгенерировать один пример. */
    public LabeledSample generateOne(Task task) {
        BufferedImage img = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, size, size);

        int n;
        if (maxObjects == 0 || rnd.nextFloat() < emptyBias) {
            n = 0;
        } else {
            // перекос к малым n: n = 1 + floor(U^2 * maxObjects)
            float u = rnd.nextFloat();
            n = 1 + (int) (u * u * maxObjects);
            n = Math.min(n, maxObjects);
        }

        List<ShapeInstance> objects = new ArrayList<>(n);
        for (int attempt = 0; attempt < n * 30 && objects.size() < n; attempt++) {
            ShapeInstance s = tryRandomShape(objects);
            if (s == null) continue;
            objects.add(s);
            drawShape(g, s);
        }
        g.dispose();

        float[] inputs = toFlatRgb(img);
        float[] targets = switch (task) {
            case DETECTION -> buildDetectionTargets(objects);
            case HEATMAP   -> buildHeatmapTargets(objects, size, heatmapSigma);
            case GRID      -> buildGridTargets(objects, size, gridCells);
            case SEMSEG    -> buildSemsegTargets(objects, size);
        };

        return new LabeledSample(img, objects, task, targets, inputs);
    }

    private ShapeInstance tryRandomShape(List<ShapeInstance> placed) {
        ShapeType type = ShapeType.values()[rnd.nextInt(ShapeType.values().length)];
        int halfW = minShape + rnd.nextInt(maxShape - minShape + 1);
        int halfH = minShape + rnd.nextInt(maxShape - minShape + 1);
        int cx = halfW + rnd.nextInt(size - 2 * halfW);
        int cy = halfH + rnd.nextInt(size - 2 * halfH);
        Color color = PALETTE[rnd.nextInt(PALETTE.length)];

        Rectangle r = new Rectangle(cx - halfW, cy - halfH, 2 * halfW, 2 * halfH);
        Rectangle padded = new Rectangle(r.x - 2, r.y - 2, r.width + 4, r.height + 4);
        for (ShapeInstance s : placed) {
            if (padded.intersects(s.bounds())) return null;
        }
        return new ShapeInstance(type, cx, cy, halfW, halfH, color);
    }

    private void drawShape(Graphics2D g, ShapeInstance s) {
        g.setColor(s.color());
        int x = s.left(), y = s.top(), w = s.width(), h = s.height();
        switch (s.type()) {
            case CIRCLE -> g.fillOval(x, y, w, h);
            case SQUARE -> g.fillRect(x, y, w, h);
            case TRIANGLE -> {
                int[] xs = { s.cx(), s.left(), s.right() };
                int[] ys = { s.top(), s.bottom(), s.bottom() };
                g.fillPolygon(xs, ys, 3);
            }
            case CROSS -> {
                int armW = Math.max(2, w / 3);
                int armH = Math.max(2, h / 3);
                g.fillRect(s.cx() - armW / 2, y, armW, h);
                g.fillRect(x, s.cy() - armH / 2, w, armH);
            }
        }
    }

    // ------------------------------------------------------------------ построители целей

    /** Multi-label: {@code flat(4)} с 0/1. */
    public static float[] buildDetectionTargets(List<ShapeInstance> objects) {
        float[] t = new float[NUM_CLASSES];
        for (ShapeInstance s : objects) t[s.classIndex()] = 1f;
        return t;
    }

    /**
     * YOLO-grid: {@code flat(G*G*(5+C))}.
     *
     * <p>Раскладка ячейки: {@code [obj, cx, cy, w, h, onehot(C)]},
     * где {@code cx, cy ∈ [0,1)} — центр относительно ячейки,
     * {@code w, h ∈ (0,1]} — относительно картинки.
     *
     * <p><b>Ограничение:</b> ≤ 1 объект на ячейку (последний побеждает).
     */
    public static float[] buildGridTargets(List<ShapeInstance> objects,
                                           int size, int gridCells) {
        final int G = gridCells;
        final int n = G * G;
        final float cell = (float) size / G;
        final int per = 5 + NUM_CLASSES;   // 9

        // channel-major: t[c * n + gy*G + gx]
        float[] t = new float[per * n];

        for (ShapeInstance s : objects) {
            int gx = Math.min(G - 1, (int) (s.cx() / cell));
            int gy = Math.min(G - 1, (int) (s.cy() / cell));
            int idx = gy * G + gx;

            t[0 * n + idx] = 1f;                                   // obj
            t[1 * n + idx] = (s.cx() - gx * cell) / cell;          // cx_local
            t[2 * n + idx] = (s.cy() - gy * cell) / cell;          // cy_local
            t[3 * n + idx] = s.width()  / (float) size;            // w
            t[4 * n + idx] = s.height() / (float) size;            // h
            t[(5 + s.classIndex()) * n + idx] = 1f;                // one-hot
        }
        return t;
    }

    /**
     * CenterNet heatmap: {@code flat(C*S*S)}, гауссианы в центрах.
     * Значение в пикселе — максимум по объектам (пики не складываются).
     */
    public static float[] buildHeatmapTargets(List<ShapeInstance> objects,
                                              int size, float sigma) {
        final int n = size * size;
        float[] t = new float[NUM_CLASSES * n];
        final int r = (int) Math.ceil(3 * sigma);
        final float twoSigmaSq = 2f * sigma * sigma;

        for (ShapeInstance s : objects) {
            int base = s.classIndex() * n;
            for (int dy = -r; dy <= r; dy++) {
                int y = s.cy() + dy;
                if (y < 0 || y >= size) continue;
                for (int dx = -r; dx <= r; dx++) {
                    int x = s.cx() + dx;
                    if (x < 0 || x >= size) continue;
                    float v = (float) Math.exp(-(dx * dx + dy * dy) / twoSigmaSq);
                    int idx = base + y * size + x;
                    if (v > t[idx]) t[idx] = v;
                }
            }
        }
        return t;
    }

    /**
     * Semantic segmentation: {@code flat((C+1)*S*S)}, one-hot на пиксель.
     * Канал 0 — фон, каналы 1..C — фигуры.
     */
    public static float[] buildSemsegTargets(List<ShapeInstance> objects, int size) {
        final int C = NUM_CLASSES + 1;
        final int n = size * size;
        float[] t = new float[C * n];
        java.util.Arrays.fill(t, 0, n, 1f);   // фон по умолчанию

        for (ShapeInstance s : objects) {
            int cls = s.classIndex() + 1;
            for (int y = Math.max(0, s.top()); y <= Math.min(size - 1, s.bottom()); y++) {
                for (int x = Math.max(0, s.left()); x <= Math.min(size - 1, s.right()); x++) {
                    if (!inside(s, x, y)) continue;
                    int pix = y * size + x;
                    for (int c = 0; c < C; c++) t[c * n + pix] = 0f;
                    t[cls * n + pix] = 1f;
                }
            }
        }
        return t;
    }

    private static boolean inside(ShapeInstance s, int x, int y) {
        double dx = x - s.cx(), dy = y - s.cy();
        return switch (s.type()) {
            case CIRCLE -> (dx * dx) / (double) (s.halfW() * s.halfW())
                    + (dy * dy) / (double) (s.halfH() * s.halfH()) <= 1.0;
            case SQUARE -> Math.abs(dx) <= s.halfW() && Math.abs(dy) <= s.halfH();
            case TRIANGLE -> {
                double t = (y - s.top()) / (double) s.height();
                yield t >= 0 && t <= 1 && Math.abs(dx) <= t * s.halfW();
            }
            case CROSS -> {
                int armW = Math.max(2, s.width() / 3);
                int armH = Math.max(2, s.height() / 3);
                yield (Math.abs(dx) <= armW / 2.0 && Math.abs(dy) <= s.halfH())
                        || (Math.abs(dy) <= armH / 2.0 && Math.abs(dx) <= s.halfW());
            }
        };
    }

    // ------------------------------------------------------------------ утилиты

    /** RGB в {@code [0,1]}, плоская раскладка {@code [R(все), G(все), B(все)]}. */
    public static float[] toFlatRgb(BufferedImage img) {
        int w = img.getWidth(), h = img.getHeight();
        float[] out = new float[3 * w * h];
        int[] rgb = img.getRGB(0, 0, w, h, null, 0, w);
        int n = w * h;
        for (int i = 0; i < n; i++) {
            int c = rgb[i];
            out[i]         = ((c >> 16) & 0xFF) / 255f;
            out[n + i]     = ((c >>  8) & 0xFF) / 255f;
            out[2 * n + i] = ( c        & 0xFF) / 255f;
        }
        return out;
    }

    /** Из flat RGB обратно в BufferedImage. */
    public static BufferedImage fromFlatRgb(float[] flat, int w, int h) {
        if (flat.length != 3 * w * h) throw new IllegalArgumentException("length != 3*w*h");
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

    private static int clamp255(float v) {
        int i = Math.round(v);
        return i < 0 ? 0 : Math.min(i, 255);
    }

    /** Совместимость с {@link SampleDataset}. */
    public static SampleDataset<Sample> toSimple(List<LabeledSample> samples) {
        List<Sample> all = new ArrayList<>(samples.size());
        for (LabeledSample s : samples) all.add(new Sample(s.inputs(), s.targets()));
        return SampleDataset.split(all, 0.8f);
    }
}