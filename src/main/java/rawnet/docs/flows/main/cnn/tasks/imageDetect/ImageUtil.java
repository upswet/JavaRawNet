package rawnet.docs.flows.main.cnn.tasks.imageDetect;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import javax.imageio.ImageIO;

/**
 * Утилиты для работы с изображениями и декодирования предсказаний.
 *
 * <p>Декодеры (статические методы):
 * <ul>
 *   <li>{@link #decodeGrid} — YOLO-grid: порог + восстановление + NMS;</li>
 *   <li>{@link #decodeHeatmap} — CenterNet: сигмоида + локальный максимум;</li>
 *   <li>{@link #decodeSemseg} — SEMSEG: argmax + connected components.</li>
 * </ul>
 *
 * <p>Визуализация: {@link #drawBoxes}, {@link #triptych}.
 */
public final class ImageUtil {

    private ImageUtil() {}

    // ==================================================================
    // ======================= ДЕКОДЕРЫ ПРЕДСКАЗАНИЙ ===================
    // ==================================================================

    /**
     * YOLO-grid: {@code flat(G*G*(5+C))} логитов → список объектов.
     *
     * <p>Алгоритм: сигмоида obj, порог, восстановление центра и размеров,
     * argmax по классам, NMS внутри класса.
     */
    public static List<ImageDatasetGenerator.ShapeInstance> decodeGrid(
            float[] logits, int size, int gridCells, int numClasses,
            float confThreshold, float nmsIouThreshold) {

        final int cells = gridCells * gridCells;
        final float cell = (float) size / gridCells;
        List<ImageDatasetGenerator.ShapeInstance> raw = new ArrayList<>();

        for (int gy = 0; gy < gridCells; gy++) {
            for (int gx = 0; gx < gridCells; gx++) {
                int idx = gy * gridCells + gx;
                if (sigmoid(logits[idx]) < confThreshold) continue;   // obj

                float cxLocal = sigmoid(logits[1 * cells + idx]);
                float cyLocal = sigmoid(logits[2 * cells + idx]);
                float wRel    = sigmoid(logits[3 * cells + idx]);
                float hRel    = sigmoid(logits[4 * cells + idx]);

                int cx = Math.round((gx + cxLocal) * cell);
                int cy = Math.round((gy + cyLocal) * cell);
                int w  = Math.max(1, Math.round(wRel * size));
                int h  = Math.max(1, Math.round(hRel * size));

                int bestCls = 0;
                float bestV = logits[5 * cells + idx];
                for (int k = 1; k < numClasses; k++) {
                    float v = logits[(5 + k) * cells + idx];
                    if (v > bestV) { bestV = v; bestCls = k; }
                }

                raw.add(new ImageDatasetGenerator.ShapeInstance(
                        ImageDatasetGenerator.ShapeType.values()[bestCls],
                        cx, cy, w / 2, h / 2, Color.RED));
            }
        }
        return nms(raw, numClasses, nmsIouThreshold);
    }

    /** Рисует список найденных классов в левом верхнем углу (для DETECTION). */
    public static BufferedImage drawClassList(BufferedImage img, List<String> classNames,
                                              Color color) {
        BufferedImage copy = deepCopy(img);
        Graphics2D g = copy.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 14));
        g.setColor(color);
        int y = 20;
        if (classNames.isEmpty()) {
            g.drawString("(пусто)", 6, y);
        } else {
            for (String name : classNames) {
                g.drawString(name, 6, y);
                y += 18;
            }
        }
        g.dispose();
        return copy;
    }

    /** NMS внутри класса; «уверенность» — площадь (прокси). */
    private static List<ImageDatasetGenerator.ShapeInstance> nms(
            List<ImageDatasetGenerator.ShapeInstance> boxes,
            int numClasses, float iouThreshold) {

        List<ImageDatasetGenerator.ShapeInstance> sorted = new ArrayList<>(boxes);
        sorted.sort(Comparator.comparingInt(
                (ImageDatasetGenerator.ShapeInstance s) -> s.width() * s.height()).reversed());

        List<ImageDatasetGenerator.ShapeInstance> kept = new ArrayList<>();
        boolean[] removed = new boolean[sorted.size()];
        for (int i = 0; i < sorted.size(); i++) {
            if (removed[i]) continue;
            ImageDatasetGenerator.ShapeInstance a = sorted.get(i);
            kept.add(a);
            for (int j = i + 1; j < sorted.size(); j++) {
                if (removed[j]) continue;
                ImageDatasetGenerator.ShapeInstance b = sorted.get(j);
                if (a.type() != b.type()) continue;
                if (iou(a, b) > iouThreshold) removed[j] = true;
            }
        }
        return kept;
    }

    private static float iou(ImageDatasetGenerator.ShapeInstance a,
                             ImageDatasetGenerator.ShapeInstance b) {
        int x1 = Math.max(a.left(), b.left());
        int y1 = Math.max(a.top(), b.top());
        int x2 = Math.min(a.right(), b.right());
        int y2 = Math.min(a.bottom(), b.bottom());
        int iw = Math.max(0, x2 - x1);
        int ih = Math.max(0, y2 - y1);
        int inter = iw * ih;
        if (inter == 0) return 0f;
        int union = a.width() * a.height() + b.width() * b.height() - inter;
        return union == 0 ? 0f : (float) inter / union;
    }

    /**
     * CenterNet heatmap: {@code flat(C*S*S)} логитов → список центров.
     *
     * <p>Сигмоида → порог → локальный максимум 3×3 (tie-break по индексу).
     * Размеры не предсказываются — подставляем {@code 2*fallbackHalf}.
     */
    public static List<ImageDatasetGenerator.ShapeInstance> decodeHeatmap(
            float[] logits, int size, int numClasses,
            float peakThreshold, int fallbackHalf) {

        int n = size * size;
        float[] prob = new float[logits.length];
        for (int i = 0; i < logits.length; i++) prob[i] = sigmoid(logits[i]);

        List<ImageDatasetGenerator.ShapeInstance> out = new ArrayList<>();
        for (int c = 0; c < numClasses; c++) {
            int base = c * n;
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    int idx = base + y * size + x;
                    if (prob[idx] < peakThreshold) continue;
                    if (!isLocalMax(prob, c, x, y, size)) continue;

                    out.add(new ImageDatasetGenerator.ShapeInstance(
                            ImageDatasetGenerator.ShapeType.values()[c],
                            x, y, fallbackHalf, fallbackHalf, Color.RED));
                }
            }
        }
        return out;
    }

    private static boolean isLocalMax(float[] prob, int c, int x, int y, int size) {
        int base = c * size * size;
        int idx = base + y * size + x;
        float v = prob[idx];
        for (int dy = -1; dy <= 1; dy++) {
            int ny = y + dy;
            if (ny < 0 || ny >= size) continue;
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = x + dx;
                if (nx < 0 || nx >= size) continue;
                int nIdx = base + ny * size + nx;
                float nv = prob[nIdx];
                if (nv > v || (nv == v && nIdx < idx)) return false;
            }
        }
        return true;
    }

    /**
     * SEMSEG: {@code flat(C*S*S)} логитов → список объектов через
     * connected components по argmax-маске. Канал 0 — фон.
     */
    public static List<ImageDatasetGenerator.ShapeInstance> decodeSemseg(
            float[] logits, int size, int numClasses, int minArea) {

        int n = size * size;
        int[] mask = new int[n];
        for (int i = 0; i < n; i++) {
            int best = 0;
            float bestV = logits[i];
            for (int c = 1; c < numClasses; c++) {
                float v = logits[c * n + i];
                if (v > bestV) { bestV = v; best = c; }
            }
            mask[i] = best;
        }

        boolean[] used = new boolean[n];
        List<ImageDatasetGenerator.ShapeInstance> out = new ArrayList<>();
        int[] dx = { 1, -1, 0, 0 };
        int[] dy = { 0, 0, 1, -1 };

        for (int i = 0; i < n; i++) {
            if (used[i]) continue;
            int cls = mask[i];
            used[i] = true;
            if (cls == 0) continue;

            ArrayDeque<Integer> q = new ArrayDeque<>();
            q.add(i);
            int minX = i % size, maxX = minX, minY = i / size, maxY = minY, area = 0;
            while (!q.isEmpty()) {
                int cur = q.poll();
                int x = cur % size, y = cur / size;
                minX = Math.min(minX, x); maxX = Math.max(maxX, x);
                minY = Math.min(minY, y); maxY = Math.max(maxY, y);
                area++;
                for (int k = 0; k < 4; k++) {
                    int nx = x + dx[k], ny = y + dy[k];
                    if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
                    int ni = ny * size + nx;
                    if (used[ni] || mask[ni] != cls) continue;
                    used[ni] = true;
                    q.add(ni);
                }
            }
            if (area < minArea) continue;

            int shapeIdx = cls - 1;
            if (shapeIdx < 0 || shapeIdx >= ImageDatasetGenerator.ShapeType.values().length)
                continue;

            int w = maxX - minX + 1, h = maxY - minY + 1;
            out.add(new ImageDatasetGenerator.ShapeInstance(
                    ImageDatasetGenerator.ShapeType.values()[shapeIdx],
                    (minX + maxX) / 2, (minY + maxY) / 2,
                    Math.max(1, w / 2), Math.max(1, h / 2), Color.RED));
        }
        return out;
    }

    // ==================================================================
    // ========================= ВИЗУАЛИЗАЦИЯ ===========================
    // ==================================================================

    /** Рисует рамки + подписи поверх копии {@code img}. */
    public static BufferedImage drawBoxes(BufferedImage img,
                                          List<ImageDatasetGenerator.ShapeInstance> objects,
                                          boolean withLabels, Color boxColor) {
        BufferedImage copy = deepCopy(img);
        Graphics2D g = copy.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setStroke(new BasicStroke(2f));
        Font font = new Font(Font.SANS_SERIF, Font.BOLD, 11);
        g.setFont(font);
        FontMetrics fm = g.getFontMetrics();

        for (ImageDatasetGenerator.ShapeInstance s : objects) {
            g.setColor(boxColor);
            g.drawRect(s.left(), s.top(), s.width(), s.height());
            if (withLabels) {
                String label = s.type().name();
                int tw = fm.stringWidth(label);
                int th = fm.getHeight();
                int tx = s.left();
                int ty = Math.max(0, s.top() - th);
                g.setColor(new Color(255, 255, 255, 200));
                g.fillRect(tx, ty, tw + 4, th);
                g.setColor(boxColor.darker());
                g.drawString(label, tx + 2, ty + fm.getAscent() + 1);
            }
        }
        g.dispose();
        return copy;
    }

    /** Триптих {@code [вход | предсказание | эталон]} с заголовками. */
    public static BufferedImage triptych(BufferedImage input,
                                         BufferedImage prediction,
                                         BufferedImage groundTruth) {
        if (input.getWidth() != prediction.getWidth()
                || input.getWidth() != groundTruth.getWidth()
                || input.getHeight() != prediction.getHeight()
                || input.getHeight() != groundTruth.getHeight())
            throw new IllegalArgumentException("Все панели должны быть одного размера");

        int w = input.getWidth(), h = input.getHeight();
        int header = 20, gap = 6;
        BufferedImage out = new BufferedImage(w * 3 + gap * 2, h + header,
                BufferedImage.TYPE_INT_RGB);
        Graphics2D g = out.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, out.getWidth(), out.getHeight());

        g.setColor(Color.BLACK);
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 12));
        FontMetrics fm = g.getFontMetrics();
        String[] titles = { "Вход", "Предсказание", "Эталон" };
        BufferedImage[] panels = { input, prediction, groundTruth };

        for (int i = 0; i < 3; i++) {
            int x = i * (w + gap);
            int tw = fm.stringWidth(titles[i]);
            g.drawString(titles[i], x + (w - tw) / 2, header - 6);
            g.drawImage(panels[i], x, header, null);
            g.setColor(Color.GRAY);
            g.drawRect(x, header, w - 1, h - 1);
            g.setColor(Color.BLACK);
        }
        g.dispose();
        return out;
    }

    /** Сохранить картинку в PNG (создаёт родительские каталоги). */
    public static void savePng(BufferedImage img, File file) throws IOException {
        File parent = file.getAbsoluteFile().getParentFile();
        if (parent != null && !parent.exists()) parent.mkdirs();
        ImageIO.write(img, "png", file);
    }

    /** Рекурсивно удалить каталог (если существует). */
    public static void clearDir(File dir) {
        if (dir == null || !dir.exists()) return;
        File[] children = dir.listFiles();
        if (children != null) for (File c : children) clearDir(c);
        if (!dir.delete()) System.err.println("Не удалось удалить: " + dir);
    }

    public static BufferedImage deepCopy(BufferedImage src) {
        BufferedImage copy = new BufferedImage(src.getWidth(), src.getHeight(), src.getType());
        Graphics2D g = copy.createGraphics();
        g.drawImage(src, 0, 0, null);
        g.dispose();
        return copy;
    }

    static float sigmoid(float z) {
        if (z <= -30f) return 0f;
        if (z >=  30f) return 1f;
        return 1f / (1f + (float) Math.exp(-z));
    }

    /**
     * Оверлей для задачи DETECTION: рисует список классов, у которых
     * {@code sigmoid(pred[i]) >= threshold}, в левом верхнем углу.
     *
     * <p>Если ничего не найдено — пишет «(пусто)». Если классов много —
     * переносит на новую строку.
     */
    public static BufferedImage drawDetectionOverlay(BufferedImage img,
                                                     float[] logits,
                                                     float threshold) {
        BufferedImage copy = deepCopy(img);
        Graphics2D g = copy.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 14));
        FontMetrics fm = g.getFontMetrics();

        List<String> found = new ArrayList<>();
        for (int i = 0; i < ImageDatasetGenerator.NUM_CLASSES && i < logits.length; i++) {
            if (sigmoid(logits[i]) >= threshold) {
                found.add(ImageDatasetGenerator.ShapeType.values()[i].name());
            }
        }

        String[] lines = found.isEmpty()
                ? new String[]{ "(пусто)" }
                : found.toArray(new String[0]);

        int lineH = fm.getHeight();
        int maxW = 0;
        for (String s : lines) maxW = Math.max(maxW, fm.stringWidth(s));

        // плашка под текст
        int pad = 4;
        int boxW = maxW + 2 * pad;
        int boxH = lineH * lines.length + 2 * pad;
        g.setColor(new Color(255, 255, 255, 210));
        g.fillRect(4, 4, boxW, boxH);
        g.setColor(new Color(200, 0, 0));
        g.drawRect(4, 4, boxW, boxH);

        g.setColor(new Color(180, 0, 0));
        int y = 4 + pad + fm.getAscent();
        for (String s : lines) {
            g.drawString(s, 4 + pad, y);
            y += lineH;
        }
        g.dispose();
        return copy;
    }
}