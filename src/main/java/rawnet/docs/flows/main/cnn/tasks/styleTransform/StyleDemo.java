package rawnet.docs.flows.main.cnn.tasks.styleTransform;

import lombok.extern.slf4j.Slf4j;
import rawnet.code.net.model.Model;
import rawnet.code.net.model.sequence.Sequence;
import rawnet.docs.flows.main.cnn.tasks.imageDetect.ImageDatasetGenerator;
import rawnet.docs.flows.main.cnn.tasks.imageDetect.ImageUtil;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Демонстрация Style Transfer на <b>пользовательских</b> картинках.
 *
 * <p>Два сценария:
 * <ol>
 *   <li><b>Медленный</b> ({@link SlowStyleTransfer}): оптимизирует пиксели
 *       одной картинки, веса CNN заморожены. 300 итераций ≈ 15–30 секунд.</li>
 *   <li><b>Быстрый</b> ({@link StyleStylizerTrainer}): обучает сеть-стилизатор
 *       {@code S} на одной стилевой картинке. 2000 итераций ≈ 5–10 минут,
 *       затем применение к новой картинке — один forward (~50 мс).</li>
 * </ol>
 *
 * <h3>Входные файлы</h3>
 * Кладутся в {@code images/} рядом с корнем проекта:
 * <ul>
 *   <li>{@code images/content.png} — картинка «формы» (что рисуем);</li>
 *   <li>{@code images/style.png}   — картинка «стиля» (чем рисуем).</li>
 * </ul>
 * Обе приводятся к {@link #SIZE}×{@link #SIZE} (билинейно).
 *
 * <h3>Выходные файлы</h3>
 * В {@code out/style/}:
 * <ul>
 *   <li>{@code 01-content.png}, {@code 02-style.png} — входы (приведённые к SIZE×SIZE);</li>
 *   <li>{@code 03-slow-styled.png} — медленный результат;</li>
 *   <li>{@code 04-slow-triptych.png} — content | styled | style;</li>
 *   <li>{@code 05-fast-styled.png} — быстрый результат (на <b>новой</b> content);</li>
 *   <li>{@code 06-fast-triptych.png} — content | styled | style.</li>
 * </ul>
 */
@Slf4j
public final class StyleDemo {

    private static final int SIZE = 128;

    /** Слой для content loss: Conv128 на 16×16 — «объекты, но не класс». */
    private static final int CONTENT_LAYER = 9;

    /**
     * Слои для style loss: <b>только Conv</b>, без MaxPool.
     * 0 = Conv16 (края), 3 = Conv32 (текстуры), 6 = Conv64, 9 = Conv128.
     */
    private static final int[] STYLE_LAYERS = { 0, 3, 6, 9 };

    private static final float ALPHA = 1f;
    private static final float BETA  = 500f;

    private static final int   SLOW_ITERS = 600;
    private static final float SLOW_LR    = 0.01f;

    private static final int   FAST_ITERS = 1000;
    private static final float FAST_LR    = 1e-2f;

    private static final File OUT_DIR       = new File("out/style");
    private static final Path CNN_PATH      = Path.of("model/grid/grid-cnn.bin");
    private static final Path STYLIZER_PATH = Path.of("model/style/stylizer.bin");

    private static final String CONTENT_IMG = "images/content.png";
    private static final String STYLE_IMG   = "images/style.png";

    private StyleDemo() {}

    public static void main(String[] args) throws Exception {
        ImageUtil.clearDir(OUT_DIR);

        // ── 1. CNN для фич (заморожена) ───────────────────────────────
        if (!Files.exists(CNN_PATH)) {
            log.error("Не найдена CNN: {}. Сначала запустите ImageDetectDemo.", CNN_PATH);
            return;
        }
        Model cnn = Model.load(CNN_PATH);
        log.info("CNN загружена: {} → {}", cnn.inputShape(), cnn.outputShape());

        // ── 2. Пользовательские картинки ──────────────────────────────
        BufferedImage content = loadSquare(CONTENT_IMG, SIZE);
        BufferedImage style   = loadSquare(STYLE_IMG,   SIZE);

        ImageUtil.savePng(content, new File(OUT_DIR, "01-content.png"));
        ImageUtil.savePng(style,   new File(OUT_DIR, "02-style.png"));
        log.info("content и style сохранены в {}", OUT_DIR.getAbsolutePath());

        // ── 3. Медленный Style Transfer ───────────────────────────────
        runSlow(cnn, content, style);

        // ── 4. Быстрый (обучение + применение) ────────────────────────
        runFast(cnn, style);

        log.info("Готово. Артефакты: {}", OUT_DIR.getAbsolutePath());
    }

    // ==================================================================
    // ====================== СЦЕНАРИЙ 1: МЕДЛЕННЫЙ =====================
    // ==================================================================

    private static void runSlow(Model cnn,
                                BufferedImage content,
                                BufferedImage style) throws IOException {
        log.info("=== SLOW Style Transfer ({} итераций) ===", SLOW_ITERS);
        log.info("  contentLayer={}, styleLayers={}, α={}, β={}",
                CONTENT_LAYER, java.util.Arrays.toString(STYLE_LAYERS), ALPHA, BETA);

        long t0 = System.currentTimeMillis();
        BufferedImage styled = SlowStyleTransfer.transfer(
                cnn, content, style,
                CONTENT_LAYER, STYLE_LAYERS,
                ALPHA, BETA,
                SLOW_ITERS, SLOW_LR);
        log.info("Готово за {} мс", System.currentTimeMillis() - t0);

        ImageUtil.savePng(styled, new File(OUT_DIR, "03-slow-styled.png"));

        BufferedImage tri = triptych(content, styled, style,
                "Content", "Slow Style Transfer", "Style");
        ImageUtil.savePng(tri, new File(OUT_DIR, "04-slow-triptych.png"));
    }

    // ==================================================================
    // ======================= СЦЕНАРИЙ 2: БЫСТРЫЙ ======================
    // ==================================================================

    private static void runFast(Model cnn, BufferedImage style) throws IOException {
        Sequence stylizer;

        if (Files.exists(STYLIZER_PATH)) {
            log.info("=== FAST Style Transfer: найден {}, загружаем ===", STYLIZER_PATH);
            stylizer = (Sequence) Model.load(STYLIZER_PATH);
        } else {
            log.info("=== FAST Style Transfer: обучение ({} итераций) ===", FAST_ITERS);
            log.info("  contentLayer={}, styleLayers={}, α={}, β={}",
                    CONTENT_LAYER, java.util.Arrays.toString(STYLE_LAYERS), ALPHA, BETA);

            stylizer = StyleStylizer.build(SIZE);

            // Для обучения нужен генератор content-картинок (S учится на случайных).
            // Берём тот же генератор, что и в ImageDetectDemo — он даёт разнообразные
            // картинки (фигуры), на которых S учится «не терять форму».
            ImageDatasetGenerator gen = new ImageDatasetGenerator(
                    SIZE, /*maxObjects=*/ 4, /*minShape=*/ 10, /*maxShape=*/ 22, /*seed=*/ 42L, 0.1f);

            long t0 = System.currentTimeMillis();
            StyleStylizerTrainer.train(
                    stylizer, cnn, style, gen,
                    CONTENT_LAYER, STYLE_LAYERS,
                    ALPHA, BETA,
                    FAST_ITERS, FAST_LR);
            log.info("Обучение: {} мс", System.currentTimeMillis() - t0);

            stylizer.save(STYLIZER_PATH);
            log.info("Стилизатор сохранён: {}", STYLIZER_PATH);
        }

        // Применяем к НОВОЙ картинке (не той, что была в медленном).
        // Для демо «новой» может быть та же content — но лучше взять другую,
        // чтобы показать обобщение. Если images/ содержит только одну —
        // используем её же.
        BufferedImage freshContent;
        String freshPath = "images/content2.png";
        if (Files.exists(Path.of(freshPath))) {
            freshContent = loadSquare(freshPath, SIZE);
            log.info("Применение к новой картинке: {}", freshPath);
        } else {
            freshContent = loadSquare(CONTENT_IMG, SIZE);
            log.info("Применение к той же content (content2.png не найден)");
        }

        long t0 = System.currentTimeMillis();
        BufferedImage styled = StyleStylizer.apply(stylizer, freshContent);
        log.info("Применение: {} мс (один forward)", System.currentTimeMillis() - t0);

        ImageUtil.savePng(styled, new File(OUT_DIR, "05-fast-styled.png"));

        BufferedImage tri = triptych(freshContent, styled, style,
                "Content", "Fast Style Transfer", "Style");
        ImageUtil.savePng(tri, new File(OUT_DIR, "06-fast-triptych.png"));
    }

    // ==================================================================
    // ==================== ЗАГРУЗКА КАРТИНОК ===========================
    // ==================================================================

    /**
     * Загружает картинку и приводит к {@code size×size} (билинейно).
     *
     * <p>Не сохраняет пропорции — растягивает/сжимает. Для Style Transfer
     * это нормально: важна «текстура», а не точные пропорции.
     *
     * @param path путь к JPG/PNG
     * @param size сторона результата
     * @throws IOException если файл не найден или не читается
     */
    private static BufferedImage loadSquare(String path, int size) throws IOException {
        File f = new File(path);
        if (!f.exists())
            throw new IOException("Картинка не найдена: " + f.getAbsolutePath()
                    + "\nПоложите её в images/ рядом с корнем проекта.");

        BufferedImage src = ImageIO.read(f);
        if (src == null)
            throw new IOException("Не удалось прочитать картинку: " + f.getAbsolutePath()
                    + " (формат не поддерживается?)");

        if (src.getWidth() == size && src.getHeight() == size) return src;

        BufferedImage out = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = out.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.setRenderingHint(RenderingHints.KEY_RENDERING,
                RenderingHints.VALUE_RENDER_QUALITY);
        g.drawImage(src, 0, 0, size, size, null);
        g.dispose();
        return out;
    }

    // ==================================================================
    // =========================== ТРИПТИХ ==============================
    // ==================================================================

    private static BufferedImage triptych(BufferedImage content,
                                          BufferedImage styled,
                                          BufferedImage style,
                                          String contentTitle,
                                          String styledTitle,
                                          String styleTitle) {
        if (content.getWidth() != styled.getWidth() || content.getWidth() != style.getWidth()
                || content.getHeight() != styled.getHeight() || content.getHeight() != style.getHeight()) {
            throw new IllegalArgumentException("Все три панели должны быть одного размера");
        }

        int w = content.getWidth(), h = content.getHeight();
        int header = 24, gap = 6;
        int totalW = w * 3 + gap * 2;
        int totalH = h + header;

        BufferedImage out = new BufferedImage(totalW, totalH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = out.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING,
                RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, totalW, totalH);

        g.setColor(Color.BLACK);
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 12));
        FontMetrics fm = g.getFontMetrics();

        String[] titles = { contentTitle, styledTitle, styleTitle };
        BufferedImage[] panels = { content, styled, style };

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
}