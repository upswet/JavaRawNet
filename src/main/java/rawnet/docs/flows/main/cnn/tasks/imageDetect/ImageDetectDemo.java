package rawnet.docs.flows.main.cnn.tasks.imageDetect;

import lombok.extern.slf4j.Slf4j;
import rawnet.code.dataset.SampleDataset;
import rawnet.code.dataset.Sample;
import rawnet.code.net.TensorShape;
import rawnet.code.net.core.activate.LeakyRelu;
import rawnet.code.net.core.activate.Sigmoid;
import rawnet.code.net.layers.cnn.ConvLayer;
import rawnet.code.net.layers.cnn.MaxPoolLayer;
import rawnet.code.net.layers.cnn.UpsampleLayer;
import rawnet.code.net.layers.core.ActivationLayer;
import rawnet.code.net.layers.core.DenseLayer;
import rawnet.code.net.layers.core.DropoutLayer;
import rawnet.code.net.layers.structure.ReshapeLayer;
import rawnet.code.net.core.loss.BinaryCrossEntropy;
import rawnet.code.net.core.loss.Loss;
import rawnet.code.net.core.metric.Accuracy;
import rawnet.code.net.core.metric.F1;
import rawnet.code.net.core.metric.LossMetric;
import rawnet.code.net.core.metric.Metric;
import rawnet.code.net.core.metric.Precision;
import rawnet.code.net.core.metric.Recall;
import rawnet.code.net.model.Model;
import rawnet.code.net.model.ModelTraining;
import rawnet.code.net.model.sequence.Sequence;
import rawnet.code.net.core.optimizator.AdamW;
import rawnet.code.net.core.optimizator.Optimizer;
import rawnet.docs.flows.main.cnn.tasks.imageDetect.loss.GridLoss;
import rawnet.docs.flows.main.cnn.tasks.imageDetect.loss.HeatmapLoss;
import rawnet.docs.flows.main.cnn.tasks.imageDetect.loss.SegLoss;

import java.awt.*;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.BiPredicate;

/**
 * Демонстрация четырёх задач компьютерного зрения на синтетических фигурах.
 *
 * <p>Задачи:
 * <ul>
 *   <li><b>DETECTION</b> — multi-label: {@code flat(4)} вероятностей;</li>
 *   <li><b>GRID</b>      — YOLO-style: {@code flat(G·G·(5+C))} логитов;</li>
 *   <li><b>HEATMAP</b>   — CenterNet-style: {@code flat(C·S·S)} логитов;</li>
 *   <li><b>SEMSEG</b>    — semantic segmentation: {@code flat((C+1)·S·S)} логитов.</li>
 * </ul>
 *
 * <p>Все четыре сценария идут через один конвейер
 * {@link #loadOrTrain}: если файл модели существует — загружаем,
 * иначе строим, обучаем и сохраняем. Затем для каждого сценария
 * генерируются триптихи «вход | предсказание | эталон».
 *
 * <p>Запуск: {@code main} вызывает по очереди
 * {@link #runDetection}, {@link #runGrid}, {@link #runHeatmap},
 * {@link #runSemseg}.
 */
@Slf4j
public final class ImageDetectDemo {

    // ── общие параметры ────────────────────────────────────────────────
    private static final int SIZE         = 128;
    private static final int MAX_OBJECTS  = 8;
    private static final int MIN_SHAPE    = 5;
    private static final int MAX_SHAPE    = 13;
    private static final long SEED        = 42L;
    private static final int TRIPTYCH_CNT = 8;

    private ImageDetectDemo() {}

    public static void main(String[] args) throws Exception {
        runDetection();
        runGrid();
        runHeatmap();
        runSemseg();
        log.info("Все сценарии завершены.");
    }

    // ==================================================================
    // ===================== СЦЕНАРИЙ 1: DETECTION ======================
    // ==================================================================

    /**
     * Multi-label: 4 независимые вероятности (CIRCLE / SQUARE / CROSS / TRIANGLE).
     * Выход — {@code flat(4)}, loss — BCE.
     */
    private static void runDetection() throws Exception {
        log.info("=== DETECTION ===");
        File outDir = new File("out/detect");
        Path modelPath = Path.of("model/detect/detection-cnn.bin");

        // Для DETECTION — крупнее фигуры и больше пустых картинок,
        // иначе задача вырождается в «всегда [1,1,1,1]».
        ImageDatasetGenerator gen = new ImageDatasetGenerator(
                SIZE, /*maxObjects=*/4, /*minShape=*/10, /*maxShape=*/22, SEED, 0f);

        Sequence model = loadOrTrain(
                gen, ImageDatasetGenerator.Task.DETECTION, modelPath,
                /*trainCount=*/3000,
                ImageDetectDemo::buildDetectionModel,
                ds -> new BinaryCrossEntropy(),
                50,
                Model.earlyStop(3, 0.005f, log),
                loss -> List.of(
                        new LossMetric(loss),
                        new Accuracy(0.5f),
                        new Precision(0.5f),
                        new Recall(0.5f),
                        new F1(0.5f)),
                5e-4f
        );

        saveTriptychs(gen, model, ImageDatasetGenerator.Task.DETECTION, outDir,
                "det", (s, pred) -> {
                    // pred — flat(4) вероятностей; порог 0.5
                    List<String> found = new ArrayList<>();
                    for (int i = 0; i < ImageDatasetGenerator.NUM_CLASSES; i++) {
                        if (pred[i] >= 0.5f) {
                            found.add(ImageDatasetGenerator.ShapeType.values()[i].name());
                        }
                    }
                    var predicted = ImageUtil.drawClassList(s.image(), found, Color.RED);
                    var truth = ImageUtil.drawBoxes(s.image(), s.objects(), true, new Color(0, 140, 0));
                    return ImageUtil.triptych(s.image(), predicted, truth);
                });
    }


    private static Sequence buildDetectionModel() {
        return Sequence.of(List.of(
                new Sequence.InputDto(TensorShape.cnn(3, SIZE, SIZE)),

                new ConvLayer.Dto(16, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),
                new MaxPoolLayer.Dto(2, 2),

                new ConvLayer.Dto(32, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),
                new MaxPoolLayer.Dto(2, 2),

                new ConvLayer.Dto(64, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),
                new MaxPoolLayer.Dto(2, 2),

                new ReshapeLayer.FlattenLayer.Dto(),
                new DenseLayer.Dto(TensorShape.flat(128)),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),
                new DropoutLayer.Dto(0.3f),
                new DenseLayer.Dto(TensorShape.flat(ImageDatasetGenerator.NUM_CLASSES)),
                new ActivationLayer.Dto(new Sigmoid())
        ));
    }

    // ==================================================================
    // ======================= СЦЕНАРИЙ 2: GRID =========================
    // ==================================================================

    /**
     * YOLO-grid: выход {@code flat(G·G·(5+C))}, loss {@link GridLoss},
     * декодер {@link ImageUtil#decodeGrid}.
     */
    private static void runGrid() throws Exception {
        log.info("=== GRID ===");
        final int grid = 8;
        final int numClasses = ImageDatasetGenerator.NUM_CLASSES;
        final int perCell = 5 + numClasses;
        final float confThreshold = 0.4f;
        final float nmsIou = 0.5f;

        File outDir = new File("out/grid");
        Path modelPath = Path.of("model/grid/grid-cnn.bin");

        ImageDatasetGenerator gen = new ImageDatasetGenerator(
                0.1f, SIZE, MAX_OBJECTS, MIN_SHAPE, MAX_SHAPE, grid, 1.5f, SEED);

        Sequence model = loadOrTrain(
                gen, ImageDatasetGenerator.Task.GRID, modelPath,
                /*trainCount=*/2000,
                () -> buildGridModel(grid, numClasses),
                /*lossFactory=*/ds -> new GridLoss(grid, numClasses, 10f, 5f, 1f, 0.1f),
                /*maxEpoch=*/30,
                /*earlyStop=*/Model.earlyStop(3, 0.01f, log),
                /*metricsFactory=*/loss -> List.of(new LossMetric(loss)),
                1e-3f
        );

        saveTriptychs(gen, model, ImageDatasetGenerator.Task.GRID, outDir,
                "grid", (s, pred) -> {
                    var objs = ImageUtil.decodeGrid(pred, SIZE, grid, numClasses,
                            confThreshold, nmsIou);
                    var predicted = ImageUtil.drawBoxes(s.image(), objs, true, Color.RED);
                    var truth = ImageUtil.drawBoxes(s.image(), s.objects(), true, new Color(0, 140, 0));
                    return ImageUtil.triptych(s.image(), predicted, truth);
                });
    }

    private static Sequence buildGridModel(int grid, int numClasses) {
        final int perCell = 5 + numClasses;
        return Sequence.of(List.of(
                new Sequence.InputDto(TensorShape.cnn(3, SIZE, SIZE)),

                // энкодер: 128 → 64 → 32 → 16 → 8 (= grid)
                new ConvLayer.Dto(16, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),
                new MaxPoolLayer.Dto(2, 2),

                new ConvLayer.Dto(32, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),
                new MaxPoolLayer.Dto(2, 2),

                new ConvLayer.Dto(64, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),
                new MaxPoolLayer.Dto(2, 2),

                new ConvLayer.Dto(128, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),
                new MaxPoolLayer.Dto(2, 2),

                new ConvLayer.Dto(perCell, 1, 1, 0)
        ));
    }

    // ==================================================================
    // ===================== СЦЕНАРИЙ 3: HEATMAP ========================
    // ==================================================================

    /**
     * CenterNet heatmap: выход {@code flat(C·S·S)} логитов,
     * loss {@link HeatmapLoss}, декодер {@link ImageUtil#decodeHeatmap}.
     */
    private static void runHeatmap() throws Exception {
        log.info("=== HEATMAP ===");
        final int numClasses = ImageDatasetGenerator.NUM_CLASSES;
        final int outSize = numClasses * SIZE * SIZE;
        final float sigma = 3.0f;
        final float peakThreshold = 0.3f;
        final int fallbackHalf = MAX_SHAPE;

        File outDir = new File("out/heatmap");
        Path modelPath = Path.of("model/heatmap/heatmap-cnn.bin");

        ImageDatasetGenerator gen = new ImageDatasetGenerator(
                0.1f, SIZE, MAX_OBJECTS, MIN_SHAPE, MAX_SHAPE, 8, sigma, SEED);

        Sequence model = loadOrTrain(
                gen, ImageDatasetGenerator.Task.HEATMAP, modelPath,
                /*trainCount=*/2000,
                () -> buildHeatmapModel(numClasses, outSize),
                /*lossFactory=*/ds -> new HeatmapLoss(),
                /*maxEpoch=*/30,
                /*earlyStop=*/Model.earlyStop(3, 0.01f, log),
                /*metricsFactory=*/loss -> List.of(new LossMetric(loss)),
                5e-4f
        );

        saveTriptychs(gen, model, ImageDatasetGenerator.Task.HEATMAP, outDir,
                "heatmap", (s, pred) -> {
                    var objs = ImageUtil.decodeHeatmap(pred, SIZE, numClasses,
                            peakThreshold, fallbackHalf);
                    var predicted = ImageUtil.drawBoxes(s.image(), objs, true, Color.RED);
                    var truth = ImageUtil.drawBoxes(s.image(), s.objects(), true, new Color(0, 140, 0));
                    return ImageUtil.triptych(s.image(), predicted, truth);
                });
    }

    private static Sequence buildHeatmapModel(int numClasses, int outSize) {
        return Sequence.of(List.of(
                new Sequence.InputDto(TensorShape.cnn(3, SIZE, SIZE)),

                // энкодер: 128 → 64 → 32
                new ConvLayer.Dto(16, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),
                new MaxPoolLayer.Dto(2, 2),

                new ConvLayer.Dto(32, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),
                new MaxPoolLayer.Dto(2, 2),

                new ConvLayer.Dto(64, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),
                new ConvLayer.Dto(64, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),

                // декодер: 32 → 64 → 128
                new UpsampleLayer.Dto(2),
                new ConvLayer.Dto(32, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),
                new UpsampleLayer.Dto(2),
                new ConvLayer.Dto(16, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),

                // голова: C каналов heatmap, без активации (HeatmapLoss сам применяет sigmoid)
                new ConvLayer.Dto(numClasses, 1, 1, 0),
                new ReshapeLayer.Dto(TensorShape.flat(outSize))
        ));
    }

    // ==================================================================
    // ====================== СЦЕНАРИЙ 4: SEMSEG ========================
    // ==================================================================

    /**
     * Semantic segmentation: выход {@code flat((C+1)·S·S)} логитов
     * (канал 0 — фон), loss {@link SegLoss},
     * декодер {@link ImageUtil#decodeSemseg}.
     */
    private static void runSemseg() throws Exception {
        log.info("=== SEMSEG ===");
        final int numClasses = ImageDatasetGenerator.NUM_CLASSES + 1;   // 5: фон + 4 фигуры
        final int outSize = numClasses * SIZE * SIZE;
        final int minArea = 40;

        File outDir = new File("out/seg");
        Path modelPath = Path.of("model/seg/seg-cnn.bin");

        ImageDatasetGenerator gen = new ImageDatasetGenerator(
                SIZE, MAX_OBJECTS, MIN_SHAPE, MAX_SHAPE, SEED, 0.1f);

        Sequence model = loadOrTrain(
                gen, ImageDatasetGenerator.Task.SEMSEG, modelPath,
                /*trainCount=*/1500,
                () -> buildSemsegModel(numClasses, outSize),
                /*lossFactory=*/ds -> new SegLoss(SIZE, numClasses),
                /*maxEpoch=*/30,
                /*earlyStop=*/Model.earlyStop(2, 0.01f, log),
                /*metricsFactory=*/loss -> List.of(new LossMetric(loss)),
                1e-3f
        );

        saveTriptychs(gen, model, ImageDatasetGenerator.Task.SEMSEG, outDir,
                "seg", (s, pred) -> {
                    var objs = ImageUtil.decodeSemseg(pred, SIZE, numClasses, minArea);
                    var predicted = ImageUtil.drawBoxes(s.image(), objs, true, Color.RED);
                    var truth = ImageUtil.drawBoxes(s.image(), s.objects(), true, new Color(0, 140, 0));
                    return ImageUtil.triptych(s.image(), predicted, truth);
                });
    }

    private static Sequence buildSemsegModel(int numClasses, int outSize) {
        return Sequence.of(List.of(
                new Sequence.InputDto(TensorShape.cnn(3, SIZE, SIZE)),

                // энкодер: 128 → 64 → 32
                new ConvLayer.Dto(16, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),
                new MaxPoolLayer.Dto(2, 2),

                new ConvLayer.Dto(32, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),
                new MaxPoolLayer.Dto(2, 2),

                new ConvLayer.Dto(64, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),
                new ConvLayer.Dto(64, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),

                // декодер: 32 → 64 → 128
                new UpsampleLayer.Dto(2),
                new ConvLayer.Dto(32, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),
                new UpsampleLayer.Dto(2),
                new ConvLayer.Dto(16, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),

                // голова: C каналов на пиксель
                new ConvLayer.Dto(numClasses, 1, 1, 0),
                new ReshapeLayer.Dto(TensorShape.flat(outSize))
        ));
    }

    // ==================================================================
    // ===================== ОБЩИЙ КОНВЕЙЕР =============================
    // ==================================================================

    /**
     * Единый конвейер: загрузить модель с диска или обучить с нуля.
     *
     * <p>Шаги:
     * <ol>
     *   <li>если {@code modelPath} существует — {@code Model.load}, проверка
     *       типа {@link Sequence}, возврат;</li>
     *   <li>иначе — построить модель через {@code modelFactory};</li>
     *   <li>сгенерировать {@code trainCount} примеров, взять {@code s.inputs()}
     *       и {@code s.targets()} (цели уже построены генератором);</li>
     *   <li>{@code SampleSimple.split(0.8)}, обучить с заданными
     *       loss / optimizer / метриками / earlyStop / clipNorm;</li>
     *   <li>сохранить в {@code modelPath}.</li>
     * </ol>
     *
     * @param gen           генератор датасета
     * @param task          задача генерации
     * @param modelPath     путь для загрузки/сохранения
     * @param trainCount    число примеров для обучения
     * @param modelFactory  фабрика модели (вызывается только если нет файла)
     * @param lossFactory   фабрика loss (принимает датасет — на случай, если
     *                      loss зависит от числа классов/размера)
     * @param maxEpoch      максимум эпох
     * @param earlyStop     критерий ранней остановки
     * @param metricsFactory фабрика метрик (принимает loss)
     */
    private static Sequence loadOrTrain(
            ImageDatasetGenerator gen,
            ImageDatasetGenerator.Task task,
            Path modelPath,
            int trainCount,
            java.util.function.Supplier<Sequence> modelFactory,
            java.util.function.Function<SampleDataset<Sample>, Loss> lossFactory,
            int maxEpoch,
            BiPredicate<Model, List<Metric>> earlyStop,
            java.util.function.Function<Loss, List<Metric>> metricsFactory, float lr) {

        // ── 1. Загрузка ──
        if (Files.exists(modelPath)) {
            log.info("Найдена модель: {} — загружаем.", modelPath);
            Model loaded = Model.load(modelPath);
            if (!(loaded instanceof Sequence seq))
                throw new IllegalStateException(
                        "Ожидалась Sequence, получено: " + loaded.getClass().getName());
            log.info("Загружено. {} → {}", seq.inputShape(), seq.outputShape());
            return seq;
        }

        // ── 2. Построение ──
        log.info("Модель не найдена ({}), обучаем с нуля.", modelPath);
        Sequence model = modelFactory.get();
        log.info("Модель:\n{}", model);

        // ── 3. Датасет ──
        List<ImageDatasetGenerator.LabeledSample> all = gen.generate(trainCount, task);
        SampleDataset<Sample> ds = ImageDatasetGenerator.toSimple(all);
        log.info("Датасет: train={}, test={}",
                ds.getTrainSamples().size(), ds.getTestSamples().size());

        // ── 4. Обучение ──
        Loss loss = lossFactory.apply(ds);
        Optimizer opt = new AdamW(lr);
        List<Metric> metrics = metricsFactory.apply(loss);

        new ModelTraining<Sequence, Sample>()
                .setModel((Sequence) model)
                .setDataset(ds)
                .setLoss(loss)
                .setOptimizer(opt)
                .setMaxEpoch(maxEpoch)
                .setBatchSize(32)
                .setShowTestCount(5)
                .setMetrics(metrics)
                .setEndPredicate(earlyStop)
                .setClipNorm(5f).train();

        // ── 5. Сохранение ──
        model.save(modelPath);
        log.info("Модель сохранена: {}", modelPath);
        return model;
    }

    /**
     * Генерирует {@code TRIPTYCH_CNT} примеров, для каждого строит триптих
     * через {@code renderer} и сохраняет PNG в {@code outDir}.
     *
     * @param renderer функция {@code (sample, prediction) -> triptych}
     */
    private static void saveTriptychs(
            ImageDatasetGenerator gen,
            Sequence model,
            ImageDatasetGenerator.Task task,
            File outDir,
            String prefix,
            BiFunction<ImageDatasetGenerator.LabeledSample, float[], java.awt.image.BufferedImage> renderer)
            throws Exception {

        ImageUtil.clearDir(outDir);
        var samples = gen.generate(TRIPTYCH_CNT, task);
        for (int i = 0; i < samples.size(); i++) {
            var s = samples.get(i);
            float[] pred = model.forward(s.inputs(), false);
            var tri = renderer.apply(s, pred);
            ImageUtil.savePng(tri, new File(outDir, String.format("%s-%02d.png", prefix, i)));
        }
        log.info("Триптихи: {}", outDir.getAbsolutePath());
    }
}