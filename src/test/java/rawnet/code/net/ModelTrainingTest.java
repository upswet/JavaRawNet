package rawnet.code.net;

import rawnet.code.dataset.Sample;
import rawnet.code.dataset.SampleDataset;
import rawnet.code.net.core.activate.Sigmoid;
import rawnet.code.net.core.activate.Tanh;
import rawnet.code.net.core.loss.Mse;
import rawnet.code.net.core.optimizator.AdamW;
import rawnet.code.net.layers.core.ActivationLayer;
import rawnet.code.net.layers.core.DenseLayer;
import rawnet.code.net.model.Model;
import rawnet.code.net.model.ModelTraining;
import rawnet.code.net.model.sequence.Sequence;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static rawnet.code.net.TestDatasets.xorDataset;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Юнит-тесты {@link ModelTraining}: сохранение модели и `.model.train`,
 * удаление `.model.train` при успехе, {@code path == null} → {@code temp}.
 *
 * <p><b>Почему это важно.</b> {@code save()}/{@code load()} — критичная
 * часть (resume после Ctrl+C). Ошибка в формате `.model.train` (порядок
 * полей, base64, persistables) — тихо ломает resume, а обучение «сходится».
 */
class ModelTrainingTest {

    /**
     * Успешное обучение сохраняет модель в {@code <base>.model} и
     * удаляет {@code <base>.model.train} (индикатор «всё, готово»).
     */
    @Test
    void successfulTrainSavesModelAndDeletesTrainFile(@TempDir Path tmp) throws IOException {
        Sequence net = buildNet();
        SampleDataset<Sample> ds = xorDataset(200, new java.util.Random(42L));
        Mse loss = new Mse();

        Path base = tmp.resolve("m");

        new ModelTraining<Sequence, Sample>()
                .setModel(net)
                .setDataset(ds)
                .setLoss(loss)
                .setOptimizer(new AdamW(1e-2f))
                .setMaxEpoch(2)
                .setBatchSize(16)
                .setMetrics(List.of())
                .setClipNorm(1.0f)
                .setPath(base)
                .train();

        Path modelPath = base.resolveSibling(base.getFileName() + ".model");
        Path trainPath = base.resolveSibling(base.getFileName() + ".model.train");

        assertTrue(Files.isRegularFile(modelPath), "модель должна быть сохранена: " + modelPath);
        assertFalse(Files.exists(trainPath),
                "после успешного обучения .model.train должен быть удалён: " + trainPath);
    }

    /**
     * После {@code train()} загруженная из {@code .model} модель даёт
     * тот же выход, что и {@code trainer.getModel()}.
     */
    @Test
    void loadRoundTripPreservesPredictions(@TempDir Path tmp) throws IOException {
        Sequence net = buildNet();
        SampleDataset<Sample> ds = xorDataset(200, new java.util.Random(1L));
        Mse loss = new Mse();

        Path base = tmp.resolve("m");

        ModelTraining<Sequence, Sample> trainer = new ModelTraining<Sequence, Sample>()
                .setModel(net)
                .setDataset(ds)
                .setLoss(loss)
                .setOptimizer(new AdamW(1e-2f))
                .setMaxEpoch(2)
                .setBatchSize(16)
                .setMetrics(List.of())
                .setClipNorm(1.0f)
                .setPath(base);
        trainer.train();

        // Прогноз до/после load
        float[] x = {0.2f, 0.8f};
        float[] before = trainer.getModel().forward(x, false).clone();

        Path modelPath = base.resolveSibling(base.getFileName() + ".model");
        Model loaded = Model.load(modelPath);
        float[] after = loaded.forward(x, false).clone();

        assertEquals(before.length, after.length);
        for (int i = 0; i < before.length; i++) {
            assertEquals(before[i], after[i], 1e-6f,
                    "save/load: предсказание изменилось в позиции " + i);
        }
    }

    /**
     * {@code path == null} → модель сохраняется в {@code temp.model}
     * в текущей директории, {@code temp.model.train} удаляется при успехе.
     */
    @Test
    void nullPathUsesTempInCurrentDirectory() throws IOException {
        Sequence net = buildNet();
        SampleDataset<Sample> ds = xorDataset(200, new java.util.Random(2L));

        Path tempModel = Path.of("temp.model");
        Path tempTrain = Path.of("temp.model.train");
        Files.deleteIfExists(tempModel);
        Files.deleteIfExists(tempTrain);

        try {
            new ModelTraining<Sequence, Sample>()
                    .setModel(net)
                    .setDataset(ds)
                    .setLoss(new Mse())
                    .setOptimizer(new AdamW(1e-2f))
                    .setMaxEpoch(1)
                    .setBatchSize(16)
                    .setMetrics(List.of())
                    .setClipNorm(1.0f)
                    .train();   // без setPath

            assertTrue(Files.isRegularFile(tempModel),
                    "path == null → temp.model в текущей директории");
            assertFalse(Files.exists(tempTrain),
                    "после успеха temp.model.train должен быть удалён");
        } finally {
            Files.deleteIfExists(tempModel);
            Files.deleteIfExists(tempTrain);
        }
    }

    // ==================== утилиты ====================

    private static Sequence buildNet() {
        return Sequence.of(List.of(
                new Sequence.InputDto(2),
                new DenseLayer.Dto(TensorShape.flat(8)),
                new ActivationLayer.Dto(new Tanh()),
                new DenseLayer.Dto(TensorShape.flat(1)),
                new ActivationLayer.Dto(new Sigmoid())
        ));
    }
}