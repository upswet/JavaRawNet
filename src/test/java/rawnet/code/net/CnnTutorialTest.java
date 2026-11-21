package rawnet.code.net;

import rawnet.code.dataset.SampleDataset;
import rawnet.code.dataset.Sample;
import rawnet.code.net.core.activate.LeakyRelu;
import rawnet.code.net.core.activate.Sigmoid;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.cnn.ConvLayer;
import rawnet.code.net.layers.cnn.GlobalAveragePoolingLayer;
import rawnet.code.net.layers.cnn.MaxPoolLayer;
import rawnet.code.net.layers.cnn.UpsampleLayer;
import rawnet.code.net.layers.core.ActivationLayer;
import rawnet.code.net.layers.core.DenseLayer;
import rawnet.code.net.model.sequence.Sequence;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static rawnet.code.net.TestDatasets.trainSmall;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * «Учебник»: CNN-слои. Проверяем, что Conv/MaxPool/GAP/Upsample
 * собираются в сеть и что простая CNN обучается на мини-картинках.
 *
 * <h2>Что проверяем</h2>
 * <ol>
 *   <li>{@link #cnn_shapes_are_as_expected()} — формулы форм
 *       ({@code computeOutputShape}) для Conv/MaxPool/GAP/Upsample.
 *       Это «шпаргалка», а не обучение: дёшево и ловит регрессии
 *       в расчёте форм;</li>
 *   <li>{@link #cnn_learns_cross_vs_circle()} — CNN реально обучается
 *       на синтетических «крестик vs нолик» {@code 8×8}.</li>
 * </ol>
 *
 * <h2>Датасет</h2>
 * Синтетический: два класса по {@code n/2} примеров, с шумом ±0.1.
 * «Крестик» — две диагонали, «нолик» — рамка по периметру.
 * Задача простая: CNN должна выучить её с запасом (> 0.85).
 *
 * <h2>Гиперпараметры</h2>
 * {@code trainSmall(net, ds, 30, 1e-2f)} — 30 эпох, {@code lr=1e-2},
 * батч 16, AdamW + BCE. CNN на 500×8×8 с двумя свёртками — ~3–4 с.
 * Меньше эпох — сеть не успевает (это ограничение бюджета, а не баг).
 */
class CnnTutorialTest {

    // =====================================================================
    // ==================== 1. Формы слоёв (шпаргалка) =====================
    // =====================================================================

    /**
     * Формулы форм — «шпаргалка» по CNN-слоям. Не обучение, а быстрая
     * проверка, что {@code computeOutputShape} считает правильно:
     * <ul>
     *   <li>{@code Conv(3, stride=1, padding=1)} сохраняет H/W;</li>
     *   <li>{@code MaxPool(2, stride=2)} делит H/W на 2;</li>
     *   <li>{@code GAP} сворачивает {@code cnn(C,H,W)} в {@code flat(C)};</li>
     *   <li>{@code Upsample(scale=2)} умножает H/W на 2.</li>
     * </ul>
     * Если эти формулы сломаются, CNN-сеть соберётся с неверными формами
     * и упадёт где-то в рантайме — тест поймает это раньше.
     */
    @Test
    void cnn_shapes_are_as_expected() {
        TensorShape in = TensorShape.cnn(1, 8, 8);

        // Conv: stride=1, padding=1, kernel=3 → H/W сохраняются
        assertEquals(TensorShape.cnn(4, 8, 8),
                ConvLayer.computeOutputShape(in, 4, 3, 1, 1));

        // MaxPool(2, 2) → H/W делятся на 2
        assertEquals(TensorShape.cnn(4, 4, 4),
                MaxPoolLayer.computeOutputShape(TensorShape.cnn(4, 8, 8), 2, 2));

        // GAP: cnn(C, H, W) → flat(C)
        assertEquals(TensorShape.flat(4),
                GlobalAveragePoolingLayer.computeOutputShape(TensorShape.cnn(4, 4, 4)));

        // Upsample(scale=2) → H/W умножаются на 2
        assertEquals(TensorShape.cnn(4, 8, 8),
                UpsampleLayer.computeOutputShape(TensorShape.cnn(4, 4, 4), 2));
    }

    // =====================================================================
    // ==================== 2. Обучение CNN ================================
    // =====================================================================

    /**
     * Мини-CNN: {@code Conv → LeakyReLU → MaxPool → Conv → LeakyReLU →
     * GAP → Dense → Sigmoid}.
     *
     * <p>Проверяет всю цепочку CNN-слоёв:
     * <ul>
     *   <li>{@link ConvLayer} — свёртки с параметрами;</li>
     *   <li>{@link LeakyRelu} — нелинейность;</li>
     *   <li>{@link MaxPoolLayer} — уменьшение H/W;</li>
     *   <li>{@link GlobalAveragePoolingLayer} — сворачивание в вектор;</li>
     *   <li>{@link DenseLayer} + {@link Sigmoid} — классификационная голова.</li>
     * </ul>
     *
     * <p>Ожидаемая accuracy: ~0.90+ на «крестик vs нолик». Порог 0.85 —
     * с запасом. Если CNN не выучивает такую простую задачу — что-то
     * сломано в слоях или в связке.
     */
    @Test
    void cnn_learns_cross_vs_circle() {
        SampleDataset<Sample> ds = crossVsCircleDataset(500, new Random(42L));

        List<Layer.Dto> dtos = List.of(
                // вход: 1 канал, 8×8
                new Sequence.InputDto(TensorShape.cnn(1, 8, 8)),

                // Conv 3×3, 4 канала, stride=1, padding=1 → 4×8×8
                new ConvLayer.Dto(4, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu()),

                // MaxPool 2×2 → 4×4×4
                new MaxPoolLayer.Dto(2, 2),

                // Conv 3×3, 8 каналов, stride=1, padding=1 → 8×4×4
                new ConvLayer.Dto(8, 3, 1, 1),
                new ActivationLayer.Dto(new LeakyRelu()),

                // GAP: 8×4×4 → flat(8)
                new GlobalAveragePoolingLayer.Dto(),

                // голова: flat(8) → flat(1) → sigmoid
                new DenseLayer.Dto(TensorShape.flat(1)),
                new ActivationLayer.Dto(new Sigmoid())
        );
        Sequence net = Sequence.of(dtos);

        float acc = trainSmall(net, ds, 30, 1e-2f).get(1).result();
        assertTrue(acc > 0.85f, "CNN: ожидалось > 0.85, получено " + acc);
    }

    // =====================================================================
    // ==================== Датасет ========================================
    // =====================================================================

    /**
     * «Крестик vs нолик» на сетке {@code 8×8}: два класса, по
     * {@code n/2} примеров, с небольшим шумом.
     *
     * <p>Метка: {@code 1} — крестик, {@code 0} — нолик.
     *
     * <p><b>Почему 500, а не 200.</b> На 200 примерах CNN может
     * «запомнить» картинки и переобучиться за 30 эпох; на 500 —
     * учится обобщать. Разбиение 80/20 — train/test.
     *
     * <p><b>Шум.</b> Каждый пиксель сдвигается на ±0.1, чтобы сеть
     * не выучила точные паттерны наизусть. Без шума задача становится
     * тривиальной (CNN выучит её за 1–2 эпохи, и тест перестанет быть
     * показательным).
     */
    private static SampleDataset<Sample> crossVsCircleDataset(int n, Random rnd) {
        List<Sample> all = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            boolean cross = (i % 2 == 0);
            float[] img = new float[64];

            if (cross) {
                // Диагонали: (0,0)-(7,7) и (0,7)-(7,0)
                for (int k = 0; k < 8; k++) {
                    img[k * 8 + k] = 1f;             // главная диагональ
                    img[k * 8 + (7 - k)] = 1f;       // побочная диагональ
                }
            } else {
                // Рамка по периметру
                for (int k = 0; k < 8; k++) {
                    img[0 * 8 + k] = 1f;             // верхняя строка
                    img[7 * 8 + k] = 1f;             // нижняя строка
                    img[k * 8 + 0] = 1f;             // левый столбец
                    img[k * 8 + 7] = 1f;             // правый столбец
                }
            }

            // Небольшой шум: ±0.05
            for (int j = 0; j < img.length; j++) {
                img[j] += (rnd.nextFloat() - 0.5f) * 0.2f;
            }

            all.add(new Sample(img, new float[]{cross ? 1f : 0f}));
        }

        int split = (int) (n * 0.8);
        return new SampleDataset<Sample>(all.subList(0, split), all.subList(split, n));
    }
}