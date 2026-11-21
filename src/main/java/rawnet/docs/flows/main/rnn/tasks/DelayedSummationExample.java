package rawnet.docs.flows.main.rnn.tasks;

import rawnet.code.dataset.SampleDataset;
import rawnet.code.dataset.Sample;
import rawnet.code.net.TensorShape;
import rawnet.code.net.core.activate.Linear;
import rawnet.code.net.core.activate.Tanh;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.core.ActivationLayer;
import rawnet.code.net.layers.core.DenseLayer;
import rawnet.code.net.layers.rnn.BidirectionalLayer;
import rawnet.code.net.layers.rnn.ElmanRnnLayer;
import rawnet.code.net.layers.rnn.GruLayer;
import rawnet.code.net.layers.structure.ReshapeLayer;
import rawnet.code.net.layers.core.TimeDistributedDenseLayer;
import rawnet.code.net.core.loss.Loss;
import rawnet.code.net.core.loss.Mse;
import rawnet.code.net.core.metric.LossMetric;
import rawnet.code.net.core.metric.Metric;
import rawnet.code.net.core.metric.Rmse;
import rawnet.code.net.model.Model;
import rawnet.code.net.model.ModelTraining;
import rawnet.code.net.model.sequence.Sequence;
import rawnet.code.net.core.optimizator.AdamW;
import rawnet.code.net.core.optimizator.Optimizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Учебный пример: <b>отложенное сложение</b> (delayed summation).
 *
 * <h2>Задача</h2>
 * На вход подаются две параллельные дорожки длины {@code T}:
 * <ul>
 *   <li>{@code values[t] ∈ [0, 1)} — значения;</li>
 *   <li>{@code markers[t] ∈ {0, 1}}, ровно <b>два</b> маркера равны {@code 1}.</li>
 * </ul>
 * Нужно вернуть {@code v[m1] + v[m2]}, где {@code m1 < m2} — позиции маркеров.
 *
 * <pre>
 * t:        0     1     2   ...  m1   ...  m2   ...  49
 * values:  .21   .73   .55  ...  .42  ...  .88  ...  .31
 * markers:  0     0     0   ...   1   ...   1   ...   0
 * target = 0.42 + 0.88 = 1.30
 * </pre>
 *
 * <h2>Чем задача сложнее предыдущих</h2>
 * <ol>
 *   <li><b>Два входа.</b> Сеть должна «дождаться» <b>двух</b> маркеров
 *       и сложить значения в этих позициях. Расстояние {@code m2 - m1}
 *       может быть до {@code T-1 = 49} шагов — классическая задача на
 *       долговременную память.</li>
 *   <li><b>Регрессия.</b> Выход — вещественное число, не класс. Loss —
 *       {@link Mse}, активация выхода — {@link Linear}.</li>
 *   <li><b>Вещественный вход.</b> Не индексы токенов, а числа
 *       {@code ∈ [0, 1)}. {@code EmbeddingLayer} не подходит — нужна
 *       обычная линейная проекция.</li>
 * </ol>
 *
 * <h2>Формат входа</h2>
 * Две дорожки «сшиваются» в один плоский вектор длины {@code 2·T}
 * чередованием:
 * <pre>{@code
 * input[2t]   = values[t]
 * input[2t+1] = markers[t]
 * }</pre>
 * Это позволяет {@link ReshapeLayer} превратить {@code flat(2·T)} в
 * {@code seq(T, 2)} — по паре {@code (value, marker)} на шаг. Дальше
 * {@link TimeDistributedDenseLayer} расширяет {@code 2} признака до
 * {@code dModel}, и RNN работает с {@code seq(T, dModel)}.
 *
 * <h2>Архитектура</h2>
 * Все три модели имеют одинаковый «вход» и «выход»; различается только
 * рекуррентное ядро:
 * <pre>{@code
 * Input(flat(2·T))
 *   → Reshape(seq(T, 2))
 *   → TimeDistributedDense(dModel)
 *   → [RNN | GRU | BiRNN]
 *   → Dense(1)
 *   → Linear            (выход — вещественное число, без ограничения)
 * }</pre>
 *
 * <h2>Ожидаемые результаты (T = 50)</h2>
 * <ul>
 *   <li><b>RNN</b> (ElmanRNN): низкая точность (RMSE &gt; 0.3) — на
 *       расстоянии до 49 шагов затухающий градиент «сотрёт» {@code v[m1]}
 *       раньше, чем встретится {@code m2}; сеть часто возвращает
 *       что-то вроде среднего {@code v}.</li>
 *   <li><b>GRU</b>: заметно лучше — гейты позволяют «удерживать»
 *       {@code v[m1]} до {@code m2}; RMSE ~ 0.1–0.3.</li>
 *   <li><b>BiRNN</b>: лучший результат — обратное направление «видит»
 *       {@code m2} раньше {@code m1} и «запоминает» {@code v[m2]},
 *       а прямое — {@code v[m1]}. Склейка даёт оба значения; RMSE ~ 0.02–0.1.</li>
 * </ul>
 *
 * <p><b>Замечание.</b> Даже BiRNN не даёт RMSE = 0 — задача сложная,
 * потому что нужно <b>точно</b> определить позиции маркеров и извлечь
 * значения. Для «идеального» решения нужен attention (глава про
 * Transformer) или явный «считыватель» (pointer network). Но
 * относительный порядок «BiRNN &gt; GRU &gt; RNN» устойчив и
 * педагогически ценен.
 */
public final class DelayedSummationExample {

    private static final Logger log = LoggerFactory.getLogger(DelayedSummationExample.class);

    // =====================================================================
    // ==================== Гиперпараметры =================================
    // =====================================================================

    /** Длина последовательности (число пар value/marker). */
    static final int T = 50;

    /** Размерность эмбеддинга после TimeDistributedDense. */
    static final int D_MODEL = 16;

    /** Размер скрытого состояния RNN/GRU. */
    static final int HIDDEN = 32;

    /** Размер батча. */
    static final int BATCH = 32;

    /** Число эпох. */
    static final int EPOCHS = 30;

    /** Скорость обучения AdamW. */
    static final float LR = 1e-3f;

    /** Порог глобального клипинга градиента. */
    static final float CLIP_NORM = 1.0f;

    /** Размер train-выборки. */
    static final int N_TRAIN = 5000;

    /** Размер test-выборки. */
    static final int N_TEST = 1000;

    // =====================================================================
    // ==================== main ===========================================
    // =====================================================================

    public static void main(String[] args) {
        long seed = 42L;

        SampleDataset<Sample> ds = generateDataset(seed);

        log.info("Датасет: train={}, test={}, T={}",
                ds.getTrainSamples().size(), ds.getTestSamples().size(), T);
        log.info("Задача: v[m1] + v[m2], где m1, m2 — позиции двух маркеров");

        // Сравниваем три архитектуры.
        runAndReport("RNN",   buildRnn(),   ds);
        runAndReport("GRU",   buildGru(),   ds);
        runAndReport("BiRNN", buildBiRnn(), ds);
    }

    /** Собирает модель, обучает и печатает итоговую RMSE. */
    private static void runAndReport(String name, Model model, SampleDataset<Sample> ds) {
        log.info("=== {} ===", name);
        log.info("{}", model);

        Optimizer opt = new AdamW(LR);
        Loss loss = new Mse();

        // RMSE — более интерпретируемая метрика, чем MSE (в тех же единицах,
        // что и target). LossMetric — для контроля, что loss падает.
        List<Metric> metrics = List.of(new LossMetric(loss), new Rmse());

        new ModelTraining<Model, Sample>()
                .setModel(model)
                .setDataset(ds)
                .setLoss(loss)
                .setOptimizer(opt)
                .setMaxEpoch(EPOCHS)
                .setBatchSize(BATCH)
                .setShowTestCount(3)
                .setMetrics(metrics)
                .setEndPredicate(null)
                .setClipNorm(CLIP_NORM)
                .train();

        log.info("{}: итог {}", name, Metric.print(metrics));
    }

    // =====================================================================
    // ==================== Архитектуры ====================================
    // =====================================================================

    /**
     * RNN-модель: Reshape → TimeDistributedDense → ElmanRNN → Dense(1) → Linear.
     *
     * <p>{@link Tanh} — стандарт для RNN. {@link Linear} на выходе —
     * чтобы сеть могла выдавать любое вещественное число (target ∈ [0, 2)).
     */
    static Model buildRnn() {
        List<Layer.Dto> dtos = List.of(
                new Sequence.InputDto(TensorShape.flat(2 * T)),

                // flat(2·T) → seq(T, 2)
                new ReshapeLayer.Dto(TensorShape.seq(T, 2)),

                // seq(T, 2) → seq(T, D_MODEL)
                new TimeDistributedDenseLayer.Dto(D_MODEL),

                // seq(T, D_MODEL) → flat(HIDDEN)  (returnSequences=false)
                new ElmanRnnLayer.Dto(T, HIDDEN, new Tanh(), false),

                // flat(HIDDEN) → flat(1)
                new DenseLayer.Dto(TensorShape.flat(1)),

                // вещественный выход
                new ActivationLayer.Dto(new Linear())
        );
        return Sequence.of(dtos);
    }

    /**
     * GRU-модель: Reshape → TimeDistributedDense → GruLayer → Dense(1) → Linear.
     */
    static Model buildGru() {
        List<Layer.Dto> dtos = List.of(
                new Sequence.InputDto(TensorShape.flat(2 * T)),
                new ReshapeLayer.Dto(TensorShape.seq(T, 2)),
                new TimeDistributedDenseLayer.Dto(D_MODEL),
                new GruLayer.Dto(T, HIDDEN, false),
                new DenseLayer.Dto(TensorShape.flat(1)),
                new ActivationLayer.Dto(new Linear())
        );
        return Sequence.of(dtos);
    }

    /**
     * BiRNN-модель: Reshape → TimeDistributedDense → BidirectionalLayer(ElmanRNN) → Dense(1) → Linear.
     *
     * <p>Обратное направление читает {@code t = T-1, …, 0}, «видит»
     * {@code m2} раньше {@code m1}. Склейка {@code [h^f_{T-1} ‖ h^b_0]}
     * содержит информацию об обоих маркерах.
     */
    static Model buildBiRnn() {
        List<Layer.Dto> dtos = List.of(
                new Sequence.InputDto(TensorShape.flat(2 * T)),
                new ReshapeLayer.Dto(TensorShape.seq(T, 2)),
                new TimeDistributedDenseLayer.Dto(D_MODEL),
                new BidirectionalLayer.Dto(
                        T, HIDDEN, false,
                        new ElmanRnnLayer.Dto(T, HIDDEN, new Tanh(), false)),
                new DenseLayer.Dto(TensorShape.flat(1)),
                new ActivationLayer.Dto(new Linear())
        );
        return Sequence.of(dtos);
    }

    // =====================================================================
    // ==================== Генерация данных ===============================
    // =====================================================================

    /**
     * Генерирует датасет: каждая последовательность длины ровно {@code T}.
     * Ровно два маркера (случайные позиции {@code m1 < m2}), остальные — 0.
     *
     * <p>Вход — плоский {@code flat(2·T)}, где:
     * <pre>{@code
     * input[2t]   = values[t]
     * input[2t+1] = markers[t]
     * }</pre>
     * Target — {@code [v[m1] + v[m2]]} (один элемент).
     */
    static SampleDataset<Sample> generateDataset(long seed) {
        Random rnd = new Random(seed);
        List<Sample> train = new ArrayList<>(N_TRAIN);
        List<Sample> test  = new ArrayList<>(N_TEST);

        for (int i = 0; i < N_TRAIN + N_TEST; i++) {
            Sample s = randomSample(rnd);
            if (i < N_TRAIN) train.add(s); else test.add(s);
        }
        return new SampleDataset<Sample>(train, test);
    }

    /**
     * Один пример: случайные {@code values[t] ∈ [0, 1)}, два маркера
     * в случайных различных позициях {@code m1 < m2}, target =
     * {@code v[m1] + v[m2]}.
     */
    static Sample randomSample(Random rnd) {
        float[] values  = new float[T];
        float[] markers = new float[T];

        for (int t = 0; t < T; t++) values[t] = rnd.nextFloat();

        // Два различных маркера.
        int m1 = rnd.nextInt(T);
        int m2 = rnd.nextInt(T - 1);
        if (m2 >= m1) m2++;   // сдвигаем, чтобы m2 != m1
        if (m1 > m2) { int tmp = m1; m1 = m2; m2 = tmp; }
        markers[m1] = 1f;
        markers[m2] = 1f;

        // Сшиваем в плоский вектор чередованием: [v0, m0, v1, m1, …].
        float[] inputs = new float[2 * T];
        for (int t = 0; t < T; t++) {
            inputs[2 * t]     = values[t];
            inputs[2 * t + 1] = markers[t];
        }

        float target = values[m1] + values[m2];
        return new Sample(inputs, new float[]{ target });
    }
}