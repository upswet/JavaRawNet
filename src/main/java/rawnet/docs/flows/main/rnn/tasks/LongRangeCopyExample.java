package rawnet.docs.flows.main.rnn.tasks;

import rawnet.code.dataset.SampleDataset;
import rawnet.code.dataset.Sample;
import rawnet.code.net.TensorShape;
import rawnet.code.net.core.activate.Tanh;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.core.DenseLayer;
import rawnet.code.net.layers.rnn.BidirectionalLayer;
import rawnet.code.net.layers.rnn.ElmanRnnLayer;
import rawnet.code.net.layers.rnn.GruLayer;
import rawnet.code.net.layers.core.EmbeddingLayer;
import rawnet.code.net.core.loss.Loss;
import rawnet.code.net.core.loss.SoftmaxCrossEntropy;
import rawnet.code.net.core.metric.Accuracy;
import rawnet.code.net.core.metric.LossMetric;
import rawnet.code.net.core.metric.Metric;
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
 * Учебный пример: <b>long-range copy</b> — копирование первого токена
 * последовательности через {@code T-1} шагов шума.
 *
 * <h2>Задача</h2>
 * На вход подаётся последовательность {@code [t_0, t_1, …, t_{T-1}]}
 * целых чисел из алфавита {@code [0, V)}. <b>Только первый токен
 * содержателен</b>, все остальные — шум (случайные токены).
 * Сеть должна вернуть <b>первый токен</b>.
 *
 * <pre>
 *   [0 3 3 1 0 2 5 7 5 0 1 7 …]  →  target = 0
 *   [6 1 3 3 7 1 4 7 3 0 2 4 …]  →  target = 6
 *   [2 3 0 4 1 7 5 5 2 7 2 4 …]  →  target = 2
 * </pre>
 *
 * <p>Это классическая проверка долговременной памяти
 * (Bengio et al., 1994; Hochreiter &amp; Schmidhuber, 1997):
 * единственная релевантная информация — в начале, а ответ нужен в конце.
 * Сеть должна «донести» один токен через {@code T-1} шагов шума.
 *
 * <h2>Три архитектуры</h2>
 * Все три модели — {@link Sequence} с одинаковым входом (embedding) и
 * одинаковой головой (dense на {@code V} логитов); различается только
 * рекуррентное ядро:
 * <ol>
 *   <li><b>RNN</b> — {@link ElmanRnnLayer}. Читает слева направо,
 *       в конце берёт {@code h_{T-1}}. Из-за затухающего градиента
 *       информация о {@code x_0} «размывается» шумом на длинных {@code T};
 *       ожидаемо плохой результат на {@code T = 50}.</li>
 *   <li><b>GRU</b> — {@link GruLayer}. Гейты {@code z_t} (update) и
 *       {@code r_t} (reset) позволяют «заморозить» состояние после
 *       первого шага и игнорировать шум. Должен быть заметно лучше RNN.</li>
 *   <li><b>BiRNN</b> — {@link BidirectionalLayer} с {@link ElmanRnnLayer}
 *       в качестве ячейки. Обратное направление читает
 *       {@code x_{T-1}, …, x_0} и видит {@code x_0} <b>последним</b> —
 *       для «copy first» это естественно. Ожидаемо лучшее качество
 *       (при условии, что ElmanRNN вообще способен дочитать до конца
 *       обратной последовательности без потери {@code x_0}, что на
 *       {@code T = 50} уже под вопросом).</li>
 * </ol>
 *
 * <h2>Формат данных</h2>
 * <ul>
 *   <li>вход модели: {@code TensorShape.flat(T)} — индексы токенов
 *       (дополнения нет: все последовательности длины ровно {@code T});</li>
 *   <li>выход модели: {@code TensorShape.flat(V)} — логиты (без softmax);</li>
 *   <li>loss: {@link SoftmaxCrossEntropy} — принимает <b>логиты</b> и
 *       one-hot target, считает stable softmax и {@code dL/dz = p - t}
 *       внутри;</li>
 *   <li>метрика: {@link Accuracy} — на векторном выходе берёт argmax.</li>
 * </ul>
 *
 * <h2>Ожидаемые результаты (T = 50)</h2>
 * <ul>
 *   <li>RNN: accuracy ≈ {@code 0.10–0.20} (случайный уровень для
 *       {@code V = 10}) — ElmanRNN не удерживает информацию 50 шагов;</li>
 *   <li>GRU: accuracy ≈ {@code 0.60–0.90} — гейты помогают, но
 *       не гарантируют;</li>
 *   <li>BiRNN: accuracy ≈ {@code 0.90–0.99} — обратное направление
 *       «видит» {@code x_0} в конце.</li>
 * </ul>
 *
 * <p><b>Замечание про vanishing gradient.</b> Именно на этой задаче
 * видно, зачем нужны гейты (GRU/LSTM) и почему «читать с конца»
 * (BiRNN) — мощный приём. ElmanRNN на {@code T = 50} «сломан»
 * концептуально, а не из-за бага.
 */
public final class LongRangeCopyExample {

    private static final Logger log = LoggerFactory.getLogger(LongRangeCopyExample.class);

    // =====================================================================
    // ==================== Гиперпараметры =================================
    // =====================================================================

    /** Размер алфавита токенов. */
    static final int VOCAB = 10;

    /** Длина последовательности. */
    static final int T = 30;

    /** Размерность эмбеддинга. */
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

        // Генерируем данные один раз — все три модели учатся на одном и том же.
        SampleDataset<Sample> ds = generateDataset(seed);

        log.info("Датасет: train={}, test={}, T={}, vocab={}",
                ds.getTrainSamples().size(), ds.getTestSamples().size(), T, VOCAB);
        log.info("Задача: скопировать x[0] через {} шагов шума", T - 1);

        // Сравниваем три архитектуры.
        runAndReport("RNN",   buildRnn(),   ds);
        runAndReport("GRU",   buildGru(),   ds);
        runAndReport("BiRNN", buildBiRnn(), ds);
    }

    /** Собирает модель, обучает и печатает итоговую accuracy. */
    private static void runAndReport(String name, Model model, SampleDataset<Sample> ds) {
        log.info("=== {} ===", name);
        log.info("{}", model);

        Optimizer opt = new AdamW(LR);
        Loss loss = new SoftmaxCrossEntropy();   // ждёт логиты, не вероятности

        List<Metric> metrics = List.of(new LossMetric(loss), new Accuracy());

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
     * RNN-модель: embedding → ElmanRNN → dense(V) (логиты).
     *
     * <p>Без softmax на выходе: {@link SoftmaxCrossEntropy} принимает логиты
     * и делает softmax внутри. Это численно устойчивее и даёт простой
     * градиент {@code p - t}.
     *
     * <p>{@link Tanh} — стандартная активация для ElmanRNN (ограничена,
     * не взрывается). {@code LeakyRelu} в RNN почти всегда ведёт к
     * нестабильности при {@code T > 20}.
     */
    static Model buildRnn() {
        List<Layer.Dto> dtos = List.of(
                new Sequence.InputDto(TensorShape.flat(T)),

                // flat(T) → seq(T, D_MODEL)
                new EmbeddingLayer.Dto(VOCAB, D_MODEL),

                // seq(T, D_MODEL) → flat(HIDDEN)  (returnSequences=false)
                new ElmanRnnLayer.Dto(T, HIDDEN, new Tanh(), false),

                // flat(HIDDEN) → flat(V)  — логиты
                new DenseLayer.Dto(TensorShape.flat(VOCAB))
        );
        return Sequence.of(dtos);
    }

    /**
     * GRU-модель: embedding → GruLayer → dense(V) (логиты).
     *
     * <p>Внутри GRU уже стоят sigmoid (гейты) и tanh (candidate) —
     * отдельная активация не нужна.
     */
    static Model buildGru() {
        List<Layer.Dto> dtos = List.of(
                new Sequence.InputDto(TensorShape.flat(T)),
                new EmbeddingLayer.Dto(VOCAB, D_MODEL),

                // seq(T, D_MODEL) → flat(HIDDEN)
                new GruLayer.Dto(T, HIDDEN, false, 0f),

                new DenseLayer.Dto(TensorShape.flat(VOCAB))
        );
        return Sequence.of(dtos);
    }

    /**
     * BiRNN-модель: embedding → BidirectionalLayer(ElmanRNN) → dense(V).
     *
     * <p>Обратное направление читает {@code x_{T-1}, …, x_0} и видит
     * {@code x_0} последним — для «copy first» это естественно.
     * Выход — {@code [h^f_{T-1} ‖ h^b_0]} размера {@code 2·HIDDEN}.
     */
    static Model buildBiRnn() {
        List<Layer.Dto> dtos = List.of(
                new Sequence.InputDto(TensorShape.flat(T)),
                new EmbeddingLayer.Dto(VOCAB, D_MODEL),

                // seq(T, D_MODEL) → flat(2·HIDDEN)
                // cellDto — ElmanRNN с теми же параметрами (T, HIDDEN, Tanh, false).
                new BidirectionalLayer.Dto(
                        T, HIDDEN, false,
                        new ElmanRnnLayer.Dto(T, HIDDEN, new Tanh(), false)),

                new DenseLayer.Dto(TensorShape.flat(VOCAB))
        );
        return Sequence.of(dtos);
    }

    // =====================================================================
    // ==================== Генерация данных ===============================
    // =====================================================================

    /**
     * Генерирует датасет: каждая последовательность длины ровно {@code T}.
     * Первый токен — случайный из {@code [0, V)}, остальные — независимый
     * шум из того же алфавита.
     *
     * <p>Target — one-hot по первому токену: {@code target[x[0]] = 1},
     * остальное {@code 0}. Это то, что ожидает {@link SoftmaxCrossEntropy}.
     *
     * <p>Никакого padding'а нет: все последовательности длины {@code T},
     * поэтому можно обойтись без маски и учить «в лоб».
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
     * Один пример: последовательность длины {@code T} (первый токен —
     * «сигнал», остальные — шум), target — one-hot по первому токену.
     */
    static Sample randomSample(Random rnd) {
        float[] inputs = new float[T];
        for (int i = 0; i < T; i++) {
            inputs[i] = rnd.nextInt(VOCAB);
        }

        // Target — one-hot по первому токену.
        int first = (int) inputs[0];
        float[] targets = new float[VOCAB];
        targets[first] = 1f;

        return new Sample(inputs, targets);
    }
}