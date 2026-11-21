package rawnet.docs.flows.main.transformer.tasks;

import rawnet.code.dataset.SampleDataset;
import rawnet.code.dataset.Sample;
import rawnet.code.net.TensorShape;
import rawnet.code.net.composition.TransformerEncoderBlock;
import rawnet.code.net.composition.TransformerEncoderStack;
import rawnet.code.net.core.loss.SoftmaxCrossEntropy;
import rawnet.code.net.core.metric.Accuracy;
import rawnet.code.net.core.metric.LossMetric;
import rawnet.code.net.core.metric.Metric;
import rawnet.code.net.core.optimizator.AdamW;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.core.DenseLayer;
import rawnet.code.net.layers.core.EmbeddingLayer;
import rawnet.code.net.layers.core.PositionalEncodingLayer;
import rawnet.code.net.layers.mask.MaskingLayer;
import rawnet.code.net.layers.structure.ModelLayer;
import rawnet.code.net.layers.structure.ReshapeLayer;
import rawnet.code.net.layers.structure.SliceLayer;
import rawnet.code.net.model.Model;
import rawnet.code.net.model.ModelTraining;
import rawnet.code.net.model.sequence.Sequence;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Учебный пример: <b>associative recall</b> (mini-dictionary) через
 * Transformer-энкодер.
 *
 * <h2>Задача</h2>
 * Дана последовательность пар {@code k→v} и {@code query} (последний
 * токен). Нужно вернуть {@code v}, где {@code k == query}.
 *
 * <pre>
 *   3→5  1→7  4→2  0→6  query=4  →  2
 * </pre>
 *
 * <p>Вход — {@code 2*P + 1} токенов: {@code P} пар {@code (k_i, v_i)}
 * подряд, затем {@code query}. Выход — один токен {@code v_i}, где
 * {@code k_i == query}.
 *
 * <h2>Почему Transformer</h2>
 * Attention на позиции {@code query} смотрит на <b>все</b> ключи
 * {@code k_i} одновременно, softmax выделяет совпавший (вес ≈1),
 * FFN+Dense считывает {@code v_i} из этой позиции. RNN этого не может:
 * к моменту {@code query} (позиция {@code 2P}) информация о {@code k_0}
 * «затухает».
 *
 * <h2>Тонкость: сдвиг V</h2>
 * В self-attention V — те же позиции, что Q/K: на позиции {@code k_i}
 * лежит {@code k_i}, а не {@code v_i}. Модель должна <b>выучить</b>
 * сдвиг (что {@code v_i} — на следующей позиции). На {@code P=4},
 * {@code vocab=10}, 1 блоке, 2000 примерах — обычно получается.
 *
 * <h2>Архитектура</h2>
 * <pre>
 *   flat(T) → Masking → Embedding → PositionalEncoding
 *           → TransformerEncoderStack (1 блок)
 *           → Slice(SEQ, T-1, 1)        ← только позиция query
 *           → Flatten                    ← seq(1, dModel) → flat(dModel)
 *           → Dense(vocab)               ← логиты
 *           → SoftmaxCrossEntropy        ← one-hot target
 * </pre>
 *
 * <h2>Запуск</h2>
 * <pre>{@code
 * java rawnet.docs.code.task.transformer.AssociativeRecallExample
 * }</pre>
 *
 * <p>Печатает прогресс по эпохам, итоговую accuracy и несколько
 * примеров «предсказание vs истина».
 */
public final class AssociativeRecallExample {

    private static final Logger log = LoggerFactory.getLogger(AssociativeRecallExample.class);

    // =====================================================================
    // ==================== Гиперпараметры =================================
    // =====================================================================

    /** Число пар {@code k→v} в одном примере. */
    static final int P = 4;

    /** Размер алфавита: 0 = PAD, 1..VOCAB-1 = настоящие токены. */
    static final int VOCAB = 10;

    /** Длина последовательности: {@code 2P + 1} (пары + query). */
    static final int T = 2 * P + 1;

    /** Размерность эмбеддинга (и dModel). */
    static final int D_MODEL = 32;

    /** Число голов attention; {@code dModel % heads == 0}. */
    static final int HEADS = 4;

    /** Ширина FFN; {@code 4 * dModel}. */
    static final int D_FF = 4 * D_MODEL;

    /** Dropout в Transformer-блоке; 0 — выключен (маленькая модель). */
    static final float DROPOUT = 0.0f;

    /** Число Transformer-блоков. */
    static final int N_LAYERS = 1;

    /** Размер батча. */
    static final int BATCH = 32;

    /** Число эпох. */
    static final int EPOCHS = 50;

    /** Скорость обучения AdamW. */
    static final float LR = 1e-3f;

    /** Глобальный клипинг градиентов. */
    static final float CLIP_NORM = 1.0f;

    /** Размер train-выборки. */
    static final int N_TRAIN = 2000;

    /** Размер test-выборки. */
    static final int N_TEST = 500;

    /** Сколько примеров печатать в конце. */
    static final int SHOW_EXAMPLES = 5;

    // =====================================================================
    // ==================== main ===========================================
    // =====================================================================

    public static void main(String[] args) {
        long seed = 42L;

        SampleDataset<Sample> ds = generateDataset(seed);

        log.info("Датасет: train={}, test={}, P={}, T={}, vocab={}",
                ds.getTrainSamples().size(), ds.getTestSamples().size(), P, T, VOCAB);

        // ─────────── Сборка модели ───────────
        // Transformer-блок как готовая Model (ModelLayer.Dto, не Dto!)
        Model encoderBlock = TransformerEncoderStack.build(
                T,
                new TransformerEncoderBlock.Dto(HEADS, D_MODEL, D_FF, DROPOUT, false),
                N_LAYERS);

        List<Layer.Dto> dtos = List.of(
                new Sequence.InputDto(TensorShape.flat(T)),
                new MaskingLayer.Dto(0),                          // PAD = 0
                new EmbeddingLayer.Dto(VOCAB, D_MODEL),
                new PositionalEncodingLayer.Dto(),
                new ModelLayer.Dto(encoderBlock),                 // Transformer-блок
                new SliceLayer.Dto(SliceLayer.Axis.SEQ, T - 1, 1),// только query
                new ReshapeLayer.FlattenLayer.Dto(),                           // seq(1, dModel) → flat(dModel)
                new DenseLayer.Dto(TensorShape.flat(VOCAB))       // логиты по vocab
        );
        Sequence net = Sequence.of(dtos);

        log.info("Модель:\n{}", net);

        // ─────────── Обучение ───────────
        var loss = new SoftmaxCrossEntropy();
        var opt = new AdamW(LR);
        List<Metric> metrics = List.of(new LossMetric(loss), new Accuracy());

        new ModelTraining<Model, Sample>()
                .setModel(net)
                .setDataset(ds)
                .setLoss(loss)
                .setOptimizer(opt)
                .setMaxEpoch(EPOCHS)
                .setBatchSize(BATCH)
                .setShowTestCount(0)
                .setMetrics(metrics)
                .setClipNorm(CLIP_NORM)
                .train();

        float acc = metrics.get(1).result();
        log.info("");
        log.info("══════════════════════════════════════════════════════════════════");
        log.info("  Итог: {}", Metric.print(metrics));
        log.info("  Accuracy = {}", fmt(acc));
        log.info("══════════════════════════════════════════════════════════════════");

        // ─────────── Примеры ───────────
        log.info("");
        log.info("Примеры (prediction vs target):");
        for (int i = 0; i < SHOW_EXAMPLES && i < ds.getTestSamples().size(); i++) {
            Sample s = ds.getTestSamples().get(i);
            float[] logits = net.forward(s.getInputs(), false);

            int pred = argmax(logits);
            int truth = argmax(s.getTargets());

            log.info("  пример {}: sequence = {}, query = {}, prediction = {}, truth = {}  {}",
                    i,
                    formatSequence(s.getInputs(), T - 1),
                    (int) s.getInputs()[T - 1],
                    pred,
                    truth,
                    pred == truth ? "✓" : "✗");
        }
    }

    // =====================================================================
    // ==================== Датасет ========================================
    // =====================================================================

    /**
     * Генерирует датасет: пары {@code k→v} + {@code query}, target —
     * one-hot по {@code VOCAB} (индекс ответа = 1, остальные 0).
     *
     * <p><b>Почему one-hot.</b> {@link SoftmaxCrossEntropy} ожидает
     * {@code logits = flat(VOCAB)} и {@code target = flat(VOCAB)}
     * (вероятности или one-hot). Индекс-ответ (scalar) — <b>не</b>
     * подходит: {@code loss} трактует вход как один вектор логитов
     * длины {@code VOCAB}, а target — как вероятности. One-hot — самый
     * простой корректный формат.
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
     * Один пример: {@code P} пар {@code (k_i, v_i)} + {@code query}.
     *
     * <p>Ключи {@code k_i} — <b>уникальны</b> в примере (чтобы ответ был
     * однозначен). Значения {@code v_i} — любые из {@code 1..VOCAB-1}.
     * {@code query} — один из ключей.
     */
    static Sample randomSample(Random rnd) {
        int[] keys   = new int[P];
        int[] values = new int[P];
        boolean[] used = new boolean[VOCAB];

        for (int j = 0; j < P; j++) {
            int k;
            do { k = 1 + rnd.nextInt(VOCAB - 1); } while (used[k]);
            used[k] = true;
            keys[j] = k;
            values[j] = 1 + rnd.nextInt(VOCAB - 1);
        }

        // Query — случайный ключ
        int qIdx  = rnd.nextInt(P);
        int query = keys[qIdx];
        int answer = values[qIdx];

        // Последовательность: k0, v0, k1, v1, …, k_{P-1}, v_{P-1}, query
        float[] x = new float[T];
        for (int j = 0; j < P; j++) {
            x[2 * j]     = keys[j];
            x[2 * j + 1] = values[j];
        }
        x[T - 1] = query;

        // Target — one-hot: y[answer] = 1, остальные 0
        float[] y = new float[VOCAB];
        y[answer] = 1f;

        return new Sample(x, y);
    }

    // =====================================================================
    // ==================== Утилиты ========================================
    // =====================================================================

    /** Индекс максимума (argmax). */
    static int argmax(float[] a) {
        int best = 0;
        for (int i = 1; i < a.length; i++)
            if (a[i] > a[best]) best = i;
        return best;
    }

    /** Форматирует последовательность «k0→v0  k1→v1  …». */
    static String formatSequence(float[] x, int len) {
        StringBuilder sb = new StringBuilder();
        for (int j = 0; j + 1 < len; j += 2) {
            if (j > 0) sb.append("  ");
            sb.append((int) x[j]).append("→").append((int) x[j + 1]);
        }
        return sb.toString();
    }

    static String fmt(float v) {
        return String.format(Locale.US, "%.4f", v);
    }
}