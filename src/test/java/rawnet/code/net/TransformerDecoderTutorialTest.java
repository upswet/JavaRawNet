package rawnet.code.net;

import rawnet.code.dataset.NInputSample;
import rawnet.code.dataset.SampleDataset;
import rawnet.code.net.TensorShape;
import rawnet.code.net.composition.TransformerEncoderBlock;
import rawnet.code.net.core.loss.SequenceCrossEntropy;
import rawnet.code.net.core.metric.LossMetric;
import rawnet.code.net.core.metric.Metric;
import rawnet.code.net.core.metric.TokenAccuracy;
import rawnet.code.net.core.optimizator.AdamW;
import rawnet.code.net.layers.core.EmbeddingLayer;
import rawnet.code.net.layers.core.PositionalEncodingLayer;
import rawnet.code.net.layers.core.TimeDistributedDenseLayer;
import rawnet.code.net.layers.mask.MaskingLayer;
import rawnet.code.net.composition.TransformerDecoderBlock;
import rawnet.code.net.composition.TransformerEncoderStack;
import rawnet.code.net.model.Model;
import rawnet.code.net.model.ModelTraining;
import rawnet.code.net.model.functional.Functional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * «Учебник»: Transformer-<b>декодер</b> на seq2seq-задаче.
 *
 * <h2>Задача</h2>
 * Реверс последовательности: дана строка {@code [a, b, c, d]}, декодер
 * должен выдать {@code [d, c, b, a]}. Требует <b>настоящего</b>
 * cross-attention: декодер не может «подсмотреть» ответ в Q (Q — это
 * {@code <s>} + уже сгенерированные токены), он должен смотреть на KV
 * (энкодер). Без cross-attention задача невыполнима.
 *
 * <h2>Архитектура</h2>
 * <pre>
 *   Encoder: Masking → Embedding → PositionalEncoding
 *            → TransformerEncoderStack → (encOut)
 *
 *   Decoder: Masking → Embedding → PositionalEncoding → (Q)
 *            TransformerDecoderBlock(Q, encOut) → Dense(vocab) → logits
 * </pre>
 *
 * <p>Энкодер и декодер объединены в один {@link Functional} с двумя
 * входами: Q = target (декодер), KV = source (энкодер). Оба
 * обрабатываются внутри одного графа, и градиент от декодера доходит
 * до энкодера через {@code CrossAttentionLayer.backward2}.
 *
 * <h2>Teacher forcing</h2>
 * На обучении Q = target <b>без последнего токена</b> (то есть
 * {@code [<s>, t_1, …, t_{T-2}]}), target = target <b>без первого</b>
 * (то есть {@code [t_1, …, t_{T-1}]}). Декодер предсказывает каждый
 * следующий токен по предыдущим — стандартный teacher forcing.
 *
 * <h2>Бюджет</h2>
 * 5 эпох, {@code lr=1e-2}, 2000 примеров, {@code T=6}, {@code dModel=16},
 * {@code heads=2}, {@code 1 блок}. ~10–15 с.
 *
 * <h2>Ожидаемый результат</h2>
 * Accuracy ~0.90–0.98 (реверс — простая задача, и cross-attention
 * решает её «из коробки»).
 */
class TransformerDecoderTutorialTest {

    private static final int T = 6;          // длина source (и target)
    private static final int VOCAB = 10;     // 0 = PAD, 1 = <s>, 2..9 = токены
    private static final int D_MODEL = 16;
    private static final int HEADS = 2;
    private static final int D_FF = 32;

    // =====================================================================
    // ==================== 1. Обучение seq2seq ============================
    // =====================================================================

    /**
     * Encoder-decoder на реверсе последовательности.
     *
     * <p>Проверяет:
     * <ul>
     *   <li>{@link TransformerDecoderBlock} собирается в {@link Functional}
     *       с двумя входами;</li>
     *   <li>{@link ModelTraining} обучает через {@code forwardN}/backwardN};</li>
     *   <li>cross-attention <b>работает</b>: декодер «видит» энкодер, и
     *       реверс выучивается (без cross-attention задача невыполнима);</li>
     *   <li>градиент от декодера доходит до энкодера (backwardN → backward
     *       энкодера).</li>
     * </ul>
     */
    @Test
    void decoder_reverses_sequence() {
        SampleDataset<NInputSample> ds = reverseDataset(2000, new Random(42L));

        // Q и KV — индексы (flat(T)); Embedding/PositionalEncoding — внутри графа.
        Functional.Builder b = Functional.builder(
                TensorShape.flat(T),    // Q: индексы декодера
                TensorShape.flat(T));   // KV: индексы энкодера

        Functional.Node tgtIdx = b.input();
        Functional.Node srcIdx = b.kvInput();

        // ── Энкодер ──
        Functional.Node srcMask = b.layer(new MaskingLayer.Dto(0), srcIdx);
        Functional.Node srcEmb  = b.layer(new EmbeddingLayer.Dto(VOCAB, D_MODEL), srcMask);
        Functional.Node srcPos  = b.layer(new PositionalEncodingLayer.Dto(), srcEmb);

        Model encoderBlock = TransformerEncoderStack.build(
                T, new TransformerEncoderBlock.Dto(HEADS, D_MODEL, D_FF, 0.0f, false), 1);
        Functional.Node encOut = b.model(encoderBlock, srcPos);

        // ── Декодер ──
        Functional.Node tgtMask = b.layer(new MaskingLayer.Dto(0), tgtIdx);
        Functional.Node tgtEmb  = b.layer(new EmbeddingLayer.Dto(VOCAB, D_MODEL), tgtMask);
        Functional.Node tgtPos  = b.layer(new PositionalEncodingLayer.Dto(), tgtEmb);

        Functional.Node decOut = TransformerDecoderBlock.build(
                b, tgtPos, encOut,
                new TransformerDecoderBlock.Dto(HEADS, D_MODEL, D_FF, 0.0f, T));

        Functional.Node logits = b.layer(new TimeDistributedDenseLayer.Dto(VOCAB), decOut);
        Functional decoder = b.output(logits);

        // ── Обучение ──
        var loss = new SequenceCrossEntropy();
        var opt  = new AdamW(1e-2f);
        List<Metric> metrics = List.of(new LossMetric(loss), new TokenAccuracy());

        new ModelTraining<Functional, NInputSample>()
                .setModel(decoder)
                .setDataset(ds)
                .setLoss(loss)
                .setOptimizer(opt)
                .setMaxEpoch(5)
                .setBatchSize(32)
                .setMetrics(metrics)
                .setEndPredicate(null)
                .setClipNorm(1.0f)
                .train();

        float acc = metrics.get(1).result();
        assertTrue(acc > 0.85f,
                "seq2seq reverse: ожидалось > 0.85, получено " + acc);
    }

    // =====================================================================
    // ==================== 2. Ошибки сборки ===============================
    // =====================================================================

    /**
     * {@link TransformerDecoderBlock.Dto} с {@code kvLen}, не совпадающим
     * с длиной {@code encOut}, должен падать на сборке.
     */
    @Test
    void decoder_rejects_wrong_kvLen() {
        Functional.Builder b = Functional.builder(
                TensorShape.seq(T, D_MODEL),
                TensorShape.seq(T, D_MODEL));
        Functional.Node tgt = b.input();
        Functional.Node enc = b.kvInput();

        // kvLen = T+1, а encOut.seqLen = T → ошибка
        var dto = new TransformerDecoderBlock.Dto(HEADS, D_MODEL, D_FF, 0.0f, T + 1);

        assertThrows(IllegalArgumentException.class,
                () -> TransformerDecoderBlock.build(b, tgt, enc, dto));
    }

    /**
     * {@link TransformerDecoderBlock.Dto} с {@code dModel % heads != 0}
     * должен падать уже в конструкторе DTO.
     */
    @Test
    void decoder_dto_rejects_bad_heads() {
        assertThrows(IllegalArgumentException.class,
                () -> new TransformerDecoderBlock.Dto(3, 16, 32, 0.0f, T));
        // 16 % 3 != 0
    }

    // =====================================================================
    // ==================== 3. Сериализация ================================
    // =====================================================================

    /**
     * Двухвходовый {@code Functional} с {@link TransformerDecoderBlock}
     * сохраняется и загружается: forward2 даёт тот же результат.
     */
    @Test
    void decoder_survives_save_load(@TempDir Path tmp) {
        Functional decoder = TransformerDecoderBlock.build(
                T, T,
                new TransformerDecoderBlock.Dto(HEADS, D_MODEL, D_FF, 0.0f, T));

        float[] q = new float[T * D_MODEL];
        float[] kv = new float[T * D_MODEL];
        for (int i = 0; i < q.length; i++)  q[i]  = 0.01f * i;
        for (int i = 0; i < kv.length; i++) kv[i] = 0.02f * i;

        float[] before = decoder.forward2(q, kv, false).clone();

        Path file = tmp.resolve("decoder.bin");
        decoder.save(file);

        Functional loaded = (Functional) Model.load(file);
        float[] after = loaded.forward2(q, kv, false).clone();

        assertEquals(before.length, after.length);
        for (int i = 0; i < before.length; i++) {
            assertEquals(before[i], after[i], 1e-6f,
                    "save/load: предсказание изменилось в позиции " + i);
        }
    }

    // =====================================================================
    // ==================== Датасет: реверс ================================
    // =====================================================================

    /**
     * Датасет для реверса: source — случайная последовательность длины T-1,
     * target — {@code [<s>, reverse(source)]}.
     *
     * <p>Q декодера — target <b>без последнего</b> токена (teacher forcing),
     * target loss — target <b>без первого</b> токена (предсказываем
     * следующие). KV — source.
     *
     * <p><b>Токены:</b> 0 = PAD, 1 = {@code <s>}, 2..9 — настоящие.
     */
    private static SampleDataset<NInputSample> reverseDataset(int n, Random rnd) {
        List<NInputSample> train = new ArrayList<>();
        List<NInputSample> test  = new ArrayList<>();

        int srcLen = T - 1;

        for (int i = 0; i < n; i++) {
            float[] kv = new float[T];
            for (int j = 0; j < srcLen; j++)
                kv[j] = 2 + rnd.nextInt(VOCAB - 2);

            int[] tgt = new int[T];
            tgt[0] = 1;
            for (int j = 0; j < srcLen; j++)
                tgt[j + 1] = (int) kv[srcLen - 1 - j];

            float[] q          = new float[T];
            float[] lossTarget = new float[T];
            for (int j = 0; j < T - 1; j++) {
                q[j]          = tgt[j];
                lossTarget[j] = tgt[j + 1];
            }

            float[][] inputs = new float[][]{ q, kv };
            NInputSample sample = new NInputSample(inputs, lossTarget);

            if (i < n * 0.8) train.add(sample);
            else             test.add(sample);
        }
        return new SampleDataset<NInputSample>(train, test);
    }
}