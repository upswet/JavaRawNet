package rawnet.code.net.composition;

import rawnet.code.net.TensorShape;
import rawnet.code.net.core.activate.Gelu;
import rawnet.code.net.layers.attention.MultiHeadAttentionLayer;
import rawnet.code.net.layers.core.LayerNormLayer;
import rawnet.code.net.layers.core.DropoutLayer;
import rawnet.code.net.layers.core.TimeDistributedDenseLayer;
import rawnet.code.net.model.Model;
import rawnet.code.net.model.functional.Functional;

/**
 * Один блок Transformer-энкодера (pre-LN).
 *
 * <pre>
 *   ┌─────────────────────────────────────────────────────────┐
 *   │  x ──► LayerNorm ──► MultiHeadAttention ──► + ──► x1    │
 *   │  │                                          ▲           │
 *   │  └──────────────────────────────────────────┘ (residual)│
 *   │                                                         │
 *   │  x1 ─► LayerNorm ──► FFN (Dense→GELU→Dense) ─► + ──► y  │
 *   │  │                                            ▲         │
 *   │  └────────────────────────────────────────────┘(residual)│
 *   └─────────────────────────────────────────────────────────┘
 * </pre>
 *
 * <p>Простыми словами: блок делает две вещи — «обмен информацией между
 * токенами» (attention) и «обработку каждого токена отдельно» (FFN).
 * После каждой — residual-связь (прибавление входа) и LayerNorm
 * (нормировка перед следующим шагом). Это ядро Transformer'а.
 *
 * <h2>Pre-LN vs Post-LN</h2>
 * <ul>
 *   <li><b>Post-LN</b> (Vaswani 2017): {@code x = LN(x + Sublayer(x))}.
 *       Требует learning-rate warmup, иначе расходится;</li>
 *   <li><b>Pre-LN</b> (GPT-2/3, LLaMA, современные): {@code x = x + Sublayer(LN(x))}.
 *       Устойчив без warmup, лучше для глубоких сетей. <b>Используется здесь.</b></li>
 * </ul>
 *
 * <h2>Формат входа/выхода</h2>
 * {@code seq(T, dModel) → seq(T, dModel)} — форма не меняется. Это позволяет
 * ставить блоки «стопкой»: выход одного — вход следующего.
 *
 * <h2>Маска</h2>
 * Блок <b>прозрачен для маски</b>: {@link MultiHeadAttentionLayer} читает
 * маску ({@code MaskConsumer}), а {@code LayerNorm}/{@code TimeDistributedDense}/
 * {@code ActivationLayer} — нейтральные (маска идёт мимо). Поэтому
 * {@code TransformerEncoderBlock} можно ставить в {@code Sequence} после
 * {@code MaskingLayer} — маска дойдёт до attention.
 *
 * <h2>Как собрать</h2>
 * <pre>{@code
 * Model net = Sequence.of(List.of(
 *     new Sequence.InputDto(TensorShape.flat(T)),          // индексы токенов
 *     new MaskingLayer.Dto(PAD),
 *     new EmbeddingLayer.Dto(vocab, dModel),
 *     new PositionalEncodingLayer.Dto(),
 *     new TransformerEncoderBlock.Dto(heads, dModel, dFF, dropout),
 *     new TransformerEncoderBlock.Dto(heads, dModel, dFF, dropout),  // стопка
 *     new TokenPoolingLayer.Dto(TokenPoolingLayer.Mode.MEAN),
 *     new DenseLayer.Dto(TensorShape.flat(numClasses)),
 *     new ActivationLayer.Dto(new Sigmoid())               // или без, для SoftmaxCE
 * ));
 * }</pre>
 *
 * <h2>Гиперпараметры (правила большого пальца)</h2>
 * <ul>
 *   <li>{@code dModel}: 64–512 для реальных задач; 8–32 для учебных;</li>
 *   <li>{@code heads}: {@code dModel / 64} обычно; {@code dModel % heads == 0} обязательно;</li>
 *   <li>{@code dFF}: {@code 4 * dModel} — стандарт (FFN «шире» attention);</li>
 *   <li>{@code dropout}: 0.1 — стандарт; 0 — для маленьких/учебных.</li>
 * </ul>
 *
 * <h2>Ограничение</h2>
 * Блок реализован как <b>фабрика</b> ({@link Dto} + {@link #build}),
 * а не как {@code Layer}. Причина: внутри — DAG с residual и двумя
 * ветвями (attention/FFN), а это {@link Functional}, не {@code Sequence}.
 * Сделать {@code Layer} можно, но тогда пришлось бы вручную писать
 * backward через все под-слои — дублирование того, что уже делает
 * {@code Functional}. Фабрика возвращает готовый {@link Functional},
 * который встраивается в {@code Sequence} через {@link ModelLayer}.
 */
public final class TransformerEncoderBlock {

    private TransformerEncoderBlock() {}

    // =====================================================================
    // ==================== DTO ============================================
    // =====================================================================

    /**
     * Параметры блока Transformer-энкодера.
     *
     * <p>Иммутабельный; передаётся в {@link #build(Functional.Builder)}.
     */
    public static final class Dto {
        /** Число голов attention; {@code dModel % heads == 0}. */
        public final int heads;
        /** Размерность модели (ширина потока); {@code dModel % heads == 0}. */
        public final int dModel;
        /** Ширина FFN (обычно {@code 4 * dModel}). */
        public final int dFF;
        /** Dropout в attention и FFN; {@code 0 <= dropout < 1}. */
        public final float dropout;
        /** Causal-маска в self-attention (для декодера / language modeling). */
        public final boolean causal;

        /**
         * @param heads   число голов attention
         * @param dModel  размерность модели
         * @param dFF     ширина FFN (обычно {@code 4 * dModel})
         * @param dropout dropout; {@code 0} — выключен
         * @param causal  causal-маска (для авторегрессии)
         */
        public Dto(int heads, int dModel, int dFF, float dropout, boolean causal) {
            if (heads < 1)         throw new IllegalArgumentException("heads >= 1, получено " + heads);
            if (dModel < 1)        throw new IllegalArgumentException("dModel >= 1, получено " + dModel);
            if (dModel % heads != 0)
                throw new IllegalArgumentException(
                        "dModel (" + dModel + ") % heads (" + heads + ") != 0");
            if (dFF < 1)           throw new IllegalArgumentException("dFF >= 1, получено " + dFF);
            if (dropout < 0f || dropout >= 1f)
                throw new IllegalArgumentException("dropout in [0, 1), получено " + dropout);
            this.heads = heads;
            this.dModel = dModel;
            this.dFF = dFF;
            this.dropout = dropout;
            this.causal = causal;
        }

        /** Удобный конструктор: {@code dFF = 4 * dModel}, без causal, dropout=0.1. */
        public Dto(int heads, int dModel) {
            this(heads, dModel, 4 * dModel, 0.1f, false);
        }
    }

    // =====================================================================
    // ==================== Сборка =========================================
    // =====================================================================

    /**
     * Строит блок в заданном {@link Functional.Builder}.
     *
     * <p>Вход {@code x} должен иметь форму {@code seq(T, dModel)}.
     * Выход — тоже {@code seq(T, dModel)}. Форма не меняется, поэтому
     * блоки можно ставить стопкой.
     *
     * <p><b>Внутренняя структура (pre-LN):</b>
     * <pre>
     *   x1 = x + Dropout(MHA(LN(x)))           // attention-подслой
     *   y  = x1 + Dropout(FFN(LN(x1)))         // FFN-подслой
     * </pre>
     *
     * @param b   builder (он же граф)
     * @param x   входной узел {@code seq(T, dModel)}
     * @param dto параметры блока
     * @return выходной узел {@code seq(T, dModel)}
     */
    public static Functional.Node build(Functional.Builder b, Functional.Node x, Dto dto) {
        if (!x.shape.isSequence())
            throw new IllegalArgumentException(
                    "TransformerEncoderBlock: ожидался seq(T, dModel), получено " + x.shape);
        if (x.shape.channels() != dto.dModel)
            throw new IllegalArgumentException(
                    "TransformerEncoderBlock: x.channels=" + x.shape.channels()
                            + " != dModel=" + dto.dModel);

        // ---------- 1. Attention-подслой (pre-LN) ----------
        //   x1 = x + Dropout(MHA(LN(x)))
        Functional.Node ln1 = b.layer(new LayerNormLayer.Dto(), x);

        Functional.Node mha = b.layer(
                new MultiHeadAttentionLayer.Dto(dto.heads, dto.causal),
                ln1);

        Functional.Node mhaDropped = maybeDropout(b, mha, dto.dropout);

        Functional.Node x1 = b.add(x, mhaDropped);

        // ---------- 2. FFN-подслой (pre-LN) ----------
        //   y = x1 + Dropout(FFN(LN(x1)))
        Functional.Node ln2 = b.layer(new LayerNormLayer.Dto(), x1);

        // FFN: TimeDistributedDense(dModel → dFF) → GELU → TimeDistributedDense(dFF → dModel)
        Functional.Node ffn1 = b.layer(new TimeDistributedDenseLayer.Dto(dto.dFF), ln2);
        Functional.Node ffn2 = b.activation(ffn1, new Gelu(true));
        Functional.Node ffn3 = b.layer(new TimeDistributedDenseLayer.Dto(dto.dModel), ffn2);

        Functional.Node ffnDropped = maybeDropout(b, ffn3, dto.dropout);

        Functional.Node y = b.add(x1, ffnDropped);
        return y;
    }

    /**
     * Если {@code dropout > 0}, добавляет слой dropout; иначе возвращает
     * {@code node} как есть (не создаём пустой слой ради нуля).
     *
     * <p><b>Зачем.</b> {@code DropoutLayer.Dto(0)} технически можно, но
     * это лишний слой в графе и лишний вызов {@code forward}/{@code backward}.
     * Пропуск при {@code 0} — микро-оптимизация и чище в {@code toString}.
     */
    private static Functional.Node maybeDropout(Functional.Builder b,
                                                Functional.Node node,
                                                float p) {
        if (p <= 0f) return node;
        return b.layer(new DropoutLayer.Dto(p), node);
    }

    // =====================================================================
    // ==================== Готовый блок как Model =========================
    // =====================================================================

    /**
     * Собирает блок как самостоятельную {@link Model}.
     *
     * <p>Удобно для тестов и для вложения в другую сеть через
     * {@link ModelLayer}. Вход/выход — {@code seq(T, dModel)}.
     *
     * @param seqLen длина последовательности {@code T}
     * @param dto    параметры блока
     * @return готовая модель-блок
     */
    public static Model build(int seqLen, Dto dto) {
        Functional.Builder b = Functional.builder(TensorShape.seq(seqLen, dto.dModel));
        Functional.Node x = b.input();
        Functional.Node y = build(b, x, dto);
        return b.output(y);
    }
}