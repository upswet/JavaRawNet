package rawnet.code.net.composition;

import rawnet.code.net.TensorShape;
import rawnet.code.net.core.activate.Gelu;
import rawnet.code.net.layers.attention.CrossAttentionLayer;
import rawnet.code.net.layers.attention.MultiHeadAttentionLayer;
import rawnet.code.net.layers.core.DropoutLayer;
import rawnet.code.net.layers.core.LayerNormLayer;
import rawnet.code.net.layers.core.TimeDistributedDenseLayer;
import rawnet.code.net.model.functional.Functional;

/**
 * Блок Transformer-<b>декодера</b> (pre-LN, для seq2seq).
 *
 * <pre>
 *   x ──► LN ──► MaskedSelfAttention(causal) ──► + ──► x1
 *   │                                              ▲
 *   └──────────────────────────────────────────────┘ (residual)
 *
 *   x1 ─► LN ──► CrossAttention(Q=x1, KV=encOut) ──► + ──► x2
 *   │                                                  ▲
 *   └──────────────────────────────────────────────────┘ (residual)
 *
 *   x2 ─► LN ──► FFN(Dense→GELU→Dense) ──► + ──► y
 *   │                                      ▲
 *   └──────────────────────────────────────┘ (residual)
 * </pre>
 *
 * <p>Простыми словами: декодер делает <b>три</b> вещи, а не две, как
 * энкодер:
 * <ol>
 *   <li><b>causal self-attention</b> — токен {@code t} видит только
 *       {@code 0..t} (не подглядывает в будущее). Это нужно для
 *       авторегрессии: предсказываем токен {@code t+1} по {@code 0..t};</li>
 *   <li><b>cross-attention</b> — декодер «смотрит» на выход энкодера
 *       ({@code encOut}): Q из декодера, KV из энкодера. Это позволяет
 *       переводу «сопоставить» слова источника и цели;</li>
 *   <li><b>FFN</b> — как в энкодере, обработка каждого токена отдельно.</li>
 * </ol>
 *
 * <h2>Два входа</h2>
 * Блок принимает <b>два</b> узла: {@code x} (декодер) и {@code encOut}
 * (энкодер). Оба — {@code seq(T, dModel)}. Для cross-attention
 * {@code Tq = T} (длина декодера), {@code Tkv} — длина энкодера
 * (в DTO — {@link Dto#kvLen}).
 *
 * <p>Требует, чтобы {@code Functional} поддерживал <b>два</b> входа
 * ({@code Functional.builder(qShape, kvShape)} + {@code kvInput()}).
 * Если {@code encOut} уже существует в том же билдере (например, собран
 * из {@code x} через {@code Dense}/{@code Reshape}), блок работает и в
 * одновходовом графе — но тогда cross-attention вырождается в
 * self-attention, и это <b>не</b> seq2seq.
 *
 * <h2>Где использовать</h2>
 * Seq2seq: машинный перевод (T5, BART), summarization, captioning
 * (изображение → текст), любой «вход → выход-последовательность».
 *
 * <h2>Параметры</h2>
 * <ul>
 *   <li>{@code heads}, {@code dModel}, {@code dFF} — как у энкодера;</li>
 *   <li>{@code dropout} — в self-attention, cross-attention и FFN;</li>
 *   <li>{@code kvLen} — длина последовательности энкодера (KV) для
 *       cross-attention.</li>
 * </ul>
 *
 * <h2>Как собрать (настоящий seq2seq)</h2>
 * <pre>{@code
 * Functional.Builder dec = Functional.builder(
 *         TensorShape.seq(tgtLen, dModel),    // Q — декодер
 *         TensorShape.seq(srcLen, dModel));   // KV — энкодер
 *
 * Functional.Node tgt = dec.input();
 * Functional.Node enc = dec.kvInput();
 * Functional.Node y = TransformerDecoderBlock.build(
 *         dec, tgt, enc, new TransformerDecoderBlock.Dto(heads, dModel, dFF, 0f, srcLen));
 * Functional.Node logits = dec.layer(new TimeDistributedDenseLayer.Dto(vocab), y);
 * Functional decoder = dec.output(logits);
 *
 * // Обучение: decoder.train2(epochs, ds, loss, opt, ...)
 * // Inference: decoder.forward2(target, encoderOut, false)
 * }</pre>
 *
 * <h2>Сериализация</h2>
 * Блок — часть {@link Functional}. Двухвходовый {@code Functional}
 * сохраняется/загружается как обычно (см. {@code Functional.save/load}):
 * флаг {@code hasKvInput} + топология + параметры слоёв.
 */
public final class TransformerDecoderBlock {

    private TransformerDecoderBlock() {}

    // =====================================================================
    // ==================== DTO ============================================
    // =====================================================================

    /**
     * Параметры блока декодера.
     */
    public static final class Dto {
        /** Число голов attention; {@code dModel % heads == 0}. */
        public final int heads;
        /** Размерность модели; {@code dModel % heads == 0}. */
        public final int dModel;
        /** Ширина FFN (обычно {@code 4 * dModel}). */
        public final int dFF;
        /** Dropout в self-attention, cross-attention и FFN. */
        public final float dropout;
        /** Длина последовательности энкодера (KV). */
        public final int kvLen;

        public Dto(int heads, int dModel, int dFF, float dropout, int kvLen) {
            if (heads < 1)         throw new IllegalArgumentException("heads >= 1, получено " + heads);
            if (dModel < 1)        throw new IllegalArgumentException("dModel >= 1, получено " + dModel);
            if (dModel % heads != 0)
                throw new IllegalArgumentException(
                        "dModel (" + dModel + ") % heads (" + heads + ") != 0");
            if (dFF < 1)           throw new IllegalArgumentException("dFF >= 1, получено " + dFF);
            if (dropout < 0f || dropout >= 1f)
                throw new IllegalArgumentException("dropout in [0, 1), получено " + dropout);
            if (kvLen < 1)         throw new IllegalArgumentException("kvLen >= 1, получено " + kvLen);
            this.heads = heads;
            this.dModel = dModel;
            this.dFF = dFF;
            this.dropout = dropout;
            this.kvLen = kvLen;
        }

        /** Удобный конструктор: {@code dFF = 4·dModel}, {@code dropout = 0.1}. */
        public Dto(int heads, int dModel, int kvLen) {
            this(heads, dModel, 4 * dModel, 0.1f, kvLen);
        }
    }

    // =====================================================================
    // ==================== build (2 входа) ================================
    // =====================================================================

    /**
     * Строит блок декодера с <b>двумя</b> входами: {@code x} (декодер)
     * и {@code encOut} (энкодер).
     *
     * @param b      builder (он же граф)
     * @param x      вход декодера {@code seq(Tq, dModel)}
     * @param encOut выход энкодера {@code seq(Tkv, dModel)}
     * @param dto    параметры блока; {@code dto.kvLen == encOut.shape.seqLen()}
     * @return выход декодера {@code seq(Tq, dModel)}
     * @throws IllegalArgumentException если формы не совпадают с {@code dto}
     */
    public static Functional.Node build(Functional.Builder b,
                                        Functional.Node x,
                                        Functional.Node encOut,
                                        Dto dto) {
        if (b == null)      throw new IllegalArgumentException("TransformerDecoderBlock: b == null");
        if (dto == null)    throw new IllegalArgumentException("TransformerDecoderBlock: dto == null");
        if (x == null)      throw new IllegalArgumentException("TransformerDecoderBlock: x == null");
        if (encOut == null) throw new IllegalArgumentException("TransformerDecoderBlock: encOut == null");

        if (!x.shape.isSequence())
            throw new IllegalArgumentException(
                    "TransformerDecoderBlock: x должен быть seq(Tq, dModel), получено " + x.shape);
        if (!encOut.shape.isSequence())
            throw new IllegalArgumentException(
                    "TransformerDecoderBlock: encOut должен быть seq(Tkv, dModel), получено " + encOut.shape);
        if (x.shape.channels() != dto.dModel)
            throw new IllegalArgumentException(
                    "TransformerDecoderBlock: x.channels=" + x.shape.channels()
                            + " != dModel=" + dto.dModel);
        if (encOut.shape.channels() != dto.dModel)
            throw new IllegalArgumentException(
                    "TransformerDecoderBlock: encOut.channels=" + encOut.shape.channels()
                            + " != dModel=" + dto.dModel);
        if (encOut.shape.seqLen() != dto.kvLen)
            throw new IllegalArgumentException(
                    "TransformerDecoderBlock: encOut.seqLen=" + encOut.shape.seqLen()
                            + " != kvLen=" + dto.kvLen);

        // ---------- 1. Causal self-attention ----------
        Functional.Node ln1 = b.layer(new LayerNormLayer.Dto(), x);
        Functional.Node selfAttn = b.layer(
                new MultiHeadAttentionLayer.Dto(dto.heads, /*causal=*/ true), ln1);
        Functional.Node selfDropped = maybeDropout(b, selfAttn, dto.dropout);
        Functional.Node x1 = b.add(x, selfDropped);

        // ---------- 2. Cross-attention (Q = x1, KV = encOut) ----------
        Functional.Node ln2 = b.layer(new LayerNormLayer.Dto(), x1);
        Functional.Node cross = b.crossAttention(
                ln2, encOut,
                new CrossAttentionLayer.Dto(dto.heads, dto.dModel, dto.dModel, dto.kvLen));
        Functional.Node crossDropped = maybeDropout(b, cross, dto.dropout);
        Functional.Node x2 = b.add(x1, crossDropped);

        // ---------- 3. FFN ----------
        Functional.Node ln3 = b.layer(new LayerNormLayer.Dto(), x2);
        Functional.Node ffn1 = b.layer(new TimeDistributedDenseLayer.Dto(dto.dFF), ln3);
        Functional.Node ffn2 = b.activation(ffn1, new Gelu(true));
        Functional.Node ffn3 = b.layer(new TimeDistributedDenseLayer.Dto(dto.dModel), ffn2);
        Functional.Node ffnDropped = maybeDropout(b, ffn3, dto.dropout);
        Functional.Node y = b.add(x2, ffnDropped);

        return y;
    }

    /**
     * Собирает блок декодера как готовую {@link Functional} с <b>двумя</b>
     * входами: Q (декодер) и KV (энкодер).
     *
     * <p>Удобно для тестов и для использования как самостоятельной модели:
     * <pre>{@code
     * Functional dec = TransformerDecoderBlock.build(
     *         tgtLen, srcLen,
     *         new TransformerDecoderBlock.Dto(heads, dModel, dFF, 0f, srcLen));
     * }</pre>
     *
     * @param tgtLen длина последовательности декодера {@code Tq}
     * @param srcLen длина последовательности энкодера {@code Tkv}
     * @param dto    параметры блока; {@code dto.kvLen == srcLen}
     * @return {@code Functional} с двумя входами: Q и KV
     */
    public static Functional build(int tgtLen, int srcLen, Dto dto) {
        if (dto == null)
            throw new IllegalArgumentException("TransformerDecoderBlock: dto == null");
        if (dto.kvLen != srcLen)
            throw new IllegalArgumentException(
                    "TransformerDecoderBlock: dto.kvLen=" + dto.kvLen
                            + " != srcLen=" + srcLen);

        Functional.Builder b = Functional.builder(
                TensorShape.seq(tgtLen, dto.dModel),   // Q
                TensorShape.seq(srcLen, dto.dModel));  // KV
        Functional.Node tgt = b.input();
        Functional.Node enc = b.kvInput();
        Functional.Node y = build(b, tgt, enc, dto);
        return b.output(y);
    }

    private static Functional.Node maybeDropout(Functional.Builder b,
                                                Functional.Node node,
                                                float p) {
        if (p <= 0f) return node;
        return b.layer(new DropoutLayer.Dto(p), node);
    }
}