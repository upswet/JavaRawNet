package rawnet.code.net.composition;

import rawnet.code.net.TensorShape;
import rawnet.code.net.layers.core.LayerNormLayer;
import rawnet.code.net.model.Model;
import rawnet.code.net.model.functional.Functional;
import rawnet.code.net.model.sequence.Sequence;

/**
 * Стопка из N одинаковых {@link TransformerEncoderBlock} + финальный
 * {@code LayerNorm}.
 *
 * <p>Простыми словами: энкодер Transformer'а — это не один блок, а
 * <b>много</b> блоков подряд (BERT-base — 12, BERT-large — 24, GPT-3 — 96).
 * Каждый блок «смотрит» на все токены и обрабатывает их; стопка
 * постепенно строит всё более абстрактные представления. Финальный
 * {@code LayerNorm} нормирует выход стопки — так делают все современные
 * Transformer'ы (pre-LN).
 *
 * <h2>Композиция, а не наследование</h2>
 * {@code Stack} <b>использует</b> {@link TransformerEncoderBlock}, но не
 * расширяет его:
 * <ul>
 *   <li>{@code Block} — «кирпич»: один слой, возвращает {@code Node}
 *       (внутри билдера) или {@code Model} (готовый);</li>
 *   <li>{@code Stack} — «стена»: N блоков + финальный {@code LN},
 *       возвращает {@code Model}.</li>
 * </ul>
 * Финальный {@code LN} — <b>не часть блока</b>: в pre-LN каждый блок
 * нормирует <i>вход</i> подслоя, а выход последнего блока остаётся
 * ненормированным. {@code Stack} добавляет {@code LN} после всех блоков —
 * это стандарт BERT/GPT.
 *
 * <h2>Блоки не разделяют веса</h2>
 * Каждый {@link TransformerEncoderBlock#build(int, TransformerEncoderBlock.Dto)}
 * создаёт <b>новый</b> {@link Functional} со своими параметрами. Это
 * стандарт для Transformer'ов (BERT/GPT не разделяют веса между слоями).
 * Weight sharing (ALBERT-стиль) — отдельная задача, здесь не поддержано.
 *
 * <h2>Сколько блоков</h2>
 * <ul>
 *   <li>{@code 2–4} — учебные задачи, маленькие данные;</li>
 *   <li>{@code 6} — BERT-small, реальные небольшие задачи;</li>
 *   <li>{@code 12} — BERT-base, стандарт;</li>
 *   <li>{@code 24+} — большие модели (BERT-large, GPT-2).</li>
 * </ul>
 * Каждый блок — это ~{@code 4·dModel² + 8·dModel·dFF} параметров; для
 * {@code dModel=256, dFF=1024} это ~3M параметров на блок. 12 блоков —
 * ~36M, как BERT-base.
 *
 * <h2>Как собрать</h2>
 * <pre>{@code
 * // Как готовый Model (для ModelLayer в Sequence):
 * Model encoder = TransformerEncoderStack.build(
 *         T, new TransformerEncoderBlock.Dto(heads, dModel, dFF, 0.1f, false), 6);
 *
 * Model net = Sequence.of(List.of(
 *     new Sequence.InputDto(TensorShape.flat(T)),
 *     new MaskingLayer.Dto(PAD),
 *     new EmbeddingLayer.Dto(vocab, dModel),
 *     new PositionalEncodingLayer.Dto(),
 *     new ModelLayer.Dto(encoder),                // 6 блоков + LN
 *     new TokenPoolingLayer.Dto(TokenPoolingLayer.Mode.MEAN),
 *     new DenseLayer.Dto(TensorShape.flat(1)),
 *     new ActivationLayer.Dto(new Sigmoid())
 * ));
 * }</pre>
 *
 * <pre>{@code
 * // Или внутри существующего Functional (без промежуточного Model):
 * Functional.Builder b = Functional.builder(TensorShape.seq(T, dModel));
 * Functional.Node x = b.input();
 * Functional.Node y = TransformerEncoderStack.build(b, x, dto, 6);
 * Model encoder = b.output(y);
 * }</pre>
 *
 * <h2>Сериализация</h2>
 * {@code Stack} — это {@link Functional}, внутри которого N
 * {@code ModelLayer}-обёрток (каждая — свой {@code TransformerEncoderBlock})
 * и один {@code LayerNormLayer}. Сохраняется/загружается как обычный
 * {@code Functional} — без специального кода.
 */
public final class TransformerEncoderStack {

    private TransformerEncoderStack() {}

    // =====================================================================
    // ==================== build → Model ==================================
    // =====================================================================

    /**
     * Собирает стопку из {@code nLayers} блоков + финальный {@code LayerNorm}
     * как <b>готовую модель</b>.
     *
     * <p>Удобно для встраивания в {@link Sequence}
     * через {@link ModelLayer} или для использования как самостоятельной
     * модели.
     *
     * @param seqLen  длина последовательности {@code T}
     * @param dto     параметры каждого блока (одинаковые для всех)
     * @param nLayers число блоков; {@code >= 1}
     * @return готовая модель: {@code seq(T, dModel) → seq(T, dModel)}
     * @throws IllegalArgumentException если {@code nLayers < 1} или
     *                                  {@code dto == null}
     */
    public static Model build(int seqLen, TransformerEncoderBlock.Dto dto, int nLayers) {
        if (dto == null)
            throw new IllegalArgumentException("TransformerEncoderStack: dto == null");
        if (nLayers < 1)
            throw new IllegalArgumentException("TransformerEncoderStack: nLayers >= 1, получено " + nLayers);

        Functional.Builder b = Functional.builder(TensorShape.seq(seqLen, dto.dModel));
        Functional.Node x = b.input();
        Functional.Node y = build(b, x, dto, nLayers);
        return b.output(y);
    }

    // =====================================================================
    // ==================== build → Node ===================================
    // =====================================================================

    /**
     * Добавляет стопку из {@code nLayers} блоков + финальный {@code LayerNorm}
     * в <b>существующий</b> {@link Functional.Builder}.
     *
     * <p>Полезно, когда вся сеть собирается в одном {@code Functional}
     * (без промежуточных {@code Model}): например, ViT-стиль
     * {@code Conv→Reshape→Stack→Pooling→Dense}. Тогда не нужно создавать
     * отдельный {@code Model} для стопки и оборачивать его в {@code ModelLayer}.
     *
     * <p><b>Вход и выход:</b> {@code x} должен быть {@code seq(T, dModel)};
     * возвращаемый узел — тоже {@code seq(T, dModel)}. Форма не меняется,
     * поэтому стопку можно ставить «стопкой» с другими seq-слоями.
     *
     * @param b       builder (он же граф)
     * @param x       входной узел {@code seq(T, dModel)}
     * @param dto     параметры каждого блока
     * @param nLayers число блоков; {@code >= 1}
     * @return выходной узел {@code seq(T, dModel)}
     * @throws IllegalArgumentException если {@code x} не {@code seq},
     *                                  {@code x.channels != dModel},
     *                                  {@code nLayers < 1} или {@code dto == null}
     */
    public static Functional.Node build(Functional.Builder b,
                                        Functional.Node x,
                                        TransformerEncoderBlock.Dto dto,
                                        int nLayers) {
        if (b == null)
            throw new IllegalArgumentException("TransformerEncoderStack: b == null");
        if (dto == null)
            throw new IllegalArgumentException("TransformerEncoderStack: dto == null");
        if (nLayers < 1)
            throw new IllegalArgumentException("TransformerEncoderStack: nLayers >= 1, получено " + nLayers);
        if (!x.shape.isSequence())
            throw new IllegalArgumentException(
                    "TransformerEncoderStack: x должен быть seq(T, dModel), получено " + x.shape);
        if (x.shape.channels() != dto.dModel)
            throw new IllegalArgumentException(
                    "TransformerEncoderStack: x.channels=" + x.shape.channels()
                            + " != dModel=" + dto.dModel);

        // N блоков подряд. Каждый build(b, x, dto) — свой набор параметров;
        // веса между блоками не разделяются (стандарт BERT/GPT).
        Functional.Node cur = x;
        for (int i = 0; i < nLayers; i++) {
            cur = TransformerEncoderBlock.build(b, cur, dto);
        }

        // Финальный LayerNorm: в pre-LN выход последнего блока не нормирован.
        // Без него следующий слой (голова) получает сигнал с большим разбросом.
        return b.layer(new LayerNormLayer.Dto(), cur);
    }
}