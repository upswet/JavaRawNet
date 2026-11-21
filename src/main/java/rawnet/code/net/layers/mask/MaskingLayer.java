package rawnet.code.net.layers.mask;

import rawnet.code.net.TensorShape;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.contract.MaskSource;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Слой-источник маски последовательности.
 *
 * <p>Стоит <b>до</b> {@code EmbeddingLayer} (на индексах токенов) и
 * строит маску {@code float[T]}, где {@code 1} — настоящий токен,
 * {@code 0} — PAD. Данные не меняются: {@code forward} возвращает
 * копию входа.
 *
 * <pre>
 *   flat(T)   — индексы токенов, PAD = padIndex
 *   → Masking  — flat(T) + lastMask()
 *   → Embedding → seq(T, dModel)   // маска идёт мимо (нейтральный слой)
 *   → Attention / RNN / Pooling    // MaskConsumer читает маску
 * </pre>
 *
 * <h3>Почему до Embedding</h3>
 * После embedding'а {@code input[t]} — вектор {@code E[idx]}, и сравнение
 * с {@code padIndex} невозможно. До embedding'а {@code input[t]} — сам
 * индекс, сравнение точное.
 *
 * <h3>Backward</h3>
 * Градиент проходит как есть: {@code prevDelta = deltaNext.clone()}.
 * Индексы дискретны, градиент по ним не имеет смысла; {@code Embedding}
 * в backward заполнит {@code prevDelta} нулями.
 *
 * <h3>Сериализация</h3>
 * {@link #save(DataOutputStream)} пишет формы (через {@code super.save})
 * и {@code padIndex}. {@link #load(DataInputStream)} читает формы,
 * перезаписывает {@code padIndex} из потока и пересоздаёт буферы через
 * {@link #allocateBuffers()}.
 *
 * <p>Слой имеет публичный конструктор без аргументов, чтобы
 * {@code Sequence.load} / {@code Functional.load} могли создать его
 * через рефлексию. {@code padIndex} по умолчанию {@code 0} — если
 * в модели PAD не {@code 0}, он будет восстановлен из потока.
 */
public class MaskingLayer extends Layer implements MaskSource {

    /** DTO слоя маски. */
    public static class Dto extends Layer.Dto {
        /** Индекс PAD-токена. */
        public final int padIndex;

        public Dto(int padIndex) { this.padIndex = padIndex; }

        @Override
        public Layer create(TensorShape inputShape) {
            if (!inputShape.isFlat())
                throw new IllegalArgumentException(
                        "MaskingLayer.Dto: ожидался flat-вход, получено " + inputShape);
            return new MaskingLayer()
                    .setPadIndex(padIndex)
                    .setInputShape(inputShape)
                    .setOutputShape(inputShape)
                    .allocateBuffers();
        }
    }

    private int padIndex;
    private int size;

    private float[] out;
    private float[] prevDelta;
    private float[] lastMask;

    /** Пустой конструктор для рефлексии ({@code Sequence.load}, {@code Functional.load}). */
    public MaskingLayer() { this(0); }

    public MaskingLayer(int padIndex) { this.padIndex = padIndex; }

    /** @return {@code this} для цепочек */
    public MaskingLayer setPadIndex(int padIndex) {
        this.padIndex = padIndex;
        return this;
    }

    /** @return индекс PAD-токена */
    public int padIndex() { return padIndex; }

    /**
     * Валидирует размеры, выделяет буферы.
     *
     * @return {@code this}
     * @throws IllegalStateException    если формы не заданы
     * @throws IllegalArgumentException если размеры входа и выхода не совпадают
     *                                  или вход не {@code flat}
     */
    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("MaskingLayer.create: inputShape и outputShape должны быть заданы");
        if (!inputShape.isFlat())
            throw new IllegalArgumentException("MaskingLayer: ожидался flat-вход, получено " + inputShape);
        if (inputShape.size() != outputShape.size())
            throw new IllegalArgumentException(
                    "MaskingLayer: размеры входа и выхода должны совпадать, получено "
                            + inputShape.size() + " и " + outputShape.size());

        this.size = inputShape.size();
        this.out = new float[size];
        this.prevDelta = new float[size];
        this.lastMask = new float[size];
        return this;
    }

    /**
     * Строит маску по входу и возвращает копию входа.
     *
     * @param input    индексы токенов длины {@code T}
     * @param training не используется
     * @return копия {@code input} (данные не меняются)
     */
    @Override
    public float[] forward(float[] input, boolean training) {
        if (input.length != size)
            throw new IllegalArgumentException("MaskingLayer.forward: ожидалось " + size + ", получено " + input.length);

        if (size > 0) {
            float v0 = input[0];
            if (v0 != Math.round(v0) || v0 < 0f)
                throw new IllegalStateException("MaskingLayer.forward: вход[0]=" + v0 + " не является индексом токена (целое >= 0). " + "MaskingLayer должен стоять ДО IndexConsumer-слоя (Embedding и т.п.).");
        }

        for (int t = 0; t < size; t++)
            lastMask[t] = (input[t] == padIndex) ? 0f : 1f;

        System.arraycopy(input, 0, out, 0, size);
        return out;
    }

    @Override
    public float[] backward(float[] deltaNext) {
        if (deltaNext.length != size)
            throw new IllegalArgumentException(
                    "MaskingLayer.backward: ожидалось " + size + ", получено " + deltaNext.length);
        System.arraycopy(deltaNext, 0, prevDelta, 0, size);
        return prevDelta;
    }

    /** @return маска последнего {@link #forward}: {@code 1} — настоящий токен, {@code 0} — PAD. */
    @Override
    public float[] lastMask() { return lastMask; }

    /**
     * Сохраняет: имя класса, формы (через {@code super.save}) и {@code padIndex}.
     */
    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);
        out.writeInt(padIndex);
    }

    /**
     * Загружает: формы (через {@code super.load}), {@code padIndex},
     * затем пересоздаёт буферы через {@link #allocateBuffers()}.
     */
    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);
        this.padIndex = in.readInt();
        allocateBuffers();
    }
}