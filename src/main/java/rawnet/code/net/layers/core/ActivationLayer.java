package rawnet.code.net.layers.core;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;
import rawnet.code.net.core.activate.*;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.TensorShape;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Слой только с активацией: {@code y = f(x)}. Без параметров.
 *
 * <p>Форма входа и выхода совпадают (поэлементная операция).
 */
@Accessors(chain = true)
@NoArgsConstructor
public class ActivationLayer extends Layer {

    /** DTO слоя активации. */
    @RequiredArgsConstructor
    public static class Dto extends Layer.Dto {
        /** Функция активации; не {@code null}. */
        public final Activate activate;

        @Override
        public Layer create(TensorShape inputShape) {
            return new ActivationLayer()
                    .setActivate(activate)
                    .setInputShape(inputShape)
                    .setOutputShape(inputShape)
                    .allocateBuffers();
        }
    }

    @Setter @Getter
    private Activate activate;

    private int size;
    private float[] out, prevDelta, deriv, lastInput;

    /**
     * Валидирует размеры, сохраняет {@code size} и выделяет буферы.
     *
     * @return {@code this}
     * @throws IllegalStateException    если формы не заданы
     * @throws IllegalArgumentException если размеры входа и выхода не совпадают
     *                                  или {@code activate == null}
     */
    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("ActivationLayer.create: inputShape и outputShape должны быть заданы");
        if (inputShape.size() != outputShape.size())
            throw new IllegalArgumentException(
                    "ActivationLayer: размеры входа и выхода должны совпадать, получено "
                            + inputShape.size() + " и " + outputShape.size());
        if (activate == null)
            throw new IllegalArgumentException("ActivationLayer: activate не может быть null");

        this.size = inputShape.size();

        this.out       = new float[size];
        this.prevDelta = new float[size];
        this.deriv     = new float[size];
        this.lastInput = new float[size];

        return this;
    }

    /**
     * Прямой проход: {@code y = f(x)}.
     *
     * <p>Копирует {@code input} в {@code lastInput}, так как
     * {@link Activate#apply} может перезаписывать входной буфер, а для
     * {@link #backward} нужен именно вход.
     *
     * @param input    входной вектор длины {@code size}
     * @param training не используется (у слоя нет параметров)
     * @return выходной вектор длины {@code size}; внутренний буфер, перезаписывается
     *         при следующем вызове
     * @throws IllegalArgumentException если длина {@code input} не равна {@code size}
     */
    @Override
    public float[] forward(float[] input, boolean training) {
        if (input.length != size)
            throw new IllegalArgumentException("ActivationLayer.forward: ожидалось " + size + ", получено " + input.length);

        System.arraycopy(input, 0, lastInput, 0, size);
        activate.apply(input, out);
        return out;
    }

    /**
     * Обратный проход: {@code deltaPrev = deltaNext ⊙ f'(lastInput)}.
     *
     * <p>Требует предшествующего {@link #forward}.
     *
     * @param deltaNext градиент по выходу слоя длины {@code size}
     * @return градиент по входу слоя длины {@code size}
     * @throws IllegalStateException    если {@code forward} не вызывался
     * @throws IllegalArgumentException если длина {@code deltaNext} не равна {@code size}
     */
    @Override
    public float[] backward(float[] deltaNext) {
        if (lastInput == null)
            throw new IllegalStateException("ActivationLayer.backward: forward не был вызван");
        if (deltaNext.length != size)
            throw new IllegalArgumentException("ActivationLayer.backward: ожидалось " + size + ", получено " + deltaNext.length);

        // deriv = f'(lastInput); второй аргумент — уже вычисленный выход (может
        // использоваться реализациями, например сигмоидой).
        activate.derivative(lastInput, out, deriv);
        for (int i = 0; i < size; i++) {
            prevDelta[i] = deltaNext[i] * deriv[i];
        }
        return prevDelta;
    }

    /**
     * Сохраняет тип активации (и её параметры, например {@code alpha} у
     * {@link LeakyRelu}) после общих полей слоя.
     */
    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);
        writeActivate(out, activate);
    }

    /**
     * Загружает тип активации и пересоздаёт буферы через {@link #allocateBuffers()}.
     */
    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);
        activate = readActivate(in);
        allocateBuffers();
    }

    /** Код активации {@link LeakyRelu} в потоке. */
    private static final byte CODE_LEAKY_RELU = 1;
    /** Код активации {@link Linear} в потоке. */
    private static final byte CODE_LINEAR     = 2;
    /** Код активации {@link Sigmoid} в потоке. */
    private static final byte CODE_SIGMOID    = 3;
    /** Код активации {@link Tanh} в потоке. */
    private static final byte CODE_TANH       = 4;
    /** Код активации {@link Gelu} в потоке. */
    private static final byte CODE_GELU  = 5;
    /** Код активации {@link Silu} в потоке. */
    private static final byte CODE_SILU  = 6;
    /** Код активации {@link Mish} в потоке. */
    private static final byte CODE_MISH  = 7;

    /**
     * Записывает активацию в поток: байт-код типа и, при необходимости,
     * параметры.
     *
     * @param out поток для записи
     * @param a   активация; не {@code null}
     * @throws IOException если тип активации неизвестен или произошла ошибка записи
     */
    public static void writeActivate(DataOutputStream out, Activate a) throws IOException {
        if (a instanceof LeakyRelu lr)      { out.writeByte(CODE_LEAKY_RELU); out.writeFloat(lr.getAlpha()); }
        else if (a instanceof Linear)       { out.writeByte(CODE_LINEAR); }
        else if (a instanceof Sigmoid)      { out.writeByte(CODE_SIGMOID); }
        else if (a instanceof Tanh)         { out.writeByte(CODE_TANH); }
        else if (a instanceof Gelu g)       { out.writeByte(CODE_GELU); out.writeBoolean(g.isApproximate()); }
        else if (a instanceof Silu)         { out.writeByte(CODE_SILU); }
        else if (a instanceof Mish)         { out.writeByte(CODE_MISH); }
        else throw new IOException("Неизвестная активация: " + a.getClass().getName());
    }

    /**
     * Читает активацию из потока (обратно {@link #writeActivate}).
     *
     * @param in поток для чтения
     * @return восстановленная активация
     * @throws IOException если код некорректен или произошла ошибка чтения
     */
    public static Activate readActivate(DataInputStream in) throws IOException {
        return switch (in.readByte()) {
            case CODE_LEAKY_RELU -> new LeakyRelu(in.readFloat());
            case CODE_LINEAR     -> new Linear();
            case CODE_SIGMOID    -> new Sigmoid();
            case CODE_TANH       -> new Tanh();
            case CODE_GELU       -> new Gelu(in.readBoolean());
            case CODE_SILU       -> new Silu();
            case CODE_MISH       -> new Mish();
            default -> throw new IOException("Некорректный код активации");
        };
    }
}