package rawnet.code.net.layers.core;

import lombok.NoArgsConstructor;
import rawnet.code.net.TensorShape;
import rawnet.code.net.layers.Layer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Dropout: во время обучения с вероятностью {@code p} «выключает» нейрон,
 * а выжившие масштабирует на {@code 1 / (1 - p)}, чтобы матожидание выхода
 * совпало со входом (inverted dropout).
 *
 * <p>Во время инференса слой — тождественный: {@code y = x}.
 *
 * <h3>Параметры</h3>
 * <ul>
 *   <li>{@code p} — вероятность «выключить» элемент, {@code 0 <= p < 1}.
 *       {@code p = 0} — no-op; типичные значения 0.1–0.5; {@code p >= 1}
 *       запрещено (деление на ноль).</li>
 * </ul>
 *
 * <h3>Режимы</h3>
 * <ul>
 *   <li>{@code training = true} — генерируется маска Бернулли({@code 1 - p}),
 *       выход масштабируется; маска сохраняется для {@link #backward};</li>
 *   <li>{@code training = false} — слой возвращает копию входа, маска не
 *       используется.</li>
 * </ul>
 *
 * <h3>Backward</h3>
 * {@code deltaPrev[i] = deltaNext[i] * mask[i] / (1 - p)}. Параметров нет,
 * {@link #zeroGrad()}/{@link #update} — no-op.
 *
 * <p><b>Совместимость с BatchNorm/InstanceNorm.</b> Inverted dropout
 * масштабирует активации на обучении, но не на инференсе — статистики
 * норм, накопленные на обучении, уже учитывают этот масштаб. Менять
 * порядок «norm → dropout» не нужно.
 */
@NoArgsConstructor
public class DropoutLayer extends Layer {

    /** DTO dropout-слоя. */
    public static class Dto extends Layer.Dto {
        /** Вероятность выключить элемент; {@code 0 <= p < 1}. */
        public final float p;

        public Dto(float p) {
            if (p < 0f || p >= 1f)
                throw new IllegalArgumentException("DropoutLayer.Dto: p in [0, 1), получено " + p);
            this.p = p;
        }

        @Override
        public Layer create(TensorShape inputShape) {
            return new DropoutLayer()
                    .setP(p)
                    .setInputShape(inputShape)
                    .setOutputShape(inputShape)
                    .allocateBuffers();
        }
    }

    private float p;
    private int size;

    private float[] out;
    private float[] prevDelta;
    /** Маска последнего forward в режиме обучения; {@code 1/(1-p)} для выживших, 0 для выключенных. */
    private float[] mask;
    /** {@code true}, если последний {@link #forward} был в режиме обучения. */
    private boolean lastTraining;

    public DropoutLayer setP(float p) {
        if (p < 0f || p >= 1f)
            throw new IllegalArgumentException("DropoutLayer: p in [0, 1), получено " + p);
        this.p = p;
        return this;
    }

    public float p() { return p; }

    /**
     * Валидирует размеры и выделяет буферы.
     *
     * @return {@code this}
     * @throws IllegalStateException    если формы не заданы
     * @throws IllegalArgumentException если размеры входа и выхода не совпадают
     */
    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("DropoutLayer.create: inputShape и outputShape должны быть заданы");
        if (inputShape.size() != outputShape.size())
            throw new IllegalArgumentException(
                    "DropoutLayer: размеры входа и выхода должны совпадать, получено "
                            + inputShape.size() + " и " + outputShape.size());
        if (p < 0f || p >= 1f)
            throw new IllegalArgumentException("DropoutLayer.create: p in [0, 1), получено " + p);

        this.size = inputShape.size();
        this.out       = new float[size];
        this.prevDelta = new float[size];
        this.mask      = new float[size];
        return this;
    }

    /**
     * {@inheritDoc}
     *
     * <p>В режиме обучения генерирует маску Бернулли и сохраняет её для
     * {@link #backward}. В режиме инференса возвращает копию входа.
     *
     * @throws IllegalArgumentException если длина {@code input} не равна {@code size}
     */
    @Override
    public float[] forward(float[] input, boolean training) {
        if (input.length != size)
            throw new IllegalArgumentException(
                    "DropoutLayer.forward: ожидалось " + size + ", получено " + input.length);

        this.lastTraining = training;

        if (!training || p == 0f) {
            System.arraycopy(input, 0, out, 0, size);
            return out;
        }

        float keep = 1f - p;
        float scale = 1f / keep;
        Random rnd = ThreadLocalRandom.current();
        for (int i = 0; i < size; i++) {
            if (rnd.nextFloat() < keep) {
                mask[i] = scale;
                out[i]  = input[i] * scale;
            } else {
                mask[i] = 0f;
                out[i]  = 0f;
            }
        }
        return out;
    }

    /**
     * {@inheritDoc}
     *
     * <p>В режиме инференса (или при {@code p == 0}) — просто копия
     * {@code deltaNext}. В режиме обучения умножает на сохранённую маску.
     *
     * @throws IllegalStateException    если {@link #forward} не вызывался
     * @throws IllegalArgumentException если длина {@code deltaNext} не равна {@code size}
     */
    @Override
    public float[] backward(float[] deltaNext) {
        if (prevDelta == null)
            throw new IllegalStateException("DropoutLayer.backward: forward не был вызван");
        if (deltaNext.length != size)
            throw new IllegalArgumentException(
                    "DropoutLayer.backward: ожидалось " + size + ", получено " + deltaNext.length);

        if (!lastTraining || p == 0f) {
            System.arraycopy(deltaNext, 0, prevDelta, 0, size);
        } else {
            for (int i = 0; i < size; i++) prevDelta[i] = deltaNext[i] * mask[i];
        }
        return prevDelta;
    }

    /**
     * Сохраняет {@code p} после общих полей слоя.
     */
    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);
        out.writeFloat(p);
    }

    /**
     * Загружает {@code p} и пересоздаёт буферы через {@link #allocateBuffers()}.
     */
    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);
        p = in.readFloat();
        allocateBuffers();
    }
}