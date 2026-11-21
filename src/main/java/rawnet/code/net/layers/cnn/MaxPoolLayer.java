package rawnet.code.net.layers.cnn;

import lombok.RequiredArgsConstructor;
import rawnet.code.net.TensorShape;
import rawnet.code.net.layers.Layer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;

/**
 * Слой максимального пулинга (2D). Без параметров.
 *
 * <p>Вход/выход — {@link TensorShape} с {@code isCnn() == true}. Форма
 * выхода:
 * <pre>
 *   outH = (inH - poolSize) / stride + 1
 *   outW = (inW - poolSize) / stride + 1
 *   outShape = cnn(channels, outH, outW)
 * </pre>
 *
 * <h3>Backward</h3>
 * Градиент идёт только в позиции максимума (сохранённые в {@code argmax}
 * последнего forward). Если окно неполное (последний столбец/строка при
 * {@code stride > 1}), {@code argmax == -1} — такой выход не участвует в
 * backward (это не ошибка, а следствие того, что окно не покрывает ни
 * одного элемента).
 *
 * <p>Параметров нет: {@code zeroGrad}/{@code update}/{@code gradNormSq}/
 * {@code scaleGrad} — no-op/{@code 0}.
 */
public class MaxPoolLayer extends Layer {

    /** DTO max-pool слоя. */
    @RequiredArgsConstructor
    public static class Dto extends Layer.Dto {
        public final int poolSize;
        public final int stride;

        @Override
        public Layer create(TensorShape inputShape) {
            return new MaxPoolLayer()
                    .setPoolSize(poolSize)
                    .setStride(stride)
                    .setInputShape(inputShape)
                    .allocateBuffers();
        }
    }

    private int poolSize;
    private int stride;

    private int channels, inH, inW;
    private int outH, outW;

    private int[] argmax;
    private float[] output;
    private float[] prevDelta;

    public MaxPoolLayer() {}

    public MaxPoolLayer setPoolSize(int poolSize) { this.poolSize = poolSize; return this; }
    public MaxPoolLayer setStride(int stride)     { this.stride = stride; return this; }

    /**
     * Вычисляет форму выхода по параметрам пулинга.
     *
     * @throws IllegalArgumentException если вход не CNN, размеры некорректны
     *                                  или выход {@code <= 0}
     */
    public static TensorShape computeOutputShape(TensorShape in, int poolSize, int stride) {
        if (in == null) throw new IllegalArgumentException("MaxPoolLayer: inputShape == null");
        if (!in.isCnn())
            throw new IllegalArgumentException("MaxPoolLayer: ожидался CNN-вход, получено " + in);
        if (poolSize < 1) throw new IllegalArgumentException("MaxPoolLayer: poolSize >= 1, получено " + poolSize);
        if (stride < 1)   throw new IllegalArgumentException("MaxPoolLayer: stride >= 1, получено " + stride);

        int outH = (in.height() - poolSize) / stride + 1;
        int outW = (in.width()  - poolSize) / stride + 1;
        if (outH <= 0 || outW <= 0)
            throw new IllegalArgumentException(
                    "MaxPoolLayer: размер выхода <= 0 (outH=" + outH + ", outW=" + outW + ")");
        return TensorShape.cnn(in.channels(), outH, outW);
    }

    /**
     * Валидирует формы и выделяет буферы.
     *
     * @return {@code this}
     * @throws IllegalStateException    если формы не заданы
     * @throws IllegalArgumentException если вход не CNN
     */
    public Layer allocateBuffers() {
        if (inputShape == null)
            throw new IllegalStateException("MaxPoolLayer.allocateBuffers: inputShape должен быть заданы");

        outputShape = computeOutputShape(inputShape, poolSize, stride);

        if (!inputShape.isCnn())
            throw new IllegalArgumentException("MaxPoolLayer.allocateBuffers: ожидался CNN-вход, получено " + inputShape);
        if (!outputShape.isCnn())
            throw new IllegalArgumentException("MaxPoolLayer.allocateBuffers: ожидался CNN-выход, получено " + outputShape);

        this.channels = inputShape.channels();
        this.inH      = inputShape.height();
        this.inW      = inputShape.width();
        this.outH     = outputShape.height();
        this.outW     = outputShape.width();

        int outSize = channels * outH * outW;
        int inSize  = channels * inH * inW;
        this.argmax    = new int[outSize];
        this.output    = new float[outSize];
        this.prevDelta = new float[inSize];
        Arrays.fill(argmax, -1);
        return this;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Запоминает индекс максимума для backward.
     *
     * @throws IllegalArgumentException если длина {@code input} не равна {@code inputShape.size()}
     */
    @Override
    public float[] forward(float[] input, boolean training) {
        if (input.length != inputShape.size())
            throw new IllegalArgumentException(
                    "MaxPoolLayer.forward: ожидалось " + inputShape.size() + ", получено " + input.length);

        final int outHW = outH * outW;
        final int inHW  = inH  * inW;
        final boolean windowsFit =
                (outH - 1) * stride + poolSize <= inH &&
                        (outW - 1) * stride + poolSize <= inW;

        for (int c = 0; c < channels; c++) {
            final int inCOff  = c * inHW;
            final int outCOff = c * outHW;
            for (int oh = 0; oh < outH; oh++) {
                final int ih0 = oh * stride;
                for (int ow = 0; ow < outW; ow++) {
                    final int iw0 = ow * stride;
                    final int outIdx = outCOff + oh * outW + ow;

                    if (windowsFit) {
                        int bestIdx = inCOff + ih0 * inW + iw0;
                        float best = input[bestIdx];
                        for (int kh = 0; kh < poolSize; kh++) {
                            final int rowBase = inCOff + (ih0 + kh) * inW + iw0;
                            final int kwStart = (kh == 0) ? 1 : 0;
                            for (int kw = kwStart; kw < poolSize; kw++) {
                                final int idx = rowBase + kw;
                                if (input[idx] > best) { best = input[idx]; bestIdx = idx; }
                            }
                        }
                        output[outIdx] = best;
                        argmax[outIdx] = bestIdx;
                    } else {
                        float best = Float.NEGATIVE_INFINITY;
                        int bestIdx = -1;
                        for (int kh = 0; kh < poolSize; kh++) {
                            final int ih = ih0 + kh;
                            if (ih >= inH) continue;
                            for (int kw = 0; kw < poolSize; kw++) {
                                final int iw = iw0 + kw;
                                if (iw >= inW) continue;
                                final int idx = inCOff + ih * inW + iw;
                                if (input[idx] > best) { best = input[idx]; bestIdx = idx; }
                            }
                        }
                        output[outIdx] = best;
                        argmax[outIdx] = bestIdx;
                    }
                }
            }
        }
        return output;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Пропускает {@code deltaNext} в позиции максимумов, остальные
     * элементы {@code prevDelta} — нули.
     *
     * @throws IllegalArgumentException если длина {@code deltaNext} не равна {@code outputShape.size()}
     */
    @Override
    public float[] backward(float[] deltaNext) {
        if (deltaNext.length != outputShape.size())
            throw new IllegalArgumentException(
                    "MaxPoolLayer.backward: ожидалось " + outputShape.size() + ", получено " + deltaNext.length);
        Arrays.fill(prevDelta, 0f);
        for (int i = 0; i < output.length; i++) {
            final int src = argmax[i];
            if (src >= 0) prevDelta[src] += deltaNext[i];
        }
        return prevDelta;
    }

    /**
     * Сериализация: только имя класса и формы через {@code super.save}.
     * Параметров нет, {@code argmax} не сохраняется (пересчитывается в forward).
     */
    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);          // className + inputShape + outputShape
        out.writeInt(poolSize);
        out.writeInt(stride);
    }

    /**
     * Загрузка: читает формы (через {@code super.load}) и пересоздаёт буферы.
     * Архитектурные {@code poolSize}/{@code stride} должны быть заданы в DTO
     * (или через {@link #setPoolSize}/{@link #setStride}) до вызова.
     */
    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);           // className + inputShape + outputShape

        this.poolSize = in.readInt();
        this.stride   = in.readInt();

        allocateBuffers();
    }
}