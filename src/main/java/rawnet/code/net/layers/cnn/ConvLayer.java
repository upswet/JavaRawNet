package rawnet.code.net.layers.cnn;

import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;
import rawnet.code.net.layers.core.ActivationLayer;
import rawnet.code.net.TensorShape;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.core.activate.Activate;
import rawnet.code.net.core.optimizator.Optimizer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

/**
 * Свёрточный слой (2D) <b>без активации</b>: {@code z = W * x + b}.
 *
 * <p>Выход — сырые логиты {@code z}; активацию применяет следующий слой
 * (обычно {@link ActivationLayer},
 * который уже совместим с CNN-формами, поскольку работает поэлементно).
 *
 * <h3>Форма</h3>
 * Вход и выход — {@link TensorShape} с {@code isCnn() == true}
 * (плоские формы и sequence не поддерживаются). Форма выхода:
 * <pre>
 *   outH = (inH + 2·padding - kernelSize) / stride + 1
 *   outW = (inW + 2·padding - kernelSize) / stride + 1
 *   outShape = cnn(outChannels, outH, outW)
 * </pre>
 *
 * <h3>Параметры</h3>
 * <ul>
 *   <li>{@code weights[oc][ic][kh][kw]} — ядро свёртки;</li>
 *   <li>{@code biases[oc]} — смещение выходного канала;</li>
 *   <li>инициализация — Xavier/Glorot (через {@link Layer#initLimit}).
 *       Если передать {@link Dto#initActivate}, инициализация подстроится
 *       под неё (He для LeakyRelu и т.п.), но <b>применяться</b> активация
 *       не будет — это только подсказка для {@code initLimit}.</li>
 * </ul>
 *
 * <h3>Параллелизация</h3>
 * При числе MAC-ов выше {@link #PARALLEL_MIN_MACS} forward и backward
 * распараллеливаются по каналам. Потоки пишут в непересекающиеся участки
 * {@code z}/{@code prevDelta}/{@code weightGrad}/{@code biasGrad} —
 * race-free без синхронизации.
 *
 * <h3>Сериализация</h3>
 * {@link #save(DataOutputStream)} пишет имя класса и формы (через
 * {@code super.save}), затем веса и смещения. Архитектурные параметры
 * (kernelSize, stride, padding, outChannels) должны быть заданы через DTO
 * или {@code setXxx} до {@link #load(DataInputStream)}; {@code outChannels}
 * дополнительно восстанавливается из {@code outputShape.channels()}.
 */
@Accessors(chain = true)
public class ConvLayer extends Layer {

    /** Порог MAC-ов, после которого forward/backward распараллеливается по каналам. */
    protected static final long PARALLEL_MIN_MACS = 200_000L;

    /**
     * DTO свёрточного слоя без активации.
     *
     * <p>{@code initActivate} — <b>только подсказка для инициализации</b>
     * весов (Xavier vs He). В forward не применяется.
     */
    @RequiredArgsConstructor
    public static class Dto extends Layer.Dto {
        /** Число выходных каналов; {@code >= 1}. */
        public final int outChannels;
        /** Размер квадратного ядра свёртки; {@code >= 1}. */
        public final int kernelSize;
        /** Шаг свёртки; {@code >= 1}. */
        public final int stride;
        /** Дополнение нулями; {@code >= 0}. */
        public final int padding;
        /**Будущая функция активации - для инициализации весов.. Проставиться при сборке сети из следующего ActivationLayer если он есть*/
        public Activate pendingActivate;

        @Override
        public Layer create(TensorShape inputShape) {
            return new ConvLayer()
                    .setOutChannels(outChannels)
                    .setKernelSize(kernelSize)
                    .setStride(stride)
                    .setPadding(padding)
                    .setInputShape(inputShape)
                    .setOutputShape(computeOutputShape(inputShape, outChannels, kernelSize, stride, padding))
                    .allocateBuffers().initWeights(pendingActivate);
        }
    }

    // --- Архитектурные параметры (задаются через setXxx до create()) ---
    @Setter private int outChannels;
    @Setter private int kernelSize;
    @Setter private int stride;
    @Setter private int padding;
    /**
     * Подсказка для инициализации весов. В forward не применяется.
     * Хранится отдельно от Dto, чтобы после {@code load} инициализация
     * не зависела от того, передан ли DTO.
     */
    private Activate initActivate;

    // --- Размеры (вычисляются в create()) ---
    private int inChannels, inH, inW;
    private int kH, kW;
    private int outH, outW, outHW, inHW;
    private long forwardMacs;

    // --- Обучаемые параметры ---
    /** {@code [outChannels][inChannels][kH][kW]} */
    float[][][][] weights;
    /** {@code [outChannels]} */
    float[] biases;
    /** Накопители градиентов; те же формы. */
    private float[][][][] weightGrad;
    private float[] biasGrad;

    // --- Рабочие буферы ---
    private transient float[] input;
    /** {@code z = W*x + b} последнего forward. */
    private float[] z;
    /** Что возвращает forward — копия {@code z}. */
    private float[] output;
    private float[] prevDelta;

    // --- Плоские проекции для Optimizer.update ---
    private float[][] flatWeights;
    private float[][] flatGrad;

    public ConvLayer() {}

    /**
     * Вычисляет форму выхода по параметрам свёртки.
     *
     * @throws IllegalArgumentException если вход не CNN, размеры некорректны
     *                                  или выход получается {@code <= 0}
     */
    public static TensorShape computeOutputShape(TensorShape in, int outChannels,
                                                 int kernelSize, int stride, int padding) {
        if (in == null) throw new IllegalArgumentException("ConvLayer: inputShape == null");
        if (!in.isCnn())
            throw new IllegalArgumentException("ConvLayer: ожидался CNN-вход, получено " + in);
        if (outChannels < 1) throw new IllegalArgumentException("ConvLayer: outChannels >= 1, получено " + outChannels);
        if (kernelSize < 1)  throw new IllegalArgumentException("ConvLayer: kernelSize >= 1, получено " + kernelSize);
        if (stride < 1)      throw new IllegalArgumentException("ConvLayer: stride >= 1, получено " + stride);
        if (padding < 0)     throw new IllegalArgumentException("ConvLayer: padding >= 0, получено " + padding);

        int outH = (in.height() + 2 * padding - kernelSize) / stride + 1;
        int outW = (in.width()  + 2 * padding - kernelSize) / stride + 1;
        if (outH <= 0 || outW <= 0)
            throw new IllegalArgumentException(
                    "ConvLayer: размер выхода <= 0 (outH=" + outH + ", outW=" + outW + ")");
        return TensorShape.cnn(outChannels, outH, outW);
    }

    /**
     * Валидирует формы, выделяет буферы
     *
     * <p>Идемпотентен: повторный вызов пересоздаёт всё заново (нужно после
     * {@link #load(DataInputStream)}, когда формы читаются из потока).
     *
     * @return {@code this}
     * @throws IllegalStateException    если формы не заданы
     * @throws IllegalArgumentException если вход не CNN или выход {@code <= 0}
     */
    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("ConvLayer.create: inputShape и outputShape должны быть заданы");
        if (!inputShape.isCnn())
            throw new IllegalArgumentException("ConvLayer.create: ожидался CNN-вход, получено " + inputShape);
        if (!outputShape.isCnn())
            throw new IllegalArgumentException("ConvLayer.create: ожидался CNN-выход, получено " + outputShape);
        if (outputShape.channels() != outChannels)
            throw new IllegalArgumentException(
                    "ConvLayer.create: outputShape.channels=" + outputShape.channels()
                            + " != outChannels=" + outChannels);

        this.inChannels = inputShape.channels();
        this.inH        = inputShape.height();
        this.inW        = inputShape.width();
        this.kH         = kernelSize;
        this.kW         = kernelSize;
        this.outH       = outputShape.height();
        this.outW       = outputShape.width();
        this.outHW      = outH * outW;
        this.inHW       = inH  * inW;
        this.forwardMacs = (long) outChannels * inChannels * kH * kW * outH * outW;

        this.weights    = new float[outChannels][inChannels][kH][kW];
        this.biases     = new float[outChannels];
        this.weightGrad = new float[outChannels][inChannels][kH][kW];
        this.biasGrad   = new float[outChannels];

        int outSize = outChannels * outHW;
        int inSize  = inChannels  * inHW;
        this.z         = new float[outSize];
        this.output    = new float[outSize];
        this.prevDelta = new float[inSize];

        int flatSize = inChannels * kH * kW;
        this.flatWeights = new float[outChannels][flatSize];
        this.flatGrad    = new float[outChannels][flatSize];

        return this;
    }

    @Override
    public Layer initWeights(Activate activate) {
        int fanIn  = inChannels  * kH * kW;
        int fanOut = outChannels * kH * kW;
        float limit = Layer.initLimit(activate, fanIn, fanOut);
        Random rnd = ThreadLocalRandom.current();
        for (int oc = 0; oc < outChannels; oc++) {
            for (int ic = 0; ic < inChannels; ic++)
                for (int i = 0; i < kH; i++)
                    for (int j = 0; j < kW; j++)
                        weights[oc][ic][i][j] = rnd.nextFloat() * 2 * limit - limit;
            biases[oc] = 0f;
        }

        return this;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Возвращает копию {@code z} во внутреннем буфере {@code output}.
     * Активация не применяется — это ответственность следующего слоя.
     *
     * @throws IllegalArgumentException если длина {@code input} не равна {@code inputShape.size()}
     */
    @Override
    public float[] forward(float[] input, boolean training) {
        if (input.length != inputShape.size())
            throw new IllegalArgumentException(
                    "ConvLayer.forward: ожидалось " + inputShape.size() + ", получено " + input.length);
        this.input = input;

        for (int oc = 0; oc < outChannels; oc++)
            Arrays.fill(z, oc * outHW, (oc + 1) * outHW, biases[oc]);

        if (forwardMacs >= PARALLEL_MIN_MACS)
            IntStream.range(0, outChannels).parallel().forEach(oc -> accumulateChannel(oc, input));
        else
            for (int oc = 0; oc < outChannels; oc++) accumulateChannel(oc, input);

        System.arraycopy(z, 0, output, 0, z.length);
        return output;
    }

    private void accumulateChannel(int oc, float[] in) {
        final int ocOutOff = oc * outHW;
        for (int ic = 0; ic < inChannels; ic++) {
            final int icInOff = ic * inHW;
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    final float w = weights[oc][ic][kh][kw];
                    if (w == 0f) continue;
                    if (stride == 1) {
                        final int owStart = Math.max(0, padding - kw);
                        final int owEnd   = Math.min(outW, inW + padding - kw);
                        if (owStart >= owEnd) continue;
                        final int colShift = kw - padding;
                        for (int oh = 0; oh < outH; oh++) {
                            final int ih = oh + kh - padding;
                            if (ih < 0 || ih >= inH) continue;
                            final int baseIn = icInOff + ih * inW + colShift;
                            final int rowOut = ocOutOff + oh * outW;
                            for (int ow = owStart; ow < owEnd; ow++)
                                z[rowOut + ow] += w * in[baseIn + ow];
                        }
                    } else {
                        for (int oh = 0; oh < outH; oh++) {
                            final int ih = oh * stride + kh - padding;
                            if (ih < 0 || ih >= inH) continue;
                            final int rowIn  = icInOff + ih * inW;
                            final int rowOut = ocOutOff + oh * outW;
                            for (int ow = 0; ow < outW; ow++) {
                                final int iw = ow * stride + kw - padding;
                                if (iw < 0 || iw >= inW) continue;
                                z[rowOut + ow] += w * in[rowIn + iw];
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * {@inheritDoc}
     *
     * <p>Активация не применяется: {@code deltaNext} — это {@code dL/dz}.
     * Накопливает {@code weightGrad}/{@code biasGrad}, возвращает
     * градиент по входу.
     *
     * @throws IllegalStateException    если {@link #forward} не вызывался
     * @throws IllegalArgumentException если длина {@code deltaNext} не равна {@code outputShape.size()}
     */
    @Override
    public float[] backward(float[] deltaNext) {
        if (deltaNext.length != outputShape.size())
            throw new IllegalArgumentException(
                    "ConvLayer.backward: ожидалось " + outputShape.size() + ", получено " + deltaNext.length);
        if (input == null)
            throw new IllegalStateException("ConvLayer.backward: forward не был вызван");

        Arrays.fill(prevDelta, 0f);

        if (forwardMacs >= PARALLEL_MIN_MACS) {
            IntStream.range(0, outChannels).parallel().forEach(oc -> accumulateBiasGrad(oc, deltaNext));
        } else {
            for (int oc = 0; oc < outChannels; oc++) accumulateBiasGrad(oc, deltaNext);
        }

        if (forwardMacs >= PARALLEL_MIN_MACS) {
            IntStream.range(0, inChannels).parallel().forEach(ic -> accumulateBackwardChannel(ic, deltaNext));
        } else {
            for (int ic = 0; ic < inChannels; ic++) accumulateBackwardChannel(ic, deltaNext);
        }

        return prevDelta;
    }

    private void accumulateBiasGrad(int oc, float[] dz) {
        final int off = oc * outHW;
        float s = 0f;
        for (int i = 0; i < outHW; i++) s += dz[off + i];
        biasGrad[oc] += s;
    }

    private void accumulateBackwardChannel(int ic, float[] dz) {
        final int icInOff = ic * inHW;
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                final int owStart, owEnd;
                if (stride == 1) {
                    owStart = Math.max(0, padding - kw);
                    owEnd   = Math.min(outW, inW + padding - kw);
                    if (owStart >= owEnd) continue;
                } else { owStart = 0; owEnd = outW; }

                for (int oc = 0; oc < outChannels; oc++) {
                    final float w = weights[oc][ic][kh][kw];
                    final int ocOutOff = oc * outHW;
                    float wGradAcc = 0f;

                    if (stride == 1) {
                        final int colShift = kw - padding;
                        for (int oh = 0; oh < outH; oh++) {
                            final int ih = oh + kh - padding;
                            if (ih < 0 || ih >= inH) continue;
                            final int baseIn = icInOff + ih * inW + colShift;
                            final int rowOut = ocOutOff + oh * outW;
                            for (int ow = owStart; ow < owEnd; ow++) {
                                final float delta = dz[rowOut + ow];
                                prevDelta[baseIn + ow] += w * delta;
                                wGradAcc += delta * input[baseIn + ow];
                            }
                        }
                    } else {
                        for (int oh = 0; oh < outH; oh++) {
                            final int ih = oh * stride + kh - padding;
                            if (ih < 0 || ih >= inH) continue;
                            final int rowIn  = icInOff + ih * inW;
                            final int rowOut = ocOutOff + oh * outW;
                            for (int ow = 0; ow < outW; ow++) {
                                final int iw = ow * stride + kw - padding;
                                if (iw < 0 || iw >= inW) continue;
                                final float delta = dz[rowOut + ow];
                                prevDelta[rowIn + iw] += w * delta;
                                wGradAcc += delta * input[rowIn + iw];
                            }
                        }
                    }
                    weightGrad[oc][ic][kh][kw] += wGradAcc;
                }
            }
        }
    }

    @Override
    public void zeroGrad() {
        for (float[][][] byOut : weightGrad)
            for (float[][] byIn : byOut)
                for (float[] row : byIn)
                    Arrays.fill(row, 0f);
        Arrays.fill(biasGrad, 0f);
    }

    @Override
    public void update(Optimizer optimizer, int batchSize) {
        if (batchSize <= 0)
            throw new IllegalArgumentException("batchSize должен быть > 0, получено " + batchSize);
        for (int oc = 0; oc < outChannels; oc++) {
            int idx = 0;
            for (int ic = 0; ic < inChannels; ic++)
                for (int kh = 0; kh < kH; kh++)
                    for (int kw = 0; kw < kW; kw++) {
                        flatWeights[oc][idx] = weights[oc][ic][kh][kw];
                        flatGrad[oc][idx]    = weightGrad[oc][ic][kh][kw];
                        idx++;
                    }
        }
        optimizer.update(flatWeights, flatGrad, biases, biasGrad, batchSize);
        for (int oc = 0; oc < outChannels; oc++) {
            int idx = 0;
            for (int ic = 0; ic < inChannels; ic++)
                for (int kh = 0; kh < kH; kh++)
                    for (int kw = 0; kw < kW; kw++)
                        weights[oc][ic][kh][kw] = flatWeights[oc][idx++];
        }
    }

    @Override
    public double gradNormSq() {
        double sum = 0;
        for (float[][][] oc : weightGrad)
            for (float[][] ic : oc)
                for (float[] row : ic)
                    for (float v : row) sum += v * v;
        for (float v : biasGrad) sum += v * v;
        return sum;
    }

    @Override
    public void scaleGrad(float scale) {
        for (float[][][] oc : weightGrad)
            for (float[][] ic : oc)
                for (float[] row : ic)
                    for (int j = 0; j < row.length; j++) row[j] *= scale;
        for (int i = 0; i < biasGrad.length; i++) biasGrad[i] *= scale;
    }

    /**
     * Сохраняет архитектурные параметры и формы (через {@code super.save}),
     * затем веса и смещения.
     *
     * <p>{@code initActivate} <b>не сохраняется</b>: это подсказка для
     * инициализации, а после обучения веса уже не переинициализируются.
     */
    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);                   // className + inputShape + outputShape

        out.writeInt(kernelSize);
        out.writeInt(stride);
        out.writeInt(padding);
        out.writeInt(outChannels);

        // веса
        for (float[][][] ocW : weights)
            for (float[][] icW : ocW)
                for (float[] row : icW)
                    for (float v : row) out.writeFloat(v);
        for (float b : biases) out.writeFloat(b);
    }

    /**
     * Загружает формы (через {@code super.load}), пересоздаёт буферы и
     * читает веса и смещения.
     *
     * <p>Архитектурные параметры {@code kernelSize/stride/padding} должны
     * быть заданы до вызова (через DTO или {@code setXxx});
     * {@code outChannels} восстанавливается из {@code outputShape.channels()}.
     */
    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);                    // className + inputShape + outputShape

        // Архитектурные параметры — из потока
        this.kernelSize  = in.readInt();
        this.stride      = in.readInt();
        this.padding     = in.readInt();
        this.outChannels = in.readInt();

        allocateBuffers();
        // веса
        for (float[][][] ocW : weights)
            for (float[][] icW : ocW)
                for (float[] row : icW)
                    for (int k = 0; k < row.length; k++) row[k] = in.readFloat();
        for (int i = 0; i < biases.length; i++) biases[i] = in.readFloat();
    }
}