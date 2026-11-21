package rawnet.code.net.layers.core;

import lombok.NoArgsConstructor;
import rawnet.code.net.TensorShape;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.core.optimizator.Optimizer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;

/**
 * Instance Normalization: нормирует каждый канал каждого примера
 * отдельно, затем применяет обучаемые {@code gamma} (масштаб) и
 * {@code beta} (сдвиг) — общие для всех примеров.
 *
 * <p>Форма входа интерпретируется как {@code (C, H, W, seqLen)}:
 * <ul>
 *   <li>{@code C} — число каналов (ведущая ось);</li>
 *   <li>{@code S = H * W * seqLen} — число элементов в канале;</li>
 *   <li>для каждого канала {@code c} и каждого примера считаются
 *       {@code mean} и {@code var} по {@code S} элементам;</li>
 *   <li>{@code gamma[c]}, {@code beta[c]} — свои для канала.</li>
 * </ul>
 *
 * <p>Для плоского входа {@code flat(N)} это эквивалентно
 * {@link LayerNormLayer}: один канал, {@code S = N}. Отличие возникает
 * для CNN-входа {@code cnn(C,H,W)}, где каждый канал нормируется
 * независимо — это и есть InstanceNorm.
 *
 * <p><b>Чем отличается от BatchNorm.</b> BatchNorm считает статистики
 * по батчу и накапливает running-статистики для инференса; InstanceNorm
 * считает статистики по одному примеру и не накапливает ничего. Поэтому
 * режимы {@code training} и инференса идентичны.
 *
 * <h3>Параметры</h3>
 * <ul>
 *   <li>{@code gamma[c]} — обучаемый масштаб канала {@code c} (инициализируется единицей);</li>
 *   <li>{@code beta[c]}  — обучаемый сдвиг канала {@code c} (инициализируется нулём);</li>
 *   <li>{@code eps}      — малое число для численной устойчивости,
 *       по умолчанию {@code 1e-5}.</li>
 * </ul>
 *
 * <p><b>Backward.</b> Стандартная формула для instance-нормы: см.
 * комментарий в {@link #backward(float[])}.
 *
 * <p><b>Когда использовать.</b> InstanceNorm — стандарт для style transfer
 * и генеративных CNN (Fast Neural Style, AdaIN). Для задач классификации
 * обычно лучше BatchNorm (статистики по батчу дают лучший сигнал).
 */
@NoArgsConstructor
public class InstanceNormLayer extends Layer {

    /** DTO instancenorm-слоя. */
    public static class Dto extends Layer.Dto {
        /** Малое число для численной устойчивости. */
        public final float eps;

        /** Эквивалентно {@code new Dto(1e-5f)}. */
        public Dto() { this(1e-5f); }

        public Dto(float eps) {
            if (eps <= 0f)
                throw new IllegalArgumentException("InstanceNormLayer.Dto: eps > 0, получено " + eps);
            this.eps = eps;
        }

        @Override
        public Layer create(TensorShape inputShape) {
            return new InstanceNormLayer()
                    .setEps(eps)
                    .setInputShape(inputShape)
                    .setOutputShape(inputShape)
                    .allocateBuffers();
        }
    }

    private float eps = 1e-5f;
    private int channels;
    /** Число элементов в одном канале: {@code H * W * (seqLen == 0 ? 1 : seqLen)}. */
    private int perChannel;

    /** Обучаемый масштаб; {@code gamma[c]} для канала {@code c}. */
    float[] gamma;
    /** Обучаемый сдвиг; {@code beta[c]} для канала {@code c}. */
    float[] beta;

    private float[] gammaGrad;
    private float[] betaGrad;

    /**
     * Обёртки {@code [1][channels]} для вызова матричной версии
     * {@link Optimizer#update(float[][], float[][], float[], float[], int)}.
     *
     * <p><b>Важно:</b> {@code gammaAsMatrix[0] == gamma} и
     * {@code gammaGradAsMatrix[0] == gammaGrad} — это те же самые массивы,
     * а не копии. Обёртки создаются в {@link #allocateBuffers()} и больше не
     * пересоздаются, поэтому ссылочное равенство сохраняется на всё время
     * жизни слоя.
     */
    private float[][] gammaAsMatrix;
    private float[][] gammaGradAsMatrix;

    private float[] out;
    private float[] prevDelta;

    /** {@code xhat[c][i]} — нормированный вход канала {@code c}. */
    private float[][] xhat;
    /** {@code invStd[c]} — {@code 1 / sqrt(var_c + eps)}. */
    private float[] invStd;

    public InstanceNormLayer setEps(float eps) {
        if (eps <= 0f)
            throw new IllegalArgumentException("InstanceNormLayer: eps > 0, получено " + eps);
        this.eps = eps;
        return this;
    }

    /**
     * Валидирует формы, выделяет буферы, инициализирует {@code gamma=1},
     * {@code beta=0} и создаёт матричные обёртки для {@link #update}.
     *
     * @return {@code this}
     * @throws IllegalStateException    если формы не заданы
     * @throws IllegalArgumentException если размеры входа и выхода не совпадают
     */
    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("InstanceNormLayer.create: inputShape и outputShape должны быть заданы");
        if (inputShape.size() != outputShape.size())
            throw new IllegalArgumentException(
                    "InstanceNormLayer: размеры входа и выхода должны совпадать, получено "
                            + inputShape.size() + " и " + outputShape.size());
        if (eps <= 0f)
            throw new IllegalArgumentException("InstanceNormLayer.create: eps > 0, получено " + eps);

        this.channels   = inputShape.channels();
        this.perChannel = inputShape.height() * inputShape.width()
                * (inputShape.seqLen() == 0 ? 1 : inputShape.seqLen());

        this.gamma     = new float[channels];
        this.beta      = new float[channels];
        this.gammaGrad = new float[channels];
        this.betaGrad  = new float[channels];

        Arrays.fill(gamma, 1f);

        // Обёртки [1][channels] для матричной версии Optimizer.update.
        // Внутри — те же массивы, не копии.
        this.gammaAsMatrix     = new float[][]{ gamma };
        this.gammaGradAsMatrix = new float[][]{ gammaGrad };

        int size = inputShape.size();
        this.out       = new float[size];
        this.prevDelta = new float[size];
        this.xhat      = new float[channels][perChannel];
        this.invStd    = new float[channels];
        return this;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Статистики считаются для каждого канала каждого примера;
     * режим {@code training} не влияет на результат.
     *
     * @throws IllegalArgumentException если длина {@code input} не равна {@code size}
     */
    @Override
    public float[] forward(float[] input, boolean training) {
        int size = channels * perChannel;
        if (input.length != size)
            throw new IllegalArgumentException(
                    "InstanceNormLayer.forward: ожидалось " + size + ", получено " + input.length);

        for (int c = 0; c < channels; c++) {
            int base = c * perChannel;

            // среднее по каналу
            float mean = 0f;
            for (int i = 0; i < perChannel; i++) mean += input[base + i];
            mean /= perChannel;

            // дисперсия по каналу
            float var = 0f;
            for (int i = 0; i < perChannel; i++) {
                float d = input[base + i] - mean;
                var += d * d;
            }
            var /= perChannel;

            float inv = (float) (1.0 / Math.sqrt(var + eps));
            invStd[c] = inv;

            float g = gamma[c];
            float b = beta[c];
            for (int i = 0; i < perChannel; i++) {
                float xh = (input[base + i] - mean) * inv;
                xhat[c][i] = xh;
                out[base + i] = g * xh + b;
            }
        }
        return out;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Для каждого канала:
     * <pre>{@code
     * dgamma[c] += sum_i(deltaNext[i] * xhat[c][i])
     * dbeta[c]  += sum_i(deltaNext[i])
     * dxhat[i]   = deltaNext[i] * gamma[c]
     * dx[i]      = (1/S) * invStd[c] * (S * dxhat[i]
     *              - sum(dxhat) - xhat[i] * sum(dxhat * xhat))
     * }</pre>
     * Градиенты {@code gamma}/{@code beta} накапливаются по каналам и
     * примерам.
     *
     * @throws IllegalStateException    если {@link #forward} не вызывался
     * @throws IllegalArgumentException если длина {@code deltaNext} не равна {@code size}
     */
    @Override
    public float[] backward(float[] deltaNext) {
        if (xhat == null)
            throw new IllegalStateException("InstanceNormLayer.backward: forward не был вызван");
        int size = channels * perChannel;
        if (deltaNext.length != size)
            throw new IllegalArgumentException(
                    "InstanceNormLayer.backward: ожидалось " + size + ", получено " + deltaNext.length);

        for (int c = 0; c < channels; c++) {
            int base = c * perChannel;
            float g = gamma[c];
            float inv = invStd[c];
            float[] xh = xhat[c];

            float sumDxhat = 0f;
            float sumDxhatXhat = 0f;

            // dgamma, dbeta, dxhat
            float[] dxhat = new float[perChannel];
            for (int i = 0; i < perChannel; i++) {
                float d = deltaNext[base + i];
                gammaGrad[c] += d * xh[i];
                betaGrad[c]  += d;
                dxhat[i] = d * g;
                sumDxhat     += dxhat[i];
                sumDxhatXhat += dxhat[i] * xh[i];
            }

            float invS = 1f / perChannel;
            for (int i = 0; i < perChannel; i++) {
                prevDelta[base + i] = invS * inv
                        * (perChannel * dxhat[i] - sumDxhat - xh[i] * sumDxhatXhat);
            }
        }
        return prevDelta;
    }

    @Override
    public void zeroGrad() {
        Arrays.fill(gammaGrad, 0f);
        Arrays.fill(betaGrad, 0f);
    }

    /**
     * {@inheritDoc}
     *
     * <p>Использует матричную версию {@link Optimizer#update(float[][], float[][], float[], float[], int)}
     * через обёртки {@code [1][channels]}, созданные в {@link #allocateBuffers()}.
     * Внутри обёрток — те же массивы {@code gamma}/{@code gammaGrad},
     * что и в forward/backward, поэтому оптимизатор обновляет именно их.
     */
    @Override
    public void update(Optimizer optimizer, int batchSize) {
        if (batchSize <= 0)
            throw new IllegalArgumentException("batchSize должен быть > 0, получено " + batchSize);
        optimizer.update(gammaAsMatrix, gammaGradAsMatrix, beta, betaGrad, batchSize);
    }

    @Override
    public double gradNormSq() {
        double sum = 0;
        for (float v : gammaGrad) sum += v * v;
        for (float v : betaGrad)  sum += v * v;
        return sum;
    }

    @Override
    public void scaleGrad(float scale) {
        for (int c = 0; c < channels; c++) {
            gammaGrad[c] *= scale;
            betaGrad[c]  *= scale;
        }
    }

    /**
     * Сохраняет {@code eps}, {@code gamma} и {@code beta} после общих полей.
     */
    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);
        out.writeFloat(eps);
        for (float v : gamma) out.writeFloat(v);
        for (float v : beta)  out.writeFloat(v);
    }

    /**
     * Загружает {@code eps}, пересоздаёт буферы (включая матричные
     * обёртки) и читает {@code gamma}, {@code beta}.
     */
    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);
        eps = in.readFloat();
        allocateBuffers();
        for (int c = 0; c < channels; c++) gamma[c] = in.readFloat();
        for (int c = 0; c < channels; c++) beta[c]  = in.readFloat();
    }
}