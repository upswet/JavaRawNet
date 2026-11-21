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
 * Layer Normalization: нормирует каждый пример по всем его признакам,
 * затем применяет обучаемые {@code gamma} (масштаб) и {@code beta} (сдвиг).
 *
 * <p>Для одного примера длины {@code N}:
 * <pre>{@code
 * mu    = mean(x)
 * var   = mean((x - mu)^2)
 * xhat  = (x - mu) / sqrt(var + eps)
 * y     = gamma * xhat + beta
 * }</pre>
 *
 * <p>В отличие от BatchNorm-стиля нормировки по батчу, здесь статистики
 * <b>не накапливаются</b>: они вычисляются из текущего примера заново.
 * Поэтому режимы {@code training} и инференса идентичны, и слой можно
 * применять с батчем из одного примера.
 *
 * <h3>Параметры</h3>
 * <ul>
 *   <li>{@code gamma[i]} — обучаемый масштаб (инициализируется единицей);</li>
 *   <li>{@code beta[i]}  — обучаемый сдвиг (инициализируется нулём);</li>
 *   <li>{@code eps}      — малое число для численной устойчивости,
 *       по умолчанию {@code 1e-5}.</li>
 * </ul>
 *
 * <h3>Backward</h3>
 * Стандартная формула для layernorm:
 * <pre>{@code
 * dgamma[i] += deltaNext[i] * xhat[i]
 * dbeta[i]  += deltaNext[i]
 * dxhat[i]  = deltaNext[i] * gamma[i]
 * dx[i]     = (1 / N) * (1 / sigma) * (N * dxhat[i]
 *             - sum(dxhat) - xhat[i] * sum(dxhat * xhat))
 * }</pre>
 * Градиенты {@code gamma}/{@code beta} <b>накапливаются</b> (как у Dense).
 *
 * <p><b>Когда использовать.</b> LayerNorm — стандарт для Transformer'ов
 * (нормировка по dModel в каждом токене) и для MLP с сильным разбросом
 * масштабов признаков. Для CNN обычно лучше BatchNorm/InstanceNorm.
 */
@NoArgsConstructor
public class LayerNormLayer extends Layer {

    /** DTO layernorm-слоя. */
    public static class Dto extends Layer.Dto {
        /** Малое число для численной устойчивости. */
        public final float eps;

        /** Эквивалентно {@code new Dto(1e-5f)}. */
        public Dto() { this(1e-5f); }

        public Dto(float eps) {
            if (eps <= 0f)
                throw new IllegalArgumentException("LayerNormLayer.Dto: eps > 0, получено " + eps);
            this.eps = eps;
        }

        @Override
        public Layer create(TensorShape inputShape) {
            return new LayerNormLayer()
                    .setEps(eps)
                    .setInputShape(inputShape)
                    .setOutputShape(inputShape)
                    .allocateBuffers();
        }
    }

    private float eps = 1e-5f;
    private int size;

    /** Обучаемый масштаб; {@code gamma[i]} для признака {@code i}. */
    float[] gamma;
    /** Обучаемый сдвиг; {@code beta[i]} для признака {@code i}. */
    float[] beta;

    private float[] gammaGrad;
    private float[] betaGrad;

    /**
     * Обёртки {@code [1][size]} для вызова матричной версии
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
    private float[] xhat;
    /** {@code 1 / sqrt(var + eps)} последнего forward. */
    private float invStd;

    public LayerNormLayer setEps(float eps) {
        if (eps <= 0f)
            throw new IllegalArgumentException("LayerNormLayer: eps > 0, получено " + eps);
        this.eps = eps;
        return this;
    }

    /**
     * Валидирует размеры, выделяет буферы, инициализирует {@code gamma=1},
     * {@code beta=0} и создаёт матричные обёртки для {@link #update}.
     *
     * @return {@code this}
     * @throws IllegalStateException    если формы не заданы
     * @throws IllegalArgumentException если размеры входа и выхода не совпадают
     */
    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("LayerNormLayer.create: inputShape и outputShape должны быть заданы");
        if (inputShape.size() != outputShape.size())
            throw new IllegalArgumentException(
                    "LayerNormLayer: размеры входа и выхода должны совпадать, получено "
                            + inputShape.size() + " и " + outputShape.size());
        if (eps <= 0f)
            throw new IllegalArgumentException("LayerNormLayer.create: eps > 0, получено " + eps);

        this.size = inputShape.size();

        this.gamma     = new float[size];
        this.beta      = new float[size];
        this.gammaGrad = new float[size];
        this.betaGrad  = new float[size];

        Arrays.fill(gamma, 1f);

        // Обёртки [1][size] для матричной версии Optimizer.update.
        // Внутри — те же массивы, не копии.
        this.gammaAsMatrix     = new float[][]{ gamma };
        this.gammaGradAsMatrix = new float[][]{ gammaGrad };

        this.out       = new float[size];
        this.prevDelta = new float[size];
        this.xhat      = new float[size];
        return this;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Статистики вычисляются из текущего примера; режим {@code training}
     * не влияет на результат.
     *
     * @throws IllegalArgumentException если длина {@code input} не равна {@code size}
     */
    @Override
    public float[] forward(float[] input, boolean training) {
        if (input.length != size)
            throw new IllegalArgumentException(
                    "LayerNormLayer.forward: ожидалось " + size + ", получено " + input.length);

        // 1) среднее
        float mean = 0f;
        for (int i = 0; i < size; i++) mean += input[i];
        mean /= size;

        // 2) дисперсия
        float var = 0f;
        for (int i = 0; i < size; i++) {
            float d = input[i] - mean;
            var += d * d;
        }
        var /= size;

        this.invStd = (float) (1.0 / Math.sqrt(var + eps));

        // 3) нормировка + масштаб/сдвиг
        for (int i = 0; i < size; i++) {
            xhat[i] = (input[i] - mean) * invStd;
            out[i]  = gamma[i] * xhat[i] + beta[i];
        }
        return out;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Накапливает {@code dGamma} и {@code dBeta}; возвращает градиент
     * по входу.
     *
     * @throws IllegalStateException    если {@link #forward} не вызывался
     * @throws IllegalArgumentException если длина {@code deltaNext} не равна {@code size}
     */
    @Override
    public float[] backward(float[] deltaNext) {
        if (xhat == null)
            throw new IllegalStateException("LayerNormLayer.backward: forward не был вызван");
        if (deltaNext.length != size)
            throw new IllegalArgumentException(
                    "LayerNormLayer.backward: ожидалось " + size + ", получено " + deltaNext.length);

        // dgamma, dbeta и dxhat = deltaNext * gamma
        float[] dxhat = new float[size];
        for (int i = 0; i < size; i++) {
            float d = deltaNext[i];
            gammaGrad[i] += d * xhat[i];
            betaGrad[i]  += d;
            dxhat[i]      = d * gamma[i];
        }

        // суммы для формулы dx
        float sumDxhat = 0f;
        float sumDxhatXhat = 0f;
        for (int i = 0; i < size; i++) {
            sumDxhat      += dxhat[i];
            sumDxhatXhat  += dxhat[i] * xhat[i];
        }

        float invN = 1f / size;
        for (int i = 0; i < size; i++) {
            prevDelta[i] = invN * invStd
                    * (size * dxhat[i] - sumDxhat - xhat[i] * sumDxhatXhat);
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
     * через обёртки {@code [1][size]}, созданные в {@link #allocateBuffers()}.
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
        for (int i = 0; i < size; i++) {
            gammaGrad[i] *= scale;
            betaGrad[i]  *= scale;
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
        for (int i = 0; i < size; i++) gamma[i] = in.readFloat();
        for (int i = 0; i < size; i++) beta[i]  = in.readFloat();
    }
}