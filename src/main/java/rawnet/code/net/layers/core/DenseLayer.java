package rawnet.code.net.layers.core;

import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.TensorShape;
import rawnet.code.net.core.activate.Activate;
import rawnet.code.net.core.optimizator.Optimizer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Полносвязный слой без активации {@code y = W·x + b}.
 *
 * <p>Форма входа — {@link TensorShape#flat(int)} размера {@code inputSize},
 * форма выхода — {@link TensorShape#flat(int)} размера {@code outputSize}.
 *
 * <p>Параметры слоя: матрица весов {@code W} размера
 * {@code [outputSize][inputSize]} и вектор смещений {@code b} размера
 * {@code [outputSize]}. Инициализация — Xavier/Glorot (см.
 * {@link Layer#initLimit}).
 */
@NoArgsConstructor
@Accessors(chain = true)
public class DenseLayer extends Layer {

    /** DTO полносвязного слоя без активации. */
    @RequiredArgsConstructor
    public static class Dto extends Layer.Dto {
        /** Форма выхода слоя. */
        public final TensorShape outputShape;
        /**Будущая функция активации - для инициализации весов.. Проставиться при сборке сети из следующего ActivationLayer если он есть*/
        public Activate pendingActivate;

        public Dto (int outputSize){this(TensorShape.flat(outputSize));}

        @Override
        public Layer create(TensorShape inputShape) {
            return new DenseLayer()
                    .setInputShape(inputShape)
                    .setOutputShape(outputShape)
                    .allocateBuffers().initWeights(pendingActivate);
        }
    }

    /** Матрица весов: {@code weights[i][j]} — вес входа {@code j} на нейрон {@code i}. */
    public float[][] weights;   // [outputSize][inputSize]

    /** Вектор смещений: {@code biases[i]} — смещение нейрона {@code i}. */
    public float[] biases;      // [outputSize]

    // Накопители градиентов для мини-батчей
    /** Накопитель градиентов весов; суммируется в {@link #backward}. */
    public float[][] weightGrad; // [outputSize][inputSize]

    /** Накопитель градиентов смещений; суммируется в {@link #backward}. */
    public float[] biasGrad;     // [outputSize]

    // Буферы для прямого/обратного прохода (выделяются один раз)
    /** Вход последнего {@link #forward}; нужен для {@link #backward}. */
    private float[] input;

    /** Рабочая область выхода; возвращается из {@link #forward}. */
    private float[] output;

    /** Градиент по входу; возвращается из {@link #backward}. */
    private float[] prevDelta;

    private int inputSize;
    private int outputSize;

    @Override
    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("DenseLayer.create: inputShape и outputShape должны быть заданы");
        if (!inputShape.isFlat() || !outputShape.isFlat())
            throw new IllegalArgumentException("DenseLayer хочет строго плоские вход и выход.");

        this.inputSize  = inputShape.size();
        this.outputSize = outputShape.size();

        weights    = new float[outputSize][inputSize];
        biases     = new float[outputSize];
        weightGrad = new float[outputSize][inputSize];
        biasGrad   = new float[outputSize];

        this.output    = new float[outputSize];
        this.prevDelta = new float[inputSize];

        return this;
    }

    @Override
    public Layer initWeights(Activate activate) {
        float limit = Layer.initLimit(activate, inputSize, outputSize);
        Random rnd = ThreadLocalRandom.current();
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = rnd.nextFloat() * 2 * limit - limit;
            }
            biases[i] = rnd.nextFloat() * 0.2f - 0.1f;
        }

        return this;
    }

    /**
     * Прямой проход: {@code y = W·x + b}.
     *
     * <p>Сохраняет {@code input} для последующего {@link #backward}.
     * Возвращает внутренний буфер {@code output} (не алиас {@code input}):
     * его содержимое перезаписывается при следующем вызове {@code forward}.
     *
     * @param input    входной вектор длины {@code inputSize}
     * @param training не используется (у слоя нет dropout/BatchNorm)
     * @return выходной вектор длины {@code outputSize}
     * @throws IllegalArgumentException если длина {@code input} не равна {@code inputSize}
     */
    @Override
    public float[] forward(float[] input, boolean training) {
        if (input.length != inputSize)
            throw new IllegalArgumentException(
                    "DenseLayer.forward: ожидалось " + inputSize + ", получено " + input.length);

        this.input = input;

        for (int i = 0; i < outputSize; i++) {
            float sum = biases[i];
            float[] w = weights[i];
            for (int j = 0; j < inputSize; j++) {
                sum += w[j] * input[j];
            }
            output[i] = sum;
        }

        return output;
    }

    /**
     * Обратный проход: {@code deltaPrev = Wᵀ·deltaNext}, накопление
     * {@code dW += deltaNext·inputᵀ} и {@code db += deltaNext}.
     *
     * <p>Требует предшествующего {@link #forward} для того же примера.
     *
     * @param deltaNext градиент по выходу слоя длины {@code outputSize}
     * @return градиент по входу слоя длины {@code inputSize}
     * @throws IllegalStateException    если {@code forward} не вызывался
     * @throws IllegalArgumentException если длина {@code deltaNext} не равна {@code outputSize}
     */
    @Override
    public float[] backward(float[] deltaNext) {
        if (this.input == null)
            throw new IllegalStateException("DenseLayer.backward: forward не был вызван");
        if (deltaNext.length != outputSize)
            throw new IllegalArgumentException(
                    "DenseLayer.backward: ожидалось " + outputSize + ", получено " + deltaNext.length);

        Arrays.fill(prevDelta, 0f);

        for (int i = 0; i < outputSize; i++) {
            float delta = deltaNext[i];

            // Градиент по смещению: db_i += delta_i
            biasGrad[i] += delta;

            float[] w  = weights[i];        // строка весов
            float[] wg = weightGrad[i];     // строка накопителя градиентов
            for (int j = 0; j < inputSize; j++) {
                // Градиент по входу: prevDelta = Wᵀ · delta
                prevDelta[j] += w[j] * delta;

                // Градиент по весу: dL/dw_ij = delta_i * input_j
                wg[j] += delta * input[j];
            }
        }
        return prevDelta;
    }

    @Override
    public void zeroGrad() {
        for (int i = 0; i < outputSize; i++) {
            Arrays.fill(weightGrad[i], 0f);
            biasGrad[i] = 0f;
        }
    }

    @Override
    public void update(Optimizer optimizer, int batchSize) {
        if (batchSize <= 0)
            throw new IllegalArgumentException("batchSize должен быть > 0, получено " + batchSize);
        optimizer.update(weights, weightGrad, biases, biasGrad, batchSize);
    }

    /**
     * Сохраняет параметры слоя: сначала формы (через {@code super}),
     * затем веса и смещения.
     */
    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);
        for (float[] row : weights) {
            for (float v : row) {
                out.writeFloat(v);
            }
        }
        for (float b : biases) {
            out.writeFloat(b);
        }
    }

    /**
     * Загружает параметры слоя: сначала формы (через {@code super}),
     * затем пересоздаёт буферы по новым формам и читает веса и смещения.
     *
     * <p>{@link #allocateBuffers()} вызывается, чтобы привести размеры массивов
     * в соответствие с прочитанными формами (и выделить накопители
     * градиентов), даже если слой был сконструирован с другими размерами.
     */
    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);
        allocateBuffers();

        //перезапишем веса из файла
        for (float[] row : weights) {
            for (int j = 0; j < row.length; j++) {
                row[j] = in.readFloat();
            }
        }
        for (int i = 0; i < biases.length; i++) {
            biases[i] = in.readFloat();
        }
    }

    @Override
    public double gradNormSq() {
        double sum = 0;
        for (float[] row : weightGrad) {
            for (float v : row) {
                sum += v * v;
            }
        }
        for (float v : biasGrad) {
            sum += v * v;
        }
        return sum;
    }

    @Override
    public void scaleGrad(float scale) {
        for (float[] row : weightGrad) {
            for (int j = 0; j < row.length; j++) {
                row[j] *= scale;
            }
        }
        for (int i = 0; i < biasGrad.length; i++) {
            biasGrad[i] *= scale;
        }
    }
}