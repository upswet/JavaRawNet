package rawnet.docs.flows.main.mlp.glava2.code;

import rawnet.code.dataset.SampleDataset;
import rawnet.code.dataset.Sample;
import rawnet.docs.flows.main.mlp.tasks.Iris;

import java.nio.file.Path;
import java.util.Random;

/**
 * Решение задачи классификации ирисов в матричной концепции:
 * сеть — это набор матриц весов и векторов смещений,
 * прямой проход — умножение вектора на матрицу + поэлементная активация.
 */
public class IrisMatrix {

    static final int INPUT_SIZE  = 4; //размер входного слоя
    static final int HIDDEN_SIZE = 10;//размер скрытого слоя
    static final int OUTPUT_SIZE = 3; //размер выходного слоя
    static final float LEARNING_RATE = 0.01f;
    static final int EPOCHS   = 2000;
    static final int LOG_EVERY = 200;

    static final Random RND = new Random(42);

    // -------------------- Параметры сети --------------------
    static float[][] weightsHidden = randomMatrix(INPUT_SIZE,  HIDDEN_SIZE);
    static float[][] weightsOutput = randomMatrix(HIDDEN_SIZE, OUTPUT_SIZE);
    static float[]   biasHidden    = randomVector(HIDDEN_SIZE);
    static float[]   biasOutput    = randomVector(OUTPUT_SIZE);

    // -------------------- Инициализация --------------------

    static float randomWeight() {
        return RND.nextFloat() * 0.2f - 0.1f;
    }

    static float[][] randomMatrix(int rows, int cols) {
        float[][] matrix = new float[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i][j] = randomWeight();
        return matrix;
    }

    static float[] randomVector(int size) {
        float[] vector = new float[size];
        for (int i = 0; i < size; i++) vector[i] = randomWeight();
        return vector;
    }

    // -------------------- Активация --------------------

    static float sigmoid(float x) {
        return 1f / (1f + (float) Math.exp(-x));
    }

    static float[] sigmoidAll(float[] values) {
        float[] result = new float[values.length];
        for (int i = 0; i < values.length; i++) result[i] = sigmoid(values[i]);
        return result;
    }

    // -------------------- Матричные операции --------------------

    /** Умножение вектора-строки на матрицу весов плюс смещение. */
    static float[] multiplyByWeights(float[] input, float[][] weights, float[] bias) {
        int outputSize = weights[0].length;
        float[] result = new float[outputSize];
        for (int j = 0; j < outputSize; j++) {
            float sum = bias[j];
            for (int i = 0; i < input.length; i++) sum += input[i] * weights[i][j];
            result[j] = sum;
        }
        return result;
    }

    // -------------------- Прямой проход --------------------

    static float[] forward(float[] input) {
        float[] hidden = sigmoidAll(multiplyByWeights(input, weightsHidden, biasHidden));
        return sigmoidAll(multiplyByWeights(hidden, weightsOutput, biasOutput));
    }

    // -------------------- Один шаг обучения --------------------

    static void trainOneEpoch(SampleDataset<Sample> dataset) {
        for (Sample sample : dataset.getTrainSamples()) {
            float[] input  = sample.getInputs();
            float[] target = sample.getTargets();

            /** прямой проход с сохранением промежуточных значений*/
            float[] hiddenPreActivation = multiplyByWeights(input, weightsHidden, biasHidden);
            float[] hidden              = sigmoidAll(hiddenPreActivation);
            float[] outputPreActivation = multiplyByWeights(hidden, weightsOutput, biasOutput);
            float[] output              = sigmoidAll(outputPreActivation);

            /** обратный проход*/

            // дельты выходного слоя: (output - target) * σ'(z)
            float[] outputDeltas = new float[OUTPUT_SIZE];
            for (int j = 0; j < OUTPUT_SIZE; j++)
                outputDeltas[j] = (output[j] - target[j]) * output[j] * (1 - output[j]);

            // дельты скрытого слоя: (outputDeltas · weightsOutputᵀ) * σ'(z)
            float[] hiddenDeltas = new float[HIDDEN_SIZE];
            for (int k = 0; k < HIDDEN_SIZE; k++) {
                float weightedSum = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++)
                    weightedSum += outputDeltas[j] * weightsOutput[k][j];
                hiddenDeltas[k] = weightedSum * hidden[k] * (1 - hidden[k]);
            }

            // обновление весов и смещений выходного слоя
            for (int k = 0; k < HIDDEN_SIZE; k++)
                for (int j = 0; j < OUTPUT_SIZE; j++)
                    weightsOutput[k][j] -= LEARNING_RATE * outputDeltas[j] * hidden[k];
            for (int j = 0; j < OUTPUT_SIZE; j++)
                biasOutput[j] -= LEARNING_RATE * outputDeltas[j];

            // обновление весов и смещений скрытого слоя
            for (int i = 0; i < INPUT_SIZE; i++)
                for (int k = 0; k < HIDDEN_SIZE; k++)
                    weightsHidden[i][k] -= LEARNING_RATE * hiddenDeltas[k] * input[i];
            for (int k = 0; k < HIDDEN_SIZE; k++)
                biasHidden[k] -= LEARNING_RATE * hiddenDeltas[k];
        }
    }

    // -------------------- Оценка --------------------

    //считать потерю
    static float mseLoss(float[] predicted, float[] target) {
        float sum = 0;
        for (int i = 0; i < predicted.length; i++) {
            float diff = predicted[i] - target[i];
            sum += diff * diff;
        }
        return sum / predicted.length;
    }

    static int argmax(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }

    static void test(int epoch, SampleDataset<Sample> dataset) {
        float lossSum = 0;
        int correct = 0;
        int total = 0;
        for (Sample sample : dataset.getTestSamples()) {
            float[] output = forward(sample.getInputs());
            lossSum += mseLoss(output, sample.getTargets());
            if (argmax(output) == argmax(sample.getTargets())) correct++;
            total++;
        }
        float averageLoss = lossSum / total;
        float accuracy = 100f * correct / total;
        System.out.printf("Epoch %4d: loss=%.6f, accuracy=%.1f%%%n", epoch, averageLoss, accuracy);
    }

    // -------------------- main --------------------

    public static void main(String[] args) {
        SampleDataset<Sample> dataset = SampleDataset.split(Iris.prepareDataForIris(Path.of("d:\\Work\\Project\\0files\\iris\\iris.data.csv")), 0.9f);

        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            trainOneEpoch(dataset);
            if (epoch % LOG_EVERY == 0 || epoch == EPOCHS) test(epoch, dataset);
        }
    }
}