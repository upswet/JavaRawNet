package rawnet.docs.flows.main.rnn.glava1.code;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Простая рекуррентная нейронная сеть (RNN) для демонстрации решения задачи:
 * "каких символов больше во входной последовательности из 0 и 1 — нулей или единиц".
 * Сеть обучается выдавать 1, если единиц больше, и 0 в противном случае (включая равенство).
 */
public class RnnClassifier {

    /**
     * Внутренний класс, реализующий RNN с одним скрытым слоем.
     * Используется активация tanh для скрытого слоя и сигмоида для выходного слоя.
     * Обучение производится методом обратного распространения ошибки во времени (BPTT).
     */
    public static class SimpleRNN {
        private final int hiddenSize;
        private final double[] Wxh;            // веса входа -> скрытого слоя [hiddenSize]
        private final double[][] Whh;          // веса скрытый -> скрытый [hiddenSize][hiddenSize]
        private final double[] Why;            // веса скрытый -> выход [hiddenSize]
        private final double[] bh;             // смещения скрытого слоя [hiddenSize]
        private double by;                     // смещение выходного слоя (скаляр)

        /**
         * Создаёт RNN с заданным числом нейронов скрытого слоя и начальной случайной инициализацией весов.
         *
         * @param hiddenSize количество нейронов в скрытом слое
         * @param randomSeed зерно для воспроизводимой случайной инициализации
         */
        public SimpleRNN(int hiddenSize, long randomSeed) {
            this.hiddenSize = hiddenSize;
            Random rng = new Random(randomSeed);

            Wxh = new double[hiddenSize];
            Whh = new double[hiddenSize][hiddenSize];
            Why = new double[hiddenSize];
            bh = new double[hiddenSize];
            by = 0.0;

            // Инициализируем веса небольшими случайными значениями в диапазоне [-0.1, 0.1]
            for (int i = 0; i < hiddenSize; i++) {
                Wxh[i] = (rng.nextDouble() * 0.2) - 0.1;
                Why[i] = (rng.nextDouble() * 0.2) - 0.1;
                bh[i] = 0.0;
                for (int j = 0; j < hiddenSize; j++) {
                    Whh[i][j] = (rng.nextDouble() * 0.2) - 0.1;
                }
            }
        }

        /** Сигмоидная функция. */
        private double sigmoid(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }

        /**
         * Прямой проход (forward) через сеть для заданной последовательности.
         *
         * @param sequence   последовательность нулей и единиц
         * @param returnStates если true, сохранять и возвращать промежуточные скрытые состояния
         * @return результат, содержащий выходную вероятность и (опционально) сохранённые состояния
         */
        public ForwardResult forward(int[] sequence, boolean returnStates) {
            int T = sequence.length;
            List<double[]> hStates = new ArrayList<>();
            List<double[]> hPrevs = new ArrayList<>();
            double[] hPrev = new double[hiddenSize]; // начальное состояние — нулевой вектор

            for (int t = 0; t < T; t++) {
                double x = sequence[t];
                // Сохраняем состояние с предыдущего шага (h_{t-1}) до его изменения
                if (returnStates) {
                    hPrevs.add(hPrev.clone());
                }

                double[] h = new double[hiddenSize];
                // Вычисляем скрытое состояние: h_t = tanh(Wxh*x + Whh*h_{t-1} + bh)
                for (int i = 0; i < hiddenSize; i++) {
                    double sum = Wxh[i] * x + bh[i];
                    for (int j = 0; j < hiddenSize; j++) {
                        sum += Whh[i][j] * hPrev[j];
                    }
                    h[i] = Math.tanh(sum);
                }

                if (returnStates) {
                    hStates.add(h);
                }
                hPrev = h; // теперь текущее состояние становится предыдущим для следующего шага
            }

            // Выходной слой: y_pred = sigmoid(Why · h_T + by)
            double z = by;
            for (int i = 0; i < hiddenSize; i++) {
                z += Why[i] * hPrev[i];
            }
            double yPred = sigmoid(z);

            return new ForwardResult(yPred, hStates, hPrevs);
        }

        /**
         * Один шаг обучения методом стохастического градиентного спуска (BPTT).
         *
         * @param sequence     последовательность нулей и единиц
         * @param targetLabel  целевая метка: 1, если единиц больше, иначе 0
         * @param learningRate скорость обучения
         */
        public void train(int[] sequence, int targetLabel, double learningRate) {
            // Прямой проход с сохранением промежуточных состояний
            ForwardResult result = forward(sequence, true);
            double yPred = result.output;
            List<double[]> hStates = result.hStates;
            List<double[]> hPrevs = result.hPrevs;
            int T = sequence.length;

            // 1. Производная функции потерь (кросс-энтропия + сигмоида) по выходу до активации
            double delta = yPred - targetLabel;

            // 2. Градиенты выходного слоя
            double[] dWhy = new double[hiddenSize];
            double[] lastH = hStates.get(T - 1);
            for (int i = 0; i < hiddenSize; i++) {
                dWhy[i] = delta * lastH[i];
            }
            double dby = delta;

            // 3. Градиент, приходящий на последнее скрытое состояние h_T
            double[] dh = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                dh[i] = delta * Why[i];
            }

            // 4. Накопители градиентов для параметров скрытого слоя
            double[] dWxh = new double[hiddenSize];
            double[][] dWhh = new double[hiddenSize][hiddenSize];
            double[] dbh = new double[hiddenSize];

            // 5. Обратное распространение ошибки во времени (BPTT)
            for (int t = T - 1; t >= 0; t--) {
                double[] h = hStates.get(t);
                double[] hPrev = hPrevs.get(t);
                double x = sequence[t];

                // Производная tanh: dh_in = dh ⊙ (1 - h^2)
                double[] dh_in = new double[hiddenSize];
                for (int i = 0; i < hiddenSize; i++) {
                    dh_in[i] = dh[i] * (1.0 - h[i] * h[i]);
                }

                // Накапливаем градиенты для Wxh, Whh, bh на текущем шаге
                for (int i = 0; i < hiddenSize; i++) {
                    dWxh[i] += dh_in[i] * x;
                    dbh[i]  += dh_in[i];
                    for (int j = 0; j < hiddenSize; j++) {
                        dWhh[i][j] += dh_in[i] * hPrev[j];
                    }
                }

                // Вычисляем градиент для h_{t-1} (чтобы передать на предыдущий шаг)
                if (t > 0) {
                    double[] newDh = new double[hiddenSize];
                    // dh_{t-1} = Whh^T * dh_in
                    for (int j = 0; j < hiddenSize; j++) {
                        double sum = 0.0;
                        for (int i = 0; i < hiddenSize; i++) {
                            sum += Whh[i][j] * dh_in[i];
                        }
                        newDh[j] = sum;
                    }
                    dh = newDh;
                }
            }

            // 6. Обновление весов (градиентный спуск)
            for (int i = 0; i < hiddenSize; i++) {
                Wxh[i] -= learningRate * dWxh[i];
                Why[i] -= learningRate * dWhy[i];
                bh[i]  -= learningRate * dbh[i];
                for (int j = 0; j < hiddenSize; j++) {
                    Whh[i][j] -= learningRate * dWhh[i][j];
                }
            }
            by -= learningRate * dby;
        }

        /**
         * Предсказание вероятности того, что единиц больше (класс 1).
         *
         * @param sequence последовательность нулей и единиц
         * @return вероятность принадлежности классу 1
         */
        public double predict(int[] sequence) {
            return forward(sequence, false).output;
        }

        /**
         * Контейнер для результатов прямого прохода.
         */
        static class ForwardResult {
            final double output;
            final List<double[]> hStates;
            final List<double[]> hPrevs;

            ForwardResult(double output, List<double[]> hStates, List<double[]> hPrevs) {
                this.output = output;
                this.hStates = hStates;
                this.hPrevs = hPrevs;
            }
        }
    }

    /**
     * Демонстрация обучения и проверки RNN.
     */
    public static void main(String[] args) {
        int hiddenSize =16;
        double learningRate = 0.1;
        int epochs = 20000;
        long seed = 42;

        SimpleRNN rnn = new SimpleRNN(hiddenSize, seed);
        Random dataRng = new Random(123);

        System.out.println("Обучение RNN...");
        for (int e = 0; e < epochs; e++) {
            // Генерируем случайную последовательность длиной от 1 до 20
            int length = dataRng.nextInt(20) + 1;
            int[] seq = new int[length];
            int ones = 0;
            for (int i = 0; i < length; i++) {
                seq[i] = dataRng.nextInt(2); // 0 или 1
                if (seq[i] == 1) ones++;
            }
            int zeros = length - ones;
            int label = (ones > zeros) ? 1 : 0;

            // Один шаг обучения
            rnn.train(seq, label, learningRate);

            // Вывод прогресса каждые 1000 эпох
            if ((e + 1) % 1000 == 0) {
                double pred = rnn.predict(seq);
                System.out.printf("Эпоха %d, длина %d, предсказание: %.4f, истина: %d%n",
                        e + 1, length, pred, label);
            }
        }

        // Проверка на нескольких тестовых последовательностях
        System.out.println("\nТестирование обученной сети:");
        int[][] testSeqs = {
                {0, 0, 1},          // нулей больше -> класс 0
                {1, 0, 0, 1, 1},    // единиц больше -> класс 1
                {1, 1, 1},          // класс 1
                {0, 0, 0},          // класс 0
                {0, 1, 0, 1},       // поровну -> по условию класс 0
                {1, 1, 0, 0, 0},    // нулей больше -> класс 0
                {1}                 // класс 1
        };

        for (int[] seq : testSeqs) {
            double pred = rnn.predict(seq);
            int decidedClass = pred >= 0.5 ? 1 : 0;
            int zeros = 0, ones = 0;
            for (int b : seq) {
                if (b == 1) ones++; else zeros++;
            }
            System.out.print("Последовательность: ");
            for (int b : seq) System.out.print(b + " ");
            System.out.printf("-> нулей:%d, единиц:%d, предсказание: %.4f (класс %d)%n",
                    zeros, ones, pred, decidedClass);
        }
    }
}