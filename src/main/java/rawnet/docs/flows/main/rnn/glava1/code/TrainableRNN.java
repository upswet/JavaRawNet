package rawnet.docs.flows.main.rnn.glava1.code;

import java.util.*;

/**
 * Простая рекуррентная нейронная сеть (RNN) для бинарной классификации последовательностей цифр.
 * Определяет, содержит ли входная строка подстроки "123" и/или "789".
 *
 * <p>Архитектура: входной one-hot слой (10 нейронов, цифры '0'–'9'),
 * один скрытый рекуррентный слой (сигмоида), выходной слой (2 нейрона с сигмоидой).
 * Обучение – стохастический градиентный спуск с обратным распространением ошибки во времени (BPTT).
 *
 * <p>Для борьбы с дисбалансом классов генератор данных создаёт сбалансированные примеры,
 * а gradient clipping предотвращает взрыв градиентов.
 */
public class TrainableRNN {

    /** Размер входного вектора (one-hot для 10 цифр). */
    private static final int INPUT_SIZE = 10;
    /** Количество нейронов в скрытом рекуррентном слое. */
    private static final int HIDDEN_SIZE = 12;
    /** Количество выходных нейронов (вероятности наличия "123" и "789"). */
    private static final int OUTPUT_SIZE = 2;
    /** Скорость обучения для SGD. */
    private static final double LEARNING_RATE = 0.5;
    /** Порог отсечения градиентов (gradient clipping). */
    private static final double GRAD_CLIP = 5.0;

    // Веса сети
    /** Веса из входного слоя в скрытый: Wxh[i][j] – вес от j-го входа к i-му скрытому нейрону. */
    private double[][] Wxh;
    /** Рекуррентные веса скрытого слоя: Whh[i][j] – вес от j-го скрытого нейрона (предыдущее состояние) к i-му. */
    private double[][] Whh;
    /** Смещения скрытого слоя. */
    private double[] bh;
    /** Веса из скрытого слоя в выходной: Why[i][j] – вес от j-го скрытого нейрона к i-му выходному. */
    private double[][] Why;
    /** Смещения выходного слоя. */
    private double[] by;

    // Накопленные градиенты (для одного примера)
    private double[][] dWxh, dWhh, dWhy;
    private double[] dbh, dby;

    /**
     * Сохранённые скрытые состояния на каждом временном шаге при прямом проходе.
     * hiddenStates.get(0) – начальное состояние h0 (нули), hiddenStates.get(t) – состояние после t-го символа.
     */
    private List<double[]> hiddenStates;
    /** Входные one-hot векторы для каждого временного шага (индексы 0..T-1). */
    private List<double[]> inputs;

    /** Генератор псевдослучайных чисел с фиксированным зерном для воспроизводимости. */
    private static final Random rng = new Random(42);

    /** Отображение символа цифры в индекс one-hot вектора. */
    private static final Map<Character, Integer> charToIndex = new HashMap<>();
    static {
        for (int i = 0; i < 10; i++) charToIndex.put((char) ('0' + i), i);
    }

    /**
     * Создаёт необученную RNN со случайными начальными весами.
     * Веса инициализируются малыми значениями из равномерного распределения [-0.1, 0.1].
     */
    public TrainableRNN() {
        Wxh = randomMatrix(HIDDEN_SIZE, INPUT_SIZE, 0.1);
        Whh = randomMatrix(HIDDEN_SIZE, HIDDEN_SIZE, 0.1);
        bh = new double[HIDDEN_SIZE];          // нули
        Why = randomMatrix(OUTPUT_SIZE, HIDDEN_SIZE, 0.1);
        by = new double[OUTPUT_SIZE];          // нули

        dWxh = new double[HIDDEN_SIZE][INPUT_SIZE];
        dWhh = new double[HIDDEN_SIZE][HIDDEN_SIZE];
        dbh = new double[HIDDEN_SIZE];
        dWhy = new double[OUTPUT_SIZE][HIDDEN_SIZE];
        dby = new double[OUTPUT_SIZE];
    }

    /**
     * Создаёт матрицу размера rows × cols, заполненную случайными числами в диапазоне [-scale, scale].
     * @param rows количество строк
     * @param cols количество столбцов
     * @param scale полуширина диапазона
     * @return матрица случайных весов
     */
    private double[][] randomMatrix(int rows, int cols, double scale) {
        double[][] m = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i][j] = (rng.nextDouble() - 0.5) * 2 * scale;
        return m;
    }

    /** Логистическая сигмоида. */
    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * Прямой проход сети: обрабатывает последовательность символов и возвращает предсказания.
     *
     * @param sequence входная строка (цифры и, возможно, другие символы – они кодируются нулевым вектором)
     * @param training если {@code true}, сохраняет скрытые состояния и входы для последующего обучения
     * @return массив из двух вероятностей: [P("123"), P("789")]
     */
    public double[] forward(String sequence, boolean training) {
        // Инициализация структур только в режиме обучения
        if (training) {
            hiddenStates = new ArrayList<>();
            inputs = new ArrayList<>();
        } else {
            // В режиме предсказания этого не требуется
            hiddenStates = null;
            inputs = null;
        }

        // Начальное скрытое состояние h0 – нулевой вектор
        double[] h = new double[HIDDEN_SIZE];
        if (training) {
            hiddenStates.add(h.clone());
        }

        // Обработка последовательности посимвольно
        for (char c : sequence.toCharArray()) {
            // one-hot кодирование символа; для не-цифр – нулевой вектор
            double[] x = new double[INPUT_SIZE];
            Integer idx = charToIndex.get(c);
            if (idx != null) x[idx] = 1.0;

            if (training) {
                inputs.add(x);
            }

            // Вычисление нового скрытого состояния: h_t = sigmoid(Wxh * x_t + Whh * h_{t-1} + bh)
            double[] hNew = new double[HIDDEN_SIZE];
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                double sum = bh[i];
                for (int j = 0; j < INPUT_SIZE; j++) sum += Wxh[i][j] * x[j];
                for (int j = 0; j < HIDDEN_SIZE; j++) sum += Whh[i][j] * h[j];
                hNew[i] = sigmoid(sum);
            }
            h = hNew;

            if (training) {
                hiddenStates.add(h.clone()); // сохраняем h_t
            }
        }

        // Выходной слой: y = sigmoid(Why * h_T + by)
        double[] y = new double[OUTPUT_SIZE];
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            double sum = by[i];
            for (int j = 0; j < HIDDEN_SIZE; j++) sum += Why[i][j] * h[j];
            y[i] = sigmoid(sum);
        }
        return y;
    }

    /**
     * Один шаг обучения на одном примере.
     * Выполняет прямой проход (с сохранением состояний), обратное распространение ошибки во времени (BPTT)
     * и обновление весов методом SGD.
     *
     * @param sequence   входная строка
     * @param target123  целевая вероятность наличия "123" (0.0 или 1.0)
     * @param target789  целевая вероятность наличия "789" (0.0 или 1.0)
     */
    public void train(String sequence, double target123, double target789) {
        // Прямой проход
        double[] output = forward(sequence, true);
        double[] target = {target123, target789};

        // Обнуляем накопленные градиенты
        zeroGradients();

        // ----- 1. Градиенты выходного слоя -----
        // Для бинарной кросс-энтропии с сигмоидой: dL/dy_i = (y_i - t_i)
        double[] deltaOut = new double[OUTPUT_SIZE];
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            deltaOut[i] = (output[i] - target[i]);
        }

        // Градиенты по Why и by
        double[] hLast = hiddenStates.get(hiddenStates.size() - 1);
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                dWhy[i][j] += deltaOut[i] * hLast[j];
            }
            dby[i] += deltaOut[i];
        }

        // ----- 2. Распространение ошибки назад во времени (BPTT) -----
        // Вычисляем вклад выхода в градиент последнего скрытого состояния
        double[] dhNext = new double[HIDDEN_SIZE];
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                dhNext[j] += deltaOut[i] * Why[i][j];
            }
        }

        // Количество временных шагов (от 1 до T, где T – длина последовательности)
        int T = hiddenStates.size() - 1;
        // Идём в обратном порядке: от последнего шага к первому
        for (int t = T; t >= 1; t--) {
            double[] hCurr = hiddenStates.get(t);   // h_t
            double[] hPrev = hiddenStates.get(t - 1); // h_{t-1}
            double[] x = inputs.get(t - 1);          // x_t

            // Вычисляем локальный градиент для текущего скрытого состояния:
            // dh_curr = dh_next * (h_curr * (1 - h_curr)) – производная сигмоиды
            double[] dhCurr = new double[HIDDEN_SIZE];
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                dhCurr[i] = dhNext[i] * hCurr[i] * (1.0 - hCurr[i]);
            }

            // Накапливаем градиенты по Whh, Wxh и bh
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    dWhh[i][j] += dhCurr[i] * hPrev[j]; // ∂h_t/∂Whh = h_{t-1}
                }
                for (int j = 0; j < INPUT_SIZE; j++) {
                    dWxh[i][j] += dhCurr[i] * x[j];     // ∂h_t/∂Wxh = x_t
                }
                dbh[i] += dhCurr[i];
            }

            // Готовим dh для предыдущего шага: dh_prev = dh_curr * Whh (транспонированное умножение)
            double[] dhPrev = new double[HIDDEN_SIZE];
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    dhPrev[j] += dhCurr[i] * Whh[i][j];
                }
            }
            dhNext = dhPrev;
        }

        // ----- 3. Gradient clipping и обновление весов -----
        clipGradients();
        updateWeights();
    }

    /** Обнуляет все массивы градиентов. */
    private void zeroGradients() {
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            Arrays.fill(dWxh[i], 0);
            Arrays.fill(dWhh[i], 0);
            dbh[i] = 0;
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            Arrays.fill(dWhy[i], 0);
            dby[i] = 0;
        }
    }

    /** Ограничивает значения градиентов для предотвращения взрыва (clipping). */
    private void clipGradients() {
        clipArray(dWxh); clipArray(dWhh); clipArray(dbh);
        clipArray(dWhy); clipArray(dby);
    }

    private void clipArray(double[][] arr) {
        for (double[] row : arr) clipArray(row);
    }

    private void clipArray(double[] arr) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] > GRAD_CLIP) arr[i] = GRAD_CLIP;
            else if (arr[i] < -GRAD_CLIP) arr[i] = -GRAD_CLIP;
        }
    }

    /** Обновляет все веса и смещения по накопленным градиентам (SGD). */
    private void updateWeights() {
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) Wxh[i][j] -= LEARNING_RATE * dWxh[i][j];
            for (int j = 0; j < HIDDEN_SIZE; j++) Whh[i][j] -= LEARNING_RATE * dWhh[i][j];
            bh[i] -= LEARNING_RATE * dbh[i];
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) Why[i][j] -= LEARNING_RATE * dWhy[i][j];
            by[i] -= LEARNING_RATE * dby[i];
        }
    }

    // ================== Генерация обучающих данных ==================

    /**
     * Генерирует случайную строку заданной длины, состоящую только из цифр '0'–'9'.
     * @param length желаемая длина
     * @return случайная цифровая строка
     */
    private static String randomDigits(int length) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) sb.append((char) ('0' + rng.nextInt(10)));
        return sb.toString();
    }

    /**
     * Вставляет подстроку {@code sub} в случайную позицию строки {@code base}, затирая существующие символы.
     * @param base исходная строка
     * @param sub  вставляемая подстрока
     * @return новая строка с внедрённой подстрокой
     */
    private static String insertSubstring(String base, String sub) {
        int pos = rng.nextInt(base.length() - sub.length() + 1);
        return base.substring(0, pos) + sub + base.substring(pos + sub.length());
    }

    /**
     * Демонстрация обучения и тестирования RNN.
     *
     * <p>Генерирует сбалансированный набор строк (примерно поровну содержащих "123", "789" и без них),
     * обучает сеть в течение 30 эпох и выводит предсказания на тестовых примерах.
     */
    public static void main(String[] args) {
        TrainableRNN rnn = new TrainableRNN();

        final int samples = 2000;
        List<String> seqs = new ArrayList<>();
        List<Double> l123 = new ArrayList<>();
        List<Double> l789 = new ArrayList<>();

        // Генерация сбалансированных данных
        for (int i = 0; i < samples; i++) {
            int len = 5 + rng.nextInt(4); // длина от 5 до 8
            String base = randomDigits(len);

            double has123 = 0, has789 = 0;
            double dice = rng.nextDouble();
            // ~40% содержат "123", ~40% – "789", ~20% – ни одной
            if (dice < 0.4) {
                base = insertSubstring(base, "123");
                has123 = 1;
            } else if (dice < 0.8) {
                base = insertSubstring(base, "789");
                has789 = 1;
            }
            // Иногда (10%) добавляем вторую подстроку
            if (rng.nextDouble() < 0.1) {
                if (has123 == 0) {
                    base = insertSubstring(base, "123");
                    has123 = 1;
                }
                if (has789 == 0) {
                    base = insertSubstring(base, "789");
                    has789 = 1;
                }
            }
            seqs.add(base);
            l123.add(has123);
            l789.add(has789);
        }

        // Цикл обучения
        int epochs = 10;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double loss = 0;
            // Перемешивание данных для лучшей сходимости
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < seqs.size(); i++) indices.add(i);
            Collections.shuffle(indices, rng);

            for (int idx : indices) {
                rnn.train(seqs.get(idx), l123.get(idx), l789.get(idx));
                // Вычисление loss на этом же примере (для отладки)
                double[] pred = rnn.forward(seqs.get(idx), false);
                double t1 = l123.get(idx), t2 = l789.get(idx);
                //функция потерь бинарная кросс‑энтропия ошибку
                //loss = -[ t * log(p) + (1 - t) * log(1 - p) ].
                //Она измеряет, насколько сильно предсказанная вероятность p расходится с истинной меткой t:
                loss += -t1 * Math.log(pred[0] + 1e-8) - (1 - t1) * Math.log(1 - pred[0] + 1e-8);
                loss += -t2 * Math.log(pred[1] + 1e-8) - (1 - t2) * Math.log(1 - pred[1] + 1e-8);
            }
            System.out.printf("Epoch %d, avg loss: %.4f%n", epoch + 1, loss / seqs.size());
        }

        // Тестирование на фиксированных примерах
        System.out.println("\nРезультаты после обучения:");
        String[] tests = {
                "123", "789", "012345", "67890", "123789",
                "1112333", "7778999", "000", "124", "12x34y789z"
        };
        for (String t : tests) {
            double[] p = rnn.forward(t, false);
            System.out.printf("Строка: \"%s\" -> P(123)=%.4f, P(789)=%.4f%n",
                    t, p[0], p[1]);
        }
    }
}