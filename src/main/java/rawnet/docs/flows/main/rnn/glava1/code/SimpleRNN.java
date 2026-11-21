package rawnet.docs.flows.main.rnn.glava1.code;

import java.util.HashMap;
import java.util.Map;

/**
 * Простейшая RNN для обнаружения подстрок "123" и "789" в строке.
 * Архитектура: 6 скрытых нейронов (0-2 для "123", 3-5 для "789"), 2 выходных.
 * Веса подобраны так, чтобы при непрерывном появлении цифр формировались чёткие 0/1.
 */
public class SimpleRNN {

    //x (вход) -> h (рекурентный слой) - >y (выходной слой)
    //также h замыкается сам на себя
    private static final int INPUT_SIZE = 10;   // '0'..'9'
    private static final int HIDDEN_SIZE = 6;
    private static final int OUTPUT_SIZE = 2;

    private final double[][] Wxh = new double[HIDDEN_SIZE][INPUT_SIZE]; //используется для всех связок xt → ht
    private final double[][] Whh = new double[HIDDEN_SIZE][HIDDEN_SIZE]; //используется для всех связок ht-1 → ht
    private final double[] bh = new double[HIDDEN_SIZE]; //смещение для скрытого рекурентного слоя h
    private final double[][] Why = new double[OUTPUT_SIZE][HIDDEN_SIZE]; //используется для всех связок ht → yt
    private final double[] by = new double[OUTPUT_SIZE]; //смещение для выходного слоя y

    private static final Map<Character, Integer> charToIndex = new HashMap<>();
    static {
        for (int i = 0; i < 10; i++) {
            charToIndex.put((char) ('0' + i), i);
        }
    }

    public SimpleRNN() {
        // --- Детектор "123" ---
        // Нейрон 0: активен, когда на входе '1'
        Wxh[0][charToIndex.get('1')] = 10.0;
        bh[0] = -5.0;

        // Нейрон 1: активен, когда на входе '2' и нейрон 0 был активен на предыдущем шаге
        Wxh[1][charToIndex.get('2')] = 10.0;
        Whh[1][0] = 10.0;
        bh[1] = -15.0;

        // Нейрон 2: флаг найденной подстроки "123"
        Wxh[2][charToIndex.get('3')] = 10.0;
        Whh[2][1] = 10.0;
        Whh[2][2] = 20.0;   // самоподдержка
        bh[2] = -15.0;

        // --- Детектор "789" ---
        Wxh[3][charToIndex.get('7')] = 10.0;
        bh[3] = -5.0;

        Wxh[4][charToIndex.get('8')] = 10.0;
        Whh[4][3] = 10.0;
        bh[4] = -15.0;

        Wxh[5][charToIndex.get('9')] = 10.0;
        Whh[5][4] = 10.0;
        Whh[5][5] = 20.0;
        bh[5] = -15.0;

        // Выходной слой
        Why[0][2] = 10.0;   // h2 -> выход "123"
        Why[1][5] = 10.0;   // h5 -> выход "789"
        by[0] = -5.0;
        by[1] = -5.0;
    }

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * Прямой проход по строке.
     * Не‑цифры сбрасывают прогресс (подаётся нулевой входной вектор).
     * @param input строка из любых символов
     * @return [P("123"), P("789")]
     */
    public double[] forward(String input) {
        double[] h = new double[HIDDEN_SIZE];      // текущее скрытое состояние
        double[] hNext = new double[HIDDEN_SIZE];

        for (char c : input.toCharArray()) {
            Integer idx = charToIndex.get(c);

            for (int i = 0; i < HIDDEN_SIZE; i++) {
                double sum = bh[i];
                // вход от текущего символа (для не‑цифр равен 0)
                if (idx != null) {
                    sum += Wxh[i][idx];
                }
                // рекуррентный вход
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    sum += Whh[i][j] * h[j];
                }
                hNext[i] = sigmoid(sum);
            }

            // переключить буферы
            double[] tmp = h;
            h = hNext;
            hNext = tmp;
        }

        // выход
        double[] y = new double[OUTPUT_SIZE];
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            double sum = by[i];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                sum += Why[i][j] * h[j];
            }
            y[i] = sigmoid(sum);
        }
        return y;
    }

    public static void main(String[] args) {
        SimpleRNN rnn = new SimpleRNN();
        String[] tests = {
                "123", "789", "012345", "67890", "123789",
                "1112333", "7778999", "000", "124", "12x34y789z"
        };
        for (String t : tests) {
            double[] p = rnn.forward(t);
            System.out.printf("Строка: \"%s\" -> P(123)=%.4f, P(789)=%.4f%n",
                    t, p[0], p[1]);
        }
    }
}