package rawnet.code.rl;

import java.util.Random;

/**
 * Сэмплирование из категориального распределения (softmax) и связанные
 * с ним утилиты: {@code softmax}, {@code logProb}.
 *
 * <h2>Зачем</h2>
 * Политика {@code π(a|s) = softmax(Dense(s))} выдаёт <b>логиты</b>
 * {@code z[0..K-1]}. Чтобы выбрать действие, нужно <b>сэмплировать</b>
 * индекс по вероятностям {@code p = softmax(z)}. Не {@code argmax}!
 * Если всегда брать лучшее действие — агент <b>никогда</b> не попробует
 * альтернативы, и может <b>застрять</b> в локальном оптимуме. Это
 * <b>исследование</b> (exploration) — ключевое отличие RL от supervised.
 *
 * <h2>Что внутри</h2>
 * <ul>
 *   <li>{@link #sample} — сэмплировать индекс из {@code softmax(logits)};</li>
 *   <li>{@link #logProb} — логарифм вероятности действия (для loss);</li>
 *   <li>{@link #softmax} — softmax (для отладки / инференса).</li>
 * </ul>
 *
 * <h2>Численная устойчивость</h2>
 * {@code exp(z - max)} — стандартный max-trick, чтобы {@code exp} не
 * переполнялся. Без него при {@code z} ~ 100 {@code exp(100)} = inf.
 */
public final class Categorical {

    private Categorical() {}

    /**
     * Сэмплирует индекс из {@code softmax(logits)}.
     *
     * <p><b>Алгоритм.</b> Считаем {@code p = softmax(z)}, затем
     * {@code r = random()}, идём по накопленной сумме, возвращаем
     * первый индекс, где {@code acc >= r}. Это стандартный inverse-CDF.
     *
     * @param logits логиты длины {@code K}; не {@code null}, не пуст
     * @param rnd    генератор случайных чисел
     * @return индекс в {@code [0, K)}
     */
    public static int sample(float[] logits, Random rnd) {
        float max = logits[0];
        for (int i = 1; i < logits.length; i++)
            if (logits[i] > max) max = logits[i];

        double sum = 0.0;
        double[] probs = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = Math.exp(logits[i] - max);
            sum += probs[i];
        }
        double r = rnd.nextDouble() * sum;
        double acc = 0.0;
        for (int i = 0; i < logits.length; i++) {
            acc += probs[i];
            if (r <= acc) return i;
        }
        return logits.length - 1;
    }

    /**
     * Логарифм вероятности действия {@code action} при логитах
     * {@code logits}: {@code log softmax(logits)[action]}.
     *
     * <p>Используется в REINFORCE loss:
     * {@code loss = -Σ log π(a_t|s_t) · G_t}. Через {@code logProb}
     * это {@code -Σ logProb(logits_t, a_t) · G_t}.
     */
    public static float logProb(float[] logits, int action) {
        float max = logits[0];
        for (int i = 1; i < logits.length; i++)
            if (logits[i] > max) max = logits[i];
        double sum = 0.0;
        for (float z : logits) sum += Math.exp(z - max);
        return (float) (logits[action] - max - Math.log(sum));
    }

    /** Softmax (для отладки / инференса). */
    public static float[] softmax(float[] logits) {
        float max = logits[0];
        for (int i = 1; i < logits.length; i++)
            if (logits[i] > max) max = logits[i];
        float[] p = new float[logits.length];
        double sum = 0.0;
        for (int i = 0; i < logits.length; i++) {
            p[i] = (float) Math.exp(logits[i] - max);
            sum += p[i];
        }
        float inv = (float) (1.0 / sum);
        for (int i = 0; i < p.length; i++) p[i] *= inv;
        return p;
    }
}