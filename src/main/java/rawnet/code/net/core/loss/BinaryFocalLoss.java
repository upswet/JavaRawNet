package rawnet.code.net.core.loss;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Бинарная Focal Loss (Lin et al., 2017) для задач с сильным дисбалансом
 * классов, где метки жёсткие: {@code y ∈ {0, 1}}.
 * <p>
 * Идея: обычная Binary Cross-Entropy (BCE) штрафует все ошибки одинаково,
 * и на сильно разбалансированных данных сеть «скатывается» к предсказанию
 * доминирующего класса (высокая accuracy, но бесполезная модель).
 * Focal Loss гасит вклад «лёгких» примеров (там, где сеть уверена и права)
 * и оставляет полный штраф для «трудных» (где уверена и ошибается).
 * <p>
 * Формула для одного элемента:
 * <pre>
 *   L = -α_t · (1 - p_t)^γ · log(p_t)
 * </pre>
 * где {@code p_t} — вероятность правильного класса:
 * <pre>
 *   p_t = p·y + (1-p)·(1-y)
 * </pre>
 * а {@code α_t} — весовой коэффициент класса:
 * <pre>
 *   α_t = α·y + (1-α)·(1-y)
 * </pre>
 * <p>
 * <b>Параметры:</b>
 * <ul>
 *   <li>{@code alpha = 0.25} — баланс положительных/отрицательных классов
 *       (в оригинальной статье для положительного класса берут 0.25,
 *       для отрицательного — 0.75, потому что положительных обычно
 *       в разы меньше);</li>
 *   <li>{@code gamma = 2.0} — сила фокусировки. При {@code gamma = 0}
 *       Focal Loss вырождается в обычную BCE. Чем больше {@code gamma},
 *       тем сильнее подавляются лёгкие примеры.</li>
 * </ul>
 * <p>
 * <b>Важно:</b> это <i>не</i> CenterNet-style heatmap-loss. Тот вариант
 * работает с мягкими метками (гауссианами) и имеет другую формулу
 * (с параметром {@code beta} и весом {@code (1-y)^beta}). Если вам нужен
 * heatmap-loss для детекции объектов — используйте отдельный класс
 * {@code CenterNetFocalLoss}.
 * <p>
 * <b>Производная</b> возвращается в пространстве вероятностей {@code dL/dp}.
 * Последующий слой Sigmoid домножит её на {@code p(1-p)}, получив корректный
 * {@code dL/dz}. В композиции с Sigmoid градиент численно устойчив, даже
 * когда {@code p} близко к 0 или 1.
 */
public class BinaryFocalLoss implements Loss {
    /** Малая константа для клиппинга. 1e-7f — безопасно для float
     *  (не субнормальное число, в отличие от 1e-15f). */
    private static final float EPS = 1e-7f;

    private float alpha;
    private float gamma;

    public BinaryFocalLoss() {
        this(0.25f, 2f);
    }

    public BinaryFocalLoss(float alpha, float gamma) {
        if (alpha < 0f || alpha > 1f) {
            throw new IllegalArgumentException("alpha must be in [0, 1], got " + alpha);
        }
        if (gamma < 0f) {
            throw new IllegalArgumentException("gamma must be >= 0, got " + gamma);
        }
        this.alpha = alpha;
        this.gamma = gamma;
    }

    @Override
    public float loss(float[] p, float[] t) {
        double s = 0;
        for (int i = 0; i < p.length; i++) {
            float pi = Math.max(Math.min(p[i], 1f - EPS), EPS);
            float ti = t[i];
            float pt = pi * ti + (1f - pi) * (1f - ti);   // «уверенность в правильном»
            float at = alpha * ti + (1f - alpha) * (1f - ti);
            s -= at * Math.pow(1 - pt, gamma) * Math.log(Math.max(pt, EPS));
        }
        return (float) (s / p.length);
    }

    @Override
    public void derivative(float[] p, float[] t, float[] d) {
        for (int i = 0; i < p.length; i++) {
            // Клиппинг для численной устойчивости.
            float pi = Math.max(Math.min(p[i], 1f - EPS), EPS);
            float ti = t[i];

            // pt — вероятность правильного класса.
            float pt = pi * ti + (1f - pi) * (1f - ti);
            // at — весовой коэффициент (alpha для положительного, 1-alpha для отрицательного).
            float at = alpha * ti + (1f - alpha) * (1f - ti);
            // sign = +1 если ti = 1, и -1 если ti = 0.
            // Это d(pt)/dp: при y=1 p_t = p, при y=0 p_t = 1-p.
            float sign = 2f * ti - 1f;

            // Защита от gamma < 1 и pt -> 1: Math.pow(0, отрицательное) = Infinity.
            float oneMinusPt = Math.max(1f - pt, EPS);

            // term1 — вклад от дифференцирования (1-pt)^gamma:
            //   d/dp[(1-pt)^gamma] = -gamma * (1-pt)^(gamma-1) * sign
            //   после умножения на log(pt) и внесения минуса получаем +gamma * ...
            float term1 = gamma
                    * (float) Math.pow(oneMinusPt, gamma - 1f)
                    * (float) Math.log(Math.max(pt, EPS));

            // term2 — вклад от дифференцирования log(pt):
            //   d/dp[log(pt)] = sign / pt
            //   после внесения минуса получаем - (1-pt)^gamma / pt
            float term2 = (float) Math.pow(oneMinusPt, gamma) / pt;

            // Итог: dL/dp = at * sign * (term1 - term2).
            d[i] = at * sign * (term1 - term2);
        }
    }

    @Override
    public void save(DataOutputStream out) throws IOException {
        out.writeFloat(alpha);
        out.writeFloat(gamma);
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        float a = in.readFloat();
        float g = in.readFloat();
        if (a < 0f || a > 1f) {
            throw new IOException("Invalid alpha loaded: " + a);
        }
        if (g < 0f) {
            throw new IOException("Invalid gamma loaded: " + g);
        }
        this.alpha = a;
        this.gamma = g;
    }
}