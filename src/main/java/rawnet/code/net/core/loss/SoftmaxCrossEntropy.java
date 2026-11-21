package rawnet.code.net.core.loss;

/**
 * Перекрёстная энтропия, объединённая с softmax.
 * <p>
 * <b>Ожидает на входе логиты</b> (выход последнего Dense-слоя <b>без активации</b>),
 * а не вероятности. Softmax считается внутри.
 * <pre>
 *   z      = logits                         (вход)
 *   p_i    = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))
 *   L      = -Σ_i t_i · log(p_i)
 *          = -Σ_i t_i · z_i + logZ · Σ_i t_i,   где logZ = log(Σ_j exp(z_j - max)) + max
 *   dL/dz  = p - t                          (то, что нужно слою назад)
 * </pre>
 *
 * <h2>Зачем объединять softmax и cross-entropy</h2>
 * <ul>
 *   <li><b>Численная устойчивость.</b> Всё считается в логитах: max-trick
 *       не даёт {@code exp} переполниться, {@code logZ} всегда конечен
 *       (в сумме есть хотя бы {@code exp(0) = 1}), нет {@code log(0)}.</li>
 *   <li><b>Простой градиент.</b> {@code dL/dz = p - t} — без деления на
 *       {@code p_i}, без якобиана softmax, без цепного правила из двух шагов.</li>
 *   <li><b>Скорость.</b> Не нужен отдельный {@code SoftmaxLayer},
 *       не нужна промежуточная копия {@code p} в графе слоёв.</li>
 * </ul>
 *
 * <h2>Как использовать</h2>
 * <pre>{@code
 * // Functional
 * Node logits = b.dense(b.input(), 10);      // без активации!
 * Functional net = b.output(logits);
 * net.train(epochs, ds, new SoftmaxCrossEntropy(), optimizer, ...);
 *
 * // Sequence
 * List<Layer.Dto> layers = List.of(
 *     new Sequence.InputDto(784),
 *     new DenseLayer.Dto(128),
 *     new ActivationLayer.Dto(new LeakyRelu()),
 *     new DenseLayer.Dto(10)                 // логиты, без softmax/активации
 * );
 * Model net = new Sequence(layers);
 * net.train(epochs, ds, new SoftmaxCrossEntropy(), optimizer, ...);
 * }</pre>
 *
 * <h2>Инференс</h2>
 * Сеть возвращает логиты. Для {@code argmax} этого достаточно (порядок
 * не меняется под действием softmax), а если нужны сами вероятности —
 * вызовите {@link #softmax(float[])} вручную.
 *
 * <h2>Соглашения</h2>
 * <ul>
 *   <li>{@code target} — вектор той же длины, что и {@code logits}.
 *       Обычно one-hot; допускаются «мягкие» метки {@code t_i ∈ [0,1]}.</li>
 * </ul>
 */
public class SoftmaxCrossEntropy implements Loss {

    @Override
    public float loss(float[] logits, float[] target) {
        final int n = logits.length;
        float max = logits[0];
        for (int i = 1; i < n; i++) if (logits[i] > max) max = logits[i];

        double sumExp = 0.0;
        for (int i = 0; i < n; i++) sumExp += Math.exp(logits[i] - max);
        // В сумме всегда есть exp(0) = 1, поэтому sumExp >= 1, logZ конечен.
        final double logZ = Math.log(sumExp) + max;

        double tSum = 0.0;
        double tz   = 0.0;
        for (int i = 0; i < n; i++) {
            tSum += target[i];
            tz   += target[i] * logits[i];
        }
        return (float) (logZ * tSum - tz);
    }

    @Override
    public void derivative(float[] logits, float[] target, float[] deriv) {
        final int n = logits.length;
        float max = logits[0];
        for (int i = 1; i < n; i++) if (logits[i] > max) max = logits[i];

        double sumExp = 0.0;
        for (int i = 0; i < n; i++) sumExp += Math.exp(logits[i] - max);
        final double inv = 1.0 / sumExp;

        for (int i = 0; i < n; i++) {
            deriv[i] = (float) (Math.exp(logits[i] - max) * inv) - target[i];
        }
    }

    /**
     * Вспомогательный softmax для инференса (когда нужны вероятности, а не argmax).
     * Возвращает новый массив; вход не модифицируется.
     *
     * float[] logits = net.forward(x, false);
     * // "Насколько сеть уверена?" — нужно число в [0,1]
     * float[] p = SoftmaxCrossEntropy.softmax(logits);
     * System.out.println("уверенность в классе 3: " + p[3]);   // 0.87
     */
    public static float[] softmax(float[] logits) {
        final int n = logits.length;
        float max = logits[0];
        for (int i = 1; i < n; i++) if (logits[i] > max) max = logits[i];

        float[] p = new float[n];
        double sumExp = 0.0;
        for (int i = 0; i < n; i++) {
            p[i] = (float) Math.exp(logits[i] - max);
            sumExp += p[i];
        }
        final float inv = (float) (1.0 / sumExp);
        for (int i = 0; i < n; i++) p[i] *= inv;
        return p;
    }
}