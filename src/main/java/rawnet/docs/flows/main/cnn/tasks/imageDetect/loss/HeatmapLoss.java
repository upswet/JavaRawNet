package rawnet.docs.flows.main.cnn.tasks.imageDetect.loss;

import rawnet.code.net.core.loss.Loss;

/**
 * Focal Loss для CenterNet-style heatmap (мягкие гауссианы).
 *
 * <p>Формула (CenterNet, Zhou et al.):
 * <ul>
 *   <li>positive ({@code t > 0.99}): {@code -(1-p)^α · log(p)}</li>
 *   <li>negative (иначе):          {@code -(1-t)^β · p^α · log(1-p)}</li>
 * </ul>
 * где {@code p = σ(z)}, {@code α = gamma}, {@code β = beta}.
 *
 * <p><b>Нормировка — на число позитивов</b> {@code N_pos}, как в оригинале.
 * Это делает loss сопоставимым между батчами с разным числом объектов
 * и не «размывает» градиент на 99% фона.
 *
 * <p>{@code loss()} и {@code derivative()} используют <b>один и тот же</b>
 * множитель {@code 1/N_pos} — это критично для согласованности
 * аналитического градиента с численным.
 */
public class HeatmapLoss implements Loss {

    private static final float EPS = 1e-15f;

    private final float alpha;   // вес для positive-части
    private final float beta;    // вес для negative-части (focusing)
    private final float gamma;   // focusing для positive-части

    /** CenterNet-параметры: α=2, β=4. */
    public HeatmapLoss() { this(2f, 4f, 2f); }

    public HeatmapLoss(float alpha, float beta, float gamma) {
        if (alpha < 0f) throw new IllegalArgumentException("alpha >= 0");
        if (beta  < 0f) throw new IllegalArgumentException("beta >= 0");
        if (gamma < 0f) throw new IllegalArgumentException("gamma >= 0");
        this.alpha = alpha;
        this.beta  = beta;
        this.gamma = gamma;
    }

    @Override
    public float loss(float[] logits, float[] target) {
        checkLengths(logits, target);

        int nPos = 0;
        for (float t : target) if (t > 0.99f) nPos++;
        final float invN = 1f / Math.max(1, nPos);

        double s = 0;
        for (int i = 0; i < logits.length; i++) {
            float p = sigmoid(logits[i]);
            float t = clamp01(target[i]);

            if (t > 0.99f) {
                // positive: -(1-p)^alpha * log(p)
                s -= Math.pow(1 - p, alpha) * Math.log(Math.max(p, EPS));
            } else {
                // negative: -(1-t)^beta * p^alpha * log(1-p)
                s -= Math.pow(1 - t, beta) * Math.pow(p, alpha)
                        * Math.log(Math.max(1 - p, EPS));
            }
        }
        return (float) (s * invN);
    }

    @Override
    public void derivative(float[] logits, float[] target, float[] deriv) {
        checkLengths(logits, target);
        if (deriv.length != logits.length)
            throw new IllegalArgumentException("HeatmapLoss: deriv.length != logits.length");

        int nPos = 0;
        for (float t : target) if (t > 0.99f) nPos++;
        final float invN = 1f / Math.max(1, nPos);

        for (int i = 0; i < logits.length; i++) {
            float z = logits[i];
            float p = sigmoid(z);
            float t = clamp01(target[i]);

            // dp/dz = p(1-p); домножаем производную фокусной части на него.
            float d;
            if (t > 0.99f) {
                // d/dp [ -(1-p)^a · log(p) ]
                //   = a(1-p)^(a-1)·log(p) - (1-p)^a / p
                float om = 1f - p;
                float dPos = alpha * (float) Math.pow(om, alpha - 1)
                        * (float) Math.log(Math.max(p, EPS))
                        - (float) Math.pow(om, alpha) / Math.max(p, EPS);
                d = dPos * p * om;
            } else {
                // d/dp [ -(1-t)^b · p^a · log(1-p) ]
                //   = -(1-t)^b · [ a·p^(a-1)·log(1-p) - p^a/(1-p) ]
                float om = 1f - p;
                float dNeg = -(float) Math.pow(1 - t, beta)
                        * (alpha * (float) Math.pow(p, alpha - 1)
                        * (float) Math.log(Math.max(om, EPS))
                        - (float) Math.pow(p, alpha) / Math.max(om, EPS));
                d = dNeg * p * om;
            }

            deriv[i] = d * invN;
            if (Float.isNaN(deriv[i]) || Float.isInfinite(deriv[i]))
                throw new IllegalStateException(
                        "HeatmapLoss: NaN/Inf в deriv[" + i + "] (z=" + z + ", t=" + t + ")");
        }
    }

    private static void checkLengths(float[] logits, float[] target) {
        if (logits.length != target.length)
            throw new IllegalArgumentException(
                    "HeatmapLoss: logits.length=" + logits.length
                            + " != target.length=" + target.length);
    }

    private static float sigmoid(float z) {
        if (z <= -30f) return 0f;
        if (z >=  30f) return 1f;
        return 1f / (1f + (float) Math.exp(-z));
    }

    private static float clamp01(float v) {
        if (v < 0f) return 0f;
        if (v > 1f) return 1f;
        return v;
    }
}