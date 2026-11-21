package rawnet.code.net.core.metric;

/**
 * Коэффициент детерминации R². Считается по <b>всем</b> значениям всех
 * примеров: 1 — идеальное предсказание, 0 — как константа (среднее),
 * отрицательные значения — хуже среднего.
 *
 * Наказывает за большие ошибки
 */
public class R2 implements Metric {
    private double ssRes;
    private double sumTarget;
    private double sumTargetSq;
    private long n;

    @Override
    public void update(float[] prediction, float[] target) {
        for (int i = 0; i < prediction.length; i++) {
            double d = prediction[i] - target[i];
            ssRes += d * d;
            sumTarget += target[i];
            sumTargetSq += (double) target[i] * target[i];
        }
        n += prediction.length;
    }

    @Override
    public float result() {
        if (n == 0) return 0f;
        double mean = sumTarget / n;
        double ssTot = sumTargetSq - n * mean * mean;
        if (ssTot <= 1e-12) return 0f;
        return (float) (1.0 - ssRes / ssTot);
    }

    @Override public void reset() { ssRes = 0; sumTarget = 0; sumTargetSq = 0; n = 0; }
    @Override public String name() { return "R2"; }
}