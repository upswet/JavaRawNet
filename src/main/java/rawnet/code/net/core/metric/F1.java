package rawnet.code.net.core.metric;

/**гармоническое среднее Precision и Recall — компромисс, который наказывает перекос в любую сторону. Если P = 1, R = 0.1, то F1 ≈ 0.18, а не 0.55 (как было бы у арифметического среднего)
 *
 * Macro-F1: гармоническое среднее Precision и Recall, усреднённое по классам. */
public class F1 extends ClassificationStats {
    public F1() { this(0.5f); }
    public F1(float threshold) { super(threshold); }

    @Override
    public float result() {
        float sum = 0f; int cnt = 0;
        for (int c = 0; c < numClasses; c++) {
            int pDenom = tp[c] + fp[c];
            int rDenom = tp[c] + fn[c];
            if (pDenom == 0 || rDenom == 0) continue;
            float p = (float) tp[c] / pDenom;
            float r = (float) tp[c] / rDenom;
            if (p + r == 0) continue;
            sum += 2 * p * r / (p + r);
            cnt++;
        }
        return cnt == 0 ? 0f : sum / cnt;
    }

    @Override public String name() { return "F1(macro)"; }
}
