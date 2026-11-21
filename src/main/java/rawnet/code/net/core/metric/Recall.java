package rawnet.code.net.core.metric;

/**Полнота
 *
 * Из всех примеров, которые реально класс c, какую долю сеть нашла?
 *
 * Macro-Recall: среднее TP/(TP+FN) по классам. */
public class Recall extends ClassificationStats {
    public Recall() { this(0.5f); }
    public Recall(float threshold) { super(threshold); }

    @Override
    public float result() {
        float sum = 0f; int cnt = 0;
        for (int c = 0; c < numClasses; c++) {
            int denom = tp[c] + fn[c];
            if (denom == 0) continue;
            sum += (float) tp[c] / denom;
            cnt++;
        }
        return cnt == 0 ? 0f : sum / cnt;
    }

    @Override public String name() { return "Recall(macro)"; }
}
