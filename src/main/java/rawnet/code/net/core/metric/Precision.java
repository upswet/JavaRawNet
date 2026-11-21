package rawnet.code.net.core.metric;

/**Точность.
 * Насколько можно доверять положительному ответу модели?
 *
 * Macro-Precision: среднее TP/(TP+FP) по классам (нулевые — пропускаются). */
public class Precision extends ClassificationStats {
    public Precision() { this(0.5f); }
    public Precision(float threshold) { super(threshold); }

    @Override
    public float result() {
        float sum = 0f; int cnt = 0;
        for (int c = 0; c < numClasses; c++) {
            int denom = tp[c] + fp[c];
            if (denom == 0) continue;
            sum += (float) tp[c] / denom;
            cnt++;
        }
        return cnt == 0 ? 0f : sum / cnt;
    }

    @Override public String name() { return "Precision(macro)"; }
}

