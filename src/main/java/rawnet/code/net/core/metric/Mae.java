package rawnet.code.net.core.metric;

import lombok.NoArgsConstructor;

/** Средняя абсолютная ошибка (MAE). */
@NoArgsConstructor
public class Mae implements Metric {
    private double sum;
    private long n;

    @Override
    public void update(float[] prediction, float[] target) {
        for (int i = 0; i < prediction.length; i++)
            sum += Math.abs(prediction[i] - target[i]);
        n += prediction.length;
    }

    @Override public float result() { return n == 0 ? 0f : (float) (sum / n); }
    @Override public void reset() { sum = 0; n = 0; }
    @Override public String name() { return "MAE"; }
}
