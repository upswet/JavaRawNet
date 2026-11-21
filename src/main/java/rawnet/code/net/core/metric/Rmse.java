package rawnet.code.net.core.metric;

/** Корень из средней квадратичной ошибки (RMSE). */
public class Rmse implements Metric {
    private double sum;
    private long n;

    @Override
    public void update(float[] prediction, float[] target) {
        for (int i = 0; i < prediction.length; i++) {
            double d = prediction[i] - target[i];
            sum += d * d;
        }
        n += prediction.length;
    }

    @Override public float result() { return n == 0 ? 0f : (float) Math.sqrt(sum / n); }
    @Override public void reset() { sum = 0; n = 0; }
    @Override public String name() { return "RMSE"; }
}