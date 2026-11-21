package rawnet.code.net.core.metric;

public class TokenAccuracy implements Metric, MaskAcceptMetric {

    private long correct = 0;
    private long total = 0;

    @Override
    public void update(float[] prediction, float[] target) {
        updateMasked(prediction, target, null);
    }

    @Override
    public void updateMasked(float[] prediction, float[] target, float[] mask) {
        int T = target.length;
        if (prediction.length % T != 0)
            throw new IllegalArgumentException(
                    "TokenAccuracy: prediction.length=" + prediction.length
                            + " не кратно T=" + T);
        boolean useMask = mask != null && mask.length > 0;
        if (useMask && mask.length != T)
            throw new IllegalArgumentException(
                    "TokenAccuracy: mask.length=" + mask.length + " != T=" + T);
        int vocab = prediction.length / T;

        for (int t = 0; t < T; t++) {
            if (useMask && mask[t] <= 0f) continue;

            int off = t * vocab;
            int best = 0;
            for (int v = 1; v < vocab; v++)
                if (prediction[off + v] > prediction[off + best]) best = v;

            int tgt = Math.round(target[t]);
            if (best == tgt) correct++;
            total++;
        }
    }

    @Override public float result() { return total == 0 ? 0f : (float) correct / total; }
    @Override public void reset()  { correct = 0; total = 0; }
    @Override public String name() { return "TokenAccuracy"; }
}