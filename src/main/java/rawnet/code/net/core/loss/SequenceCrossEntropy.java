package rawnet.code.net.core.loss;

public class SequenceCrossEntropy implements Loss, MaskAcceptLoss {

    private static final float EPS = 1e-15f;

    @Override
    public float loss(float[] logits, float[] target) {
        return lossMasked(logits, target, null);
    }

    @Override
    public void derivative(float[] logits, float[] target, float[] deriv) {
        derivativeMasked(logits, target, deriv, null);
    }

    @Override
    public float lossMasked(float[] logits, float[] target, float[] mask) {
        int T = target.length;
        if (logits.length % T != 0)
            throw new IllegalArgumentException(
                    "SequenceCrossEntropy: logits.length=" + logits.length
                            + " не кратно T=" + T);
        int vocab = logits.length / T;
        boolean useMask = mask != null && mask.length > 0;
        if (useMask && mask.length != T)
            throw new IllegalArgumentException(
                    "SequenceCrossEntropy: mask.length=" + mask.length + " != T=" + T);

        double total = 0.0;
        int valid = 0;
        for (int t = 0; t < T; t++) {
            if (useMask && mask[t] <= 0f) continue;

            int off = t * vocab;
            int tgt = Math.round(target[t]);
            if (tgt < 0 || tgt >= vocab)
                throw new IllegalArgumentException(
                        "SequenceCrossEntropy: target[" + t + "]=" + target[t]
                                + " вне [0, " + vocab + ")");

            float max = logits[off];
            for (int v = 1; v < vocab; v++)
                if (logits[off + v] > max) max = logits[off + v];

            double sumExp = 0.0;
            for (int v = 0; v < vocab; v++)
                sumExp += Math.exp(logits[off + v] - max);
            double logZ = Math.log(sumExp) + max;

            total += -logits[off + tgt] + logZ;
            valid++;
        }

        if (valid == 0) return 0f;
        return (float) (total / valid);
    }

    @Override
    public void derivativeMasked(float[] logits, float[] target, float[] deriv, float[] mask) {
        int T = target.length;
        if (logits.length % T != 0)
            throw new IllegalArgumentException(
                    "SequenceCrossEntropy: logits.length=" + logits.length
                            + " не кратно T=" + T);
        if (deriv.length != logits.length)
            throw new IllegalArgumentException(
                    "SequenceCrossEntropy: deriv.length=" + deriv.length
                            + " != logits.length=" + logits.length);
        int vocab = logits.length / T;
        boolean useMask = mask != null && mask.length > 0;
        if (useMask && mask.length != T)
            throw new IllegalArgumentException(
                    "SequenceCrossEntropy: mask.length=" + mask.length + " != T=" + T);

        int valid = 0;
        for (int t = 0; t < T; t++)
            if (!useMask || mask[t] > 0f) valid++;
        float invValid = valid > 0 ? 1f / valid : 0f;

        for (int t = 0; t < T; t++) {
            int off = t * vocab;
            if (useMask && mask[t] <= 0f) {
                for (int v = 0; v < vocab; v++) deriv[off + v] = 0f;
                continue;
            }

            int tgt = Math.round(target[t]);
            if (tgt < 0 || tgt >= vocab)
                throw new IllegalArgumentException(
                        "SequenceCrossEntropy: target[" + t + "]=" + target[t]
                                + " вне [0, " + vocab + ")");

            float max = logits[off];
            for (int v = 1; v < vocab; v++)
                if (logits[off + v] > max) max = logits[off + v];

            double sumExp = 0.0;
            for (int v = 0; v < vocab; v++)
                sumExp += Math.exp(logits[off + v] - max);
            double inv = 1.0 / sumExp;

            for (int v = 0; v < vocab; v++) {
                float soft = (float) (Math.exp(logits[off + v] - max) * inv);
                float onehot = (v == tgt) ? 1f : 0f;
                deriv[off + v] = invValid * (soft - onehot);
            }
        }
    }
}