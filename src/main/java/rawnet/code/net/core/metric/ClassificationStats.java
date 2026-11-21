package rawnet.code.net.core.metric;

import lombok.NoArgsConstructor;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;

/**
 * Общая база для классификационных метрик (Precision / Recall / F1).
 * Накапливает TP/FP/FN по каждому классу; для скалярного выхода
 * (len == 1) автоматически считает его бинарным с двумя классами.
 */
@NoArgsConstructor
public abstract class ClassificationStats implements Metric {
    protected float threshold;
    protected int numClasses = -1;
    protected int[] tp, fp, fn;

    protected ClassificationStats(float threshold) { this.threshold = threshold; }

    protected void ensureCapacity(int n) {
        if (numClasses >= n) return;
        int old = Math.max(numClasses, 0);
        numClasses = n;
        int[] ntp = new int[n], nfp = new int[n], nfn = new int[n];
        if (tp != null) {
            System.arraycopy(tp, 0, ntp, 0, old);
            System.arraycopy(fp, 0, nfp, 0, old);
            System.arraycopy(fn, 0, nfn, 0, old);
        }
        tp = ntp; fp = nfp; fn = nfn;
    }

    @Override
    public void update(float[] prediction, float[] target) {
        int n = Math.max(numClassesOf(prediction), numClassesOf(target));
        ensureCapacity(n);

        int pc = classify(prediction);
        int tc = classify(target);
        if (pc == tc) {
            tp[pc]++;
        } else {
            fp[pc]++;
            fn[tc]++;
        }
    }

    protected int numClassesOf(float[] arr) { return arr.length == 1 ? 2 : arr.length; }

    protected int classify(float[] arr) {
        if (arr.length == 1) return arr[0] >= threshold ? 1 : 0;
        int best = 0;
        for (int i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
        return best;
    }

    @Override
    public void reset() {
        if (tp != null) {
            Arrays.fill(tp, 0);
            Arrays.fill(fp, 0);
            Arrays.fill(fn, 0);
        }
    }

    @Override
    public void save(DataOutputStream out) throws IOException {
        out.writeFloat(threshold);
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        this.threshold = in.readFloat();
    }
}
