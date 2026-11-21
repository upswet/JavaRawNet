package rawnet.code.net.core.metric;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Доля правильных ответов.
 * <ul>
 *   <li>для скалярного выхода (Sigmoid, len == 1) — порог {@code threshold};</li>
 *   <li>для векторного (Softmax, len > 1) — argmax.</li>
 * </ul>
 */
public class Accuracy implements Metric {
    private float threshold;
    private int correct;
    private int total;

    public Accuracy() { this(0.5f); }
    public Accuracy(float threshold) { this.threshold = threshold; }

    @Override
    public void update(float[] prediction, float[] target) {
        if (classify(prediction) == classify(target)) correct++;
        total++;
    }

    private int classify(float[] arr) {
        if (arr.length == 1) return arr[0] >= threshold ? 1 : 0;
        int best = 0;
        for (int i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
        return best;
    }

    @Override public float result() { return total == 0 ? 0f : (float) correct / total; }
    @Override public void reset() { correct = 0; total = 0; }
    @Override public String name() { return "Accuracy"; }

    @Override
    public void save(DataOutputStream out) throws IOException {
        out.writeFloat(threshold);
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        this.threshold = in.readFloat();
    }
}
