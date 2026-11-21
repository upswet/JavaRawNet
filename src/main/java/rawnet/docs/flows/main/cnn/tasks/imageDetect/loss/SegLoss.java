package rawnet.docs.flows.main.cnn.tasks.imageDetect.loss;

import rawnet.code.net.core.loss.Loss;
import rawnet.code.net.core.loss.SoftmaxCrossEntropy;

//SegLoss — CE по пикселям под SEMSEG
//Формат target: flat(C·S·S), one-hot на пиксель, фон = все нули. Loss = softmax-CE по пикселю.

/**
 * Попопиксельная кросс-энтропия для
 *
 * <p>Формат: {@code flat(C*S*S)}, one-hot на пиксель, фон — все нули.
 * В отличие от {@link SoftmaxCrossEntropy},
 * работает с <b>канальной</b> раскладкой {@code [c*S*S + y*S + x]} и
 * считает softmax по каналам в каждом пикселе.
 *
 * <p>Ожидает на входе логиты (выход Dense без активации).
 */
public class SegLoss implements Loss {
    private final int size;
    private final int numClasses;

    public SegLoss(int size, int numClasses) {
        if (size < 1)       throw new IllegalArgumentException("size >= 1");
        if (numClasses < 1) throw new IllegalArgumentException("numClasses >= 1");
        this.size = size;
        this.numClasses = numClasses;
    }

    @Override
    public float loss(float[] p, float[] t) {
        final int n = size * size;
        double sum = 0;
        for (int i = 0; i < n; i++) {
            // softmax по каналам
            float max = p[i];
            for (int c = 1; c < numClasses; c++) {
                float v = p[c * n + i];
                if (v > max) max = v;
            }
            double z = 0;
            for (int c = 0; c < numClasses; c++)
                z += Math.exp(p[c * n + i] - max);
            double logZ = Math.log(z) + max;

            double tSum = 0, tz = 0;
            for (int c = 0; c < numClasses; c++) {
                double tc = t[c * n + i];
                tSum += tc;
                tz   += tc * p[c * n + i];
            }
            // Если пиксель — фон (tSum==0), CE не вносит вклада;
            // для «фонового» класса можно завести отдельный канал.
            sum += logZ * tSum - tz;
        }
        return (float) (sum / n);
    }

    @Override
    public void derivative(float[] p, float[] t, float[] d) {
        final int n = size * size;
        final float invN = 1f / n;
        for (int i = 0; i < n; i++) {
            float max = p[i];
            for (int c = 1; c < numClasses; c++) {
                float v = p[c * n + i];
                if (v > max) max = v;
            }
            double z = 0;
            for (int c = 0; c < numClasses; c++)
                z += Math.exp(p[c * n + i] - max);
            double invZ = 1.0 / z;
            for (int c = 0; c < numClasses; c++) {
                double sm = Math.exp(p[c * n + i] - max) * invZ;
                d[c * n + i] = (float) ((sm - t[c * n + i]) * invN);
            }
        }
    }
}
