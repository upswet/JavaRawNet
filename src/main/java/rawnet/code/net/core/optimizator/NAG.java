package rawnet.code.net.core.optimizator;

import lombok.NoArgsConstructor;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.IdentityHashMap;
import java.util.Map;

/**
        * Nesterov Accelerated Gradient (NAG).
        * <p>
 * Модификация Momentum с "взглядом вперёд": сначала делается шаг по
 * накопленной скорости, а градиент как бы берётся в точке будущего положения.
        * В используемой форме обновление выглядит так:
        * <pre>
 *   g = grad / batchSize
 *   v = momentum * v + g
 *   d = g + momentum * v
 *   param = param - lr * d
 * </pre>
        * Это даёт более раннее торможение при смене направления по сравнению
 * с обычным Momentum.
 */
@NoArgsConstructor
public class NAG implements Optimizer {
    private float lr;
    private float momentum;

    public NAG(float lr, float momentum) { this.lr = lr; this.momentum = momentum; }

    private final Map<float[][], float[][]> weightVelocity = new IdentityHashMap<>();
    private final Map<float[], float[]> biasVelocity = new IdentityHashMap<>();

    @Override
    public void updateWeights(float[][] weights, float[][] weightGrad, int batchSize) {
        float scale = 1f / batchSize;
        float[][] wv = weightVelocity.computeIfAbsent(
                weights, k -> new float[weights.length][weights[0].length]);

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                float g = weightGrad[i][j] * scale;
                float v = momentum * wv[i][j] + g;
                wv[i][j] = v;
                float d = g + momentum * v;
                weights[i][j] -= lr * d;
            }
        }
    }

    @Override
    public void updateBiases(float[] biases, float[] biasGrad, int batchSize) {
        float scale = 1f / batchSize;
        float[] bv = biasVelocity.computeIfAbsent(biases, k -> new float[biases.length]);

        for (int i = 0; i < biases.length; i++) {
            float gb = biasGrad[i] * scale;
            float vb = momentum * bv[i] + gb;
            bv[i] = vb;
            float db = gb + momentum * vb;
            biases[i] -= lr * db;
        }
    }

    @Override
    public void save(DataOutputStream out) throws IOException {
        out.writeFloat(lr);
        out.writeFloat(momentum);
        // состояние velocity НЕ сохраняется — см. javadoc
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        this.lr       = in.readFloat();
        this.momentum = in.readFloat();
        // velocity начнётся с нулей
    }
}
