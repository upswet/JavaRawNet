package rawnet.code.net.core.optimizator;

import lombok.NoArgsConstructor;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.IdentityHashMap;
import java.util.Map;

/**
 * SGD с импульсом (Momentum).
 * <p>
 * Кроме текущего градиента хранит "скорость" — экспоненциально
 * сглаженную историю градиентов:
 * <pre>
 *   v = momentum * v + grad / batchSize
 *   param = param - lr * v
 * </pre>
 * Импульс помогает быстрее проходить вдоль устойчивых направлений
 * и гасить колебания в поперечных направлениях.
 */
@NoArgsConstructor
public class Momentum implements Optimizer {
    /** Скорость обучения. */
    private float lr;

    /** Коэффициент импульса, обычно 0.9. */
    private float momentum;

    public Momentum(float lr, float momentum) { this.lr = lr; this.momentum = momentum; }

    /**
     * Скорости для весов. Ключ — сам массив весов, поэтому используется
     * IdentityHashMap: состояние привязано к конкретному объекту-параметру.
     */
    private final Map<float[][], float[][]> weightVelocity = new IdentityHashMap<>();
    /** Скорости для смещений. */
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
                weights[i][j] -= lr * v;
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
            biases[i] -= lr * vb;
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
