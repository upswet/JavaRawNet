package rawnet.code.net.core.optimizator;

import lombok.NoArgsConstructor;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Стохастический градиентный спуск (SGD).
 * <p>
 * Простейший оптимизатор: параметр сдвигается в сторону,
 * противоположную усреднённому градиенту:
 * <pre>
 *   param = param - lr * grad / batchSize
 * </pre>
 * Состояния между шагами не хранит.
 */
@NoArgsConstructor
public class SGD implements Optimizer {
    private float lr;

    public SGD(float lr) { this.lr = lr; }

    @Override
    public void updateWeights(float[][] weights, float[][] weightGrad, int batchSize) {
        float scale = lr / batchSize;
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] -= scale * weightGrad[i][j];
            }
        }
    }

    @Override
    public void updateBiases(float[] biases, float[] biasGrad, int batchSize) {
        float scale = lr / batchSize;
        for (int i = 0; i < biases.length; i++) {
            biases[i] -= scale * biasGrad[i];
        }
    }

    @Override
    public void save(DataOutputStream out) throws IOException {
        out.writeFloat(lr);
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        this.lr = in.readFloat();
    }
}
