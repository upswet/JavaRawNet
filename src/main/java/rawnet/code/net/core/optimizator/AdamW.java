package rawnet.code.net.core.optimizator;

import lombok.Getter;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.IdentityHashMap;
import java.util.Map;

/**
 * AdamW — Adam с раздельным (decoupled) weight decay.
 * <p>
 * Хранит первые и вторые моменты градиентов:
 * <pre>
 *   m = beta1 * m + (1 - beta1) * g
 *   v = beta2 * v + (1 - beta2) * g^2
 * </pre>
 * Затем применяет bias correction:
 * <pre>
 *   mHat = m / (1 - beta1^t)
 *   vHat = v / (1 - beta2^t)
 *   update = mHat / (sqrt(vHat) + eps)
 *   w = w - lr * (update + weightDecay * w)
 * </pre>
 * Weight decay применяется только к весам и не применяется к bias.
 */
public class AdamW implements Optimizer {
    @Getter private float lr;     //Скорость обучения
    @Getter private float beta1;  //Экспоненциальное затухание для первого момента
    @Getter private float beta2;  //Экспоненциальное затухание для второго момента.
    @Getter private float eps;    //Малая константа для численной устойчивости
    @Getter private float weightDecay; //Коэффициент L2-штрафа (weight decay).  Способ регуляризации модели, который «наказывает» её за слишком большие значения весов. Его цель — не дать модели переобучиться, сделав её более устойчивой к шуму в данных.

    private final Map<float[][], float[][]> mW = new IdentityHashMap<>();   //Первые моменты для весов
    private final Map<float[][], float[][]> vW = new IdentityHashMap<>();   //Вторые моменты для весов
    private final Map<float[], float[]> mB = new IdentityHashMap<>();       //Первые моменты для смещений
    private final Map<float[], float[]> vB = new IdentityHashMap<>();       //Вторые моменты для смещений
    private final Map<Object, Integer> steps = new IdentityHashMap<>();     //Номера шагов для bias correction; ключ — массив параметров

    private final Map<float[][], Integer> weightSteps = new IdentityHashMap<>();
    private final Map<float[], Integer> biasSteps = new IdentityHashMap<>();

    /** Конструктор без аргументов — для {@code Persistables.load} (рефлексия) и для {@code load(DataInputStream)}. */
    public AdamW() {
        this(0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
    }

    /** Только learning rate, остальные — стандартные дефолты AdamW. */
    public AdamW(float lr) {
        this(lr, 0.9f, 0.999f, 1e-8f, 0.01f);
    }

    /** Полный набор гиперпараметров. */
    public AdamW(float lr, float beta1, float beta2, float eps, float weightDecay) {
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
    }

    @Override
    public void updateWeights(float[][] weights, float[][] weightGrad, int batchSize) {
        float scale = 1f / batchSize;

        float[][] mw = mW.computeIfAbsent(
                weights, k -> new float[weights.length][weights[0].length]);
        float[][] vw = vW.computeIfAbsent(
                weights, k -> new float[weights.length][weights[0].length]);
        int stepW = weightSteps.merge(weights, 1, Integer::sum);
        float bc1W = (float) (1 - Math.pow(beta1, stepW));
        float bc2W = (float) (1 - Math.pow(beta2, stepW));

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                float g = weightGrad[i][j] * scale;

                mw[i][j] = beta1 * mw[i][j] + (1 - beta1) * g;
                vw[i][j] = beta2 * vw[i][j] + (1 - beta2) * g * g;

                float mHat = mw[i][j] / bc1W;
                float vHat = vw[i][j] / bc2W;
                float update = mHat / (float) (Math.sqrt(vHat) + eps);

                weights[i][j] -= lr * (update + weightDecay * weights[i][j]);
            }
        }
    }

    @Override
    public void updateBiases(float[] biases, float[] biasGrad, int batchSize) {
        float scale = 1f / batchSize;

        float[] mb = mB.computeIfAbsent(biases, k -> new float[biases.length]);
        float[] vb = vB.computeIfAbsent(biases, k -> new float[biases.length]);
        int stepB = biasSteps.merge(biases, 1, Integer::sum);
        float bc1B = (float) (1 - Math.pow(beta1, stepB));
        float bc2B = (float) (1 - Math.pow(beta2, stepB));

        for (int i = 0; i < biases.length; i++) {
            float gb = biasGrad[i] * scale;

            mb[i] = beta1 * mb[i] + (1 - beta1) * gb;
            vb[i] = beta2 * vb[i] + (1 - beta2) * gb * gb;

            float mHatB = mb[i] / bc1B;
            float vHatB = vb[i] / bc2B;
            float updateB = mHatB / (float) (Math.sqrt(vHatB) + eps);

            biases[i] -= lr * updateB;
        }
    }

    @Override
    public void save(DataOutputStream out) throws IOException {
        out.writeFloat(lr);
        out.writeFloat(beta1);
        out.writeFloat(beta2);
        out.writeFloat(eps);
        out.writeFloat(weightDecay);
        // моменты m/v и step НЕ сохраняются — см. javadoc
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        this.lr          = in.readFloat();
        this.beta1       = in.readFloat();
        this.beta2       = in.readFloat();
        this.eps         = in.readFloat();
        this.weightDecay = in.readFloat();
    }
}
