package rawnet.code.net.layers.attention;   // или layers.core

import rawnet.code.net.TensorShape;
import rawnet.code.net.core.activate.Activate;
import rawnet.code.net.core.optimizator.Optimizer;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.contract.NInputLayer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Слой negative sampling для word2vec-подобных эмбеддингов.
 *
 * <p>Принимает <b>два</b> входа одинаковой длины {@code dModel}:
 * <ul>
 *   <li>{@code ctx} — вектор контекста (из {@code E_in});</li>
 *   <li>{@code tgt} — вектор target-слова (из {@code E_out}).</li>
 * </ul>
 * Возвращает <b>одно число</b>: {@code p = sigmoid(<ctx, tgt>)} —
 * вероятность, что пара «настоящая». Loss (BCE/Focal) — снаружи.
 *
 * <p><b>Зачем отдельный слой.</b> {@code Dense + Sigmoid + BCE} не даёт
 * word2vec-градиент: {@code Dense} смешивает все эмбеддинги контекста
 * одной матрицей, а не считает {@code <ctx, tgt>} — скалярное
 * произведение <b>двух</b> обучаемых векторов. Здесь — ровно
 * {@code <ctx, tgt>}, как в word2vec.
 *
 * <p><b>Два EmbeddingLayer.</b> Слой сам матриц не хранит — он получает
 * готовые векторы. Значит, в сети нужны <b>два</b> {@code EmbeddingLayer}:
 * один для контекста (обучает {@code E_in}), второй для target (обучает
 * {@code E_out}). Оба — обычные {@code EmbeddingLayer.Dto(vocab, dModel)},
 * стоят рядом, соединены через {@code NInputLayer}.
 */
public class NegativeSamplingLayer extends Layer implements NInputLayer {

    public static class Dto extends Layer.Dto {
        public final int dModel;

        public Dto(int dModel) {
            if (dModel < 1) throw new IllegalArgumentException("dModel >= 1");
            this.dModel = dModel;
        }

        @Override
        public Layer create(TensorShape inputShape) {
            if (inputShape.size() != dModel)
                throw new IllegalArgumentException(
                        "NegativeSampling.Dto: ожидался flat(" + dModel + "), получено " + inputShape);
            return new NegativeSamplingLayer()
                    .setInputShape(inputShape)
                    .setOutputShape(TensorShape.flat(1))
                    .allocateBuffers();
        }
    }

    private int dModel;
    private float[] ctxRef, tgtRef;
    private float[] output = new float[1];
    private float[] prevDelta;          // dL/dInput — но у N входов, поэтому не используется

    public NegativeSamplingLayer() {}

    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("NegativeSampling.create: формы не заданы");
        this.dModel = inputShape.size();
        this.output = new float[1];
        return this;
    }

    @Override
    public Layer initWeights(Activate activate) { return this; }   // параметров нет

    // ---------- N входов ----------

    @Override
    public float[] forward(float[] input, boolean training) {
        throw new UnsupportedOperationException(
                "NegativeSamplingLayer требует 2 входа; используйте forwardN(...)");
    }

    @Override
    public float[] forwardN(float[][] inputs, boolean training) {
        if (inputs.length != 2)
            throw new IllegalArgumentException(
                    "NegativeSampling.forwardN: ожидалось 2 входа (ctx, tgt), получено " + inputs.length);
        if (inputs[0].length != dModel || inputs[1].length != dModel)
            throw new IllegalArgumentException(
                    "NegativeSampling.forwardN: каждый вход длины " + dModel);
        this.ctxRef = inputs[0];
        this.tgtRef = inputs[1];
        float s = 0f;
        for (int i = 0; i < dModel; i++) s += ctxRef[i] * tgtRef[i];
        output[0] = sigmoid(s);
        return output;
    }

    @Override
    public float[] backward(float[] deltaNext) {
        throw new UnsupportedOperationException(
                "NegativeSamplingLayer требует 2 входа; используйте backwardN(...)");
    }

    /**
     * @param deltaNext {@code dL/dp} — градиент по вероятности (от BCE)
     * @return {@code {dL/dctx, dL/dtgt}}
     */
    @Override
    public float[][] backwardN(float[] deltaNext) {
        if (deltaNext.length != 1)
            throw new IllegalArgumentException(
                    "NegativeSampling.backwardN: ожидалось 1, получено " + deltaNext.length);
        float p = output[0];
        float dScore = deltaNext[0] * p * (1f - p);   // dL/ds = dL/dp · σ'(s)
        float[] dCtx = new float[dModel];
        float[] dTgt = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            dCtx[i] = dScore * tgtRef[i];
            dTgt[i] = dScore * ctxRef[i];
        }
        return new float[][]{ dCtx, dTgt };
    }

    // ---------- параметры (нет) ----------

    @Override public void zeroGrad() {}
    @Override public void update(Optimizer optimizer, int batchSize) {}
    @Override public double gradNormSq() { return 0.0; }
    @Override public void scaleGrad(float scale) {}

    // ---------- сериализация ----------

    @Override
    public void save(DataOutputStream out) throws IOException { super.save(out); }

    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);
        allocateBuffers();
    }

    private static float sigmoid(float x) {
        if (x >= 0f) { float e = (float) Math.exp(-x); return 1f / (1f + e); }
        else          { float e = (float) Math.exp(x);  return e / (1f + e); }
    }
}