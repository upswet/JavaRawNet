package rawnet.code.net.layers.attention;

import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;
import rawnet.code.net.TensorShape;
import rawnet.code.net.core.activate.Activate;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.contract.MaskConsumer;
import rawnet.code.net.core.optimizator.Optimizer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Multi-Head Self-Attention (2D, sequence-формат).
 *
 * <pre>
 *   Q = X·Wq,  K = X·Wk,  V = X·Wv           // [T, dModel] → [T, dModel]
 *   head_i:  Q_i, K_i, V_i — срезы по dModel/heads
 *   attn_i = softmax(Q_i·K_iᵀ / sqrt(dHead) + mask)
 *   out_i  = attn_i·V_i
 *   out    = concat(out_i)·Wo
 * </pre>
 *
 * <h2>Маска</h2>
 * Реализует {@link MaskConsumer}: маска {@code float[T]} (1 — настоящий
 * токен, 0 — PAD). PAD-<b>ключи</b> исключаются из softmax ({@code -inf} в
 * scores), PAD-<b>запросы</b> не обучают {@code Wq} (обнуление {@code dQ}).
 * Опциональная causal-маска задаётся флагом {@code causal}.
 *
 * <p>Порядок в {@code forward}: сначала causal-условие, затем PAD-условие.
 * Если ключ и causal, и PAD — он одинаково не участвует.
 */
@Accessors(chain = true)
public class MultiHeadAttentionLayer extends Layer implements MaskConsumer {

    @RequiredArgsConstructor
    public static class Dto extends Layer.Dto {
        public final int heads;
        public final boolean causal;

        @Override
        public Layer create(TensorShape inputShape) {
            return new MultiHeadAttentionLayer()
                    .setHeads(heads)
                    .setCausal(causal)
                    .setInputShape(inputShape)
                    .setOutputShape(inputShape)
                    .allocateBuffers().initWeights(null);
        }
    }

    @Setter private int heads;
    @Setter private boolean causal;
    @Setter private float dropout = 0f;

    private int T, dModel, dHead;

    public float[][] Wq;
    public float[][] Wk;
    public float[][] Wv;
    public float[][] Wo;
    public float[] bq;
    public float[] bk;
    public float[] bv;
    public float[] bo;
    private float[][] WqG;
    private float[][] WkG;
    private float[][] WvG;
    public float[][] WoG;
    private float[] bqG;
    private float[] bkG;
    private float[] bvG;
    public float[] boG;

    private transient float[] inputRef;
    private float[] Q, K, V;
    private float[] attn;
    private float[] outHeads;
    private float[] output;

    private float[] dQ, dK, dV;
    private float[] dAttn;
    private float[] dOutHeads;
    private float[] prevDelta;

    public MultiHeadAttentionLayer() {}

    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("MultiHeadAttentionLayer.create: формы не заданы");
        if (!inputShape.isSequence())
            throw new IllegalArgumentException("MultiHeadAttentionLayer: ожидался sequence-вход, получено " + inputShape);
        if (heads < 1)
            throw new IllegalArgumentException("MultiHeadAttentionLayer: heads >= 1, получено " + heads);

        this.T      = inputShape.seqLen();
        this.dModel = inputShape.channels();
        if (dModel % heads != 0)
            throw new IllegalArgumentException(
                    "MultiHeadAttentionLayer: dModel (" + dModel + ") % heads (" + heads + ") != 0");
        this.dHead = dModel / heads;

        this.Wq = new float[dModel][dModel]; this.bq = new float[dModel];
        this.Wk = new float[dModel][dModel]; this.bk = new float[dModel];
        this.Wv = new float[dModel][dModel]; this.bv = new float[dModel];
        this.Wo = new float[dModel][dModel]; this.bo = new float[dModel];
        this.WqG = new float[dModel][dModel]; this.bqG = new float[dModel];
        this.WkG = new float[dModel][dModel]; this.bkG = new float[dModel];
        this.WvG = new float[dModel][dModel]; this.bvG = new float[dModel];
        this.WoG = new float[dModel][dModel]; this.boG = new float[dModel];

        int TT = T * T;
        int size = T * dModel;
        this.Q = new float[size];
        this.K = new float[size];
        this.V = new float[size];
        this.attn = new float[heads * TT];
        this.outHeads = new float[size];
        this.output = new float[size];

        this.dQ = new float[size];
        this.dK = new float[size];
        this.dV = new float[size];
        this.dAttn = new float[heads * TT];
        this.dOutHeads = new float[size];
        this.prevDelta = new float[size];
        return this;
    }

    @Override
    public Layer initWeights(Activate activate) {
        float limit = Layer.initLimit(null, dModel, dModel);
        Random rnd = ThreadLocalRandom.current();
        for (int i = 0; i < dModel; i++)
            for (int j = 0; j < dModel; j++) {
                Wq[i][j] = rnd.nextFloat() * 2 * limit - limit;
                Wk[i][j] = rnd.nextFloat() * 2 * limit - limit;
                Wv[i][j] = rnd.nextFloat() * 2 * limit - limit;
                Wo[i][j] = rnd.nextFloat() * 2 * limit - limit;
            }
        return this;
    }

    // ==================== forward ====================

    /** Без маски: все токены считаются настоящими. */
    @Override
    public float[] forward(float[] input, boolean training) {
        return forwardMasked(input, null, training);
    }

    /**
     * Прямой проход с маской.
     *
     * @param input {@code seq(T, dModel)}
     * @param mask  {@code float[T]} (1 — настоящий, 0 — PAD) или {@code null}
     * @param training не используется
     */
    @Override
    public float[] forwardMasked(float[] input, float[] mask, boolean training) {
        if (input.length != inputShape.size())
            throw new IllegalArgumentException(
                    "MultiHeadAttentionLayer.forwardMasked: ожидалось " + inputShape.size() + ", получено " + input.length);
        if (mask != null && mask.length != T)
            throw new IllegalArgumentException(
                    "MultiHeadAttentionLayer.forwardMasked: маска длины " + T + ", получено " + mask.length);
        this.inputRef = input;

        linear(input, Wq, bq, Q);
        linear(input, Wk, bk, K);
        linear(input, Wv, bv, V);

        final float scale = 1f / (float) Math.sqrt(dHead);
        for (int h = 0; h < heads; h++) {
            final int off = h * dHead;
            final int attnOff = h * T * T;
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    if (causal && j > i) {
                        attn[attnOff + i * T + j] = Float.NEGATIVE_INFINITY;
                        continue;
                    }
                    if (mask != null && mask[j] == 0f) {
                        attn[attnOff + i * T + j] = Float.NEGATIVE_INFINITY;
                        continue;
                    }
                    float s = 0f;
                    for (int k = 0; k < dHead; k++)
                        s += Q[i * dModel + off + k] * K[j * dModel + off + k];
                    attn[attnOff + i * T + j] = s * scale;
                }
                softmaxRow(attn, attnOff + i * T, T);
            }
            for (int i = 0; i < T; i++) {
                for (int k = 0; k < dHead; k++) {
                    float s = 0f;
                    for (int j = 0; j < T; j++)
                        s += attn[attnOff + i * T + j] * V[j * dModel + off + k];
                    outHeads[i * dModel + off + k] = s;
                }
            }
        }

        linear(outHeads, Wo, bo, output);
        return output;
    }

    // ==================== backward ====================

    /** Без маски: все токены считаются настоящими. */
    @Override
    public float[] backward(float[] deltaNext) {
        return backwardMasked(deltaNext, null);
    }

    /**
     * Обратный проход с маской.
     *
     * <p>PAD-<b>ключи</b>: {@code dK = dV = 0}. PAD-<b>запросы</b>:
     * {@code dQ = 0} (PAD-запрос не должен обучать {@code Wq}).
     */
    @Override
    public float[] backwardMasked(float[] deltaNext, float[] mask) {
        if (deltaNext.length != outputShape.size())
            throw new IllegalArgumentException(
                    "MultiHeadAttentionLayer.backwardMasked: ожидалось " + outputShape.size() + ", получено " + deltaNext.length);
        if (mask != null && mask.length != T)
            throw new IllegalArgumentException(
                    "MultiHeadAttentionLayer.backwardMasked: маска длины " + T + ", получено " + mask.length);
        if (inputRef == null)
            throw new IllegalStateException("MultiHeadAttentionLayer.backwardMasked: forward не был вызван");

        // dWo, dbo, dOutHeads
        Arrays.fill(boG, 0f);
        for (float[] row : WoG) Arrays.fill(row, 0f);
        Arrays.fill(dOutHeads, 0f);
        for (int i = 0; i < T; i++) {
            for (int o = 0; o < dModel; o++) {
                float d = deltaNext[i * dModel + o];
                boG[o] += d;
                for (int k = 0; k < dModel; k++) {
                    WoG[o][k] += d * outHeads[i * dModel + k];
                    dOutHeads[i * dModel + k] += d * Wo[o][k];
                }
            }
        }

        // dAttn, dV
        Arrays.fill(dAttn, 0f);
        Arrays.fill(dV, 0f);
        for (int h = 0; h < heads; h++) {
            final int off = h * dHead;
            final int attnOff = h * T * T;
            for (int i = 0; i < T; i++) {
                for (int k = 0; k < dHead; k++) {
                    float d = dOutHeads[i * dModel + off + k];
                    for (int j = 0; j < T; j++) {
                        dAttn[attnOff + i * T + j] += d * V[j * dModel + off + k];
                        dV[j * dModel + off + k]      += d * attn[attnOff + i * T + j];
                    }
                }
            }
            // softmax backward
            for (int i = 0; i < T; i++) {
                int row = attnOff + i * T;
                float dot = 0f;
                for (int j = 0; j < T; j++) dot += dAttn[row + j] * attn[row + j];
                for (int j = 0; j < T; j++)
                    dAttn[row + j] = attn[row + j] * (dAttn[row + j] - dot);
            }
        }

        // dQ, dK
        Arrays.fill(dQ, 0f);
        Arrays.fill(dK, 0f);
        final float scale = 1f / (float) Math.sqrt(dHead);
        for (int h = 0; h < heads; h++) {
            final int off = h * dHead;
            final int attnOff = h * T * T;
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    float d = dAttn[attnOff + i * T + j] * scale;
                    if (d == 0f) continue;
                    for (int k = 0; k < dHead; k++) {
                        dQ[i * dModel + off + k] += d * K[j * dModel + off + k];
                        dK[j * dModel + off + k] += d * Q[i * dModel + off + k];
                    }
                }
            }
        }

        // PAD-запросы: dQ = 0; PAD-ключи: dK = dV = 0
        if (mask != null) {
            for (int t = 0; t < T; t++) if (mask[t] == 0f) {
                int off = t * dModel;
                for (int k = 0; k < dModel; k++) {
                    dQ[off + k] = 0f;
                    dK[off + k] = 0f;
                    dV[off + k] = 0f;
                }
            }
        }

        // dWq = Xᵀ·dQ; dbq = sum(dQ); dX += dQ·Wqᵀ; аналогично для K, V
        Arrays.fill(prevDelta, 0f);
        accumLinearGrad(inputRef, dQ, WqG, bqG, Wq, prevDelta, dModel, T);
        accumLinearGrad(inputRef, dK, WkG, bkG, Wk, prevDelta, dModel, T);
        accumLinearGrad(inputRef, dV, WvG, bvG, Wv, prevDelta, dModel, T);
        return prevDelta;
    }

    // --- утилиты ---

    private void linear(float[] x, float[][] W, float[] b, float[] out) {
        for (int i = 0; i < T; i++) {
            for (int o = 0; o < dModel; o++) {
                float s = b[o];
                float[] w = W[o];
                for (int k = 0; k < dModel; k++) s += w[k] * x[i * dModel + k];
                out[i * dModel + o] = s;
            }
        }
    }

    private void accumLinearGrad(float[] x, float[] dOut, float[][] WG, float[] bG,
                                 float[][] W, float[] dPrev, int dIn, int T) {
        for (int i = 0; i < T; i++) {
            for (int o = 0; o < dModel; o++) {
                float d = dOut[i * dModel + o];
                bG[o] += d;
                for (int k = 0; k < dIn; k++) {
                    WG[o][k] += d * x[i * dIn + k];
                    dPrev[i * dIn + k] += d * W[o][k];
                }
            }
        }
    }

    /**
     * Softmax по строке с поддержкой {@code -inf} (PAD-ключи, causal).
     *
     * <p>Если все элементы строки {@code -inf} — заполняет нулями
     * (защита от {@code NaN}: все токены PAD или causal-срез пустой).
     */
    private static void softmaxRow(float[] a, int off, int n) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) if (a[off + i] > max) max = a[off + i];

        if (max == Float.NEGATIVE_INFINITY) {
            Arrays.fill(a, off, off + n, 0f);
            return;
        }

        float sum = 0f;
        for (int i = 0; i < n; i++) {
            if (a[off + i] == Float.NEGATIVE_INFINITY) {
                a[off + i] = 0f;
            } else {
                a[off + i] = (float) Math.exp(a[off + i] - max);
                sum += a[off + i];
            }
        }
        if (sum > 0f) {
            float inv = 1f / sum;
            for (int i = 0; i < n; i++) a[off + i] *= inv;
        }
    }

    // ==================== параметры ====================

    @Override
    public void zeroGrad() {
        for (float[] r : WqG) Arrays.fill(r, 0f); Arrays.fill(bqG, 0f);
        for (float[] r : WkG) Arrays.fill(r, 0f); Arrays.fill(bkG, 0f);
        for (float[] r : WvG) Arrays.fill(r, 0f); Arrays.fill(bvG, 0f);
        for (float[] r : WoG) Arrays.fill(r, 0f); Arrays.fill(boG, 0f);
    }

    @Override
    public void update(Optimizer optimizer, int batchSize) {
        if (batchSize <= 0) throw new IllegalArgumentException("batchSize > 0");
        optimizer.update(Wq, WqG, bq, bqG, batchSize);
        optimizer.update(Wk, WkG, bk, bkG, batchSize);
        optimizer.update(Wv, WvG, bv, bvG, batchSize);
        optimizer.update(Wo, WoG, bo, boG, batchSize);
    }

    @Override
    public double gradNormSq() {
        double s = 0;
        for (float[] r : WqG) for (float v : r) s += v * v;
        for (float v : bqG) s += v * v;
        for (float[] r : WkG) for (float v : r) s += v * v;
        for (float v : bkG) s += v * v;
        for (float[] r : WvG) for (float v : r) s += v * v;
        for (float v : bvG) s += v * v;
        for (float[] r : WoG) for (float v : r) s += v * v;
        for (float v : boG) s += v * v;
        return s;
    }

    @Override
    public void scaleGrad(float scale) {
        for (float[] r : WqG) for (int j = 0; j < r.length; j++) r[j] *= scale;
        for (int i = 0; i < bqG.length; i++) bqG[i] *= scale;
        for (float[] r : WkG) for (int j = 0; j < r.length; j++) r[j] *= scale;
        for (int i = 0; i < bkG.length; i++) bkG[i] *= scale;
        for (float[] r : WvG) for (int j = 0; j < r.length; j++) r[j] *= scale;
        for (int i = 0; i < bvG.length; i++) bvG[i] *= scale;
        for (float[] r : WoG) for (int j = 0; j < r.length; j++) r[j] *= scale;
        for (int i = 0; i < boG.length; i++) boG[i] *= scale;
    }

    // ==================== сериализация ====================

    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);
        out.writeInt(heads);
        out.writeBoolean(causal);
        out.writeFloat(dropout);
        writeMat(out, Wq); writeVec(out, bq);
        writeMat(out, Wk); writeVec(out, bk);
        writeMat(out, Wv); writeVec(out, bv);
        writeMat(out, Wo); writeVec(out, bo);
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);
        this.heads   = in.readInt();
        this.causal  = in.readBoolean();
        this.dropout = in.readFloat();
        allocateBuffers();
        readMat(in, Wq); readVec(in, bq);
        readMat(in, Wk); readVec(in, bk);
        readMat(in, Wv); readVec(in, bv);
        readMat(in, Wo); readVec(in, bo);
    }

    private static void writeMat(DataOutputStream out, float[][] m) throws IOException {
        for (float[] r : m) for (float v : r) out.writeFloat(v);
    }
    private static void writeVec(DataOutputStream out, float[] v) throws IOException {
        for (float x : v) out.writeFloat(x);
    }
    private static void readMat(DataInputStream in, float[][] m) throws IOException {
        for (float[] r : m) for (int j = 0; j < r.length; j++) r[j] = in.readFloat();
    }
    private static void readVec(DataInputStream in, float[] v) throws IOException {
        for (int i = 0; i < v.length; i++) v[i] = in.readFloat();
    }
}