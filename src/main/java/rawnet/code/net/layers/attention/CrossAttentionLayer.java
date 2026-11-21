package rawnet.code.net.layers.attention;

import lombok.Setter;
import lombok.experimental.Accessors;
import rawnet.code.net.TensorShape;
import rawnet.code.net.core.activate.Activate;
import rawnet.code.net.core.optimizator.Optimizer;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.contract.NInputLayer;
import rawnet.code.net.layers.contract.NInputMaskedLayer;
import rawnet.code.net.model.functional.Functional;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Multi-Head <b>Cross</b>-Attention через контракт {@link NInputLayer}
 * (ровно 2 входа: Q и KV).
 *
 * <p>Функционально <b>идентичен</b> {@link CrossAttentionLayer}:
 * те же формулы, те же матрицы {@code Wq/Wk/Wv/Wo}, та же семантика масок.
 * Отличие — в интерфейсе: {@link #forwardN(float[][], boolean)} /
 * {@link #backwardN(float[])} вместо {@code forward2}/{@code backward2}.
 *
 * <h2>Зачем два класса</h2>
 * <ul>
 *   <li>{@link CrossAttentionLayer} — для существующего кода и
 *       {@code Functional.crossAttention(...)} (контракт
 *       {@code TwoInputLayer}/{@code TwoInputMaskedLayer});</li>
 *       может стоять рядом с {@link NInputAttentionLayer} и другими
 *       {@code NInput*}-слоями в одном графе без смешивания
 *       контрактов.</li>
 * </ul>
 * Оба класса можно использовать одновременно в одном
 * {@link Functional}: маршрутизация
 * в {@code Functional} различает их по типу (NInput* приоритетнее
 * TwoInput*, но конкретный слой реализует только один контракт).
 *
 * <h2>Маски</h2>
 * То же, что у {@link CrossAttentionLayer}:
 * <ul>
 *   <li>PAD-<b>ключи</b> KV исключаются из softmax ({@code -inf});</li>
 *   <li>PAD-<b>запросы</b> Q не обучают {@code Wq} (обнуление {@code dQ});</li>
 *   <li>PAD-<b>ключи/значения</b> KV не обучают {@code Wk}/{@code Wv}
 *       (обнуление {@code dK}/{@code dV}).</li>
 * </ul>
 *
 * <h2>Форма входа/выхода</h2>
 * <ul>
 *   <li>вход 0 (Q):  {@code seq(Tq, dModel)};</li>
 *   <li>вход 1 (KV): {@code seq(Tkv, dModel)}; {@code Tkv == dto.kvLen};</li>
 *   <li>выход: {@code seq(Tq, dModel)} (как у Q).</li>
 * </ul>
 */
@Accessors(chain = true)
public class CrossAttentionLayer extends Layer implements NInputLayer, NInputMaskedLayer {

    /** DTO cross-attention для N-входового контракта. */
    public static class Dto extends Layer.Dto {
        public final int heads;
        public final int dModel;
        public final int kvModel;
        public final int kvLen;

        public Dto(int heads, int dModel, int kvModel, int kvLen) {
            if (heads < 1) throw new IllegalArgumentException("heads >= 1");
            if (dModel < 1) throw new IllegalArgumentException("dModel >= 1");
            if (kvModel != dModel)
                throw new IllegalArgumentException(
                        "CrossAttentionN: dModel Q (" + dModel + ") != dModel KV (" + kvModel + ")");
            if (kvLen < 1) throw new IllegalArgumentException("kvLen >= 1");
            this.heads = heads;
            this.dModel = dModel;
            this.kvModel = kvModel;
            this.kvLen = kvLen;
        }

        @Override
        public Layer create(TensorShape inputShape) {
            if (!inputShape.isSequence())
                throw new IllegalArgumentException(
                        "CrossAttentionN.Dto: ожидался sequence-вход Q, получено " + inputShape);
            if (inputShape.channels() != dModel)
                throw new IllegalArgumentException(
                        "CrossAttentionN.Dto: Q.channels=" + inputShape.channels()
                                + " != dModel=" + dModel);
            return new CrossAttentionLayer()
                    .setHeads(heads)
                    .setKvLen(kvLen)
                    .setInputShape(inputShape)
                    .setOutputShape(inputShape)
                    .allocateBuffers().initWeights(null);
        }
    }

    @Setter private int heads;
    @Setter private int kvLen;
    @Setter private float dropout = 0f;

    private int Tq, Tkv, dModel, dHead;

    public float[][] Wq;
    public float[][] Wk;
    public float[][] Wv;
    public float[][] Wo;
    public float[] bq;
    public float[] bk;
    public float[] bv;
    public float[] bo;
    public float[][] WqG;
    public float[][] WkG;
    public float[][] WvG;
    public float[][] WoG;
    private float[] bqG, bkG, bvG, boG;

    private transient float[] inputQ;
    private transient float[] inputKV;
    private float[] Q, K, V;
    private float[] attn;
    private float[] outHeads;
    private float[] output;

    private float[] dQ, dK, dV;
    private float[] dAttn;
    private float[] dOutHeads;
    private float[] prevDeltaQ;
    private float[] prevDeltaKV;

    /** Последние маски: {maskQ, maskKV}. null — все валидны. */
    private float[][] lastMasks;

    public CrossAttentionLayer() {}

    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("CrossAttentionN.create: формы не заданы");
        if (!inputShape.isSequence())
            throw new IllegalArgumentException("CrossAttentionN: ожидался sequence-вход Q, получено " + inputShape);
        if (heads < 1)
            throw new IllegalArgumentException("CrossAttentionN: heads >= 1, получено " + heads);
        if (kvLen < 1)
            throw new IllegalArgumentException("CrossAttentionN: kvLen >= 1, получено " + kvLen);

        this.Tq     = inputShape.seqLen();
        this.dModel = inputShape.channels();
        this.Tkv    = kvLen;
        if (dModel % heads != 0)
            throw new IllegalArgumentException(
                    "CrossAttentionN: dModel (" + dModel + ") % heads (" + heads + ") != 0");
        this.dHead = dModel / heads;

        this.Wq = new float[dModel][dModel]; this.bq = new float[dModel];
        this.Wk = new float[dModel][dModel]; this.bk = new float[dModel];
        this.Wv = new float[dModel][dModel]; this.bv = new float[dModel];
        this.Wo = new float[dModel][dModel]; this.bo = new float[dModel];
        this.WqG = new float[dModel][dModel]; this.bqG = new float[dModel];
        this.WkG = new float[dModel][dModel]; this.bkG = new float[dModel];
        this.WvG = new float[dModel][dModel]; this.bvG = new float[dModel];
        this.WoG = new float[dModel][dModel]; this.boG = new float[dModel];

        int TqTkv  = Tq * Tkv;
        int sizeQ  = Tq  * dModel;
        int sizeKV = Tkv * dModel;

        this.Q = new float[sizeQ];
        this.K = new float[sizeKV];
        this.V = new float[sizeKV];
        this.attn = new float[heads * TqTkv];
        this.outHeads = new float[sizeQ];
        this.output = new float[sizeQ];

        this.dQ = new float[sizeQ];
        this.dK = new float[sizeKV];
        this.dV = new float[sizeKV];
        this.dAttn = new float[heads * TqTkv];
        this.dOutHeads = new float[sizeQ];
        this.prevDeltaQ = new float[sizeQ];
        this.prevDeltaKV = new float[sizeKV];

        this.lastMasks = new float[2][];
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

    @Override
    public float[] forward(float[] input, boolean training) {
        throw new UnsupportedOperationException(
                "CrossAttentionNLayer требует 2 входа; используйте forwardN(inputs, training)");
    }

    @Override
    public float[] forwardN(float[][] inputs, boolean training) {
        return forwardNMasked(inputs, new float[2][], training);
    }

    @Override
    public float[] forwardNMasked(float[][] inputs, float[][] masks, boolean training) {
        if (inputs == null || inputs.length != 2)
            throw new IllegalArgumentException(
                    "CrossAttentionN.forwardNMasked: ожидалось 2 входа (Q, KV), получено "
                            + (inputs == null ? "null" : inputs.length));
        if (masks == null || masks.length != 2)
            throw new IllegalArgumentException(
                    "CrossAttentionN.forwardNMasked: masks.length должен быть 2");
        float[] q = inputs[0], kv = inputs[1];
        float[] maskQ = masks[0], maskKV = masks[1];
        if (q.length != Tq * dModel)
            throw new IllegalArgumentException(
                    "CrossAttentionN.forwardNMasked: Q ожидалось " + (Tq * dModel) + ", получено " + q.length);
        if (kv.length != Tkv * dModel)
            throw new IllegalArgumentException(
                    "CrossAttentionN.forwardNMasked: KV ожидалось " + (Tkv * dModel) + ", получено " + kv.length);
        if (maskQ != null && maskQ.length != Tq)
            throw new IllegalArgumentException(
                    "CrossAttentionN.forwardNMasked: maskQ длины " + Tq + ", получено " + maskQ.length);
        if (maskKV != null && maskKV.length != Tkv)
            throw new IllegalArgumentException(
                    "CrossAttentionN.forwardNMasked: maskKV длины " + Tkv + ", получено " + maskKV.length);

        this.inputQ  = q;
        this.inputKV = kv;
        this.lastMasks = masks;

        linear(q,  Wq, bq, Q,  Tq,  dModel);
        linear(kv, Wk, bk, K,  Tkv, dModel);
        linear(kv, Wv, bv, V,  Tkv, dModel);

        final float scale = 1f / (float) Math.sqrt(dHead);
        final int TqTkv = Tq * Tkv;
        for (int h = 0; h < heads; h++) {
            final int off = h * dHead;
            final int attnOff = h * TqTkv;
            for (int i = 0; i < Tq; i++) {
                boolean qValid = (maskQ == null) || (maskQ[i] > 0f);
                for (int j = 0; j < Tkv; j++) {
                    boolean kvValid = (maskKV == null) || (maskKV[j] > 0f);
                    if (!qValid || !kvValid) {
                        attn[attnOff + i * Tkv + j] = Float.NEGATIVE_INFINITY;
                        continue;
                    }
                    float s = 0f;
                    for (int k = 0; k < dHead; k++)
                        s += Q[i * dModel + off + k] * K[j * dModel + off + k];
                    attn[attnOff + i * Tkv + j] = s * scale;
                }
                softmaxRowMasked(attn, attnOff + i * Tkv, Tkv);
            }
            for (int i = 0; i < Tq; i++) {
                for (int k = 0; k < dHead; k++) {
                    float s = 0f;
                    for (int j = 0; j < Tkv; j++)
                        s += attn[attnOff + i * Tkv + j] * V[j * dModel + off + k];
                    outHeads[i * dModel + off + k] = s;
                }
            }
        }

        linear(outHeads, Wo, bo, output, Tq, dModel);
        return output;
    }

    // ==================== backward ====================

    @Override
    public float[] backward(float[] deltaNext) {
        throw new UnsupportedOperationException(
                "CrossAttentionNLayer требует 2 входа; используйте backwardN(deltaNext)");
    }

    @Override
    public float[][] backwardN(float[] deltaNext) {
        return backwardNMasked(deltaNext, lastMasks);
    }

    @Override
    public float[][] backwardNMasked(float[] deltaNext, float[][] masks) {
        if (deltaNext.length != Tq * dModel)
            throw new IllegalArgumentException(
                    "CrossAttentionN.backwardNMasked: ожидалось " + (Tq * dModel)
                            + ", получено " + deltaNext.length);
        if (inputQ == null || inputKV == null)
            throw new IllegalStateException("CrossAttentionN.backwardNMasked: forwardNMasked не был вызван");
        if (masks == null) masks = new float[2][];
        float[] maskQ = masks[0], maskKV = masks[1];

        // dWo, dbo, dOutHeads
        Arrays.fill(boG, 0f);
        for (float[] row : WoG) Arrays.fill(row, 0f);
        Arrays.fill(dOutHeads, 0f);
        for (int i = 0; i < Tq; i++) {
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
        final int TqTkv = Tq * Tkv;
        for (int h = 0; h < heads; h++) {
            final int off = h * dHead;
            final int attnOff = h * TqTkv;
            for (int i = 0; i < Tq; i++) {
                for (int k = 0; k < dHead; k++) {
                    float d = dOutHeads[i * dModel + off + k];
                    for (int j = 0; j < Tkv; j++) {
                        dAttn[attnOff + i * Tkv + j] += d * V[j * dModel + off + k];
                        dV[j * dModel + off + k]   += d * attn[attnOff + i * Tkv + j];
                    }
                }
            }
            for (int i = 0; i < Tq; i++) {
                int row = attnOff + i * Tkv;
                float dot = 0f;
                for (int j = 0; j < Tkv; j++) dot += dAttn[row + j] * attn[row + j];
                for (int j = 0; j < Tkv; j++)
                    dAttn[row + j] = attn[row + j] * (dAttn[row + j] - dot);
            }
        }

        // dQ, dK
        Arrays.fill(dQ, 0f);
        Arrays.fill(dK, 0f);
        final float scale = 1f / (float) Math.sqrt(dHead);
        for (int h = 0; h < heads; h++) {
            final int off = h * dHead;
            final int attnOff = h * TqTkv;
            for (int i = 0; i < Tq; i++) {
                for (int j = 0; j < Tkv; j++) {
                    float d = dAttn[attnOff + i * Tkv + j] * scale;
                    if (d == 0f) continue;
                    for (int k = 0; k < dHead; k++) {
                        dQ[i * dModel + off + k] += d * K[j * dModel + off + k];
                        dK[j * dModel + off + k] += d * Q[i * dModel + off + k];
                    }
                }
            }
        }

        // PAD-обнуление
        if (maskQ != null) {
            for (int t = 0; t < Tq; t++) if (maskQ[t] == 0f) {
                int off = t * dModel;
                for (int k = 0; k < dModel; k++) dQ[off + k] = 0f;
            }
        }
        if (maskKV != null) {
            for (int t = 0; t < Tkv; t++) if (maskKV[t] == 0f) {
                int off = t * dModel;
                for (int k = 0; k < dModel; k++) {
                    dK[off + k] = 0f;
                    dV[off + k] = 0f;
                }
            }
        }

        // dWq/dbq/dXq; dWk/dbk/dXkv; dWv/dbv/dXkv
        Arrays.fill(prevDeltaQ, 0f);
        Arrays.fill(prevDeltaKV, 0f);
        accumLinearGrad(inputQ,  dQ, WqG, bqG, Wq, prevDeltaQ,  dModel, Tq);
        accumLinearGrad(inputKV, dK, WkG, bkG, Wk, prevDeltaKV, dModel, Tkv);
        accumLinearGrad(inputKV, dV, WvG, bvG, Wv, prevDeltaKV, dModel, Tkv);

        return new float[][]{ prevDeltaQ, prevDeltaKV };
    }

    // ==================== утилиты ====================

    private void linear(float[] x, float[][] W, float[] b, float[] out, int T, int d) {
        for (int i = 0; i < T; i++) {
            for (int o = 0; o < d; o++) {
                float s = b[o];
                float[] w = W[o];
                for (int k = 0; k < d; k++) s += w[k] * x[i * d + k];
                out[i * d + o] = s;
            }
        }
    }

    private void accumLinearGrad(float[] x, float[] dOut, float[][] WG, float[] bG,
                                 float[][] W, float[] dPrev, int d, int T) {
        for (int i = 0; i < T; i++) {
            for (int o = 0; o < d; o++) {
                float dd = dOut[i * d + o];
                bG[o] += dd;
                for (int k = 0; k < d; k++) {
                    WG[o][k] += dd * x[i * d + k];
                    dPrev[i * d + k] += dd * W[o][k];
                }
            }
        }
    }

    private static void softmaxRowMasked(float[] a, int off, int n) {
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
        out.writeInt(kvLen);
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
        this.kvLen   = in.readInt();
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