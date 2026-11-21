package rawnet.code.net.layers.attention;

import lombok.Setter;
import lombok.experimental.Accessors;
import rawnet.code.net.TensorShape;
import rawnet.code.net.core.activate.Activate;
import rawnet.code.net.core.optimizator.Optimizer;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.contract.NInputLayer;
import rawnet.code.net.layers.contract.NInputMaskedLayer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Multi-Head Attention, принимающий <b>Q, K, V отдельными входами</b>.
 *
 * <pre>
 *   вход 0: Q — seq(Tq,  dModel)   (уже спроецирован снаружи)
 *   вход 1: K — seq(Tkv, dModel)
 *   вход 2: V — seq(Tkv, dModel)
 *
 *   head_h:  attn_h = softmax(Q_h·K_hᵀ / √dHead + mask)
 *            out_h  = attn_h·V_h
 *   out = concat(out_h)·Wo + bo
 * </pre>
 *
 * <p>Простыми словами: «голый» scaled dot-product attention. В отличие
 * от {@link MultiHeadAttentionLayer}, где Q/K/V считаются из одного X
 * внутри слоя, здесь Q, K, V — <b>готовые</b> последовательности,
 * поданные снаружи. Это позволяет:
 * <ul>
 *   <li>делать разные проекции для Q/K/V (например, разные Dense-слои
 *       перед attention — стандарт в некоторых реализациях Transformer);</li>
 *   <li>подавать K и V из другого источника, чем Q (cross-attention
 *       без отдельного класса);</li>
 *   <li>переиспользовать K/V между несколькими Q (multi-query
 *       attention, если K/V посчитаны один раз).</li>
 * </ul>
 *
 * <h2>Отличия от {@link MultiHeadAttentionLayer}</h2>
 * <ul>
 *   <li>{@link MultiHeadAttentionLayer} — 1 вход, внутри {@code Q=X·Wq},
 *       {@code K=X·Wk}, {@code V=X·Wv};</li>
 *   <li>{@code QKVAttentionLayer} — 3 входа, внутри <b>только</b>
 *       attention и выходная проекция {@code Wo}; {@code Wq/Wk/Wv}
 *       отсутствуют (проекции — снаружи).</li>
 * </ul>
 *
 * <h2>Маски</h2>
 * <ul>
 *   <li>PAD-<b>ключи</b> K исключаются из softmax ({@code -inf});</li>
 *   <li>PAD-<b>запросы</b> Q не участвуют в выходе (обнуление {@code dQ});</li>
 *   <li>PAD-<b>значения</b> V не вносят вклад в выход (обнуление {@code dV}
 *       для PAD-позиций).</li>
 * </ul>
 * Маска K и V обычно совпадает (это одна и та же последовательность
 * {@code Tkv}), но контракт позволяет разные — слой использует
 * {@code maskK[j]} для исключения из softmax и {@code maskV[j]} для
 * обнуления {@code dV[j]}.
 *
 * <h2>Форма входа/выхода</h2>
 * <ul>
 *   <li>Q: {@code seq(Tq, dModel)};</li>
 *   <li>K: {@code seq(Tkv, dModel)};</li>
 *   <li>V: {@code seq(Tkv, dModel)};</li>
 *   <li>выход: {@code seq(Tq, dModel)}.</li>
 * </ul>
 */
@Accessors(chain = true)
public class QKVAttentionLayer extends Layer implements NInputLayer, NInputMaskedLayer {

    /**
     * DTO QKV-attention.
     */
    public static class Dto extends Layer.Dto {
        public final int heads;
        public final int dModel;
        public final int kvLen;
        public final boolean causal;
        public final float dropout;

        public Dto(int heads, int dModel, int kvLen) {
            this(heads, dModel, kvLen, false, 0f);
        }

        public Dto(int heads, int dModel, int kvLen, boolean causal, float dropout) {
            if (heads < 1) throw new IllegalArgumentException("heads >= 1");
            if (dModel < 1) throw new IllegalArgumentException("dModel >= 1");
            if (dModel % heads != 0)
                throw new IllegalArgumentException(
                        "dModel (" + dModel + ") % heads (" + heads + ") != 0");
            if (kvLen < 1) throw new IllegalArgumentException("kvLen >= 1");
            if (dropout < 0f || dropout >= 1f)
                throw new IllegalArgumentException("dropout in [0, 1), получено " + dropout);
            this.heads = heads;
            this.dModel = dModel;
            this.kvLen = kvLen;
            this.causal = causal;
            this.dropout = dropout;
        }

        @Override
        public Layer create(TensorShape inputShape) {
            if (!inputShape.isSequence())
                throw new IllegalArgumentException(
                        "QKVAttention.Dto: ожидался sequence-вход Q, получено " + inputShape);
            if (inputShape.channels() != dModel)
                throw new IllegalArgumentException(
                        "QKVAttention.Dto: Q.channels=" + inputShape.channels()
                                + " != dModel=" + dModel);
            return new QKVAttentionLayer()
                    .setHeads(heads)
                    .setKvLen(kvLen)
                    .setCausal(causal)
                    .setDropout(dropout)
                    .setInputShape(inputShape)
                    .setOutputShape(inputShape)
                    .allocateBuffers().initWeights(null);
        }
    }

    @Setter
    private int heads;
    @Setter
    private int kvLen;
    @Setter
    private boolean causal;
    @Setter
    private float dropout = 0f;

    private int Tq, Tkv, dModel, dHead;

    /**
     * Только выходная проекция: Q/K/V приходят готовыми.
     */
    public float[][] Wo;
    public float[] bo;
    public float[][] WoG;
    public float[] boG;

    private transient float[] inputQ;
    private transient float[] inputK;
    private transient float[] inputV;

    private float[] attn;          // [heads*Tq*Tkv]
    private float[] outHeads;      // [Tq*dModel]
    private float[] output;        // [Tq*dModel]

    private float[] dAttn;
    private float[] dOutHeads;
    private float[] dQ;            // [Tq*dModel]
    private float[] dK;            // [Tkv*dModel]
    private float[] dV;            // [Tkv*dModel]
    private float[] prevDeltaQ;
    private float[] prevDeltaK;
    private float[] prevDeltaV;

    /**
     * Последние маски: {maskQ, maskK, maskV}. null — все валидны.
     */
    private float[][] lastMasks;

    public QKVAttentionLayer() {
    }

    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("QKVAttention.create: формы не заданы");
        if (!inputShape.isSequence())
            throw new IllegalArgumentException("QKVAttention: ожидался sequence-вход Q, получено " + inputShape);
        if (heads < 1)
            throw new IllegalArgumentException("QKVAttention: heads >= 1, получено " + heads);
        if (kvLen < 1)
            throw new IllegalArgumentException("QKVAttention: kvLen >= 1, получено " + kvLen);

        this.Tq = inputShape.seqLen();
        this.dModel = inputShape.channels();
        this.Tkv = kvLen;
        if (dModel % heads != 0)
            throw new IllegalArgumentException(
                    "QKVAttention: dModel (" + dModel + ") % heads (" + heads + ") != 0");
        this.dHead = dModel / heads;

        this.Wo = new float[dModel][dModel];
        this.bo = new float[dModel];
        this.WoG = new float[dModel][dModel];
        this.boG = new float[dModel];

        int TqTkv = Tq * Tkv;
        int sizeQ = Tq * dModel;
        int sizeKV = Tkv * dModel;

        this.attn = new float[heads * TqTkv];
        this.outHeads = new float[sizeQ];
        this.output = new float[sizeQ];

        this.dAttn = new float[heads * TqTkv];
        this.dOutHeads = new float[sizeQ];
        this.dQ = new float[sizeQ];
        this.dK = new float[sizeKV];
        this.dV = new float[sizeKV];
        this.prevDeltaQ = new float[sizeQ];
        this.prevDeltaK = new float[sizeKV];
        this.prevDeltaV = new float[sizeKV];

        this.lastMasks = new float[3][];
        return this;
    }

    @Override
    public Layer initWeights(Activate activate) {
        float limit = Layer.initLimit(null, dModel, dModel);
        Random rnd = ThreadLocalRandom.current();
        for (int i = 0; i < dModel; i++)
            for (int j = 0; j < dModel; j++)
                Wo[i][j] = rnd.nextFloat() * 2 * limit - limit;
        return this;
    }

// ==================== forward ====================

    @Override
    public float[] forward(float[] input, boolean training) {
        throw new UnsupportedOperationException(
                "QKVAttentionLayer требует 3 входа (Q, K, V); используйте forwardN(inputs, training)");
    }

    @Override
    public float[] forwardN(float[][] inputs, boolean training) {
        return forwardNMasked(inputs, new float[3][], training);
    }

    @Override
    public float[] forwardNMasked(float[][] inputs, float[][] masks, boolean training) {
        if (inputs == null || inputs.length != 3)
            throw new IllegalArgumentException(
                    "QKVAttention.forwardNMasked: ожидалось 3 входа (Q, K, V), получено "
                            + (inputs == null ? "null" : inputs.length));
        if (masks == null || masks.length != 3)
            throw new IllegalArgumentException(
                    "QKVAttention.forwardNMasked: masks.length должен быть 3");
        float[] q = inputs[0], k = inputs[1], v = inputs[2];
        float[] maskQ = masks[0], maskK = masks[1], maskV = masks[2];

        if (q.length != Tq * dModel)
            throw new IllegalArgumentException(
                    "QKVAttention.forwardNMasked: Q ожидалось " + (Tq * dModel) + ", получено " + q.length);
        if (k.length != Tkv * dModel)
            throw new IllegalArgumentException(
                    "QKVAttention.forwardNMasked: K ожидалось " + (Tkv * dModel) + ", получено " + k.length);
        if (v.length != Tkv * dModel)
            throw new IllegalArgumentException(
                    "QKVAttention.forwardNMasked: V ожидалось " + (Tkv * dModel) + ", получено " + v.length);
        if (maskQ != null && maskQ.length != Tq)
            throw new IllegalArgumentException(
                    "QKVAttention.forwardNMasked: maskQ длины " + Tq + ", получено " + maskQ.length);
        if (maskK != null && maskK.length != Tkv)
            throw new IllegalArgumentException(
                    "QKVAttention.forwardNMasked: maskK длины " + Tkv + ", получено " + maskK.length);
        if (maskV != null && maskV.length != Tkv)
            throw new IllegalArgumentException(
                    "QKVAttention.forwardNMasked: maskV длины " + Tkv + ", получено " + maskV.length);

        this.inputQ = q;
        this.inputK = k;
        this.inputV = v;
        this.lastMasks = masks;

        final float scale = 1f / (float) Math.sqrt(dHead);
        final int TqTkv = Tq * Tkv;
        for (int h = 0; h < heads; h++) {
            final int off = h * dHead;
            final int attnOff = h * TqTkv;
            for (int i = 0; i < Tq; i++) {
                boolean qValid = (maskQ == null) || (maskQ[i] > 0f);
                for (int j = 0; j < Tkv; j++) {
                    // causal: ключ j не виден запросу i, если j > i
                    if (causal && j > i) {
                        attn[attnOff + i * Tkv + j] = Float.NEGATIVE_INFINITY;
                        continue;
                    }
                    boolean kvValid = (maskK == null) || (maskK[j] > 0f);
                    if (!qValid || !kvValid) {
                        attn[attnOff + i * Tkv + j] = Float.NEGATIVE_INFINITY;
                        continue;
                    }
                    float s = 0f;
                    for (int c = 0; c < dHead; c++)
                        s += q[i * dModel + off + c] * k[j * dModel + off + c];
                    attn[attnOff + i * Tkv + j] = s * scale;
                }
                softmaxRowMasked(attn, attnOff + i * Tkv, Tkv);
            }
            for (int i = 0; i < Tq; i++) {
                for (int c = 0; c < dHead; c++) {
                    float s = 0f;
                    for (int j = 0; j < Tkv; j++)
                        s += attn[attnOff + i * Tkv + j] * v[j * dModel + off + c];
                    outHeads[i * dModel + off + c] = s;
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
                "QKVAttentionLayer требует 3 входа; используйте backwardN(deltaNext)");
    }

    @Override
    public float[][] backwardN(float[] deltaNext) {
        return backwardNMasked(deltaNext, lastMasks);
    }

    @Override
    public float[][] backwardNMasked(float[] deltaNext, float[][] masks) {
        if (deltaNext.length != Tq * dModel)
            throw new IllegalArgumentException(
                    "QKVAttention.backwardNMasked: ожидалось " + (Tq * dModel)
                            + ", получено " + deltaNext.length);
        if (inputQ == null || inputK == null || inputV == null)
            throw new IllegalStateException("QKVAttention.backwardNMasked: forwardNMasked не был вызван");
        if (masks == null) masks = new float[3][];
        float[] maskQ = masks[0], maskK = masks[1], maskV = masks[2];

        // dWo, dbo, dOutHeads
        Arrays.fill(boG, 0f);
        for (float[] row : WoG) Arrays.fill(row, 0f);
        Arrays.fill(dOutHeads, 0f);
        for (int i = 0; i < Tq; i++) {
            for (int o = 0; o < dModel; o++) {
                float d = deltaNext[i * dModel + o];
                boG[o] += d;
                for (int c = 0; c < dModel; c++) {
                    WoG[o][c] += d * outHeads[i * dModel + c];
                    dOutHeads[i * dModel + c] += d * Wo[o][c];
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
                for (int c = 0; c < dHead; c++) {
                    float d = dOutHeads[i * dModel + off + c];
                    for (int j = 0; j < Tkv; j++) {
                        dAttn[attnOff + i * Tkv + j] += d * inputV[j * dModel + off + c];
                        dV[j * dModel + off + c] += d * attn[attnOff + i * Tkv + j];
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
                    for (int c = 0; c < dHead; c++) {
                        dQ[i * dModel + off + c] += d * inputK[j * dModel + off + c];
                        dK[j * dModel + off + c] += d * inputQ[i * dModel + off + c];
                    }
                }
            }
        }

        // PAD-обнуление
        if (maskQ != null) {
            for (int t = 0; t < Tq; t++)
                if (maskQ[t] == 0f) {
                    int off = t * dModel;
                    for (int c = 0; c < dModel; c++) dQ[off + c] = 0f;
                }
        }
        if (maskK != null) {
            for (int t = 0; t < Tkv; t++)
                if (maskK[t] == 0f) {
                    int off = t * dModel;
                    for (int c = 0; c < dModel; c++) dK[off + c] = 0f;
                }
        }
        if (maskV != null) {
            for (int t = 0; t < Tkv; t++)
                if (maskV[t] == 0f) {
                    int off = t * dModel;
                    for (int c = 0; c < dModel; c++) dV[off + c] = 0f;
                }
        }

        // Q/K/V уже готовы, поэтому dX = dQ/dK/dV (никаких Wq/Wk/Wv).
        System.arraycopy(dQ, 0, prevDeltaQ, 0, dQ.length);
        System.arraycopy(dK, 0, prevDeltaK, 0, dK.length);
        System.arraycopy(dV, 0, prevDeltaV, 0, dV.length);

        return new float[][]{prevDeltaQ, prevDeltaK, prevDeltaV};
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
        for (float[] r : WoG) Arrays.fill(r, 0f);
        Arrays.fill(boG, 0f);
    }

    @Override
    public void update(Optimizer optimizer, int batchSize) {
        if (batchSize <= 0) throw new IllegalArgumentException("batchSize > 0");
        optimizer.update(Wo, WoG, bo, boG, batchSize);
    }

    @Override
    public double gradNormSq() {
        double s = 0;
        for (float[] r : WoG) for (float v : r) s += v * v;
        for (float v : boG) s += v * v;
        return s;
    }

    @Override
    public void scaleGrad(float scale) {
        for (float[] r : WoG) for (int j = 0; j < r.length; j++) r[j] *= scale;
        for (int i = 0; i < boG.length; i++) boG[i] *= scale;
    }

// ==================== сериализация ====================

    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);
        out.writeInt(heads);
        out.writeInt(kvLen);
        out.writeBoolean(causal);
        out.writeFloat(dropout);
        writeMat(out, Wo);
        writeVec(out, bo);
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);
        this.heads = in.readInt();
        this.kvLen = in.readInt();
        this.causal = in.readBoolean();
        this.dropout = in.readFloat();
        allocateBuffers();
        readMat(in, Wo);
        readVec(in, bo);
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