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
 * Multi-Head Attention с <b>N входами</b>: Q + (N−1) KV-групп.
 *
 * <pre>
 *   Q  = X₀·Wq + bq                       X₀ — seq(Tq, dModel)
 *   Kᵢ = Xᵢ·Wkᵢ + bkᵢ                     Xᵢ — seq(Tkvᵢ, dModel), i = 1..N−1
 *   Vᵢ = Xᵢ·Wvᵢ + bvᵢ
 *   K  = concat(K₁, …, K_{N−1}) по оси T  seq(ΣTkvᵢ, dModel)
 *   V  = concat(V₁, …, V_{N−1}) по оси T
 *   head_h:  attn_h = softmax(Q_h·K_hᵀ / √dHead + mask)
 *            out_h  = attn_h·V_h
 *   out = concat(out_h)·Wo + bo
 * </pre>
 *
 * <p>Простыми словами: несколько источников контекста (например,
 * текст + изображение + аудио) проецируются в общее пространство
 * K/V, склеиваются в одну длинную последовательность, и Q «смотрит»
 * на всё сразу. Каждая KV-группа имеет <b>свою</b> пару матриц
 * {@code (Wkᵢ, Wvᵢ)} — это позволяет модели учитывать разную природу
 * источников.
 *
 * <h2>Отличия от {@link CrossAttentionLayer}</h2>
 * <ul>
 *   <li>KV-групп не одна, а N−1; каждая со своими {@code Wkᵢ/Wvᵢ};</li>
 *   <li>ключи/значения склеиваются по оси T <b>до</b> attention —
 *       Q видит все источники одновременно;</li>
 *   <li>маски KV — массив: у каждой группы своя длина {@code Tkvᵢ};</li>
 *   <li>реализует {@link NInputLayer}/{@link NInputMaskedLayer},
 *       а не {@code TwoInput*}.</li>
 * </ul>
 *
 * <h2>Маски</h2>
 * <ul>
 *   <li>PAD-<b>ключи</b> любой KV-группы исключаются из softmax
 *       ({@code -inf});</li>
 *   <li>PAD-<b>запросы</b> Q не обучают {@code Wq} (обнуление {@code dQ});</li>
 *   <li>PAD-<b>ключи/значения</b> группы i не обучают {@code Wkᵢ/Wvᵢ}
 *       (обнуление {@code dKᵢ/dVᵢ}).</li>
 * </ul>
 *
 * <h2>Форма входа/выхода</h2>
 * <ul>
 *   <li>вход 0 (Q): {@code seq(Tq, dModel)};</li>
 *   <li>вход i (KVᵢ), i = 1..N−1: {@code seq(Tkvᵢ, dModel)};
 *       {@code Tkvᵢ == dto.kvLens[i−1]};</li>
 *   <li>выход: {@code seq(Tq, dModel)} (как у Q).</li>
 * </ul>
 */
@Accessors(chain = true)
public class NInputAttentionLayer extends Layer implements NInputLayer, NInputMaskedLayer {

    /** DTO N-входового attention. */
    public static class Dto extends Layer.Dto {
        /** Число голов; {@code dModel % heads == 0}. */
        public final int heads;
        /** Размерность модели (общая для всех входов и выхода). */
        public final int dModel;
        /** Длины KV-групп, {@code kvLens[i]} — длина входа i+1; {@code length >= 1}. */
        public final int[] kvLens;
        /** Dropout на attention-весах (0 — выключен). */
        public final float dropout;

        public Dto(int heads, int dModel, int[] kvLens) { this(heads, dModel, kvLens, 0f); }

        public Dto(int heads, int dModel, int[] kvLens, float dropout) {
            if (heads < 1) throw new IllegalArgumentException("heads >= 1, получено " + heads);
            if (dModel < 1) throw new IllegalArgumentException("dModel >= 1, получено " + dModel);
            if (dModel % heads != 0)
                throw new IllegalArgumentException(
                        "dModel (" + dModel + ") % heads (" + heads + ") != 0");
            if (kvLens == null || kvLens.length < 1)
                throw new IllegalArgumentException("kvLens: нужно >= 1 группу");
            for (int i = 0; i < kvLens.length; i++)
                if (kvLens[i] < 1)
                    throw new IllegalArgumentException("kvLens[" + i + "] >= 1, получено " + kvLens[i]);
            if (dropout < 0f || dropout >= 1f)
                throw new IllegalArgumentException("dropout in [0, 1), получено " + dropout);
            this.heads = heads;
            this.dModel = dModel;
            this.kvLens = kvLens.clone();
            this.dropout = dropout;
        }

        @Override
        public Layer create(TensorShape inputShape) {
            if (!inputShape.isSequence())
                throw new IllegalArgumentException(
                        "NInputAttention.Dto: ожидался sequence-вход Q, получено " + inputShape);
            if (inputShape.channels() != dModel)
                throw new IllegalArgumentException(
                        "NInputAttention.Dto: Q.channels=" + inputShape.channels()
                                + " != dModel=" + dModel);
            return new NInputAttentionLayer()
                    .setHeads(heads)
                    .setKvLens(kvLens)
                    .setDropout(dropout)
                    .setInputShape(inputShape)
                    .setOutputShape(inputShape)
                    .allocateBuffers().initWeights(null);
        }
    }

    @Setter private int heads;
    @Setter private int[] kvLens = new int[0];
    @Setter private float dropout = 0f;

    private int Tq, dModel, dHead;
    /** Сумма длин KV-групп: {@code Σ kvLens[i]}. */
    private int TkvTotal;
    /** Длина KV-группы i (i = 0..N−2), т.е. входа i+1. */
    private int[] kvLensActual;

    // Wq/bq — для Q; Wk[i]/Wv[i] — для каждой KV-группы; Wo/bo — для выхода
    private float[][] Wq;
    private float[] bq;
    private float[][][] Wk, Wv;   // [group][dModel][dModel]
    private float[][] bk, bv;     // [group][dModel]
    private float[][] Wo;
    private float[] bo;

    private float[][] WqG;
    private float[] bqG;
    private float[][][] WkG, WvG;
    private float[][] bkG, bvG;
    private float[][] WoG;
    private float[] boG;

    private transient float[][] inputRefs;   // [N][size]
    private float[] Q;                       // [Tq*dModel]
    private float[] K, V;                    // [TkvTotal*dModel]
    private float[] attn;                    // [heads*Tq*TkvTotal]
    private float[] outHeads;                // [Tq*dModel]
    private float[] output;                  // [Tq*dModel]

    private float[] dQ;
    private float[] dK, dV;
    private float[] dAttn;
    private float[] dOutHeads;
    private float[] prevDeltaQ;
    private float[][] prevDeltaKV;           // [group][size]

    /** Последние маски: {maskQ, maskKV_1, …, maskKV_{N−1}}. null — все валидны. */
    private float[][] lastMasks;

    public NInputAttentionLayer() {}

    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("NInputAttention.create: формы не заданы");
        if (!inputShape.isSequence())
            throw new IllegalArgumentException("NInputAttention: ожидался sequence-вход Q, получено " + inputShape);
        if (heads < 1)
            throw new IllegalArgumentException("NInputAttention: heads >= 1, получено " + heads);
        if (kvLens == null || kvLens.length < 1)
            throw new IllegalArgumentException("NInputAttention: нужно >= 1 KV-группу");
        for (int i = 0; i < kvLens.length; i++)
            if (kvLens[i] < 1)
                throw new IllegalArgumentException("NInputAttention: kvLens[" + i + "] >= 1, получено " + kvLens[i]);

        this.Tq     = inputShape.seqLen();
        this.dModel = inputShape.channels();
        if (dModel % heads != 0)
            throw new IllegalArgumentException(
                    "NInputAttention: dModel (" + dModel + ") % heads (" + heads + ") != 0");
        this.dHead = dModel / heads;
        this.kvLensActual = kvLens.clone();
        this.TkvTotal = 0;
        for (int l : kvLensActual) TkvTotal += l;

        int groups = kvLensActual.length;

        this.Wq = new float[dModel][dModel]; this.bq = new float[dModel];
        this.Wo = new float[dModel][dModel]; this.bo = new float[dModel];
        this.WqG = new float[dModel][dModel]; this.bqG = new float[dModel];
        this.WoG = new float[dModel][dModel]; this.boG = new float[dModel];

        this.Wk = new float[groups][dModel][dModel];
        this.Wv = new float[groups][dModel][dModel];
        this.bk = new float[groups][dModel];
        this.bv = new float[groups][dModel];
        this.WkG = new float[groups][dModel][dModel];
        this.WvG = new float[groups][dModel][dModel];
        this.bkG = new float[groups][dModel];
        this.bvG = new float[groups][dModel];

        int sizeQ  = Tq * dModel;
        int sizeKV = TkvTotal * dModel;

        this.Q = new float[sizeQ];
        this.K = new float[sizeKV];
        this.V = new float[sizeKV];
        this.attn = new float[heads * Tq * TkvTotal];
        this.outHeads = new float[sizeQ];
        this.output = new float[sizeQ];

        this.dQ = new float[sizeQ];
        this.dK = new float[sizeKV];
        this.dV = new float[sizeKV];
        this.dAttn = new float[heads * Tq * TkvTotal];
        this.dOutHeads = new float[sizeQ];
        this.prevDeltaQ = new float[sizeQ];
        this.prevDeltaKV = new float[groups][];
        for (int g = 0; g < groups; g++) prevDeltaKV[g] = new float[kvLensActual[g] * dModel];

        this.lastMasks = new float[groups + 1][];
        return this;
    }

    @Override
    public Layer initWeights(Activate activate) {
        float limit = Layer.initLimit(null, dModel, dModel);
        Random rnd = ThreadLocalRandom.current();
        for (int i = 0; i < dModel; i++)
            for (int j = 0; j < dModel; j++) {
                Wq[i][j] = rnd.nextFloat() * 2 * limit - limit;
                Wo[i][j] = rnd.nextFloat() * 2 * limit - limit;
            }
        for (int g = 0; g < kvLensActual.length; g++)
            for (int i = 0; i < dModel; i++)
                for (int j = 0; j < dModel; j++) {
                    Wk[g][i][j] = rnd.nextFloat() * 2 * limit - limit;
                    Wv[g][i][j] = rnd.nextFloat() * 2 * limit - limit;
                }
        return this;
    }

    // ==================== forward ====================

    @Override
    public float[] forward(float[] input, boolean training) {
        throw new UnsupportedOperationException(
                "NInputAttentionLayer требует N входов; используйте forwardN(inputs, training)");
    }

    @Override
    public float[] forwardN(float[][] inputs, boolean training) {
        return forwardNMasked(inputs, new float[inputs.length][], training);
    }

    @Override
    public float[] forwardNMasked(float[][] inputs, float[][] masks, boolean training) {
        int groups = kvLensActual.length;
        if (inputs.length != groups + 1)
            throw new IllegalArgumentException(
                    "NInputAttention.forwardNMasked: ожидалось " + (groups + 1)
                            + " входов (Q + " + groups + " KV), получено " + inputs.length);
        if (masks == null || masks.length != inputs.length)
            throw new IllegalArgumentException(
                    "NInputAttention.forwardNMasked: masks.length должен быть " + inputs.length);
        if (inputs[0].length != Tq * dModel)
            throw new IllegalArgumentException(
                    "NInputAttention.forwardNMasked: Q ожидалось " + (Tq * dModel) + ", получено " + inputs[0].length);
        for (int g = 0; g < groups; g++) {
            int expected = kvLensActual[g] * dModel;
            if (inputs[g + 1].length != expected)
                throw new IllegalArgumentException(
                        "NInputAttention.forwardNMasked: KV[" + g + "] ожидалось " + expected
                                + ", получено " + inputs[g + 1].length);
            if (masks[g + 1] != null && masks[g + 1].length != kvLensActual[g])
                throw new IllegalArgumentException(
                        "NInputAttention.forwardNMasked: maskKV[" + g + "] длины " + kvLensActual[g]
                                + ", получено " + masks[g + 1].length);
        }
        if (masks[0] != null && masks[0].length != Tq)
            throw new IllegalArgumentException(
                    "NInputAttention.forwardNMasked: maskQ длины " + Tq + ", получено " + masks[0].length);

        this.inputRefs = inputs;
        this.lastMasks = masks;

        // Q = X₀·Wq + bq
        linear(inputs[0], Wq, bq, Q, Tq, dModel);

        // K/V = concat по T от (Xᵢ·Wkᵢ + bkᵢ)
        int kvOff = 0;
        for (int g = 0; g < groups; g++) {
            int Tkv = kvLensActual[g];
            linearInto(inputs[g + 1], Wk[g], bk[g], K, kvOff, Tkv, dModel);
            linearInto(inputs[g + 1], Wv[g], bv[g], V, kvOff, Tkv, dModel);
            kvOff += Tkv * dModel;
        }

        // Attention: softmax по TkvTotal ключам (все группы вместе)
        final float scale = 1f / (float) Math.sqrt(dHead);
        for (int h = 0; h < heads; h++) {
            final int off = h * dHead;
            final int attnOff = h * Tq * TkvTotal;
            for (int i = 0; i < Tq; i++) {
                boolean qValid = (masks[0] == null) || (masks[0][i] > 0f);
                int kvGlobalIdx = 0;   // индекс в объединённой KV-последовательности
                for (int g = 0; g < groups; g++) {
                    int Tkv = kvLensActual[g];
                    for (int j = 0; j < Tkv; j++) {
                        boolean kvValid = (masks[g + 1] == null) || (masks[g + 1][j] > 0f);
                        if (!qValid || !kvValid) {
                            attn[attnOff + i * TkvTotal + kvGlobalIdx] = Float.NEGATIVE_INFINITY;
                        } else {
                            float s = 0f;
                            for (int k = 0; k < dHead; k++)
                                s += Q[i * dModel + off + k] * K[kvGlobalIdx * dModel + off + k];
                            attn[attnOff + i * TkvTotal + kvGlobalIdx] = s * scale;
                        }
                        kvGlobalIdx++;
                    }
                }
                softmaxRowMasked(attn, attnOff + i * TkvTotal, TkvTotal);
            }
            for (int i = 0; i < Tq; i++) {
                for (int k = 0; k < dHead; k++) {
                    float s = 0f;
                    for (int j = 0; j < TkvTotal; j++)
                        s += attn[attnOff + i * TkvTotal + j] * V[j * dModel + off + k];
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
                "NInputAttentionLayer требует N входов; используйте backwardN(deltaNext)");
    }

    @Override
    public float[][] backwardN(float[] deltaNext) {
        return backwardNMasked(deltaNext, lastMasks);
    }

    @Override
    public float[][] backwardNMasked(float[] deltaNext, float[][] masks) {
        if (deltaNext.length != Tq * dModel)
            throw new IllegalArgumentException(
                    "NInputAttention.backwardNMasked: ожидалось " + (Tq * dModel)
                            + ", получено " + deltaNext.length);
        if (inputRefs == null)
            throw new IllegalStateException("NInputAttention.backwardNMasked: forwardNMasked не был вызван");
        int groups = kvLensActual.length;
        if (masks == null) masks = new float[groups + 1][];

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
        for (int h = 0; h < heads; h++) {
            final int off = h * dHead;
            final int attnOff = h * Tq * TkvTotal;
            for (int i = 0; i < Tq; i++) {
                for (int k = 0; k < dHead; k++) {
                    float d = dOutHeads[i * dModel + off + k];
                    for (int j = 0; j < TkvTotal; j++) {
                        dAttn[attnOff + i * TkvTotal + j] += d * V[j * dModel + off + k];
                        dV[j * dModel + off + k]      += d * attn[attnOff + i * TkvTotal + j];
                    }
                }
            }
            for (int i = 0; i < Tq; i++) {
                int row = attnOff + i * TkvTotal;
                float dot = 0f;
                for (int j = 0; j < TkvTotal; j++) dot += dAttn[row + j] * attn[row + j];
                for (int j = 0; j < TkvTotal; j++)
                    dAttn[row + j] = attn[row + j] * (dAttn[row + j] - dot);
            }
        }

        // dQ, dK
        Arrays.fill(dQ, 0f);
        Arrays.fill(dK, 0f);
        final float scale = 1f / (float) Math.sqrt(dHead);
        for (int h = 0; h < heads; h++) {
            final int off = h * dHead;
            final int attnOff = h * Tq * TkvTotal;
            for (int i = 0; i < Tq; i++) {
                for (int j = 0; j < TkvTotal; j++) {
                    float d = dAttn[attnOff + i * TkvTotal + j] * scale;
                    if (d == 0f) continue;
                    for (int k = 0; k < dHead; k++) {
                        dQ[i * dModel + off + k] += d * K[j * dModel + off + k];
                        dK[j * dModel + off + k] += d * Q[i * dModel + off + k];
                    }
                }
            }
        }

        // PAD-обнуление
        if (masks[0] != null) {
            for (int t = 0; t < Tq; t++) if (masks[0][t] == 0f) {
                int off = t * dModel;
                for (int k = 0; k < dModel; k++) dQ[off + k] = 0f;
            }
        }
        int kvOff = 0;
        for (int g = 0; g < groups; g++) {
            int Tkv = kvLensActual[g];
            if (masks[g + 1] != null) {
                for (int t = 0; t < Tkv; t++) if (masks[g + 1][t] == 0f) {
                    int off = (kvOff + t) * dModel;
                    for (int k = 0; k < dModel; k++) {
                        dK[off + k] = 0f;
                        dV[off + k] = 0f;
                    }
                }
            }
            kvOff += Tkv;
        }

        // dWq/dbq/dX₀
        Arrays.fill(prevDeltaQ, 0f);
        accumLinearGrad(inputRefs[0], dQ, WqG, bqG, Wq, prevDeltaQ, dModel, Tq);

        // dWkᵢ/dbkᵢ/dXᵢ, dWvᵢ/dbvᵢ/dXᵢ (складываются в prevDeltaKV[g])
        kvOff = 0;
        for (int g = 0; g < groups; g++) {
            int Tkv = kvLensActual[g];
            int kvSize = Tkv * dModel;
            // срез dK/dV для группы g
            float[] dKg = Arrays.copyOfRange(dK, kvOff * dModel, (kvOff + Tkv) * dModel);
            float[] dVg = Arrays.copyOfRange(dV, kvOff * dModel, (kvOff + Tkv) * dModel);
            Arrays.fill(prevDeltaKV[g], 0f);
            accumLinearGrad(inputRefs[g + 1], dKg, WkG[g], bkG[g], Wk[g], prevDeltaKV[g], dModel, Tkv);
            accumLinearGrad(inputRefs[g + 1], dVg, WvG[g], bvG[g], Wv[g], prevDeltaKV[g], dModel, Tkv);
            kvOff += Tkv;
        }

        // Собираем результат: {dQ, dKV_1, …, dKV_{N−1}}
        float[][] result = new float[groups + 1][];
        result[0] = prevDeltaQ;
        for (int g = 0; g < groups; g++) result[g + 1] = prevDeltaKV[g];
        return result;
    }

    // ==================== утилиты ====================

    /** out[i*d + o] = b[o] + Σ_k W[o][k]·x[i*d + k], i = 0..T−1. */
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

    /** out[baseOff + i*d + o] = b[o] + Σ_k W[o][k]·x[i*d + k], i = 0..T−1. */
    private void linearInto(float[] x, float[][] W, float[] b, float[] out, int baseOff, int T, int d) {
        for (int i = 0; i < T; i++) {
            for (int o = 0; o < d; o++) {
                float s = b[o];
                float[] w = W[o];
                for (int k = 0; k < d; k++) s += w[k] * x[i * d + k];
                out[baseOff + i * d + o] = s;
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
        for (int g = 0; g < kvLensActual.length; g++) {
            for (float[] r : WkG[g]) Arrays.fill(r, 0f);
            for (float[] r : WvG[g]) Arrays.fill(r, 0f);
            Arrays.fill(bkG[g], 0f);
            Arrays.fill(bvG[g], 0f);
        }
        for (float[] r : WoG) Arrays.fill(r, 0f); Arrays.fill(boG, 0f);
    }

    @Override
    public void update(Optimizer optimizer, int batchSize) {
        if (batchSize <= 0) throw new IllegalArgumentException("batchSize > 0");
        optimizer.update(Wq, WqG, bq, bqG, batchSize);
        for (int g = 0; g < kvLensActual.length; g++) {
            optimizer.update(Wk[g], WkG[g], bk[g], bkG[g], batchSize);
            optimizer.update(Wv[g], WvG[g], bv[g], bvG[g], batchSize);
        }
        optimizer.update(Wo, WoG, bo, boG, batchSize);
    }

    @Override
    public double gradNormSq() {
        double s = 0;
        for (float[] r : WqG) for (float v : r) s += v * v;
        for (float v : bqG) s += v * v;
        for (int g = 0; g < kvLensActual.length; g++) {
            for (float[] r : WkG[g]) for (float v : r) s += v * v;
            for (float v : bkG[g]) s += v * v;
            for (float[] r : WvG[g]) for (float v : r) s += v * v;
            for (float v : bvG[g]) s += v * v;
        }
        for (float[] r : WoG) for (float v : r) s += v * v;
        for (float v : boG) s += v * v;
        return s;
    }

    @Override
    public void scaleGrad(float scale) {
        for (float[] r : WqG) for (int j = 0; j < r.length; j++) r[j] *= scale;
        for (int i = 0; i < bqG.length; i++) bqG[i] *= scale;
        for (int g = 0; g < kvLensActual.length; g++) {
            for (float[] r : WkG[g]) for (int j = 0; j < r.length; j++) r[j] *= scale;
            for (int i = 0; i < bkG[g].length; i++) bkG[g][i] *= scale;
            for (float[] r : WvG[g]) for (int j = 0; j < r.length; j++) r[j] *= scale;
            for (int i = 0; i < bvG[g].length; i++) bvG[g][i] *= scale;
        }
        for (float[] r : WoG) for (int j = 0; j < r.length; j++) r[j] *= scale;
        for (int i = 0; i < boG.length; i++) boG[i] *= scale;
    }

    // ==================== сериализация ====================

    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);
        out.writeInt(heads);
        out.writeFloat(dropout);
        out.writeInt(kvLens.length);
        for (int l : kvLens) out.writeInt(l);
        writeMat(out, Wq); writeVec(out, bq);
        for (int g = 0; g < kvLens.length; g++) {
            writeMat(out, Wk[g]); writeVec(out, bk[g]);
            writeMat(out, Wv[g]); writeVec(out, bv[g]);
        }
        writeMat(out, Wo); writeVec(out, bo);
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);
        this.heads   = in.readInt();
        this.dropout = in.readFloat();
        int groups = in.readInt();
        this.kvLens = new int[groups];
        for (int i = 0; i < groups; i++) kvLens[i] = in.readInt();
        allocateBuffers();
        readMat(in, Wq); readVec(in, bq);
        for (int g = 0; g < groups; g++) {
            readMat(in, Wk[g]); readVec(in, bk[g]);
            readMat(in, Wv[g]); readVec(in, bv[g]);
        }
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