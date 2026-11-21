package rawnet.code.net.layers.rnn;

import lombok.AllArgsConstructor;
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
 * Gated Recurrent Unit (Cho et al., 2014) с BPTT.
 *
 * <pre>
 *   z_t = σ(b_z + W_z·x_t + U_z·h_{t-1})
 *   r_t = σ(b_r + W_r·x_t + U_r·h_{t-1})
 *   n_t = tanh(b_n + W_n·x_t + r_t ⊙ (U_n·h_{t-1}))
 *   h_t = (1 − z_t) ⊙ n_t + z_t ⊙ h_{t-1}
 * </pre>
 *
 * <h2>Маска</h2>
 * Реализует {@link MaskConsumer}: на PAD-шагах {@code h_t = h_{t-1}},
 * гейты не считаются, веса не накапливаются, градиент проходит сквозь
 * ({@code dhPrev = dhCurr}).
 */
@Accessors(chain = true)
public class GruLayer extends Layer implements MaskConsumer {

    @RequiredArgsConstructor
    @AllArgsConstructor
    public static class Dto extends Layer.Dto {
        public final int seqLen;
        public final int hidden;
        public final boolean returnSequences;
        public float clipGradNorm = 1f;

        @Override
        public Layer create(TensorShape inputShape) {
            return new GruLayer()
                    .setSeqLen(seqLen)
                    .setHidden(hidden)
                    .setReturnSequences(returnSequences)
                    .setClipGradNorm(clipGradNorm)
                    .setInputShape(inputShape)
                    .setOutputShape(computeOutputShape(inputShape, seqLen, hidden, returnSequences))
                    .allocateBuffers().initWeights(null);
        }
    }

    @Setter private int seqLen;
    @Setter private int hidden;
    @Setter private boolean returnSequences;
    @Setter private float clipGradNorm = 1f;

    private int features;

    float[][] weights;
    float[] biases;
    private float[][] weightGrad;
    private float[] biasGrad;

    private transient float[] inputRef;
    private float[][] hCache;
    private float[][] zCache;
    private float[][] rCache;
    private float[][] nCache;
    private float[][] sNCache;
    private boolean[] validCache;

    private float[] output;
    private float[] danBuf, darBuf, dazBuf;
    private float[] dxBuf;
    private float[] prevDelta;
    private float[] dhCurr;
    private float[] dhPrev;
    private float[] proj;

    public GruLayer() {}

    public static TensorShape computeOutputShape(TensorShape in, int seqLen,
                                                 int hidden, boolean returnSequences) {
        if (in == null) throw new IllegalArgumentException("GruLayer: inputShape == null");
        if (!in.isSequence())
            throw new IllegalArgumentException("GruLayer: ожидался sequence-вход, получено " + in);
        if (seqLen < 1) throw new IllegalArgumentException("GruLayer: seqLen >= 1, получено " + seqLen);
        if (hidden < 1) throw new IllegalArgumentException("GruLayer: hidden >= 1, получено " + hidden);
        if (in.seqLen() != seqLen)
            throw new IllegalArgumentException(
                    "GruLayer: seqLen слоя (" + seqLen + ") != seqLen формы (" + in.seqLen() + ")");
        return returnSequences
                ? TensorShape.seq(seqLen, hidden)
                : TensorShape.flat(hidden);
    }

    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("GruLayer.create: inputShape и outputShape должны быть заданы");
        if (!inputShape.isSequence())
            throw new IllegalArgumentException("GruLayer.create: ожидался sequence-вход, получено " + inputShape);
        if (inputShape.seqLen() != seqLen)
            throw new IllegalArgumentException(
                    "GruLayer.create: inputShape.seqLen=" + inputShape.seqLen()
                            + " != seqLen=" + seqLen);
        if (hidden < 1)
            throw new IllegalArgumentException("GruLayer.create: hidden >= 1, получено " + hidden);

        this.features = inputShape.channels();

        final int rows = 3 * hidden;
        final int cols = features + hidden;

        this.weights    = new float[rows][cols];
        this.biases     = new float[rows];
        this.weightGrad = new float[rows][cols];
        this.biasGrad   = new float[rows];

        this.hCache     = new float[seqLen + 1][hidden];
        this.zCache     = new float[seqLen][hidden];
        this.rCache     = new float[seqLen][hidden];
        this.nCache     = new float[seqLen][hidden];
        this.sNCache    = new float[seqLen][hidden];
        this.validCache = new boolean[seqLen];

        this.output    = new float[outputShape.size()];
        this.danBuf    = new float[hidden];
        this.darBuf    = new float[hidden];
        this.dazBuf    = new float[hidden];
        this.dxBuf     = new float[features];
        this.prevDelta = new float[seqLen * features];
        this.dhCurr    = new float[hidden];
        this.dhPrev    = new float[hidden];
        this.proj      = new float[seqLen * 3 * hidden];
        return this;
    }

    @Override
    public Layer initWeights(Activate activate) {
        final int rows = 3 * hidden;
        final int cols = features + hidden;
        float limit = (float) Math.sqrt(6.0 / (cols + hidden));
        Random rnd = ThreadLocalRandom.current();
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                weights[i][j] = rnd.nextFloat() * 2 * limit - limit;
        return this;
    }

    // ==================== forward ====================

    @Override
    public float[] forward(float[] input, boolean training) {
        return forwardMasked(input, null, training);
    }

    @Override
    public float[] forwardMasked(float[] input, float[] mask, boolean training) {
        if (input.length != inputShape.size())
            throw new IllegalArgumentException(
                    "GruLayer.forwardMasked: ожидалось " + inputShape.size() + ", получено " + input.length);
        if (mask != null && mask.length != seqLen)
            throw new IllegalArgumentException(
                    "GruLayer.forwardMasked: маска длины " + seqLen + ", получено " + mask.length);

        this.inputRef = input;
        Arrays.fill(hCache[0], 0f);

        final int stride = 3 * hidden;

        // 1. Проекции входа: W·x для всех шагов (PAD тоже — дёшево, проще)
        for (int t = 0; t < seqLen; t++) {
            final int xOff = t * features;
            final int pOff = t * stride;
            for (int k = 0; k < stride; k++) {
                final float[] w = weights[k];
                float s = 0f;
                for (int j = 0; j < features; j++) s += w[j] * input[xOff + j];
                proj[pOff + k] = s;
            }
        }

        // 2. Рекуррентная часть
        for (int t = 0; t < seqLen; t++) {
            boolean valid = (mask == null) || (mask[t] > 0f);
            validCache[t] = valid;

            final float[] hPrev  = hCache[t];
            final float[] hCurr  = hCache[t + 1];
            final int pOff       = t * stride;

            if (!valid) {
                // PAD-шаг: h_t = h_{t-1} (копия, не алиас)
                System.arraycopy(hPrev, 0, hCurr, 0, hidden);
            } else {
                final float[] z  = zCache[t];
                final float[] r  = rCache[t];
                final float[] n  = nCache[t];
                final float[] sN = sNCache[t];

                for (int k = 0; k < hidden; k++) {
                    final float[] wz = weights[k];
                    final float[] wr = weights[hidden + k];
                    final float[] wn = weights[2 * hidden + k];

                    float az = biases[k] + proj[pOff + k];
                    for (int j = 0; j < hidden; j++) az += wz[features + j] * hPrev[j];
                    z[k] = sigmoid(az);

                    float ar = biases[hidden + k] + proj[pOff + hidden + k];
                    for (int j = 0; j < hidden; j++) ar += wr[features + j] * hPrev[j];
                    r[k] = sigmoid(ar);

                    float sn = 0f;
                    for (int j = 0; j < hidden; j++) sn += wn[features + j] * hPrev[j];
                    sN[k] = sn;

                    float an = biases[2 * hidden + k] + proj[pOff + 2 * hidden + k] + r[k] * sn;
                    n[k] = tanh(an);

                    hCurr[k] = (1f - z[k]) * n[k] + z[k] * hPrev[k];
                }
            }

            if (returnSequences)
                System.arraycopy(hCurr, 0, output, t * hidden, hidden);
        }

        if (!returnSequences)
            System.arraycopy(hCache[seqLen], 0, output, 0, hidden);

        return output;
    }

    // ==================== backward ====================

    @Override
    public float[] backward(float[] deltaNext) {
        return backwardMasked(deltaNext, null);
    }

    @Override
    public float[] backwardMasked(float[] deltaNext, float[] mask) {
        if (deltaNext.length != outputShape.size())
            throw new IllegalArgumentException(
                    "GruLayer.backwardMasked: ожидалось " + outputShape.size() + ", получено " + deltaNext.length);
        if (mask != null && mask.length != seqLen)
            throw new IllegalArgumentException(
                    "GruLayer.backwardMasked: маска длины " + seqLen + ", получено " + mask.length);
        if (inputRef == null)
            throw new IllegalStateException("GruLayer.backwardMasked: forward не был вызван");

        Arrays.fill(prevDelta, 0f);
        Arrays.fill(dhCurr, 0f);

        for (int t = seqLen - 1; t >= 0; t--) {
            boolean valid = (mask == null) || (mask[t] > 0f);
            final int dOff = returnSequences ? t * hidden : 0;
            final boolean last = (t == seqLen - 1);

            // Добавить внешний градиент
            if (returnSequences) {
                for (int k = 0; k < hidden; k++) dhCurr[k] += deltaNext[dOff + k];
            } else if (last) {
                for (int k = 0; k < hidden; k++) dhCurr[k] += deltaNext[k];
            }

            if (!valid) {
                // PAD-шаг: dhPrev = dhCurr, без обновлений
                System.arraycopy(dhCurr, 0, dhPrev, 0, hidden);
                float[] tmp = dhCurr; dhCurr = dhPrev; dhPrev = tmp;
                continue;
            }

            // Обычный шаг
            for (int k = 0; k < hidden; k++) {
                final float dh  = dhCurr[k];
                final float zk  = zCache[t][k];
                final float rk  = rCache[t][k];
                final float nk  = nCache[t][k];
                final float snk = sNCache[t][k];

                final float dz  = dh * (hCache[t][k] - nk);
                final float dn  = dh * (1f - zk);
                final float dan = dn * (1f - nk * nk);
                final float dar = dan * snk * rk * (1f - rk);
                final float daz = dz * zk * (1f - zk);

                danBuf[k] = dan;
                darBuf[k] = dar;
                dazBuf[k] = daz;
            }

            Arrays.fill(dhPrev, 0f);
            final int xOff = t * features;
            for (int k = 0; k < hidden; k++) {
                final float dh  = dhCurr[k];
                final float zk  = zCache[t][k];
                final float rk  = rCache[t][k];
                final float dan = danBuf[k];
                final float dar = darBuf[k];
                final float daz = dazBuf[k];

                dhPrev[k] += dh * zk;

                float[] wzg = weightGrad[k];
                for (int j = 0; j < features; j++) wzg[j] += daz * inputRef[xOff + j];
                for (int j = 0; j < hidden;  j++) wzg[features + j] += daz * hCache[t][j];
                biasGrad[k] += daz;
                for (int j = 0; j < hidden; j++) dhPrev[j] += weights[k][features + j] * daz;

                float[] wrg = weightGrad[hidden + k];
                for (int j = 0; j < features; j++) wrg[j] += dar * inputRef[xOff + j];
                for (int j = 0; j < hidden;  j++) wrg[features + j] += dar * hCache[t][j];
                biasGrad[hidden + k] += dar;
                for (int j = 0; j < hidden; j++) dhPrev[j] += weights[hidden + k][features + j] * dar;

                biasGrad[2 * hidden + k] += dan;
                float[] wng = weightGrad[2 * hidden + k];
                for (int j = 0; j < features; j++) wng[j] += dan * inputRef[xOff + j];
                final float ds_n = dan * rk;
                for (int j = 0; j < hidden; j++) wng[features + j] += ds_n * hCache[t][j];
                for (int j = 0; j < hidden; j++) dhPrev[j] += weights[2 * hidden + k][features + j] * ds_n;
            }

            Arrays.fill(dxBuf, 0f);
            for (int k = 0; k < hidden; k++) {
                final float dan = danBuf[k];
                final float dar = darBuf[k];
                final float daz = dazBuf[k];
                float[] wz = weights[k];
                float[] wr = weights[hidden + k];
                float[] wn = weights[2 * hidden + k];
                for (int j = 0; j < features; j++)
                    dxBuf[j] += wz[j] * daz + wr[j] * dar + wn[j] * dan;
            }
            System.arraycopy(dxBuf, 0, prevDelta, xOff, features);

            clipByNorm(dhPrev, clipGradNorm);

            float[] tmp = dhCurr; dhCurr = dhPrev; dhPrev = tmp;
        }
        return prevDelta;
    }

    // ==================== параметры ====================

    @Override
    public void zeroGrad() {
        for (float[] row : weightGrad) Arrays.fill(row, 0f);
        Arrays.fill(biasGrad, 0f);
    }

    @Override
    public void update(Optimizer optimizer, int batchSize) {
        if (batchSize <= 0)
            throw new IllegalArgumentException("batchSize должен быть > 0, получено " + batchSize);
        optimizer.update(weights, weightGrad, biases, biasGrad, batchSize);
    }

    @Override
    public double gradNormSq() {
        double sum = 0;
        for (float[] row : weightGrad) for (float v : row) sum += v * v;
        for (float v : biasGrad) sum += v * v;
        return sum;
    }

    @Override
    public void scaleGrad(float scale) {
        for (float[] row : weightGrad)
            for (int j = 0; j < row.length; j++) row[j] *= scale;
        for (int i = 0; i < biasGrad.length; i++) biasGrad[i] *= scale;
    }

    // ==================== сериализация ====================

    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);
        out.writeInt(seqLen);
        out.writeInt(hidden);
        out.writeBoolean(returnSequences);
        out.writeFloat(clipGradNorm);
        for (float[] row : weights) for (float v : row) out.writeFloat(v);
        for (float b : biases) out.writeFloat(b);
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);
        this.seqLen          = in.readInt();
        this.hidden          = in.readInt();
        this.returnSequences = in.readBoolean();
        this.clipGradNorm    = in.readFloat();
        allocateBuffers();
        for (float[] row : weights)
            for (int j = 0; j < row.length; j++) row[j] = in.readFloat();
        for (int i = 0; i < biases.length; i++) biases[i] = in.readFloat();
    }

    // ==================== хелперы ====================

    private static void clipByNorm(float[] v, float clipNorm) {
        if (clipNorm <= 0f) return;
        double sq = 0;
        for (float x : v) sq += x * x;
        double norm = Math.sqrt(sq);
        if (norm > clipNorm) {
            float scale = (float) (clipNorm / norm);
            for (int i = 0; i < v.length; i++) v[i] *= scale;
        }
    }

    private static float tanh(float x) {
        if (x <= -3f) return -1f;
        if (x >=  3f) return  1f;
        final float x2 = x * x;
        return x * (27f + x2) / (27f + 9f * x2);
    }

    private static float sigmoid(float x) {
        if (x <= -8f) return 0f;
        if (x >=  8f) return 1f;
        return 0.5f + 0.5f * tanh(0.5f * x);
    }
}