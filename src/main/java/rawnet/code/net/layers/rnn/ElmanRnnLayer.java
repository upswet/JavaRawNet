package rawnet.code.net.layers.rnn;

import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;
import rawnet.code.net.TensorShape;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.contract.MaskConsumer;
import rawnet.code.net.core.activate.Activate;
import rawnet.code.net.layers.core.ActivationLayer;
import rawnet.code.net.core.optimizator.Optimizer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Простой рекуррентный слой (Elman RNN) с BPTT.
 * <pre>
 *   z_t = Wx·x_t + Wh·h_{t-1} + b
 *   h_t = f(z_t)
 * </pre>
 *
 * <h2>Маска</h2>
 * Реализует {@link MaskConsumer}: на PAD-шагах ({@code mask[t] == 0})
 * скрытое состояние <b>не обновляется</b> ({@code h_t = h_{t-1}),
 * градиент проходит сквозь шаг ({@code dhPrev = dhCurr}), но
 * {@code weightGrad}/{@code biasGrad}/{@code prevDelta} не накапливаются.
 * Это позволяет RNN не «протягивать» PAD и корректно работать с
 * {@code BidirectionalLayer} (обратное направление пропускает PAD с конца).
 *
 * <p>Без маски ({@link #forward}/{@link #backward}) работает как раньше.
 */
@Accessors(chain = true)
public class ElmanRnnLayer extends Layer implements MaskConsumer {

    @RequiredArgsConstructor
    public static class Dto extends Layer.Dto {
        public final int seqLen;
        public final int hidden;
        public final Activate activate;
        public final boolean returnSequences;
        public float clipGradNorm = 1f;

        @Override
        public Layer create(TensorShape inputShape) {
            return new ElmanRnnLayer()
                    .setSeqLen(seqLen)
                    .setHidden(hidden)
                    .setActivate(activate)
                    .setReturnSequences(returnSequences)
                    .setClipGradNorm(clipGradNorm)
                    .setInputShape(inputShape)
                    .setOutputShape(computeOutputShape(inputShape, seqLen, hidden, returnSequences))
                    .allocateBuffers().initWeights(activate);
        }
    }

    @Setter private int seqLen;
    @Setter private int hidden;
    @Setter private Activate activate;
    @Setter private boolean returnSequences;
    @Setter private float clipGradNorm = 1f;

    private int features;

    /** [hidden][features + hidden] — Wx | Wh */
    float[][] weights;
    float[] biases;

    private float[][] weightGrad;
    private float[] biasGrad;

    private transient float[] inputRef;
    private float[][] hCache;
    private float[][] zCache;

    /** Кэш валидности шага: {@code true} — шаг настоящий (не PAD). */
    private boolean[] validCache;

    private float[] output;
    private float[] dActBuffer;
    private float[] dzBuffer;
    private float[] dhNextBuffer;
    private float[] dhNextTmp;
    private float[] prevDelta;

    public ElmanRnnLayer() {}

    public static TensorShape computeOutputShape(TensorShape in, int seqLen,
                                                 int hidden, boolean returnSequences) {
        if (in == null) throw new IllegalArgumentException("ElmanRnnLayer: inputShape == null");
        if (!in.isSequence())
            throw new IllegalArgumentException("ElmanRnnLayer: ожидался sequence-вход, получено " + in);
        if (seqLen < 1) throw new IllegalArgumentException("ElmanRnnLayer: seqLen >= 1, получено " + seqLen);
        if (hidden < 1) throw new IllegalArgumentException("ElmanRnnLayer: hidden >= 1, получено " + hidden);
        if (in.seqLen() != seqLen)
            throw new IllegalArgumentException(
                    "ElmanRnnLayer: seqLen слоя (" + seqLen + ") != seqLen формы (" + in.seqLen() + ")");
        return returnSequences
                ? TensorShape.seq(seqLen, hidden)
                : TensorShape.flat(hidden);
    }

    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("ElmanRnnLayer.create: inputShape и outputShape должны быть заданы");
        if (!inputShape.isSequence())
            throw new IllegalArgumentException("ElmanRnnLayer.create: ожидался sequence-вход, получено " + inputShape);
        if (inputShape.seqLen() != seqLen)
            throw new IllegalArgumentException(
                    "ElmanRnnLayer.create: inputShape.seqLen=" + inputShape.seqLen()
                            + " != seqLen=" + seqLen);
        if (hidden < 1)
            throw new IllegalArgumentException("ElmanRnnLayer.create: hidden >= 1, получено " + hidden);
        if (activate == null)
            throw new IllegalArgumentException("ElmanRnnLayer.create: activate не может быть null");

        this.features = inputShape.channels();

        int wCols = features + hidden;
        this.weights    = new float[hidden][wCols];
        this.biases     = new float[hidden];
        this.weightGrad = new float[hidden][wCols];
        this.biasGrad   = new float[hidden];

        this.hCache     = new float[seqLen + 1][hidden];
        this.zCache     = new float[seqLen][hidden];
        this.validCache = new boolean[seqLen];
        this.output     = new float[outputShape.size()];
        this.dActBuffer = new float[hidden];
        this.dzBuffer   = new float[hidden];
        this.dhNextBuffer = new float[hidden];
        this.dhNextTmp    = new float[hidden];
        this.prevDelta    = new float[seqLen * features];
        return this;
    }

    @Override
    public Layer initWeights(Activate activate) {
        int wCols = features + hidden;
        float limit = Layer.initLimit(activate, wCols, hidden);
        Random rnd = ThreadLocalRandom.current();
        for (int k = 0; k < hidden; k++) {
            for (int j = 0; j < wCols; j++) weights[k][j] = rnd.nextFloat() * 2 * limit - limit;
            biases[k] = 0f;
        }
        return this;
    }

    // ==================== forward ====================

    /** Без маски: все шаги настоящие. */
    @Override
    public float[] forward(float[] input, boolean training) {
        return forwardMasked(input, null, training);
    }

    @Override
    public float[] forwardMasked(float[] input, float[] mask, boolean training) {
        if (input.length != inputShape.size())
            throw new IllegalArgumentException(
                    "ElmanRnnLayer.forwardMasked: ожидалось " + inputShape.size() + ", получено " + input.length);
        if (mask != null && mask.length != seqLen)
            throw new IllegalArgumentException(
                    "ElmanRnnLayer.forwardMasked: маска длины " + seqLen + ", получено " + mask.length);

        this.inputRef = input;
        Arrays.fill(hCache[0], 0f);

        for (int t = 0; t < seqLen; t++) {
            boolean valid = (mask == null) || (mask[t] > 0f);
            validCache[t] = valid;

            float[] hPrev = hCache[t];
            float[] hCurr = hCache[t + 1];

            if (!valid) {
                // PAD-шаг: копия h_{t-1} → h_t (не алиас! hCache[t] может затираться)
                System.arraycopy(hPrev, 0, hCurr, 0, hidden);
            } else {
                float[] z = zCache[t];
                final int xOff = t * features;
                for (int k = 0; k < hidden; k++) {
                    float s = biases[k];
                    float[] w = weights[k];
                    for (int j = 0; j < features; j++) s += w[j] * input[xOff + j];
                    for (int j = 0; j < hidden;  j++) s += w[features + j] * hPrev[j];
                    z[k] = s;
                }
                activate.apply(z, hCurr);
            }

            if (returnSequences) {
                System.arraycopy(hCurr, 0, output, t * hidden, hidden);
            }
        }

        if (!returnSequences) {
            System.arraycopy(hCache[seqLen], 0, output, 0, hidden);
        }
        return output;
    }

    // ==================== backward ====================

    /** Без маски: все шаги настоящие. */
    @Override
    public float[] backward(float[] deltaNext) {
        return backwardMasked(deltaNext, null);
    }

    @Override
    public float[] backwardMasked(float[] deltaNext, float[] mask) {
        if (deltaNext.length != outputShape.size())
            throw new IllegalArgumentException(
                    "ElmanRnnLayer.backwardMasked: ожидалось " + outputShape.size() + ", получено " + deltaNext.length);
        if (mask != null && mask.length != seqLen)
            throw new IllegalArgumentException(
                    "ElmanRnnLayer.backwardMasked: маска длины " + seqLen + ", получено " + mask.length);
        if (inputRef == null)
            throw new IllegalStateException("ElmanRnnLayer.backwardMasked: forward не был вызван");

        Arrays.fill(prevDelta, 0f);
        Arrays.fill(dhNextBuffer, 0f);

        for (int t = seqLen - 1; t >= 0; t--) {
            boolean valid = (mask == null) || (mask[t] > 0f);
            final boolean last = (t == seqLen - 1);
            final int dOff = returnSequences ? t * hidden : 0;

            if (!valid) {
                // PAD-шаг: h_t = h_{t-1}, градиент проходит сквозь.
                // dhPrev = dhCurr (без обновлений весов).
                System.arraycopy(dhNextBuffer, 0, dhNextTmp, 0, hidden);
                // swap
                float[] swap = dhNextBuffer;
                dhNextBuffer = dhNextTmp;
                dhNextTmp    = swap;
                continue;
            }

            float[] z = zCache[t];
            float[] h = hCache[t + 1];

            activate.derivative(z, h, dActBuffer);

            for (int k = 0; k < hidden; k++) {
                float incoming = returnSequences
                        ? deltaNext[dOff + k]
                        : (last ? deltaNext[k] : 0f);
                dzBuffer[k] = (incoming + dhNextBuffer[k]) * dActBuffer[k];
            }

            float[] hPrev = hCache[t];
            final int xOff = t * features;

            for (int k = 0; k < hidden; k++) {
                float dz = dzBuffer[k];
                biasGrad[k] += dz;
                float[] wg = weightGrad[k];
                for (int j = 0; j < features; j++) wg[j]            += dz * inputRef[xOff + j];
                for (int j = 0; j < hidden;   j++) wg[features + j] += dz * hPrev[j];
            }

            // dx_t = Wxᵀ · dz
            for (int j = 0; j < features; j++) {
                float s = 0f;
                for (int k = 0; k < hidden; k++) s += weights[k][j] * dzBuffer[k];
                prevDelta[xOff + j] = s;
            }

            // dh_next = Whᵀ · dz
            for (int j = 0; j < hidden; j++) {
                float s = 0f;
                for (int k = 0; k < hidden; k++) s += weights[k][features + j] * dzBuffer[k];
                dhNextTmp[j] = s;
            }
            clipByNorm(dhNextTmp, clipGradNorm);
            float[] swap = dhNextBuffer;
            dhNextBuffer = dhNextTmp;
            dhNextTmp    = swap;
        }
        return prevDelta;
    }

    // ==================== параметры ====================

    @Override
    public void zeroGrad() {
        for (int k = 0; k < hidden; k++) {
            Arrays.fill(weightGrad[k], 0f);
            biasGrad[k] = 0f;
        }
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
        ActivationLayer.writeActivate(out, activate);
        saveParams(out);
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);
        this.seqLen          = in.readInt();
        this.hidden          = in.readInt();
        this.returnSequences = in.readBoolean();
        this.clipGradNorm    = in.readFloat();
        this.activate        = ActivationLayer.readActivate(in);
        allocateBuffers();
        loadParams(in);
    }

    /** Записывает только веса и смещения (без имени класса и форм). */
    public void saveParams(DataOutputStream out) throws IOException {
        for (float[] row : weights) for (float v : row) out.writeFloat(v);
        for (float b : biases) out.writeFloat(b);
    }

    /** Читает только веса и смещения (без имени класса и форм). */
    public void loadParams(DataInputStream in) throws IOException {
        for (float[] row : weights)
            for (int j = 0; j < row.length; j++) row[j] = in.readFloat();
        for (int i = 0; i < biases.length; i++) biases[i] = in.readFloat();
    }

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
}