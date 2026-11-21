package rawnet.code.net.layers.core;

import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;
import rawnet.code.net.TensorShape;
import rawnet.code.net.core.activate.Activate;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.core.optimizator.Optimizer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Time-distributed Dense: одна матрица {@code [dOut][dIn]} применяется
 * к каждому из {@code T} токенов независимо.
 *
 * <pre>
 *   y[t, o] = sum_i x[t, i]·W[o][i] + b[o]     для всех t
 * </pre>
 *
 * Вход/выход — {@code seq(T, dIn)} / {@code seq(T, dOut)}.
 */
@Accessors(chain = true)
public class TimeDistributedDenseLayer extends Layer {

    @RequiredArgsConstructor
    public static class Dto extends Layer.Dto {
        public final int dOut;
        @Override
        public Layer create(TensorShape inputShape) {
            return new TimeDistributedDenseLayer()
                    .setDOut(dOut)
                    .setInputShape(inputShape)
                    .setOutputShape(computeOutputShape(inputShape, dOut))
                    .allocateBuffers().initWeights(null);
        }
    }

    @Setter private int dOut;
    private int T, dIn;

    float[][] W;        // [dOut][dIn]
    float[] b;          // [dOut]
    private float[][] WG;
    private float[] bG;

    private transient float[] inputRef;
    private float[] output;
    private float[] prevDelta;

    public TimeDistributedDenseLayer() {}

    public static TensorShape computeOutputShape(TensorShape in, int dOut) {
        if (in == null) throw new IllegalArgumentException("TimeDistributedDenseLayer: inputShape == null");
        if (!in.isSequence())
            throw new IllegalArgumentException("TimeDistributedDenseLayer: ожидался sequence-вход, получено " + in);
        if (dOut < 1) throw new IllegalArgumentException("TimeDistributedDenseLayer: dOut >= 1, получено " + dOut);
        return TensorShape.seq(in.seqLen(), dOut);
    }

    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("TimeDistributedDenseLayer.create: формы не заданы");
        if (!inputShape.isSequence())
            throw new IllegalArgumentException("TimeDistributedDenseLayer.create: ожидался sequence-вход");
        this.T = inputShape.seqLen();
        this.dIn = inputShape.channels();
        if (dOut < 1) throw new IllegalArgumentException("TimeDistributedDenseLayer.create: dOut >= 1");

        this.W = new float[dOut][dIn];
        this.b = new float[dOut];
        this.WG = new float[dOut][dIn];
        this.bG = new float[dOut];

        this.output = new float[outputShape.size()];
        this.prevDelta = new float[inputShape.size()];
        return this;
    }

    @Override
    public Layer initWeights(Activate activate) {
        float limit = Layer.initLimit(null, dIn, dOut);
        Random rnd = ThreadLocalRandom.current();
        for (int o = 0; o < dOut; o++) {
            for (int i = 0; i < dIn; i++) W[o][i] = rnd.nextFloat() * 2 * limit - limit;
            b[o] = 0f;
        }
        return this;
    }

    @Override
    public float[] forward(float[] input, boolean training) {
        if (input.length != inputShape.size())
            throw new IllegalArgumentException("TimeDistributedDenseLayer.forward: неверная длина входа");
        this.inputRef = input;
        for (int t = 0; t < T; t++) {
            int xOff = t * dIn;
            int yOff = t * dOut;
            for (int o = 0; o < dOut; o++) {
                float s = b[o];
                float[] w = W[o];
                for (int i = 0; i < dIn; i++) s += w[i] * input[xOff + i];
                output[yOff + o] = s;
            }
        }
        return output;
    }

    @Override
    public float[] backward(float[] deltaNext) {
        if (deltaNext.length != outputShape.size())
            throw new IllegalArgumentException("TimeDistributedDenseLayer.backward: неверная длина deltaNext");
        if (inputRef == null)
            throw new IllegalStateException("TimeDistributedDenseLayer.backward: forward не был вызван");

        Arrays.fill(prevDelta, 0f);
        for (int t = 0; t < T; t++) {
            int xOff = t * dIn;
            int yOff = t * dOut;
            for (int o = 0; o < dOut; o++) {
                float d = deltaNext[yOff + o];
                bG[o] += d;
                float[] w = W[o];
                float[] wg = WG[o];
                for (int i = 0; i < dIn; i++) {
                    wg[i] += d * inputRef[xOff + i];
                    prevDelta[xOff + i] += d * w[i];
                }
            }
        }
        return prevDelta;
    }

    @Override
    public void zeroGrad() {
        for (float[] r : WG) Arrays.fill(r, 0f);
        Arrays.fill(bG, 0f);
    }

    @Override
    public void update(Optimizer optimizer, int batchSize) {
        if (batchSize <= 0) throw new IllegalArgumentException("batchSize > 0");
        optimizer.update(W, WG, b, bG, batchSize);
    }

    @Override
    public double gradNormSq() {
        double s = 0;
        for (float[] r : WG) for (float v : r) s += v * v;
        for (float v : bG) s += v * v;
        return s;
    }

    @Override
    public void scaleGrad(float scale) {
        for (float[] r : WG) for (int j = 0; j < r.length; j++) r[j] *= scale;
        for (int i = 0; i < bG.length; i++) bG[i] *= scale;
    }

    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);
        out.writeInt(dOut);
        for (float[] r : W) for (float v : r) out.writeFloat(v);
        for (float v : b) out.writeFloat(v);
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);
        this.dOut = in.readInt();
        allocateBuffers();
        for (float[] r : W) for (int j = 0; j < r.length; j++) r[j] = in.readFloat();
        for (int i = 0; i < b.length; i++) b[i] = in.readFloat();
    }
}