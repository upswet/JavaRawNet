package rawnet.code.net.layers.rnn;

import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;
import rawnet.code.net.TensorShape;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.core.activate.Activate;
import rawnet.code.net.layers.core.ActivationLayer;
import rawnet.code.net.core.optimizator.Optimizer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Двунаправленный рекуррентный слой (Bidirectional Elman RNN).
 *
 * <pre>
 *   forward:  h_t^f = f(Wx_f·x_t + Wh_f·h_{t-1}^f + b_f),  t = 0 … T-1,  h_{-1}^f = 0
 *   backward: h_t^b = f(Wx_b·x_t + Wh_b·h_{t+1}^b + b_b),  t = T-1 … 0,  h_T^b   = 0
 *   out[t]  = [ h_t^f ‖ h_t^b ]
 * </pre>
 *
 * <h2>Формат входа/выхода</h2>
 * Вход  — {@code TensorShape.seq(seqLen, features)}.
 * Выход — {@code seq(seqLen, 2·hidden)} при {@code returnSequences == true},
 *         иначе {@code flat(2·hidden)}.
 */
@Accessors(chain = true)
@Deprecated(since = "Используйте BidirectionalLayer")
public class BidirectionalRnnLayer extends Layer {

    @RequiredArgsConstructor
    public static class Dto extends Layer.Dto {
        public final int seqLen;
        public final int hidden;
        public final Activate activate;
        public final boolean returnSequences;
        public float clipGradNorm = 1f;

        @Override
        public Layer create(TensorShape inputShape) {
            return new BidirectionalRnnLayer()
                    .setSeqLen(seqLen)
                    .setHidden(hidden)
                    .setActivate(activate)
                    .setReturnSequences(returnSequences)
                    .setClipGradNorm(clipGradNorm)
                    .setInputShape(inputShape)
                    .setOutputShape(computeOutputShape(inputShape, seqLen, hidden, returnSequences))
                    .allocateBuffers();
        }
    }

    @Setter private int seqLen;
    @Setter private int hidden;
    @Setter private Activate activate;
    @Setter private boolean returnSequences;
    @Setter private float clipGradNorm = 1f;

    private int features;

    private ElmanRnnLayer forwardRnn;
    private ElmanRnnLayer backwardRnn;

    private float[] output;
    private float[] reversedInput;
    private float[] deltaF;
    private float[] deltaB;
    private float[] prevDelta;

    public BidirectionalRnnLayer() {}

    public static TensorShape computeOutputShape(TensorShape in, int seqLen,
                                                 int hidden, boolean returnSequences) {
        if (in == null) throw new IllegalArgumentException("BidirectionalRnnLayer: inputShape == null");
        if (!in.isSequence())
            throw new IllegalArgumentException("BidirectionalRnnLayer: ожидался sequence-вход, получено " + in);
        if (seqLen < 1) throw new IllegalArgumentException("BidirectionalRnnLayer: seqLen >= 1, получено " + seqLen);
        if (hidden < 1) throw new IllegalArgumentException("BidirectionalRnnLayer: hidden >= 1, получено " + hidden);
        if (in.seqLen() != seqLen)
            throw new IllegalArgumentException(
                    "BidirectionalRnnLayer: seqLen слоя (" + seqLen + ") != seqLen формы (" + in.seqLen() + ")");
        return returnSequences
                ? TensorShape.seq(seqLen, 2 * hidden)
                : TensorShape.flat(2 * hidden);
    }

    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("BidirectionalRnnLayer.create: inputShape и outputShape должны быть заданы");
        if (!inputShape.isSequence())
            throw new IllegalArgumentException("BidirectionalRnnLayer.create: ожидался sequence-вход, получено " + inputShape);
        if (inputShape.seqLen() != seqLen)
            throw new IllegalArgumentException(
                    "BidirectionalRnnLayer.create: inputShape.seqLen=" + inputShape.seqLen()
                            + " != seqLen=" + seqLen);
        if (hidden < 1)
            throw new IllegalArgumentException("BidirectionalRnnLayer.create: hidden >= 1, получено " + hidden);
        if (activate == null)
            throw new IllegalArgumentException("BidirectionalRnnLayer.create: activate не может быть null");

        this.features = inputShape.channels();

        // Внутренние RNN: каждый — sequence(seqLen, features) → seq(seqLen, hidden)
        // или flat(hidden). Создаём вручную (не через Dto), т.к. форма входа
        // для них всегда seq(seqLen, features), а не inputShape слоя.
        TensorShape innerIn  = TensorShape.seq(seqLen, features);
        TensorShape innerOut = returnSequences
                ? TensorShape.seq(seqLen, hidden)
                : TensorShape.flat(hidden);

        this.forwardRnn = (ElmanRnnLayer) new ElmanRnnLayer()
                .setSeqLen(seqLen)
                .setHidden(hidden)
                .setActivate(activate)
                .setReturnSequences(returnSequences)
                .setClipGradNorm(clipGradNorm)
                .setInputShape(innerIn)
                .setOutputShape(innerOut)
                .allocateBuffers();

        this.backwardRnn = (ElmanRnnLayer) new ElmanRnnLayer()
                .setSeqLen(seqLen)
                .setHidden(hidden)
                .setActivate(activate)
                .setReturnSequences(returnSequences)
                .setClipGradNorm(clipGradNorm)
                .setInputShape(innerIn)
                .setOutputShape(innerOut)
                .allocateBuffers();

        this.output        = new float[outputShape.size()];
        this.reversedInput = new float[inputShape.size()];
        this.deltaF        = new float[forwardRnn.outputShape().size()];
        this.deltaB        = new float[backwardRnn.outputShape().size()];
        this.prevDelta     = new float[inputShape.size()];
        return this;
    }

    @Override
    public float[] forward(float[] input, boolean training) {
        if (input.length != inputShape.size())
            throw new IllegalArgumentException(
                    "BidirectionalRnnLayer.forward: ожидалось " + inputShape.size() + ", получено " + input.length);

        // 1) forward-направление на исходном входе
        float[] outF = forwardRnn.forward(input, training);

        // 2) обратный вход: reversedInput[τ] = input[T-1-τ]
        for (int t = 0; t < seqLen; t++) {
            System.arraycopy(input, t * features,
                    reversedInput, (seqLen - 1 - t) * features, features);
        }
        float[] outB = backwardRnn.forward(reversedInput, training);

        // 3) склейка: out[t] = h_t^f ‖ h_t^b
        if (returnSequences) {
            for (int t = 0; t < seqLen; t++) {
                System.arraycopy(outF, t * hidden, output, t * 2 * hidden, hidden);
                System.arraycopy(outB, (seqLen - 1 - t) * hidden,
                        output, t * 2 * hidden + hidden, hidden);
            }
        } else {
            System.arraycopy(outF, 0, output, 0, hidden);       // h_{T-1}^f
            System.arraycopy(outB, 0, output, hidden, hidden);  // h_0^b
        }
        return output;
    }

    @Override
    public float[] backward(float[] deltaNext) {
        if (deltaNext.length != outputShape.size())
            throw new IllegalArgumentException(
                    "BidirectionalRnnLayer.backward: ожидалось " + outputShape.size() + ", получено " + deltaNext.length);

        if (returnSequences) {
            for (int t = 0; t < seqLen; t++) {
                System.arraycopy(deltaNext, t * 2 * hidden, deltaF, t * hidden, hidden);
                System.arraycopy(deltaNext, t * 2 * hidden + hidden,
                        deltaB, (seqLen - 1 - t) * hidden, hidden);
            }
        } else {
            System.arraycopy(deltaNext, 0, deltaF, 0, hidden);
            System.arraycopy(deltaNext, hidden, deltaB, 0, hidden);
        }

        float[] dInF = forwardRnn.backward(deltaF);
        float[] dInB = backwardRnn.backward(deltaB);

        for (int t = 0; t < seqLen; t++) {
            int dstOff = t * features;
            int srcOff = (seqLen - 1 - t) * features;
            for (int j = 0; j < features; j++) {
                prevDelta[dstOff + j] = dInF[dstOff + j] + dInB[srcOff + j];
            }
        }
        return prevDelta;
    }

    @Override public void zeroGrad() { forwardRnn.zeroGrad(); backwardRnn.zeroGrad(); }

    @Override
    public void update(Optimizer optimizer, int batchSize) {
        if (batchSize <= 0)
            throw new IllegalArgumentException("batchSize должен быть > 0, получено " + batchSize);
        forwardRnn.update(optimizer, batchSize);
        backwardRnn.update(optimizer, batchSize);
    }

    @Override
    public double gradNormSq() {
        return forwardRnn.gradNormSq() + backwardRnn.gradNormSq();
    }

    @Override
    public void scaleGrad(float scale) {
        forwardRnn.scaleGrad(scale);
        backwardRnn.scaleGrad(scale);
    }

    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);                       // className + inputShape + outputShape
        out.writeInt(seqLen);
        out.writeInt(hidden);
        out.writeBoolean(returnSequences);
        out.writeFloat(clipGradNorm);
        ActivationLayer.writeActivate(out, activate);
        forwardRnn.saveParams(out);
        backwardRnn.saveParams(out);
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);                        // className + inputShape + outputShape
        this.seqLen          = in.readInt();
        this.hidden          = in.readInt();
        this.returnSequences = in.readBoolean();
        this.clipGradNorm    = in.readFloat();
        this.activate        = ActivationLayer.readActivate(in);
        allocateBuffers();
        forwardRnn.loadParams(in);
        backwardRnn.loadParams(in);
    }
}