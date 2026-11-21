package rawnet.code.net.layers.core;

import rawnet.code.net.TensorShape;
import rawnet.code.net.layers.Layer;

import java.io.DataInputStream;
import java.io.IOException;

/**
 * Синусоидальное позиционное кодирование (Vaswani et al., 2017).
 * PE[t][2i] = sin(t/10000^{2i/d}), PE[t][2i+1] = cos(...).
 * Вход/выход — {@code seq(T, dModel)}; выход = вход + PE (поэлементно).
 * Параметров нет; в backward градиент проходит как есть.
 */
public class PositionalEncodingLayer extends Layer {

    public static class Dto extends Layer.Dto {
        @Override
        public Layer create(TensorShape inputShape) {
            return new PositionalEncodingLayer()
                    .setInputShape(inputShape)
                    .setOutputShape(inputShape)
                    .allocateBuffers();
        }
    }

    private int T, dModel;
    private float[] pe;       // [T * dModel]
    private float[] output;
    private float[] prevDelta;

    public PositionalEncodingLayer() {}

    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("PositionalEncodingLayer.create: формы не заданы");
        if (!inputShape.isSequence())
            throw new IllegalArgumentException("PositionalEncodingLayer: ожидался sequence-вход");
        this.T = inputShape.seqLen();
        this.dModel = inputShape.channels();
        this.pe = new float[T * dModel];
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < dModel; i += 2) {
                double angle = t / Math.pow(10000.0, (double) i / dModel);
                pe[t * dModel + i] = (float) Math.sin(angle);
                if (i + 1 < dModel) pe[t * dModel + i + 1] = (float) Math.cos(angle);
            }
        }
        this.output = new float[inputShape.size()];
        this.prevDelta = new float[inputShape.size()];
        return this;
    }

    @Override
    public float[] forward(float[] input, boolean training) {
        if (input.length != inputShape.size())
            throw new IllegalArgumentException("PositionalEncodingLayer.forward: неверная длина входа");
        for (int i = 0; i < input.length; i++) output[i] = input[i] + pe[i];
        return output;
    }

    @Override
    public float[] backward(float[] deltaNext) {
        if (deltaNext.length != outputShape.size())
            throw new IllegalArgumentException("PositionalEncodingLayer.backward: неверная длина deltaNext");
        System.arraycopy(deltaNext, 0, prevDelta, 0, deltaNext.length);
        return prevDelta;
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);          // читает формы
        allocateBuffers();       // пересоздаёт pe/output/prevDelta по новым T/dModel
    }
}
