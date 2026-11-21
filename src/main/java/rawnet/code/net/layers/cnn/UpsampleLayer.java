package rawnet.code.net.layers.cnn;

import lombok.RequiredArgsConstructor;
import rawnet.code.net.TensorShape;
import rawnet.code.net.layers.Layer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;

/**
 * Nearest-neighbor upsampling (2D). Без параметров.
 *
 * <p>Каждый элемент входа копируется в блок {@code scale × scale} выхода.
 * Вход/выход — {@link TensorShape} с {@code isCnn() == true}.
 *
 * <h3>Backward</h3>
 * Градиент суммируется по блоку {@code scale × scale}, соответствующему
 * одному входному элементу.
 *
 * <p>Параметров нет: {@code zeroGrad}/{@code update}/{@code gradNormSq}/
 * {@code scaleGrad} — no-op/{@code 0}.
 */
public class UpsampleLayer extends Layer {

    /** DTO upsampling-слоя. */
    @RequiredArgsConstructor
    public static class Dto extends Layer.Dto {
        /** Коэффициент увеличения; {@code >= 1}. */
        public final int scale;

        @Override
        public Layer create(TensorShape inputShape) {
            return new UpsampleLayer()
                    .setScale(scale)
                    .setInputShape(inputShape)
                    .allocateBuffers();
        }
    }

    private int scale;
    private int channels, inH, inW, outH, outW;

    private float[] output;
    private float[] prevDelta;

    public UpsampleLayer() {}

    public UpsampleLayer setScale(int scale) { this.scale = scale; return this; }

    /**
     * Вычисляет форму выхода: {@code scale} раз по каждой оси.
     *
     * @throws IllegalArgumentException если вход не CNN или {@code scale < 1}
     */
    public static TensorShape computeOutputShape(TensorShape in, int scale) {
        if (in == null) throw new IllegalArgumentException("UpsampleLayer: inputShape == null");
        if (!in.isCnn())
            throw new IllegalArgumentException("UpsampleLayer: ожидался CNN-вход, получено " + in);
        if (scale < 1) throw new IllegalArgumentException("UpsampleLayer: scale >= 1, получено " + scale);
        return TensorShape.cnn(in.channels(), in.height() * scale, in.width() * scale);
    }

    /**
     * Валидирует формы и выделяет буферы.
     *
     * @return {@code this}
     * @throws IllegalStateException    если формы не заданы
     * @throws IllegalArgumentException если вход не CNN или {@code scale < 1}
     */
    public Layer allocateBuffers() {
        if (inputShape == null)
            throw new IllegalStateException("UpsampleLayer.create: inputShape и outputShape должны быть заданы");
        if (scale < 1)
            throw new IllegalArgumentException("UpsampleLayer.create: scale >= 1, получено " + scale);

        outputShape = computeOutputShape(inputShape, scale);

        if (!inputShape.isCnn())
            throw new IllegalArgumentException("UpsampleLayer.create: ожидался CNN-вход, получено " + inputShape);


        this.channels = inputShape.channels();
        this.inH      = inputShape.height();
        this.inW      = inputShape.width();
        this.outH     = outputShape.height();
        this.outW     = outputShape.width();

        this.output    = new float[channels * outH * outW];
        this.prevDelta = new float[channels * inH * inW];
        return this;
    }

    /**
     * {@inheritDoc}
     *
     * @throws IllegalArgumentException если длина {@code input} не равна {@code inputShape.size()}
     */
    @Override
    public float[] forward(float[] input, boolean training) {
        if (input.length != inputShape.size())
            throw new IllegalArgumentException(
                    "UpsampleLayer.forward: ожидалось " + inputShape.size() + ", получено " + input.length);

        final int inHW  = inH * inW;
        final int outHW = outH * outW;
        for (int c = 0; c < channels; c++) {
            final int inCOff  = c * inHW;
            final int outCOff = c * outHW;
            for (int y = 0; y < inH; y++) {
                for (int x = 0; x < inW; x++) {
                    final float v = input[inCOff + y * inW + x];
                    for (int dy = 0; dy < scale; dy++) {
                        final int rowOut = outCOff + (y * scale + dy) * outW;
                        for (int dx = 0; dx < scale; dx++)
                            output[rowOut + x * scale + dx] = v;
                    }
                }
            }
        }
        return output;
    }

    /**
     * {@inheritDoc}
     *
     * @throws IllegalArgumentException если длина {@code deltaNext} не равна {@code outputShape.size()}
     */
    @Override
    public float[] backward(float[] deltaNext) {
        if (deltaNext.length != outputShape.size())
            throw new IllegalArgumentException(
                    "UpsampleLayer.backward: ожидалось " + outputShape.size() + ", получено " + deltaNext.length);

        final int inHW  = inH * inW;
        final int outHW = outH * outW;
        Arrays.fill(prevDelta, 0f);
        for (int c = 0; c < channels; c++) {
            final int inCOff  = c * inHW;
            final int outCOff = c * outHW;
            for (int y = 0; y < inH; y++) {
                for (int x = 0; x < inW; x++) {
                    float sum = 0f;
                    for (int dy = 0; dy < scale; dy++) {
                        final int rowOut = outCOff + (y * scale + dy) * outW;
                        for (int dx = 0; dx < scale; dx++)
                            sum += deltaNext[rowOut + x * scale + dx];
                    }
                    prevDelta[inCOff + y * inW + x] = sum;
                }
            }
        }
        return prevDelta;
    }

    /**
     * Сериализация: только имя класса и формы через {@code super.save}.
     * Параметров нет.
     */
    public void save(DataOutputStream out) throws IOException {
        super.save(out);
        out.writeInt(scale);
    }

    /**
     * Загрузка: читает формы (через {@code super.load}) и пересоздаёт буферы.
     * {@code scale} должен быть задан в DTO (или через {@link #setScale})
     * до вызова.
     */
    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);
        this.scale = in.readInt();
        allocateBuffers();
    }
}