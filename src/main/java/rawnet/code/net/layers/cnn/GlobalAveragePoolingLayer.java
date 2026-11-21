package rawnet.code.net.layers.cnn;

import rawnet.code.net.TensorShape;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.mask.TokenPoolingLayer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;

/**
 * Global Average Pooling (2D): усредняет каждый канал по пространственным
 * осям, превращая карту признаков в вектор.
 *
 * <pre>
 *   вход:  cnn(C, H, W)
 *   выход: flat(C)
 *   out[c] = (1 / (H · W)) · Σ_{h,w} x[c, h, w]
 * </pre>
 *
 * <p>Простыми словами: после нескольких свёрток/пулингов у нас есть
 * {@code C} карт размера {@code H×W}. GAP «сжимает» каждую карту в одно
 * число — среднее по всем её позициям. Получается вектор длины {@code C},
 * который подаётся в полносвязную голову (обычно {@code DenseLayer}).
 *
 * <p>Зачем: заменяет {@code FlattenLayer + DenseLayer} на входе головы.
 * Главные плюсы —
 * <ul>
 *   <li><b>меньше параметров</b>: голова видит {@code C} чисел, а не
 *       {@code C·H·W};</li>
 *   <li><b>инвариантность к размеру входа</b>: сеть с GAP может принимать
 *       картинки разного разрешения (меняются только {@code H}, {@code W});
 *       {@code Flatten} так не умеет — размер головы пришлось бы менять;</li>
 *   <li><b>устойчивость к переобучению</b>: усреднение — сильная
 *       регуляризация;</li>
 *   <li><b>интерпретируемость</b>: {@code out[c]} — «насколько в среднем
 *       активен канал {@code c}», что часто читается как уверенность в
 *       признаке.</li>
 * </ul>
 *
 * <h3>Backward</h3>
 * Производная среднего по каждой позиции одна и та же:
 * {@code dx[c, h, w] = dOut[c] / (H · W)}. То есть весь градиент канала
 * {@code c} равномерно «размазывается» по его {@code H·W} позициям.
 * Это точная производная, а не приближение.
 *
 * <h3>Параметры</h3>
 * Нет. {@code zeroGrad}/{@code update}/{@code gradNormSq}/{@code scaleGrad}
 * — no-op / {@code 0} (наследуются от {@link Layer}).
 *
 * <h3>Маска последовательности</h3>
 * Слой работает с CNN-формой ({@code isCnn()}), у которой {@code seqLen == 0}.
 * Ось SEQ сюда не доходит: sequence-ветка обрабатывается
 * {@link TokenPoolingLayer}. Поэтому
 * {@code MaskConsumer}/{@code MaskTransform} не реализуются — как у
 * {@link ConvLayer}, {@link MaxPoolLayer}, {@link UpsampleLayer}.
 *
 * <h3>Когда использовать</h3>
 * Стандарт для классификационных CNN (ResNet, MobileNet, EfficientNet):
 * {@code ... → Conv → ... → GAP → Dense(numClasses)}. Для сегментации и
 * детекции GAP не подходит — там нужны пространственные карты, а не
 * вектор.
 */
public class GlobalAveragePoolingLayer extends Layer {

    /** DTO GAP-слоя. Архитектурных параметров нет. */
    public static class Dto extends Layer.Dto {
        @Override
        public Layer create(TensorShape inputShape) {
            return new GlobalAveragePoolingLayer()
                    .setInputShape(inputShape)
                    .setOutputShape(computeOutputShape(inputShape))
                    .allocateBuffers();
        }
    }

    private int channels, inH, inW;

    private float[] output;
    private float[] prevDelta;

    public GlobalAveragePoolingLayer() {}

    /**
     * Вычисляет форму выхода: {@code cnn(C, H, W) → flat(C)}.
     *
     * @throws IllegalArgumentException если вход не CNN
     */
    public static TensorShape computeOutputShape(TensorShape in) {
        if (in == null) throw new IllegalArgumentException("GlobalAveragePoolingLayer: inputShape == null");
        if (!in.isCnn())
            throw new IllegalArgumentException(
                    "GlobalAveragePoolingLayer: ожидался CNN-вход, получено " + in);
        return TensorShape.flat(in.channels());
    }

    /**
     * Валидирует формы и выделяет буферы.
     *
     * @return {@code this}
     * @throws IllegalStateException    если формы не заданы
     * @throws IllegalArgumentException если вход не CNN или выход не совпадает
     *                                  с {@code flat(C)}
     */
    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException(
                    "GlobalAveragePoolingLayer.create: inputShape и outputShape должны быть заданы");
        if (!inputShape.isCnn())
            throw new IllegalArgumentException(
                    "GlobalAveragePoolingLayer.create: ожидался CNN-вход, получено " + inputShape);

        this.channels = inputShape.channels();
        this.inH      = inputShape.height();
        this.inW      = inputShape.width();

        if (!outputShape.isFlat() || outputShape.size() != channels)
            throw new IllegalArgumentException(
                    "GlobalAveragePoolingLayer.create: outputShape " + outputShape
                            + " не соответствует flat(" + channels + ")");

        this.output    = new float[channels];
        this.prevDelta = new float[channels * inH * inW];
        return this;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Для каждого канала считает среднее по {@code H·W} позициям.
     *
     * @throws IllegalArgumentException если длина {@code input} не равна {@code inputShape.size()}
     */
    @Override
    public float[] forward(float[] input, boolean training) {
        if (input.length != inputShape.size())
            throw new IllegalArgumentException(
                    "GlobalAveragePoolingLayer.forward: ожидалось " + inputShape.size()
                            + ", получено " + input.length);

        final int inHW = inH * inW;
        final float inv = 1f / inHW;
        for (int c = 0; c < channels; c++) {
            final int off = c * inHW;
            float s = 0f;
            for (int i = 0; i < inHW; i++) s += input[off + i];
            output[c] = s * inv;
        }
        return output;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Градиент канала {@code c} равномерно распределяется по всем
     * {@code H·W} позициям: {@code dx[c, i] = dOut[c] / (H·W)}.
     *
     * @throws IllegalArgumentException если длина {@code deltaNext} не равна {@code channels}
     */
    @Override
    public float[] backward(float[] deltaNext) {
        if (deltaNext.length != channels)
            throw new IllegalArgumentException(
                    "GlobalAveragePoolingLayer.backward: ожидалось " + channels
                            + ", получено " + deltaNext.length);

        final int inHW = inH * inW;
        final float inv = 1f / inHW;
        for (int c = 0; c < channels; c++) {
            final int off = c * inHW;
            final float d = deltaNext[c] * inv;
            Arrays.fill(prevDelta, off, off + inHW, d);
        }
        return prevDelta;
    }

    /**
     * Сериализация: имя класса и формы через {@code super.save}.
     * Своих параметров нет.
     */
    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);
    }

    /**
     * Загрузка: формы через {@code super.load}, затем пересоздание буферов.
     */
    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);
        allocateBuffers();
    }
}