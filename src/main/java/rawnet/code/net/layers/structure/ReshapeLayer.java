package rawnet.code.net.layers.structure;

import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.contract.MaskTransform;
import rawnet.code.net.TensorShape;

import java.util.Objects;

/**
 * Слой смены формы без изменения данных: {@code y = reshape(x)}.
 *
 * <p>Требует {@code inputShape.size() == outputShape.size()}. Данные
 * копируются как есть (плоская раскладка), меняется только
 * {@link TensorShape}. {@code backward} копирует дельту как есть.
 *
 * <h3>Маска</h3>
 * Реализует {@link MaskTransform}: после reshape маска
 * <b>теряется</b> ({@code transformMask} возвращает {@code null}),
 * потому что ось SEQ в общем случае меняется или исчезает.
 * Если вам нужно сохранить маску при переходе
 * {@code seq(T, d) → seq(T, d')} — используйте {@code TimeDistributedDense},
 * а не {@code Reshape}.
 */
@NoArgsConstructor
public class ReshapeLayer extends Layer implements MaskTransform {

    /** DTO слоя reshape. */
    @RequiredArgsConstructor
    public static class Dto extends Layer.Dto {
        /** Желаемая форма выхода; не {@code null}. */
        public final TensorShape outputShape;

        @Override
        public Layer create(TensorShape inputShape) {
            Objects.requireNonNull(outputShape, "outputShape");
            return new ReshapeLayer()
                    .setInputShape(inputShape)
                    .setOutputShape(outputShape)
                    .allocateBuffers();
        }
    }

    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("ReshapeLayer.create: inputShape и outputShape должны быть заданы");
        if (inputShape.size() != outputShape.size())
            throw new IllegalArgumentException(
                    "ReshapeLayer: размеры входа и выхода должны совпадать, получено "
                            + inputShape.size() + " и " + outputShape.size()
                            + " (" + inputShape + " -> " + outputShape + ")");
        return this;
    }

    @Override
    public float[] forward(float[] input, boolean training) {
        if (input.length != inputShape.size())
            throw new IllegalArgumentException(
                    "ReshapeLayer.forward: ожидалось " + inputShape.size() + ", получено " + input.length);
        return input;
    }

    @Override
    public float[] backward(float[] deltaNext) {
        if (deltaNext.length != outputShape.size())
            throw new IllegalArgumentException(
                    "ReshapeLayer.backward: ожидалось " + outputShape.size() + ", получено " + deltaNext.length);
        return deltaNext;
    }

    /** Reshape меняет раскладку — маска не имеет смысла. */
    @Override public float[] transformMask(float[] inMask) { return null; }

    // =====================================================================
    // ==================== FlattenLayer (сахар) ===========================
    // =====================================================================

    /**
     * Слой «выпрямления»: {@code y = flatten(x)} — частный случай
     * {@link ReshapeLayer} с выходом {@code flat(inputShape.size())}.
     *
     * <p>Реализует {@link MaskTransform} и, как {@link ReshapeLayer},
     * теряет маску.
     */
    @NoArgsConstructor
    public static class FlattenLayer extends Layer implements MaskTransform {

        /** DTO flatten-слоя: выход всегда {@code flat(inputShape.size())}. */
        public static class Dto extends Layer.Dto {
            @Override
            public Layer create(TensorShape inputShape) {
                return new FlattenLayer()
                        .setInputShape(inputShape)
                        .setOutputShape(TensorShape.flat(inputShape.size()))
                        .allocateBuffers();
            }
        }

        public Layer allocateBuffers() {
            if (inputShape == null)
                throw new IllegalStateException("FlattenLayer.create: inputShape должен быть задан");
            this.outputShape = TensorShape.flat(inputShape.size());
            return this;
        }

        @Override
        public float[] forward(float[] input, boolean training) {
            if (input.length != inputShape.size())
                throw new IllegalArgumentException(
                        "FlattenLayer.forward: ожидалось " + inputShape.size() + ", получено " + input.length);
            return input;
        }

        @Override
        public float[] backward(float[] deltaNext) {
            if (deltaNext.length != outputShape.size())
                throw new IllegalArgumentException(
                        "FlattenLayer.backward: ожидалось " + outputShape.size() + ", получено " + deltaNext.length);
            return deltaNext;
        }

        @Override public float[] transformMask(float[] inMask) { return null; }
    }
}