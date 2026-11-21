package rawnet.code.net.layers.structure;

import lombok.NoArgsConstructor;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.contract.MaskTransform;
import rawnet.code.net.TensorShape;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

/**
 * Слой-срез: выбирает подмножество элементов входа по одной из осей
 * {@link TensorShape}.
 *
 * <p>Режимы ({@link Axis}):
 * <ul>
 *   <li>{@link Axis#FLAT} — срез по плоской раскладке
 *       ({@code [offset, offset+length)}); работает с любой формой;</li>
 *   <li>{@link Axis#CHANNEL} — срез по ведущей оси (каналы/признаки/dModel);
 *       форма выхода сохраняет H, W, seqLen;</li>
 *   <li>{@link Axis#SEQ} — срез по оси последовательности; допустим только
 *       для {@code isSequence()}.</li>
 * </ul>
 *
 * <h3>Маска</h3>
 * Реализует {@link MaskTransform}:
 * <ul>
 *   <li>{@link Axis#FLAT} — маска теряется ({@code null}), т.к. flat не
 *       несёт ось SEQ;</li>
 *   <li>{@link Axis#CHANNEL} — маска не меняется (ось SEQ та же);</li>
 *   <li>{@link Axis#SEQ} — маска режется: {@code [offset, offset+length)}.</li>
 * </ul>
 */
@NoArgsConstructor
public class SliceLayer extends Layer implements MaskTransform {

    /** Ось, по которой режем. */
    public enum Axis { FLAT, CHANNEL, SEQ }

    /** DTO слоя-среза. */
    public static class Dto extends Layer.Dto {
        public final Axis axis;
        public final int offset;
        public final int length;

        /** Плоский срез (совместимо с прежней версией). */
        public Dto(int offset, int length) { this(Axis.FLAT, offset, length); }

        public Dto(Axis axis, int offset, int length) {
            this.axis = Objects.requireNonNull(axis, "axis");
            this.offset = offset;
            this.length = length;
        }

        @Override
        public Layer create(TensorShape inputShape) {
            return new SliceLayer()
                    .setAxis(axis)
                    .setOffset(offset)
                    .setLength(length)
                    .setInputShape(inputShape)
                    .setOutputShape(computeOutputShape(inputShape, axis, offset, length))
                    .allocateBuffers();
        }

        private static TensorShape computeOutputShape(TensorShape in, Axis axis, int offset, int length) {
            if (offset < 0)
                throw new IllegalArgumentException("SliceLayer: offset >= 0, получено " + offset);
            if (length < 1)
                throw new IllegalArgumentException("SliceLayer: length >= 1, получено " + length);
            switch (axis) {
                case FLAT -> {
                    if (offset + length > in.size())
                        throw new IllegalArgumentException(
                                "SliceLayer(FLAT): offset+length=" + (offset + length)
                                        + " > inputSize=" + in.size());
                    return TensorShape.flat(length);
                }
                case CHANNEL -> {
                    if (offset + length > in.channels())
                        throw new IllegalArgumentException(
                                "SliceLayer(CHANNEL): offset+length=" + (offset + length)
                                        + " > channels=" + in.channels());
                    return new TensorShape(length, in.height(), in.width(), in.seqLen());
                }
                case SEQ -> {
                    if (!in.isSequence())
                        throw new IllegalArgumentException(
                                "SliceLayer(SEQ): форма не sequence: " + in);
                    if (offset + length > in.seqLen())
                        throw new IllegalArgumentException(
                                "SliceLayer(SEQ): offset+length=" + (offset + length)
                                        + " > seqLen=" + in.seqLen());
                    return TensorShape.seq(length, in.channels());
                }
                default -> throw new IllegalStateException("Неизвестная ось: " + axis);
            }
        }
    }

    private Axis axis = Axis.FLAT;
    private int offset;
    private int length;

    private float[] prevDelta;

    public SliceLayer setAxis(Axis axis)     { this.axis = Objects.requireNonNull(axis); return this; }
    public SliceLayer setOffset(int offset)  { this.offset = offset; return this; }
    public SliceLayer setLength(int length)  { this.length = length; return this; }

    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("SliceLayer.create: inputShape и outputShape должны быть заданы");
        if (offset < 0)
            throw new IllegalArgumentException("SliceLayer: offset >= 0, получено " + offset);
        if (length < 1)
            throw new IllegalArgumentException("SliceLayer: length >= 1, получено " + length);

        switch (axis) {
            case FLAT -> {
                if (offset + length > inputShape.size())
                    throw new IllegalArgumentException(
                            "SliceLayer(FLAT): offset+length=" + (offset + length)
                                    + " > inputSize=" + inputShape.size());
            }
            case CHANNEL -> {
                if (offset + length > inputShape.channels())
                    throw new IllegalArgumentException(
                            "SliceLayer(CHANNEL): offset+length=" + (offset + length)
                                    + " > channels=" + inputShape.channels());
            }
            case SEQ -> {
                if (!inputShape.isSequence())
                    throw new IllegalArgumentException("SliceLayer(SEQ): форма не sequence: " + inputShape);
                if (offset + length > inputShape.seqLen())
                    throw new IllegalArgumentException(
                            "SliceLayer(SEQ): offset+length=" + (offset + length)
                                    + " > seqLen=" + inputShape.seqLen());
            }
        }

        this.prevDelta = new float[inputShape.size()];
        return this;
    }

    @Override
    public float[] forward(float[] input, boolean training) {
        if (input.length != inputShape.size())
            throw new IllegalArgumentException(
                    "SliceLayer.forward: ожидалось " + inputShape.size() + ", получено " + input.length);

        float[] out;
        switch (axis) {
            case FLAT -> {
                out = new float[length];
                System.arraycopy(input, offset, out, 0, length);
            }
            case CHANNEL -> {
                int perChannel = inputShape.height() * inputShape.width()
                        * (inputShape.seqLen() == 0 ? 1 : inputShape.seqLen());
                out = new float[length * perChannel];
                System.arraycopy(input, offset * perChannel, out, 0, length * perChannel);
            }
            case SEQ -> {
                int d = inputShape.channels();
                out = new float[length * d];
                for (int t = 0; t < length; t++) {
                    System.arraycopy(input, (offset + t) * d, out, t * d, d);
                }
            }
            default -> throw new IllegalStateException("Неизвестная ось: " + axis);
        }
        return out;
    }

    @Override
    public float[] backward(float[] deltaNext) {
        // Размер deltaNext зависит от оси: FLAT — length, CHANNEL — length*perChannel, SEQ — length*d.
        int expected;
        switch (axis) {
            case FLAT    -> expected = length;
            case CHANNEL -> expected = length * (inputShape.height() * inputShape.width()
                    * (inputShape.seqLen() == 0 ? 1 : inputShape.seqLen()));
            case SEQ     -> expected = length * inputShape.channels();
            default -> throw new IllegalStateException("Неизвестная ось: " + axis);
        }
        if (deltaNext.length != expected)
            throw new IllegalArgumentException(
                    "SliceLayer.backward: ожидалось " + expected + ", получено " + deltaNext.length);

        Arrays.fill(prevDelta, 0f);
        switch (axis) {
            case FLAT -> System.arraycopy(deltaNext, 0, prevDelta, offset, length);
            case CHANNEL -> {
                int perChannel = inputShape.height() * inputShape.width()
                        * (inputShape.seqLen() == 0 ? 1 : inputShape.seqLen());
                System.arraycopy(deltaNext, 0, prevDelta, offset * perChannel, length * perChannel);
            }
            case SEQ -> {
                int d = inputShape.channels();
                for (int t = 0; t < length; t++) {
                    System.arraycopy(deltaNext, t * d, prevDelta, (offset + t) * d, d);
                }
            }
        }
        return prevDelta;
    }

    /**
     * Преобразование маски:
     * <ul>
     *   <li>{@link Axis#FLAT} → {@code null} (flat не несёт ось SEQ);</li>
     *   <li>{@link Axis#CHANNEL} → маска не меняется (ось SEQ та же);</li>
     *   <li>{@link Axis#SEQ} → срез {@code [offset, offset+length)}.</li>
     * </ul>
     */
    @Override
    public float[] transformMask(float[] inMask) {
        if (inMask == null) return null;
        return switch (axis) {
            case FLAT -> null;
            case CHANNEL -> inMask;
            case SEQ -> {
                float[] out = new float[length];
                System.arraycopy(inMask, offset, out, 0, length);
                yield out;
            }
        };
    }

    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);
        out.writeByte(axis.ordinal());
        out.writeInt(offset);
        out.writeInt(length);
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);
        int ord = in.readByte();
        if (ord < 0 || ord >= Axis.values().length)
            throw new IOException("SliceLayer: некорректный код оси " + ord);
        axis = Axis.values()[ord];
        offset = in.readInt();
        length = in.readInt();
        allocateBuffers();
    }
}