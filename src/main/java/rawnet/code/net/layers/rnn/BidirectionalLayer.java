package rawnet.code.net.layers.rnn;

import lombok.Setter;
import lombok.experimental.Accessors;
import rawnet.code.net.TensorShape;
import rawnet.code.net.core.activate.LeakyRelu;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.contract.MaskConsumer;
import rawnet.code.net.core.optimizator.Optimizer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Objects;

/**
 * Обобщённый двунаправленный рекуррентный слой.
 *
 * <p>Оборачивает <b>любой</b> RNN, реализующий контракт {@link Layer}:
 * {@link ElmanRnnLayer}, {@link GruLayer} (а в будущем — LSTM и другие).
 * Обёртка делает только то, что специфично для двунаправленности:
 * <ul>
 *   <li>запускает два экземпляра ячейки — «прямой» и «обратный»;</li>
 *   <li>переворачивает вход для обратного прохода
 *       ({@code reversedInput[τ] = input[T-1-τ]});</li>
 *   <li>переворачивает маску для обратного прохода
 *       ({@code reversedMask[τ] = mask[T-1-τ]});</li>
 *   <li>склеивает выходы: {@code out[t] = [h_t^f ‖ h_t^b]} (при
 *       {@code returnSequences = true}) или {@code [h_{T-1}^f ‖ h_0^b]}
 *       (при {@code returnSequences = false});</li>
 *   <li>разводит градиент по двум направлениям и суммирует их на входе.</li>
 * </ul>
 *
 * <h2>Маска</h2>
 * Реализует {@link MaskConsumer}: маска {@code float[T]} (1 — настоящий
 * токен, 0 — PAD). Прямой RNN получает маску как есть; обратный RNN
 * получает перевёрнутую маску — это решает проблему «BiRNN читает PAD
 * с конца»: обратное направление пропускает PAD в начале
 * {@code reversedInput}.
 *
 * <h2>Формат входа/выхода</h2>
 * <ul>
 *   <li>вход: {@code seq(seqLen, features)};</li>
 *   <li>выход при {@code returnSequences = true}:
 *       {@code seq(seqLen, 2·hidden)};</li>
 *   <li>выход при {@code returnSequences = false}:
 *       {@code flat(2·hidden)}.</li>
 * </ul>
 *
 * <h2>Параметризация ячейки</h2>
 * Внутренняя ячейка задаётся через {@link Dto#cellDto} — любой
 * {@link Layer.Dto}, создающий слой с {@code inputShape = seq(seqLen, features)}
 * и {@code outputShape = seq(seqLen, hidden)} или {@code flat(hidden)}.
 *
 * <h2>Сериализация</h2>
 * Сохраняются: формы (через {@code super.save}), {@code seqLen},
 * {@code hidden}, {@code returnSequences}, {@code clipGradNorm},
 * <b>код типа ячейки</b> (один байт: {@code 0} — Elman, {@code 1} — GRU,
 * {@code -1} — произвольный слой), и затем параметры обоих внутренних
 * слоёв. При {@code load} тип ячейки восстанавливается из кода —
 * <b>задавать {@code cellDto} перед загрузкой не нужно</b> для встроенных
 * ячеек.
 */
@Accessors(chain = true)
public class BidirectionalLayer extends Layer implements MaskConsumer {

    /** DTO двунаправленного слоя. */
    public static class Dto extends Layer.Dto {
        public final int seqLen;
        public final int hidden;
        public final boolean returnSequences;
        public final Layer.Dto cellDto;
        public float clipGradNorm = 1f;

        public Dto(int seqLen, int hidden, boolean returnSequences, Layer.Dto cellDto) {
            if (seqLen < 1)
                throw new IllegalArgumentException("BidirectionalLayer.Dto: seqLen >= 1, получено " + seqLen);
            if (hidden < 1)
                throw new IllegalArgumentException("BidirectionalLayer.Dto: hidden >= 1, получено " + hidden);
            this.seqLen = seqLen;
            this.hidden = hidden;
            this.returnSequences = returnSequences;
            this.cellDto = Objects.requireNonNull(cellDto, "cellDto");
        }

        public Dto setClipGradNorm(float clipGradNorm) {
            this.clipGradNorm = clipGradNorm;
            return this;
        }

        @Override
        public Layer create(TensorShape inputShape) {
            return new BidirectionalLayer()
                    .setSeqLen(seqLen)
                    .setHidden(hidden)
                    .setReturnSequences(returnSequences)
                    .setCellDto(cellDto)
                    .setClipGradNorm(clipGradNorm)
                    .setInputShape(inputShape)
                    .setOutputShape(computeOutputShape(inputShape, seqLen, hidden, returnSequences))
                    .allocateBuffers();
        }
    }

    @Setter private int seqLen;
    @Setter private int hidden;
    @Setter private boolean returnSequences;
    @Setter private float clipGradNorm = 1f;
    @Setter private Layer.Dto cellDto;

    private int features;

    private Layer forwardRnn;
    private Layer backwardRnn;

    private float[] output;
    private float[] reversedInput;
    private float[] reversedMask;
    private float[] deltaF;
    private float[] deltaB;
    private float[] prevDelta;

    public BidirectionalLayer() {}

    public static TensorShape computeOutputShape(TensorShape in, int seqLen,
                                                 int hidden, boolean returnSequences) {
        if (in == null)
            throw new IllegalArgumentException("BidirectionalLayer: inputShape == null");
        if (!in.isSequence())
            throw new IllegalArgumentException(
                    "BidirectionalLayer: ожидался sequence-вход, получено " + in);
        if (seqLen < 1)
            throw new IllegalArgumentException("BidirectionalLayer: seqLen >= 1, получено " + seqLen);
        if (hidden < 1)
            throw new IllegalArgumentException("BidirectionalLayer: hidden >= 1, получено " + hidden);
        if (in.seqLen() != seqLen)
            throw new IllegalArgumentException(
                    "BidirectionalLayer: seqLen слоя (" + seqLen
                            + ") != seqLen формы (" + in.seqLen() + ")");
        return returnSequences
                ? TensorShape.seq(seqLen, 2 * hidden)
                : TensorShape.flat(2 * hidden);
    }

    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException(
                    "BidirectionalLayer.create: inputShape и outputShape должны быть заданы");
        if (cellDto == null)
            throw new IllegalStateException(
                    "BidirectionalLayer.create: cellDto должен быть задан (тип внутренней RNN-ячейки)");
        if (!inputShape.isSequence())
            throw new IllegalArgumentException(
                    "BidirectionalLayer.create: ожидался sequence-вход, получено " + inputShape);
        if (inputShape.seqLen() != seqLen)
            throw new IllegalArgumentException(
                    "BidirectionalLayer.create: inputShape.seqLen=" + inputShape.seqLen()
                            + " != seqLen=" + seqLen);
        if (hidden < 1)
            throw new IllegalArgumentException(
                    "BidirectionalLayer.create: hidden >= 1, получено " + hidden);

        this.features = inputShape.channels();

        TensorShape innerIn  = TensorShape.seq(seqLen, features);
        TensorShape innerOut = returnSequences
                ? TensorShape.seq(seqLen, hidden)
                : TensorShape.flat(hidden);

        this.forwardRnn  = createCell(innerIn, innerOut, "forward");
        this.backwardRnn = createCell(innerIn, innerOut, "backward");

        this.output        = new float[outputShape.size()];
        this.reversedInput = new float[inputShape.size()];
        this.reversedMask  = new float[seqLen];
        this.deltaF        = new float[forwardRnn.outputShape().size()];
        this.deltaB        = new float[backwardRnn.outputShape().size()];
        this.prevDelta     = new float[inputShape.size()];
        return this;
    }

    private Layer createCell(TensorShape innerIn, TensorShape innerOut, String role) {
        Layer cell = cellDto.create(innerIn);
        if (!cell.inputShape().equals(innerIn))
            throw new IllegalArgumentException(
                    "BidirectionalLayer: внутренний слой (" + role + ") вернул inputShape "
                            + cell.inputShape() + ", ожидалось " + innerIn);
        if (!cell.outputShape().equals(innerOut))
            throw new IllegalArgumentException(
                    "BidirectionalLayer: внутренний слой (" + role + ") вернул outputShape "
                            + cell.outputShape() + ", ожидалось " + innerOut);
        return cell;
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
                    "BidirectionalLayer.forwardMasked: ожидалось " + inputShape.size()
                            + ", получено " + input.length);
        if (mask != null && mask.length != seqLen)
            throw new IllegalArgumentException(
                    "BidirectionalLayer.forwardMasked: маска длины " + seqLen + ", получено " + mask.length);

        // 1) Прямое направление
        float[] outF;
        if (forwardRnn instanceof MaskConsumer mc && mask != null) {
            outF = mc.forwardMasked(input, mask, training);
        } else {
            outF = forwardRnn.forward(input, training);
        }

        // 2) Обратное направление: переворачиваем вход и маску
        for (int t = 0; t < seqLen; t++) {
            System.arraycopy(input, t * features,
                    reversedInput, (seqLen - 1 - t) * features, features);
        }
        float[] outB;
        if (backwardRnn instanceof MaskConsumer mc && mask != null) {
            for (int t = 0; t < seqLen; t++) reversedMask[t] = mask[seqLen - 1 - t];
            outB = mc.forwardMasked(reversedInput, reversedMask, training);
        } else {
            outB = backwardRnn.forward(reversedInput, training);
        }

        // 3) Склейка
        if (returnSequences) {
            for (int t = 0; t < seqLen; t++) {
                System.arraycopy(outF, t * hidden, output, t * 2 * hidden, hidden);
                System.arraycopy(outB, (seqLen - 1 - t) * hidden,
                        output, t * 2 * hidden + hidden, hidden);
            }
        } else {
            System.arraycopy(outF, 0, output, 0, hidden);
            System.arraycopy(outB, 0, output, hidden, hidden);
        }
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
                    "BidirectionalLayer.backwardMasked: ожидалось " + outputShape.size()
                            + ", получено " + deltaNext.length);
        if (mask != null && mask.length != seqLen)
            throw new IllegalArgumentException(
                    "BidirectionalLayer.backwardMasked: маска длины " + seqLen + ", получено " + mask.length);
        if (forwardRnn == null || backwardRnn == null)
            throw new IllegalStateException(
                    "BidirectionalLayer.backwardMasked: forward не был вызван");

        // 1) Разделяем deltaNext на deltaF и deltaB
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

        // 2) Пропускаем через ячейки
        float[] dInF;
        if (forwardRnn instanceof MaskConsumer mc && mask != null) {
            dInF = mc.backwardMasked(deltaF, mask);
        } else {
            dInF = forwardRnn.backward(deltaF);
        }

        float[] dInB;
        if (backwardRnn instanceof MaskConsumer mc && mask != null) {
            for (int t = 0; t < seqLen; t++) reversedMask[t] = mask[seqLen - 1 - t];
            dInB = mc.backwardMasked(deltaB, reversedMask);
        } else {
            dInB = backwardRnn.backward(deltaB);
        }

        // 3) Суммируем на входе: prevDelta[t] = dInF[t] + dInB[T-1-t]
        for (int t = 0; t < seqLen; t++) {
            int dstOff = t * features;
            int srcOff = (seqLen - 1 - t) * features;
            for (int j = 0; j < features; j++) {
                prevDelta[dstOff + j] = dInF[dstOff + j] + dInB[srcOff + j];
            }
        }
        return prevDelta;
    }

    // ==================== параметры ====================

    @Override
    public void zeroGrad() {
        forwardRnn.zeroGrad();
        backwardRnn.zeroGrad();
    }

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

    // ==================== сериализация ====================

    private static final byte CELL_ELMAN = 0;
    private static final byte CELL_GRU   = 1;
    private static final byte CELL_LSTM = 2;
    private static final byte CELL_OTHER = -1;

    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);
        out.writeInt(seqLen);
        out.writeInt(hidden);
        out.writeBoolean(returnSequences);
        out.writeFloat(clipGradNorm);

        out.writeByte(cellTypeCode(forwardRnn));

        forwardRnn.save(out);
        backwardRnn.save(out);
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);
        this.seqLen          = in.readInt();
        this.hidden          = in.readInt();
        this.returnSequences = in.readBoolean();
        this.clipGradNorm    = in.readFloat();

        byte cellCode = in.readByte();

        this.features = inputShape.channels();
        TensorShape innerIn  = TensorShape.seq(seqLen, features);
        TensorShape innerOut = returnSequences
                ? TensorShape.seq(seqLen, hidden)
                : TensorShape.flat(hidden);

        this.forwardRnn  = createCellByCode(cellCode, innerIn, innerOut);
        this.backwardRnn = createCellByCode(cellCode, innerIn, innerOut);

        forwardRnn.load(in);
        backwardRnn.load(in);

        allocateBuffersAfterLoad();
    }

    private void allocateBuffersAfterLoad() {
        this.output        = new float[outputShape.size()];
        this.reversedInput = new float[inputShape.size()];
        this.reversedMask  = new float[seqLen];
        this.deltaF        = new float[forwardRnn.outputShape().size()];
        this.deltaB        = new float[backwardRnn.outputShape().size()];
        this.prevDelta     = new float[inputShape.size()];
    }

    private static byte cellTypeCode(Layer cell) {
        if (cell instanceof ElmanRnnLayer) return CELL_ELMAN;
        if (cell instanceof GruLayer)      return CELL_GRU;
        if (cell instanceof LstmLayer)     return CELL_LSTM;
        return CELL_OTHER;
    }

    private Layer createCellByCode(byte code, TensorShape innerIn, TensorShape innerOut)
            throws IOException {
        Layer cell;
        switch (code) {
            case CELL_ELMAN -> cell = new ElmanRnnLayer.Dto(seqLen, hidden, new LeakyRelu(), returnSequences).create(innerIn);
            case CELL_GRU -> cell = new GruLayer.Dto(seqLen, hidden, returnSequences).create(innerIn);
            case CELL_LSTM -> cell = new LstmLayer.Dto(seqLen, hidden, returnSequences).create(innerIn);
            case CELL_OTHER -> {
                if (cellDto == null)
                    throw new IOException(
                            "BidirectionalLayer.load: тип ячейки CELL_OTHER, "
                                    + "но cellDto не задан до load — "
                                    + "задайте cellDto через setCellDto(...)");
                cell = cellDto.create(innerIn);
            }
            default -> throw new IOException(
                    "BidirectionalLayer.load: неизвестный код типа ячейки " + code);
        }
        if (!cell.outputShape().equals(innerOut))
            throw new IOException(
                    "BidirectionalLayer.load: ячейка вернула outputShape "
                            + cell.outputShape() + ", ожидалось " + innerOut);
        return cell;
    }
}