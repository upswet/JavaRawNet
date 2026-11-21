package rawnet.code.net;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Форма тензора одного примера.
 * <ul>
 *   <li>{@code flat(N)}          — {@code (N,1,1,0)} — MLP/RNN-вход;</li>
 *   <li>{@code cnn(C,H,W)}       — {@code (C,H,W,0)} — CNN;</li>
 *   <li>{@code seq(T,dModel)}    — {@code (dModel,1,1,T)} — Transformer/RNN-выход.</li>
 * </ul>
 * Инварианты:
 * <ul>
 *   <li>{@code channels,height,width >= 1};</li>
 *   <li>{@code seqLen == 0} — не последовательность, иначе {@code seqLen >= 1};</li>
 *   <li>{@code seqLen > 0 => height == width == 1};</li>
 *   <li>{@code size() >= 1} и не переполняет {@code int}.</li>
 * </ul>
 * <p>
 * <b>Почему одна ось «channels» для всех</b>: и CNN-канал, и MLP-признак,
 * и dModel — это «ведущая» размерность, вдоль которой работает последняя
 * линейная комбинация. Один record — один код проверок.
 * <p>
 * <b>seqLen</b> отделён от {@code channels}, потому что у Transformer
 * «пространственные» оси — это токены, а не H/W. Флаг {@link #isSequence()}
 * говорит слоям (RNN/Transformer), что перед ними.
 * <p>
 * <b>Замечание:</b> {@code cnn(C,1,1)} и {@code flat(C)} дают одинаковый
 * набор чисел и одинаковый {@link #size()}, но {@link #isFlat()} вернёт
 * {@code true} в обоих случаях. Различать их на уровне формы нельзя —
 * это ответственность вызывающего кода.
 */
public record TensorShape(int channels, int height, int width, int seqLen) {

    public TensorShape {
        if (channels < 1 || height < 1 || width < 1) throw new IllegalArgumentException("Некорректная форма: C=" + channels + ", H=" + height + ", W=" + width);
        if (seqLen < 0) throw new IllegalArgumentException("seqLen >= 0, получено " + seqLen);
        // Инвариант «одна не единичная ось для flat/seq»
        if (seqLen > 0 && (height != 1 || width != 1)) throw new IllegalArgumentException("sequence-форма требует H=W=1, получено " + height + "x" + width);
        // Защита от переполнения int при вычислении size().
        long total = (long) channels * height * width * (seqLen == 0 ? 1 : seqLen);
        if (total > Integer.MAX_VALUE) throw new IllegalArgumentException("size overflow: " + total);
    }

    /** MLP/RNN-вход: {@code (size,1,1,0)}. */
    public static TensorShape flat(int size)              { return new TensorShape(size, 1, 1, 0); }

    /** CNN-карта: {@code (c,h,w,0)}. */
    public static TensorShape cnn(int c, int h, int w)    { return new TensorShape(c, h, w, 0); }

    /**
     * Transformer/RNN: {@code (dModel,1,1,seqLen)}.
     * Требует {@code seqLen >= 1}; нулевая длина запрещена, чтобы не
     * получить молча {@code flat(dModel)}.
     */
    public static TensorShape seq(int seqLen, int dModel) {
        if (seqLen < 1) throw new IllegalArgumentException("seq: seqLen >= 1, получено " + seqLen);
        return new TensorShape(dModel, 1, 1, seqLen);
    }

    /** Полное число элементов одного примера; гарантированно {@code >= 1}. */
    public int size() {
        return channels * height * width * (seqLen == 0 ? 1 : seqLen);
    }

    /** {@code (N,1,1,0)}. Заметьте: {@code cnn(C,1,1)} тоже даёт {@code isFlat()==true}. */
    public boolean isFlat()     { return height == 1 && width == 1 && seqLen == 0; }

    /** {@code seqLen > 0}; тогда {@code height == width == 1}. */
    public boolean isSequence() { return seqLen > 0; }

    /** Не flat и не sequence: хотя бы одна из H/W > 1. */
    public boolean isCnn()      { return !isFlat() && !isSequence(); }

    /** CNN: тот же spatial, другие каналы. */
    public TensorShape withChannels(int newC) {
        requireCnn();
        return new TensorShape(newC, height, width, 0);
    }

    /** Sequence: та же длина, другой dModel. */
    public TensorShape withDModel(int newD) {
        if (!isSequence()) throw new IllegalStateException("не sequence: " + this);
        return new TensorShape(newD, 1, 1, seqLen);
    }

    /**
     * Совместимость для concat по канальной оси: совпадают H, W и seqLen.
     * Каналы (ведущая ось) могут различаться.
     */
    public boolean sameSpatialForConcat(TensorShape o) {
        return height == o.height && width == o.width && seqLen == o.seqLen;
    }

    /** @deprecated используйте {@link #sameSpatialForConcat(TensorShape)}. */
    @Deprecated
    public boolean sameLeadingSpatial(TensorShape o) {
        return sameSpatialForConcat(o);
    }

    private void requireCnn() {
        if (!isCnn()) throw new IllegalStateException("не CNN-форма: " + this);
    }

    // --- IO ---

    public void save(DataOutputStream out) throws IOException {
        out.writeInt(channels);
        out.writeInt(height);
        out.writeInt(width);
        out.writeInt(seqLen);
    }

    /**
     * Читает форму из потока. При некорректных значениях бросает
     * {@link IOException} (а не {@link IllegalArgumentException}),
     * чтобы вызывающий код мог единообразно обрабатывать ошибки сети.
     */
    public static TensorShape load(DataInputStream in) throws IOException {
        int c = in.readInt();
        int h = in.readInt();
        int w = in.readInt();
        int t = in.readInt();
        try {
            return new TensorShape(c, h, w, t);
        } catch (IllegalArgumentException e) {
            throw new IOException("Некорректная TensorShape из потока: " + e.getMessage(), e);
        }
    }

    @Override public String toString() {
        return "(" + channels + "," + height + "," + width + (seqLen > 0 ? ";T=" + seqLen : "") + ")";
    }
}