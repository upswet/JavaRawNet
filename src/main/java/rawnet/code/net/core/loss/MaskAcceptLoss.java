package rawnet.code.net.core.loss;

/**
 * Опциональная маска для {@link Loss}.
 *
 * <p>Маска — {@code float[]} длины {@code T} (по позициям последовательности):
 * {@code 1} — валидная позиция, {@code 0} — PAD/ignore (пропускается).
 * {@code null} или пустая маска — поведение как у обычного {@link #loss}/{@link #derivative}.
 *
 * <p><b>Зачем отдельный интерфейс.</b> {@link Loss} — общий контракт для BCE,
 * MSE, Focal, SequenceCE; маска имеет смысл только для <b>последовательных</b>
 * функций (SequenceCE). Отдельный интерфейс позволяет {@code ModelTraining}
 * проверить {@code loss instanceof MaskAcceptLoss} и вызвать masked-версию,
 * не зная конкретной реализации.
 */
public interface MaskAcceptLoss {

    /** Loss с маской. Позиции, где {@code mask[t] == 0}, игнорируются. */
    float lossMasked(float[] predicted, float[] target, float[] mask);

    /** Градиент с маской. Для {@code mask[t] == 0} — нули. */
    void derivativeMasked(float[] predicted, float[] target, float[] deriv, float[] mask);
}