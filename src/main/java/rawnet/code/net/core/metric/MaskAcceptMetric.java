package rawnet.code.net.core.metric;

/**
 * Опциональная маска для {@link Metric}.
 *
 * <p>Маска — {@code float[]} длины {@code T}: {@code 1} — валидная позиция,
 * {@code 0} — PAD/ignore (пропускается). {@code null} или пустая маска —
 * поведение как у обычного {@link #update}.
 */
public interface MaskAcceptMetric {

    /** Обновить статистику с маской. Позиции, где {@code mask[t] == 0}, игнорируются. */
    void updateMasked(float[] prediction, float[] target, float[] mask);
}
