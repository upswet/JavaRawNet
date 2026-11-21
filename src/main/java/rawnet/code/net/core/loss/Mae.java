package rawnet.code.net.core.loss;

/**
 * Средняя абсолютная ошибка (MAE):
 * loss = (1/N) * Σ|pred_i - target_i|
 * derivative = 1 если pred > target, -1 если pred < target и 0 если pred = target
 *
 * Сильные стороны: менее чувствительна к выбросам, чем MSE, градиент постоянен по знаку (±1).
 *
 * Слабые стороны:градиент не зависит от величины ошибки, может медленнее сходиться вблизи оптимума; не дифференцируема в нуле (здесь возвращается 0).
 *
 * Рекомендуемое сочетание: выходной слой с Linear, реже с другими.
 * Не использовать с сигмоидой (так как сигмоида не может стать точно "ноль" и поэтому всегда "тянет" вниз)
 */
public class Mae implements Loss {
    public float loss(float[] p, float[] t) {
        float s = 0;
        for (int i = 0; i < p.length; i++) s += Math.abs(p[i] - t[i]);
        return s / p.length;
    }

    public void derivative(float[] p, float[] t, float[] d) {
        for (int i = 0; i < p.length; i++) {
            float diff = p[i] - t[i];
            d[i] = diff > 0 ? 1 : diff < 0 ? -1 : 0;
        }
    }
}
