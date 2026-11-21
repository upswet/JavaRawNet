package rawnet.code.net.core.loss;

/**
 * Среднеквадратичная ошибка (MSE):
 * loss = (1/N) * Σ(pred_i - target_i)^2.
 * derivative = pred - target
 *
 * Сильные стороны: гладкая, выпуклая, штрафует большие ошибки сильнее (квадратично), градиент линеен по ошибке. Хорошо подходит для задач регрессии.
 *
 * Слабые стороны: чувствительна к выбросам, может доминировать.
 *
 * Рекомендуемое сочетание: выходной слой с Linear активацией (без нелинейности).
 * При использовании с сигмоидой/softmax может приводить к медленному обучению из-за насыщения.
 */
public class Mse implements Loss {
    public float loss(float[] p, float[] t) {
        float s = 0;
        for (int i = 0; i < p.length; i++) {
            float d = p[i] - t[i];
            s += d * d;
        }
        return s / p.length;
    }

    public void derivative(float[] p, float[] t, float[] d) {
        for (int i = 0; i < p.length; i++)
            d[i] = p[i] - t[i];
    }
}
