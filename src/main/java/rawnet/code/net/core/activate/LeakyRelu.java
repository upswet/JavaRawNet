package rawnet.code.net.core.activate;

import lombok.Getter;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Leaky ReLU:
 * Функция out = input при input > 0, иначе alpha * input.
 * Производная deriv = 1 при input >0, иначе alpha
 *
 * Сильные стороны: не насыщается для положительных input, ускоряет сходимость, решает проблему "мёртвых нейронов" (в отличие от обычной ReLU) благодаря небольшому градиенту для отрицательных z (alpha > 0, например 0.01). Проста в вычислении.
 *
 * Слабые стороны: выход не ограничен, что может приводить к большим значениям отрицательная область всё ещё линейна, а не нелинейна.
 *
 * Когда применять: в скрытых слоях глубоких сетей как замена ReLU/Tanh.
 * Хорошо работает в свёрточных и полносвязных сетях.
 */
public class LeakyRelu implements Activate {
    /**
     * @param alpha коэффициент наклона для отрицательной полуоси (обычно 0.01)
     */
    @Getter
    private float alpha;

    public LeakyRelu() { this(0.01f); }
    public LeakyRelu(float alpha) { this.alpha = alpha; }

    @Override
    public void apply(float[] z, float[] out) {
        for (int i = 0; i < z.length; i++)
            out[i] = z[i] > 0 ? z[i] : alpha * z[i];
    }

    @Override
    public void derivative(float[] z, float[] out, float[] dest) {
        for (int i = 0; i < z.length; i++)
            dest[i] = z[i] > 0 ? 1.0f : alpha;
    }

    @Override
    public void save(DataOutputStream out) throws IOException {
        out.writeFloat(alpha);
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        this.alpha = in.readFloat();
    }
}
