package rawnet.code.net.core.activate;

/**
 * SiLU (Sigmoid Linear Unit), он же Swish.
 *
 * <p>Функция: {@code out = x · σ(x)}, где {@code σ} — сигмоида.
 * Простыми словами: «взвешенный вход» — положительные значения почти
 * не меняются, отрицательные мягко гасятся, но не обнуляются.
 *
 * <p>Производная: {@code deriv = σ(x) · (1 + x · (1 − σ(x)))}.
 *
 * <h3>Сильные стороны</h3>
 * Гладкая, не монотонная около нуля (небольшой «провал» при {@code x ≈ -1.28}),
 * без «мёртвых» зон. Стандарт в EfficientNet, ConvNeXt, MobileNetV3,
 * многих современных CNN. Часто чуть лучше ReLU/GELU на CNN-задачах.
 *
 * <h3>Слабые стороны</h3>
 * Дороже ReLU ({@code exp}); выигрыш на маленьких MLP — в пределах шума.
 *
 * <h3>Когда применять</h3>
 * Скрытые слои современных CNN (EfficientNet-подобные архитектуры),
 * глубоких MLP. Для Transformer'ов чаще берут GELU, для RNN — внутренние
 * {@code tanh}/{@code sigmoid}.
 */
public class Silu implements Activate {

    @Override
    public void apply(float[] input, float[] output) {
        for (int i = 0; i < input.length; i++) {
            float x = input[i];
            output[i] = x * sigmoid(x);
        }
    }

    @Override
    public void derivative(float[] input, float[] output, float[] deriv) {
        // deriv = σ(x) · (1 + x·(1−σ(x)))
        for (int i = 0; i < input.length; i++) {
            float x = input[i];
            float s = sigmoid(x);
            deriv[i] = s * (1f + x * (1f - s));
        }
    }

    /** Сигмоида с защитой от переполнения {@code exp} на больших |x|. */
    private static float sigmoid(float x) {
        if (x >= 0f) {
            float e = (float) Math.exp(-x);
            return 1f / (1f + e);
        } else {
            float e = (float) Math.exp(x);
            return e / (1f + e);
        }
    }
}