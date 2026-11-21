package rawnet.code.net.core.activate;

/**
 * Mish.
 *
 * <p>Функция: {@code out = x · tanh(softplus(x))}, где
 * {@code softplus(x) = ln(1 + e^x)}. Простыми словами: похожа на SiLU,
 * но «мягче» и с чуть более выраженным провалом для отрицательных
 * значений; на практике часто даёт небольшой прирост в детекторах и
 * сегментаторах.
 *
 * <p>Производная:
 * {@code deriv = tanh(sp) + x · (1 − tanh²(sp)) · σ(x)},
 * где {@code sp = softplus(x)}, {@code σ} — сигмоида (производная softplus).
 *
 * <h3>Сильные стороны</h3>
 * Гладкая, не монотонная, без «мёртвых» зон. Появлялась в YOLOv4 и
 * потомках, в некоторых детекторах/сегментаторах даёт +0.5–1%.
 *
 * <h3>Слабые стороны</h3>
 * Дороже SiLU/GELU ({@code exp} + {@code log1p} + {@code tanh});
 * выигрыш нестабилен и часто в пределах шума.
 *
 * <h3>Когда применять</h3>
 * Когда SiLU/GELU уже попробованы и хочется ещё немного; в детекторах
 * и сегментаторах, где Mish — устоявшийся выбор. Для RNN и маленьких
 * MLP — обычно избыточна.
 */
public class Mish implements Activate {

    @Override
    public void apply(float[] input, float[] output) {
        for (int i = 0; i < input.length; i++) {
            float x = input[i];
            float sp = softplus(x);
            output[i] = x * (float) Math.tanh(sp);
        }
    }

    @Override
    public void derivative(float[] input, float[] output, float[] deriv) {
        // deriv = tanh(sp) + x·(1−tanh²(sp))·σ(x)
        for (int i = 0; i < input.length; i++) {
            float x = input[i];
            float sp = softplus(x);
            float t = (float) Math.tanh(sp);
            float s = sigmoid(x);
            deriv[i] = t + x * (1f - t * t) * s;
        }
    }

    /**
     * softplus с численной устойчивостью:
     * {@code ln(1+e^x) = max(x,0) + ln(1 + e^{-|x|})}.
     * Так {@code exp} не переполняется на больших {@code x}.
     */
    private static float softplus(float x) {
        float ax = Math.abs(x);
        return Math.max(x, 0f) + (float) Math.log1p(Math.exp(-ax));
    }

    /** Сигмоида с защитой от переполнения {@code exp}. */
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