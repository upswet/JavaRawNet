package rawnet.code.net.core.activate;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * GELU (Gaussian Error Linear Unit).
 *
 * <p>Функция: {@code out = x · Φ(x)}, где {@code Φ} — функция распределения
 * стандартного нормального закона. Простыми словами: активация «мягко»
 * пропускает положительные значения и «мягко» гасит отрицательные;
 * в отличие от ReLU, переход около нуля гладкий, и небольшие отрицательные
 * значения не обнуляются полностью.
 *
 * <p>Производная:
 * <ul>
 *   <li>точная: {@code deriv = Φ(x) + x · φ(x)}, где {@code φ} — плотность
 *       стандартного нормального закона;</li>
 *   <li>tanh-приближение: см. {@link #Gelu(boolean) approximate}.</li>
 * </ul>
 *
 * <h3>Две версии</h3>
 * <ul>
 *   <li><b>Точная</b> ({@code approximate = false}) — через
 *       {@link Math#erf}. Математически корректна, но {@code erf} заметно
 *       дороже {@code exp}/{@code tanh}.</li>
 *   <li><b>tanh-приближение</b> ({@code approximate = true}) — формула
 *       {@code 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))}. Стандарт
 *       де-факто (PyTorch {@code approximate='tanh'}, TensorFlow, JAX);
 *       визуально и по качеству почти не отличается от точной, но
 *       дешевле. Используется в BERT/GPT/ViT-реализациях.</li>
 * </ul>
 *
 * <h3>Сильные стороны</h3>
 * Гладкая, не монотонная около нуля (есть небольшой «провал» при
 * {@code x ≈ -0.75}), не имеет «мёртвых» зон, как ReLU. Стандарт для
 * Transformer'ов; часто даёт +0.5–2% к качеству относительно ReLU/LeakyReLU
 * в глубоких сетях.
 *
 * <h3>Слабые стороны</h3>
 * Дороже ReLU ({@code erf}/{@code tanh}/{@code exp}); для маленьких MLP
 * выигрыш может быть в пределах шума.
 *
 * <h3>Когда применять</h3>
 * Скрытые слои Transformer'ов (BERT, GPT, ViT), современных CNN
 * (EfficientNet-подобных), глубоких MLP. Для RNN-ячеек не используется —
 * там свои внутренние {@code tanh}/{@code sigmoid}.
 */
public class Gelu implements Activate {

    /**
     * {@code true} — tanh-приближение (дешевле, стандарт де-факто);
     * {@code false} — точная формула через {@link Math#erf}.
     */
    private boolean approximate;

    /** Точная версия (через {@code erf}). */
    public Gelu() { this(false); }

    /**
     * @param approximate {@code true} — tanh-приближение;
     *                    {@code false} — точная формула
     */
    public Gelu(boolean approximate) { this.approximate = approximate; }

    /** @return {@code true}, если используется tanh-приближение */
    public boolean isApproximate() { return approximate; }

    @Override
    public void apply(float[] input, float[] output) {
        if (approximate) {
            // 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
            final float c = (float) Math.sqrt(2.0 / Math.PI);
            for (int i = 0; i < input.length; i++) {
                float x = input[i];
                float inner = c * (x + 0.044715f * x * x * x);
                output[i] = 0.5f * x * (1f + (float) Math.tanh(inner));
            }
        } else {
            // x·Φ(x), Φ(x) = 0.5·(1 + erf(x/√2))
            final float invSqrt2 = (float) (1.0 / Math.sqrt(2.0));
            for (int i = 0; i < input.length; i++) {
                float x = input[i];
                float phi = 0.5f * (1f + erf(x * invSqrt2));
                output[i] = x * phi;
            }
        }
    }

    @Override
    public void derivative(float[] input, float[] output, float[] deriv) {
        if (approximate) {
            // d/dx [0.5·x·(1+tanh(u))], u = c·(x + 0.044715·x³)
            // = 0.5·(1+tanh(u)) + 0.5·x·(1−tanh²(u))·u'
            // u' = c·(1 + 3·0.044715·x²)
            final float c = (float) Math.sqrt(2.0 / Math.PI);
            for (int i = 0; i < input.length; i++) {
                float x = input[i];
                float inner = c * (x + 0.044715f * x * x * x);
                float t = (float) Math.tanh(inner);
                float du = c * (1f + 3f * 0.044715f * x * x);
                deriv[i] = 0.5f * (1f + t) + 0.5f * x * (1f - t * t) * du;
            }
        } else {
            // d/dx [x·Φ(x)] = Φ(x) + x·φ(x), φ(x) = exp(-x²/2)/√(2π)
            final float invSqrt2 = (float) (1.0 / Math.sqrt(2.0));
            final float invSqrt2Pi = (float) (1.0 / Math.sqrt(2.0 * Math.PI));
            for (int i = 0; i < input.length; i++) {
                float x = input[i];
                float phi = 0.5f * (1f + erf(x * invSqrt2));
                float pdf = invSqrt2Pi * (float) Math.exp(-0.5f * x * x);
                deriv[i] = phi + x * pdf;
            }
        }
    }

    /**
     * Аппроксимация функции ошибок {@code erf} (Abramowitz & Stegun 7.1.26).
     * Максимальная абсолютная погрешность ≈ 1.5e-7 — с запасом хватает для
     * float-вычислений; используется, чтобы не тащить {@code java.lang.Math.erf}
     * (он появился только в Java 20+; если целевая версия — 21+, можно
     * заменить на {@code Math.erf}).
     */
    private static float erf(float x) {
        // Знак: erf(-x) = -erf(x)
        float sign = (x < 0f) ? -1f : 1f;
        float ax = Math.abs(x);
        // Константы A&S 7.1.26
        float t = 1f / (1f + 0.3275911f * ax);
        float y = 1f
                - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                - 0.284496736f) * t + 0.254829592f) * t * (float) Math.exp(-ax * ax);
        return sign * y;
    }

    @Override
    public void save(DataOutputStream out) throws IOException {
        out.writeBoolean(approximate);
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        this.approximate = in.readBoolean();
    }
}