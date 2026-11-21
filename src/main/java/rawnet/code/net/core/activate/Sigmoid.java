package rawnet.code.net.core.activate;

/**
 * Сигмоида (логистическая функция)
 * Функция out = 1 / (1 + e^{-input}).
 * Производная deriv = out * (1 - out)
 *
 * Сильные стороны: гладкая, ограниченный выход (0,1), интерпретируется как вероятность.
 *
 * Слабые стороны: насыщение при больших |z| приводит к исчезающему градиенту,
 * выход не центрирован около нуля, что замедляет обучение глубоких сетей.
 *
 * Когда применять: в выходном слое для задач бинарной классификации (особенно если объект может принадлежать более одному классу)
 */
public class Sigmoid implements Activate {
    @Override
    public void apply(float[] input, float[] output) {
        for (int i = 0; i < input.length; i++)
            output[i] = 1.0f / (1.0f + (float) Math.exp(-input[i]));
    }

    @Override
    public void derivative(float[] input, float[] output, float[] deriv) {
        for (int i = 0; i < output.length; i++)
            deriv[i] = output[i] * (1.0f - output[i]);
    }
}
