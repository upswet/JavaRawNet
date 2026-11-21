package rawnet.code.net.core.activate;


/**
 * Гиперболический тангенс
 * Функция out = tanh(input).
 * Производная  deriv =  1 - out^2
 *
 * Сильные стороны: выход центрирован около нуля (-1,1), что ускоряет сходимость по сравнению с сигмоидой; градиенты более сильные в окрестности нуля.
 *
 * Слабые стороны: также страдает от насыщения и исчезающих градиентов при больших |input|.
 *
 * Когда применять: в скрытых слоях, особенно в рекуррентных сетях (RNN),где нулевое среднее выхода помогает бороться с взрывом градиентов.
 * Хорошо сочетается с любыми функциями потерь.
 */
public class Tanh implements Activate {
    @Override
    public void apply(float[] input, float[] out) {
        for (int i = 0; i < input.length; i++)
            out[i] = (float) Math.tanh(input[i]);
    }

    @Override
    public void derivative(float[] input, float[] out, float[] deriv) {
        for (int i = 0; i < out.length; i++)
            deriv[i] = 1.0f - out[i] * out[i];
    }
}
