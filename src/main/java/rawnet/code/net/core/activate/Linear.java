package rawnet.code.net.core.activate;

import java.util.Arrays;

/**
 * Линейная активация (тождественное отображение): out = input.
 * 
 * Сильные стороны: не искажает значения, градиент всегда равен 1, не страдает от насыщения.
 * 
 * Слабые стороны: несколько последовательных линейных слоёв эквивалентны одному,поэтому обычно используется только на выходном слое.
 * 
 * Когда применять: выходной слой задач регрессии (предсказание непрерывных значений).
 * Рекомендуется использовать с {@see MSELoss} или {@see MAELoss}.
 */
public class Linear implements Activate {
    @Override
    public void apply(float[] input, float[] out) {
        System.arraycopy(input, 0, out, 0, input.length);
    }

    @Override
    public void derivative(float[] input, float[] out, float[] dest) {
        Arrays.fill(dest, 1.0f);
    }
}
