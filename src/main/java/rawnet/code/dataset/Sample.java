package rawnet.code.dataset;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

/**
 * Единичный обучающй пример для MLP и CNN нейросетей
 */
@RequiredArgsConstructor
@Getter
public class Sample {
    /**Входной вектор*/
    private final float[] inputs;
    /**Выходной вектор*/
    private final float[] targets;
}