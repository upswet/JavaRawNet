package rawnet.code.dataset;

import java.util.Objects;

/**
 * Один пример с <b>N входами</b> ({@code N >= 1}) и целевым вектором.
 *
 * <p>Обобщение {@link Sample} на произвольное число входов.
 * Нужен для multi-query / multi-modal архитектур, где входов больше
 * двух: Q + N−1 KV-групп, text + image + audio и т.п.
 *
 * <p><b>Порядок входов.</b> {@code inputs[i]} соответствует входу
 * {@code i} модели в порядке {@code NInputModel.forwardN(inputs)}.
 * Порядок должен совпадать с {@code Functional.builder(shapes...)}
 * и с {@code NInputLayer.forwardN}.
 */
public final class NInputSample {
    private final float[][] inputs;
    private final float[] targets;

    /**
     * @param inputs  входы в порядке {@code forwardN}; {@code length >= 1},
     *                каждый элемент не {@code null}
     * @param targets целевой вектор; не {@code null}
     */
    public NInputSample(float[][] inputs, float[] targets) {
        Objects.requireNonNull(inputs, "inputs");
        if (inputs.length < 1)
            throw new IllegalArgumentException("NInputSample: inputs.length >= 1, получено " + inputs.length);
        for (int i = 0; i < inputs.length; i++)
            Objects.requireNonNull(inputs[i], "inputs[" + i + "]");
        this.inputs = inputs;
        this.targets = Objects.requireNonNull(targets, "targets");
    }

    public float[][] getInputs() {
        return inputs;
    }

    public float[] getTargets() {
        return targets;
    }

    /**
     * @return число входов в примере (совпадает с длиной {@code inputs})
     */
    public int nInputs() {
        return inputs.length;
    }
}
