package rawnet.code.net.core.loss;

import rawnet.code.net.core.Persistable;

/**Интерфейс функции потерь*/
public interface Loss extends Persistable {
    /**
     * Вычислить потерю
     * @param predicted выходной вектор (output нейросети)
     * @param target целевой вектор (какой должен быть выход)
     * @return потеря на данном примере (средняя потеря по всем нейронам выходного слоя)
     */
    float loss(float[] predicted, float[] target);

    /**
     * Вычислить производную функции потери
     * @param predicted выходной вектор (output нейросети)
     * @param target целевой вектор (какой должен быть выход)
     * @param deriv вектор производных
     */
    void derivative(float[] predicted, float[] target, float[] deriv);
}
