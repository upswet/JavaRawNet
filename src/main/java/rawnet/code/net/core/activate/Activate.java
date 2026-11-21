package rawnet.code.net.core.activate;

import rawnet.code.net.core.Persistable;

/**Интерфейс функции активации нейрона*/
public interface Activate extends Persistable {
    /**
     * Вычислить значение функции (fun(input) -> output)
     * @param input входной вектор
     * @param output выходной вектор
     */
    void apply(float[] input, float[] output);

    /**
     * Вычислить производную функции (fun`(input) -> deriv)
     * @param input входной вектор
     * @param output выходной вектор вычисленный ранее через apply. Может использоваться для ускорения вычислений
     * @param deriv вектор производных
     */
    void derivative(float[] input, float[] output, float[] deriv);
}
