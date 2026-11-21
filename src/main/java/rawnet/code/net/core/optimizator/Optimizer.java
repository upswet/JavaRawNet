package rawnet.code.net.core.optimizator;

import rawnet.code.net.core.Persistable;


/**Интерфейс оптимизатора*/
public interface Optimizer extends Persistable {
    /**
     * Обновляет веса по их градиентам.
     *
     * @param weights    матрица весов [rows][cols]
     * @param weightGrad градиенты весов той же размерности
     * @param batchSize  размер батча, по которому накоплены градиенты;
     *                   используется для усреднения
     */
    void updateWeights(float[][] weights, float[][] weightGrad, int batchSize);

    /**
     * Обновляет смещения по их градиентам.
     *
     * @param biases    вектор смещений
     * @param biasGrad  градиенты смещений
     * @param batchSize размер батча
     */
    void updateBiases(float[] biases, float[] biasGrad, int batchSize);

    /**
     * Комбинированное обновление: веса + biases.
     *
     * <p>По умолчанию вызывает {@link #updateWeights} и {@link #updateBiases}
     * последовательно. Оптимизаторы, у которых шаг для весов и biases
     * считается совместно (например, общий счётчик шага), могут
     * переопределить этот метод.
     *
     * <p><b>Контракт для слоёв.</b> Слои с весами и biases вызывают этот
     * метод; слои без biases (например, {@code EmbeddingLayer}) вызывают
     * только {@link #updateWeights}.
     */
    default void update(float[][] weights, float[][] weightGrad,
                        float[] biases, float[] biasGrad, int batchSize) {
        updateWeights(weights, weightGrad, batchSize);
        updateBiases(biases, biasGrad, batchSize);
    }
}
