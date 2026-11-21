package rawnet.code.net.layers.contract;

/**
 * Слой с <b>N входами</b> (N ≥ 1).
 *
 * <p>Обобщение {@link TwoInputLayer} на произвольное число входов.
 * Нужен для слоёв, которые принимают несколько источников сигнала
 * одновременно: multi-query attention (Q + несколько KV-групп),
 * FiLM-стеки, multi-modal fusion, «N голов → concat → слой».
 *
 * <h2>Контракт</h2>
 * <ul>
 *   <li>{@link #forwardN(float[][], boolean)} — прямой проход; внешний
 *       массив — список входов, каждый {@code float[]} — плоская
 *       раскладка соответствующего входа;</li>
 *   <li>{@link #backwardN(float[])} — обратный проход; возвращает
 *       {@code float[][]} градиентов <b>в том же порядке</b>, что и
 *       входы {@code forwardN}.</li>
 * </ul>
 *
 * <h2>Кто реализует</h2>
 * {@code NInputAttentionLayer}, {@code CrossAttentionNLayer},
 * {@code QKVAttentionLayer}, будущие multi-modal fusion.
 *
 * <h2>Совместимость с TwoInputLayer</h2>
 * {@link TwoInputLayer} <b>не</b> расширяет {@code NInputLayer}:
 * у них разные сигнатуры, и слой не должен реализовывать оба
 * одновременно (маршрутизация в {@code Functional} выбрала бы
 * неоднозначно). Если нужно поддержать и старый, и новый контракт —
 * реализуйте {@link NInputLayer} и делегируйте в 2-входовую версию
 * внутри.
 *
 * <h2>Маски</h2>
 * Без масок. Для слоёв, читающих маски каждого входа, используйте
 * {@link NInputMaskedLayer}.
 */
public interface NInputLayer {

    /**
     * Прямой проход с N входами.
     *
     * @param inputs массив входов; {@code inputs.length} должен совпадать
     *               с ожидаемым числом входов слоя
     * @param training режим обучения
     * @return выход слоя (плоская раскладка)
     */
    float[] forwardN(float[][] inputs, boolean training);

    /**
     * Обратный проход с N входами.
     *
     * @param deltaNext градиент по выходу
     * @return {@code float[N][]} — градиенты по каждому входу в порядке
     *         {@code forwardN}
     */
    float[][] backwardN(float[] deltaNext);
}