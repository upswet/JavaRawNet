package rawnet.code.net.layers.contract;

/**
 * Слой с N входами, который читает маски <b>каждого</b> входа.
 *
 * <p>Обобщение {@link TwoInputMaskedLayer}. У каждого входа может быть
 * своя маска (например, у Q — {@code Tq}, у каждой KV-группы — своя
 * {@code Tkv_i}). Слой сам решает, как их использовать.
 *
 * <h2>Контракт</h2>
 * <ul>
 *   <li>{@link #forwardNMasked(float[][], float[][], boolean)} —
 *       прямой проход с масками; {@code masks[i]} — маска входа
 *       {@code i} ({@code null} — вход без маски);</li>
 *   <li>{@link #backwardNMasked(float[], float[][])} — обратный
 *       проход с масками.</li>
 * </ul>
 *
 * <h2>Маршрутизация в Functional</h2>
 * <pre>{@code
 * if (layer instanceof NInputMaskedLayer nm
 *         && allMasksPresent(inputs)) {
 *     out = nm.forwardNMasked(inputs, masks, training);
 * } else if (layer instanceof NInputLayer n) {
 *     out = n.forwardN(inputs, training);
 * }
 * }</pre>
 * То есть masked-версия используется, только если маски есть у
 * <b>всех</b> входов. Иначе — обычная (как будто все токены настоящие).
 */
public interface NInputMaskedLayer {

    /**
     * Прямой проход с масками.
     *
     * @param inputs массив входов
     * @param masks  массив масок той же длины, что {@code inputs};
     *               {@code masks[i]} может быть {@code null}
     * @param training режим обучения
     * @return выход слоя
     */
    float[] forwardNMasked(float[][] inputs, float[][] masks, boolean training);

    /**
     * Обратный проход с масками.
     *
     * @param deltaNext градиент по выходу
     * @param masks     те же маски, что в {@code forwardNMasked}
     * @return {@code float[N][]} — градиенты по каждому входу
     */
    float[][] backwardNMasked(float[] deltaNext, float[][] masks);
}