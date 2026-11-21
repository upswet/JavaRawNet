package rawnet.code.net.layers;

import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;
import rawnet.code.net.TensorShape;
import rawnet.code.net.core.Persistable;
import rawnet.code.net.core.activate.*;
import rawnet.code.net.core.optimizator.Optimizer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Базовый класс слоя нейросети.
 *
 * <p>Задаёт контракт слоя:
 * <ul>
 *   <li>вычисления — {@link #forward(float[], boolean)} и {@link #backward(float[])};</li>
 *   <li>работа с весами — {@link #zeroGrad()} и {@link #update(Optimizer, int)};</li>
 *   <li>форма входа/выхода — {@link #inputShape()} и {@link #outputShape()};</li>
 *   <li>сериализация параметров — {@link #save(DataOutputStream)} и
 *       {@link #load(DataInputStream)};</li>
 *   <li>хелперы — инициализация весов ({@link #initLimit}) и клипинг градиентов
 *       ({@link #clipByNorm}, {@link #gradNormSq()}, {@link #scaleGrad(float)}).</li>
 * </ul>
 *
 * <p><b>Инварианты:</b>
 * <ul>
 *   <li>{@code inputShape} и {@code outputShape} не {@code null};</li>
 *   <li>{@code forward} возвращает массив, который может совпадать с {@code input}
 *       (алиас); вызывающий код не должен рассчитывать на отдельный буфер;</li>
 *   <li>{@code backward} вызывается только после {@code forward} для того же примера;</li>
 *   <li>{@code backward} <b>накапливает</b> градиенты весов (суммирует).</li>
 * </ul>
 *
 * <p><b>Сериализация.</b> Сохраняются только обученные параметры слоя
 * (веса, смещения, running-статистики). Архитектурные параметры — размеры,
 * активации — пишет и читает модель; слой при загрузке должен быть уже
 * сконструирован с корректными размерами.
 */
@Accessors(chain = true)
@NoArgsConstructor
public abstract class Layer implements Persistable {

    /** Базовый DTO для описания будущего слоя. */
    public abstract static class Dto {
        /**
         * Создаёт слой из данного DTO.
         *
         * @param inputShape вход для слоя (берётся из предыдущего слоя)
         * @return сконструированный слой с заданной формой входа
         */
        public abstract Layer create(TensorShape inputShape);
    }

    @Setter protected TensorShape inputShape;
    @Setter protected TensorShape outputShape;

    /**
     * Алоцируем буферы
     *
     * @return новый экземпляр слоя
     */
    public abstract Layer allocateBuffers();

    /**Инициализация весов для тех слоёв где они есть. Если известна какая ПОСЛЕ слоя будет функция активации, то подгоним инициализацию под неё*/
    public Layer initWeights(Activate activate){throw new UnsupportedOperationException();};

    // ------------------------------------------------------------- Формы

    /** Форма входа слоя; не {@code null}. */
    public final TensorShape inputShape() { return inputShape; }

    /** Форма выхода слоя; не {@code null}. */
    public final TensorShape outputShape() { return outputShape; }

    // ------------------------------------------------------ Контракт слоя

    /**
     * Прямой проход: преобразует входной вектор в выходной.
     *
     * @param input    входной вектор; не {@code null}
     * @param training режим работы слоя:
     *                 {@code true} — обучение (включает dropout, обновляет
     *                 running-статистики BatchNorm и т.п.);
     *                 {@code false} — инференс (dropout отключён, используются
     *                 накопленные статистики)
     * @return выходной вектор; реализация может вернуть тот же массив, что и
     *         {@code input} (алиас), поэтому вызывающий код не должен
     *         рассчитывать на отдельный буфер
     * @throws IllegalArgumentException если длина {@code input} не соответствует
     *                                  форме входа
     */
    public abstract float[] forward(float[] input, boolean training);

    /**
     * Обратный проход: принимает градиент функции потерь по выходу слоя
     * и возвращает градиент функции потерь по входу слоя.
     *
     * <p>Побочный эффект: если у слоя есть веса, <b>накапливает</b> их
     * градиенты (суммирует с уже накопленными в предыдущих вызовах).
     * Перед вызовом {@link #backward} для того же примера должен быть выполнен
     * {@link #forward} — слой использует сохранённые активации.
     *
     * @param deltaNext градиент функции потерь по выходу слоя
     * @return градиент функции потерь по входу слоя; передаётся в
     *         {@code backward} предыдущего слоя
     * @throws IllegalArgumentException если длина {@code deltaNext} не
     *                                  соответствует форме выхода
     * @throws IllegalStateException    если {@code backward} вызван без
     *                                  предшествующего {@code forward}
     */
    public abstract float[] backward(float[] deltaNext);

    /**
     * Обнуляет накопители градиентов весов.
     *
     * <p>Вызывается в начале каждой итерации обучения, перед серией вызовов
     * {@link #backward}. Для слоёв без весов — no-op.
     */
    public void zeroGrad() {}

    /**
     * Шаг оптимизатора: применяет накопленные градиенты к весам слоя.
     *
     * <p>Вызывается один раз на батч, после того как для всех примеров батча
     * выполнены {@link #backward}. Поскольку {@link #backward} <b>суммирует</b>
     * градиенты по примерам, {@code batchSize} используется для усреднения:
     * {@code weight -= optimizer.step(grad / batchSize)}.
     * Для слоёв без весов — no-op.
     *
     * @param optimizer оптимизатор, определяющий правило обновления
     *                  (SGD, Momentum, Adam, ...)
     * @param batchSize количество примеров, по которым накоплен градиент;
     *                  должно быть {@code > 0}
     * @throws IllegalArgumentException если {@code batchSize <= 0}
     */
    public void update(Optimizer optimizer, int batchSize) {}

    /**
     * Подбирает предел равномерной инициализации весов по типу активации:
     * Xavier/Glorot — для Sigmoid/Tanh/Linear, He/Kaiming — для LeakyReLU,
     * малое число — для остальных.
     *
     * @param activate функция активации слоя (может быть {@code null})
     * @param fanIn    число входов нейрона
     * @param fanOut   число выходов нейрона
     * @return предел равномерного распределения {@code [-limit, limit]};
     *         неотрицательное число
     */
    public static float initLimit(Activate activate, int fanIn, int fanOut) {
        if (activate == null
                || activate instanceof Sigmoid
                || activate instanceof Tanh
                || activate instanceof Linear)
            return (float) Math.sqrt(6.0 / (fanIn + fanOut));

        // He-подобный масштаб для «гладких не монотонных» активаций:
        // LeakyReLU, GELU, SiLU, Mish. Для них Xavier (fanIn+fanOut) даёт
        // слишком маленькие веса, и сеть стартует «вяло».
        if (activate instanceof LeakyRelu
                || activate instanceof Gelu
                || activate instanceof Silu
                || activate instanceof Mish)
            return (float) Math.sqrt(6.0 / fanIn);

        return 0.1f;
    }

    // ------------------------------------------------- Сохранение/загрузка

    /**
     * Сохраняет обученные параметры слоя (веса, смещения, running-статистики).
     *
     * <p>Архитектурные параметры (размеры, активации) не сохраняются — их
     * пишет модель. Дополнительно записывается полное имя класса слоя
     * ({@link Class#getName()}), а также формы входа и выхода
     * ({@link TensorShape#save(DataOutputStream)}).
     * Для слоёв без параметров — no-op.
     *
     * @param out поток для записи
     * @throws IOException при ошибке записи
     */
    public void save(DataOutputStream out) throws IOException {
        out.writeUTF(this.getClass().getName());
        inputShape.save(out);
        outputShape.save(out);
    }

    /**
     * Загружает обученные параметры слоя.
     *
     * <p>Слой должен быть уже сконструирован с корректными размерами.
     * Читает формы входа и выхода ({@link TensorShape#load(DataInputStream)});
     * имя класса, записанное в {@link #save(DataOutputStream)}, не проверяется.
     * Для слоёв без параметров — no-op.
     *
     * @param in поток для чтения
     * @throws IOException при ошибке чтения
     */
    public void load(DataInputStream in) throws IOException {
        inputShape = TensorShape.load(in);
        outputShape = TensorShape.load(in);
    }

    // --------------------------------------------------- Клипинг градиентов

    /**
     * Возвращает сумму квадратов всех накопленных градиентов параметров слоя.
     *
     * <p>По умолчанию (слой без весов) возвращает {@code 0.0}.
     *
     * @return сумма квадратов градиентов; {@code >= 0}
     */
    public double gradNormSq() { return 0.0; }

    /**
     * Умножает все накопленные градиенты параметров на {@code scale}.
     *
     * <p>По умолчанию (слой без весов) — no-op.
     *
     * @param scale множитель для градиентов
     */
    public void scaleGrad(float scale) {}
}