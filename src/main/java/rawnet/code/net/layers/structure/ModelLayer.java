package rawnet.code.net.layers.structure;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;
import rawnet.code.net.TensorShape;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.model.functional.Functional;
import rawnet.code.net.model.Model;
import rawnet.code.net.model.sequence.Sequence;
import rawnet.code.net.core.optimizator.Optimizer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Objects;

/**
 * Слой-обёртка над готовой {@link Model}: позволяет использовать любую
 * сеть как строительный блок другой сети.
 *
 * <p>Делегирует весь «слоевой» контракт вложенной модели:
 * <ul>
 *   <li>{@link #forward(float[], boolean)} → {@link Model#forward(float[], boolean)};</li>
 *   <li>{@link #backward(float[])} → {@link Model#backward(float[])};</li>
 *   <li>{@link #zeroGrad()} / {@link #update} / {@link #gradNormSq} /
 *       {@link #scaleGrad(float)} → соответствующие методы {@link Model}.</li>
 * </ul>
 *
 * <p>Вложенная модель <b>обязана</b> поддерживать слоевой контракт
 * (иначе соответствующий метод бросит {@link UnsupportedOperationException}
 * при первом обращении). Стандартные {@link Sequence} и {@link Functional}
 * его поддерживают.
 *
 * <p>Формы: {@code inputShape == model.inputShape()},
 * {@code outputShape == model.outputShape()}.
 *
 * <h3>Заморозка (transfer learning)</h3>
 * Флаг {@link #isTrainable() trainable} управляет тем, обновляются ли веса
 * вложенной модели при обучении внешней сети:
 * <ul>
 *   <li>{@code trainable = true} (по умолчанию) — блок обучается вместе
 *       с внешней сетью: {@code zeroGrad}/{@code update}/{@code gradNormSq}/
 *       {@code scaleGrad} делегируются вложенной модели;</li>
 *   <li>{@code trainable = false} — веса блока «заморожены»:
 *       {@code zeroGrad}/{@code update}/{@code scaleGrad} — no-op,
 *       {@link #gradNormSq()} возвращает {@code 0} (градиенты блока не
 *       участвуют в глобальном клипинге внешней сети).</li>
 * </ul>
 *
 * <p><b>Важно:</b> {@link #backward(float[])} работает в обоих режимах
 * одинаково. Заморозка отключает только <i>обновление</i> весов, но не
 * <i>прохождение градиента</i> сквозь блок — иначе слои внешней сети,
 * стоящие перед блоком, не смогли бы обучаться. Это стандартная семантика
 * {@code requires_grad=False} в PyTorch.
 *
 * <h3>Сериализация</h3>
 * Пишет {@link Class#getName()} и формы (как любой {@link Layer}), затем
 * флаг {@code trainable} (один байт) и вложенную модель через
 * {@link Model#save(DataOutputStream)}. При загрузке вложенная модель
 * читается через {@link Model#load(DataInputStream)}, поэтому архитектура
 * блока восстанавливается полностью.
 */
@NoArgsConstructor
public class ModelLayer extends Layer {

    /** DTO слоя-обёртки над готовой моделью. */
    @RequiredArgsConstructor
    public static class Dto extends Layer.Dto {
        /** Вложенная модель; не {@code null}. */
        public final Model model;

        /**
         * Обучаемость блока; по умолчанию {@code true}.
         * {@code false} — веса блока заморожены (transfer learning).
         */
        public final boolean trainable;

        /** Эквивалентно {@code new Dto(model, true)}. */
        public Dto(Model model) {
            this(model, true);
        }

        @Override
        public Layer create(TensorShape inputShape) {
            Objects.requireNonNull(model, "model");
            if (!model.inputShape().equals(inputShape))
                throw new IllegalArgumentException(
                        "ModelLayer.Dto: inputShape модели " + model.inputShape()
                                + " не совпадает с формой предыдущего слоя " + inputShape);
            return new ModelLayer()
                    .setModel(model)
                    .setTrainable(trainable)
                    .setInputShape(inputShape)
                    .setOutputShape(model.outputShape())
                    .allocateBuffers();
        }
    }

    @Getter private Model model;

    /**
     * Обучаемость блока. По умолчанию {@code true} — поведение, совместимое
     * с версией без флага. {@code false} — веса заморожены.
     * -- GETTER --
     *
     * @return {@code true}, если веса вложенной модели обновляются при обучении

     */
    @Getter
    private boolean trainable = true;

    /** @return {@code this} */
    public ModelLayer setModel(Model model) {
        this.model = model;
        return this;
    }

    /** @return {@code this} */
    public ModelLayer setTrainable(boolean trainable) {
        this.trainable = trainable;
        return this;
    }

    /** @return вложенная модель (может быть {@code null}, если слой не инициализирован) */
    public Model model() { return model; }

    /**
     * Валидирует, что модель задана, и синхронизирует формы.
     *
     * @return {@code this}
     * @throws IllegalStateException    если {@code model == null} или формы не заданы
     * @throws IllegalArgumentException если формы слоя не совпадают с формами модели
     */
    public Layer allocateBuffers() {
        if (model == null)
            throw new IllegalStateException("ModelLayer.create: model не задана");
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("ModelLayer.create: inputShape и outputShape должны быть заданы");
        if (!inputShape.equals(model.inputShape()))
            throw new IllegalArgumentException(
                    "ModelLayer: inputShape " + inputShape + " != model.inputShape " + model.inputShape());
        if (!outputShape.equals(model.outputShape()))
            throw new IllegalArgumentException(
                    "ModelLayer: outputShape " + outputShape + " != model.outputShape " + model.outputShape());
        return this;
    }

    /**
     * Прямой проход через вложенную модель.
     *
     * <p>Режим {@code training} пробрасывается как есть, поэтому dropout /
     * BatchNorm внутри блока работают корректно независимо от
     * {@link #isTrainable()}: заморозка касается только обновления весов.
     */
    @Override
    public float[] forward(float[] input, boolean training) {
        return model.forward(input, training);
    }

    /**
     * Обратный проход через вложенную модель.
     *
     * <p>Работает в обоих режимах {@link #isTrainable()}: градиент по входу
     * блока должен дойти до предыдущих слоёв внешней сети. В режиме
     * {@code trainable = false} вложенная модель всё равно накапливает
     * градиенты во внутренних буферах — но они не применяются
     * ({@link #update} — no-op) и не участвуют в клипинге
     * ({@link #gradNormSq()} возвращает {@code 0}). Накопленные значения
     * не растут бесконечно: {@link #zeroGrad()} в замороженном режиме —
     * no-op, поэтому буферы остаются в том состоянии, в котором их
     * оставил последний вызов {@code backward}; для инференса это
     * безопасно, для повторного обучения блока — нужно разморозить.
     */
    @Override
    public float[] backward(float[] deltaNext) {
        float[] dIn = model.backward(deltaNext);
        if (!trainable) model.zeroGrad();  // замороженный блок не копит градиенты
        return dIn;
    }

    @Override
    public void zeroGrad() {
        if (trainable) model.zeroGrad();
    }

    @Override
    public void update(Optimizer optimizer, int batchSize) {
        if (trainable) model.update(optimizer, batchSize);
    }

    /**
     * {@inheritDoc}
     *
     * <p>В замороженном режиме возвращает {@code 0}: градиенты блока не
     * должны влиять на глобальный клипинг внешней сети.
     */
    @Override
    public double gradNormSq() {
        return trainable ? model.gradNormSq() : 0.0;
    }

    @Override
    public void scaleGrad(float scale) {
        if (trainable) model.scaleGrad(scale);
    }

    /**
     * Сохраняет: имя класса, формы (через {@code super}), флаг
     * {@code trainable} (один байт) и вложенную модель.
     */
    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);                 // className + inputShape + outputShape
        out.writeBoolean(trainable);

        if (model == null)
            throw new IOException("ModelLayer.save: model == null");
        try {
            model.save(out);
        } catch (UnsupportedOperationException e) {
            throw new IOException("ModelLayer.save: вложенная модель " + model.getClass().getName() + " не поддерживает save(DataOutputStream)", e);
        }
    }

    /**
     * Загружает: формы (через {@code super}), флаг {@code trainable},
     * вложенную модель через {@link Model#load(DataInputStream)}.
     * После чтения {@link #allocateBuffers()} проверяет согласованность форм.
     */
    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);                  // формы
        trainable = in.readBoolean();
        model = Model.load(in);          // вложенная модель
        allocateBuffers();
    }
}