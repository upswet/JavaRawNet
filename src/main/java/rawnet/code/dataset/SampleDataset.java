package rawnet.code.dataset;

import lombok.Getter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Датасет: train и test — списки примеров одного типа {@code T}.
 *
 * <p>Тип {@code T} — это тип примера: {@link Sample} для
 * одного входа, {@link NInputSample} для N входов,
 * любой другой — для будущих архитектур.
 *
 * <p><b>Зачем generic.</b> Один контейнер на все типы примеров:
 * {@code Dataset<SampleSimple>} и {@code Dataset<NInputSample>} —
 * это <b>один</b> класс с разными параметрами, а не два разных
 * класса. Меньше дублирования, легче добавлять новые типы.
 *
 * <p><b>Инвариант.</b> {@code trainSamples} и {@code testSamples}
 * должны содержать примеры <b>одного</b> типа. Проверка не делается
 * в конструкторе (дженерики стираются), но компилятор гарантирует
 * это на уровне типов.
 *
 * @param <T> тип примера
 */
@Getter
public final class SampleDataset<T> {

    private final List<T> trainSamples;
    private final List<T> testSamples;

    public SampleDataset(List<T> trainSamples, List<T> testSamples) {
        this.trainSamples = Objects.requireNonNull(trainSamples, "trainSamples");
        this.testSamples  = Objects.requireNonNull(testSamples,  "testSamples");
    }

    /**
     * Разбить общий пул примеров на train/test.
     *
     * <p>Перемешивает {@code allSamples} <b>на месте</b>, затем режет
     * по {@code procent}. Возвращает <b>новый</b> {@code Dataset<T>};
     * исходный список не сохраняется в датасете (копируются подсписки).
     *
     * @param allSamples общий пул; перемешивается
     * @param procent    доля train в {@code (0, 1)}; например, {@code 0.8f}
     * @param <T>        тип примера
     * @return {@code Dataset<T>} с train (первые {@code procent})
     *         и test (остальные)
     * @throws IllegalArgumentException если {@code procent} вне {@code (0, 1)}
     */
    public static <T> SampleDataset<T> split(List<T> allSamples, float procent) {
        if (procent <= 0f || procent >= 1f)
            throw new IllegalArgumentException("procent in (0, 1), получено " + procent);
        Collections.shuffle(allSamples);
        int trainSize = (int) (allSamples.size() * procent);
        return new SampleDataset<>(
                new ArrayList<>(allSamples.subList(0, trainSize)),
                new ArrayList<>(allSamples.subList(trainSize, allSamples.size()))
        );
    }
}