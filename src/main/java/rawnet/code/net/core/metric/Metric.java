package rawnet.code.net.core.metric;

import rawnet.code.net.core.Persistable;

import java.util.List;

import static rawnet.code.net.model.ModelPrintUtil.floatToString;

/**
 * Метрика качества модели. Накапливает статистику по вызовам
 * {@link #update(float[], float[])} и агрегирует её в {@link #result()}.
 * Перед новым прогоном (эпохой/тестом) обязательно вызывается {@link #reset()}.
 * <p>
 * Метрика только <i>измеряет</i> — она не влияет на обучение и не хранится
 * в сериализованной модели.
 */
public interface Metric extends Persistable {
    /** Обработать один пример: prediction — выход сети, target — истина. */
    void update(float[] prediction, float[] target);

    /** Агрегированный результат по всем накопленным примерам. */
    float result();

    /** Сброс накопленной статистики (вызывается перед новым прогоном). */
    void reset();

    /** Имя для логов. */
    String name();

    //Печатаем метрики
    public static StringBuilder print(List<Metric> metrics) {
        StringBuilder sbMetric = new StringBuilder();
        if (metrics != null && !metrics.isEmpty()) {
            //sbMetric.append("Метрики: ");
            for (Metric m : metrics)
                sbMetric.append(m.name()).append('=').append(floatToString(m.result())).append("  ");
        }
        return sbMetric;
    }
}

