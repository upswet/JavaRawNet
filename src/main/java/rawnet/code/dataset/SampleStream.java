package rawnet.code.dataset;

import java.io.Closeable;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.function.Supplier;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * Ленивый источник примеров: не хранит всё в памяти, генерирует по
 * требованию.
 *
 * <p><b>Generic {@code <S>}</b> — чтобы работать и с
 * {@link Sample}, и с {@link NInputSample}, и с
 * любым другим типом примера.
 *
 * @param <S> тип примера
 */
public interface SampleStream<S> extends Closeable {

    Iterator<S> open();

    default int size() { return -1; }

    /**
     * Тестовая выборка для метрик.
     *
     * <p><b>Обязателен к реализации.</b> {@code SampleStream} — ленивый
     * источник train; тестовая выборка обычно небольшая и
     * материализованная. Пользователь сам решает, откуда её взять:
     * <ul>
     *   <li>отдельный файл/БД/список — загрузить в {@code getTestSamples()};</li>
     *   <li>нет тестовой выборки — вернуть {@link List#of()} (метрики
     *       на test не считаются);</li>
     *   <li>тест = часть train — вернуть подвыборку (но осторожно:
     *       это уже не «test»).</li>
     * </ul>
     *
     * <p><b>Почему не default.</b> Если бы был {@code default List.of()},
     * пользователь мог бы **забыть** реализовать и не заметить, что
     * метрики на test не считаются. Обязательный метод — компилятор
     * напомнит.
     *
     * @return тестовая выборка; не {@code null} (может быть пустой)
     */
    List<S> getTestSamples();

    @Override default void close() throws IOException {}

    // ---- фабрики ----

    static <S> SampleStream<S> of(List<S> samples, List<S> testSamples) {
        if (samples == null) throw new IllegalArgumentException("samples == null");
        return new SampleStream<>() {
            @Override public Iterator<S> open() { return samples.iterator(); }
            @Override public int size() { return samples.size(); }
            @Override public List<S> getTestSamples() { return testSamples; }
        };
    }

    static <S> SampleStream<S> fromIterator(Factory<S> factory, int size, List<S> testSamples) {
        if (factory == null) throw new IllegalArgumentException("factory == null");
        return new SampleStream<>() {
            @Override public Iterator<S> open() { return factory.create(); }
            @Override public int size() { return size; }
            @Override public List<S> getTestSamples() { return testSamples; }
        };
    }

    static <S> SampleStream<S> fromSupplier(Supplier<S> next, int size, List<S> testSamples) {
        if (next == null) throw new IllegalArgumentException("next == null");
        return new SampleStream<>() {
            @Override
            public Iterator<S> open() {
                return new Iterator<>() {
                    private S buf;
                    private boolean done;
                    @Override public boolean hasNext() {
                        if (done) return false;
                        if (buf == null) {
                            buf = next.get();
                            if (buf == null) done = true;
                        }
                        return buf != null;
                    }
                    @Override public S next() {
                        if (!hasNext()) throw new NoSuchElementException();
                        S r = buf; buf = null; return r;
                    }
                };
            }
            @Override public int size() { return size; }
            @Override public List<S> getTestSamples() { return testSamples; }
        };
    }

    default Stream<S> stream() {
        return StreamSupport.stream(
                Spliterators.spliteratorUnknownSize(open(), Spliterator.ORDERED), false);
    }

    @FunctionalInterface
    interface Factory<S> {
        Iterator<S> create();
    }
}