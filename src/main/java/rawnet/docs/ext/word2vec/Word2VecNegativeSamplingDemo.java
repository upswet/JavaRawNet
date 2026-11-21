package rawnet.docs.ext.word2vec;

import rawnet.code.dataset.NInputSample;
import rawnet.code.dataset.SampleStream;
import rawnet.code.net.TensorShape;
import rawnet.code.net.core.loss.BinaryCrossEntropy;
import rawnet.code.net.core.metric.LossMetric;
import rawnet.code.net.core.metric.Metric;
import rawnet.code.net.core.optimizator.AdamW;
import rawnet.code.net.layers.attention.NegativeSamplingLayer;
import rawnet.code.net.layers.core.EmbeddingLayer;
import rawnet.code.net.model.Model;
import rawnet.code.net.model.ModelTraining;
import rawnet.code.net.model.NInputModel;
import rawnet.code.net.model.functional.Functional;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.IntFunction;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Word2vec-NS на корпусе из директории — <b>учебный</b> вариант через
 * {@link Functional} и {@link ModelTraining}.
 *
 * <h2>Пайплайн</h2>
 * <ol>
 *   <li><b>Словарь.</b> Сканируем {@code *.txt} рекурсивно, токенизируем
 *       (буквы/цифры), считаем частоты, отбрасываем слова короче
 *       {@link #MIN_WORD_LENGTH} и реже {@link #MIN_COUNT}, сортируем по
 *       частоте, обрезаем до {@link #MAX_VOCAB}.</li>
 *   <li><b>Граф.</b> Два входа {@code ctx} и {@code tgt}; каждый идёт в
 *       свою {@link EmbeddingLayer} ({@code E_in} и {@code E_out});
 *       результаты — в {@link NegativeSamplingLayer}, который возвращает
 *       {@code sigmoid(<ctx, tgt>)}.</li>
 *   <li><b>Обучение.</b> {@link ModelTraining#train(NInputModel, int, SampleStream, List, ...)}
 *       на ленивом {@link SampleStream} из пар {@code (ctx, tgt, label)}.
 *       Positive — сосед по окну; negatives — {@link #NEGATIVES} случайных
 *       слов по {@code freq^0.75} (линейный поиск — просто и наглядно).</li>
 *   <li><b>Демонстрация.</b> Печать словаря, ближайших соседей для первых
 *       5 слов, аналогий и ближайших соседей для слов запроса.</li>
 * </ol>
 *
 * <h2>Почему dense, а не sparse</h2>
 * У {@link EmbeddingLayer} теперь есть sparse AdamW
 * ({@link EmbeddingLayer.Dto#withSparseAdamW()}), который обновляет только
 * задействованные строки. Но чтобы получить от него выигрыш, нужно
 * <b>вручную</b> собрать {@code touched}-индексы и <b>не звать</b>
 * {@code zeroGrad} в горячем цикле (см. {@code Word2VecNegativeSamplingDemoManual}).
 * {@code ModelTraining} этого не делает: он зовёт
 * {@code zeroGrad} каждый батч (O(vocab·dModel)) и
 * {@code update(opt, batchSize)}, который собирает {@code touched} из
 * {@code EG} (тоже O(vocab·dModel)). Поэтому здесь — обычный dense AdamW:
 * код короче и честно показывает, что связка «NInputModel + Functional +
 * NegativeSamplingLayer» работает. Для реальных корпусов — ручной sparse-цикл.
 *
 * <h2>Совместимость файла модели</h2>
 * {@code EmbeddingLayer.save} с недавних пор пишет дополнительные поля
 * ({@code frozenUpTo}, {@code sparseAdamW}, гиперпараметры sparse AdamW).
 * <b>Старые {@code word2vec.bin} несовместимы</b> — удалите файл при
 * первом запуске после обновления библиотеки.
 */
public final class Word2VecNegativeSamplingDemo {

    // ==================== гиперпараметры ====================

    /** Размерность эмбеддинга. */
    private static final int D_MODEL = 64;
    /** Половина окна контекста: пары {@code (t, t+d)} для {@code |d| <= WINDOW}. */
    private static final int WINDOW = 4;
    /** Сколько negative-примеров на один positive. */
    private static final int NEGATIVES = 3;
    /** Минимальная частота слова для словаря. */
    private static final int MIN_COUNT = 3;
    /** Минимальная длина слова (в символах). */
    private static final int MIN_WORD_LENGTH = 4;
    /** Размер батча (в парах). */
    private static final int BATCH = 512;
    /** Эпох обучения. */
    private static final int EPOCHS = 5;
    /** Learning rate. */
    private static final float LR = 0.01f;
    /** Максимум слов в словаре. */
    private static final int MAX_VOCAB = 30_000;
    /** Степень для negative sampling: {@code freq^0.75} — стандарт word2vec. */
    private static final double NEG_SAMPLING_POWER = 0.75;
    /** Глобальный клип L2-нормы градиента; {@code <= 0} — выключен. */
    private static final float CLIP_NORM = 5f;
    /** Размер буфера перемешивания {@link SampleStream}. */
    private static final int SHUFFLE_BUFFER = 8192;
    /** Subsampling threshold: чем больше, тем сильнее режем частые слова. */
    private static final double SUBSAMPLE_T = 1e-2;
    /** Сколько пар взять для теста (метрики). */
    private static final int TEST_PAIRS = 20_000;
    /** Файл сохранённой модели. */
    private static final String MODEL_FILE = "word2vec.bin";

    /** Токен = последовательность букв/цифр (Unicode). */
    private static final Pattern TOKEN = Pattern.compile(
            "[\\p{L}\\p{N}]+",
            Pattern.UNICODE_CHARACTER_CLASS);

    // ==================== main ====================

    public static void main(String[] args) throws Exception {
        Path corpusDir = Path.of(args.length > 0 ? args[0] : "corpus");
        if (!Files.isDirectory(corpusDir)) {
            System.err.println("Не директория: " + corpusDir.toAbsolutePath());
            return;
        }

        // ---------- 1. Словарь ----------
        System.out.println("[1/4] Сканирую корпус: " + corpusDir.toAbsolutePath());
        Vocab vocab = buildVocab(corpusDir);
        System.out.printf(Locale.US,
                "      слов в словаре: %d (minCount=%d, minLen=%d, max=%d)%n",
                vocab.size(), MIN_COUNT, MIN_WORD_LENGTH, MAX_VOCAB);
        if (vocab.size() < 10) {
            System.err.println("      слишком маленький словарь — увеличьте корпус или уменьшите MIN_COUNT");
            return;
        }

        // ---------- 2. Загрузка или обучение ----------
        Functional model;
        Path modelFile = Path.of(MODEL_FILE);
        if (Files.isRegularFile(modelFile)) {
            System.out.println("[2/4] Загружаю: " + modelFile.toAbsolutePath());
            model = (Functional) Model.load(modelFile);
        } else {
            System.out.println("[2/4] Обучаю.");
            model = trainFunctional(vocab, corpusDir);
            System.out.println("[3/4] Сохраняю: " + modelFile.toAbsolutePath());
            model.save(modelFile);
        }

        // ---------- 3. Достаём эмбеддинги ----------
        EmbeddingLayer eIn  = findEmbedding(model, 0);
        EmbeddingLayer eOut = findEmbedding(model, 1);
        if (eIn == null) { System.err.println("нет EmbeddingLayer в графе"); return; }

        // ---------- 4. Демонстрация ----------
        System.out.println();
        System.out.println("[4/4] Демонстрация эмбеддингов");

        // 4.1 Словарь — первые 40 слов с частотами.
        printVocab(vocab, 40);

        // 4.2 Ближайшие соседи для первых 5 слов словаря.
        System.out.println();
        System.out.println("Ближайшие соседи для первых 10 слов словаря (E_in):");
        int topWords = Math.min(10, vocab.size());
        for (int i = 0; i < topWords; i++) {
            String w = vocab.words.get(i);
            nearest(eIn, vocab, w, 5);
        }

        // 4.5 Соседи для слов запроса — чтобы увидеть семантику.
        System.out.println();
        checkWords(vocab, "гарри", "гермиона", "рон");
        if (eOut != null) {
            nearest(eIn, vocab, "гарри", 5);
            nearest(eIn, vocab, "гермиона", 5);
        }
    }

    // ==================== словарь ====================

    /**
     * Словарь: слово → индекс, частоты и вероятности сохранения для
     * subsampling. Индексы стабильны в рамках одного запуска и одного
     * {@code word2vec.bin}: {@code save}/{@code load} {@link EmbeddingLayer}
     * хранят только число слов, а порядок задаётся здесь. При смене
     * {@link #MIN_COUNT}/{@link #MAX_VOCAB}/{@link #MIN_WORD_LENGTH}
     * удаляйте {@code word2vec.bin}.
     */
    private static final class Vocab {
        final Map<String, Integer> wordToIdx = new HashMap<>();
        final List<String> words = new ArrayList<>();
        final List<Integer> freq  = new ArrayList<>();
        /** Вероятность оставить слово при subsampling; {@code 1.0} — не режем. */
        final List<Double> keepProb = new ArrayList<>();
        /** Сумма {@code freq^NEG_SAMPLING_POWER}; для негативного сэмплинга. */
        double negSamplingTotal = 0.0;

        int size() { return words.size(); }
    }

    /**
     * Строит словарь по корпусу: все {@code *.txt} рекурсивно, токенизация,
     * фильтр по длине и частоте, сортировка по убыванию частоты (при
     * равенстве — по слову), обрезка до {@link #MAX_VOCAB}.
     */
    private static Vocab buildVocab(Path dir) throws IOException {
        Map<String, Integer> counts = new HashMap<>();
        try (Stream<Path> files = Files.walk(dir)) {
            Iterator<Path> it = files.filter(Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".txt"))
                    .iterator();
            int filesRead = 0;
            while (it.hasNext()) {
                String text = Files.readString(it.next(), StandardCharsets.UTF_8);
                Matcher m = TOKEN.matcher(text);
                while (m.find()) {
                    String tok = m.group().toLowerCase(Locale.ROOT);
                    // Фильтр по длине: короткие слова («и», «в», «на») не нужны —
                    // они частые, но семантически пустые, и только раздувают пары.
                    if (tok.length() < MIN_WORD_LENGTH) continue;
                    counts.merge(tok, 1, Integer::sum);
                }
                if (++filesRead % 50 == 0) System.out.println("      файлов: " + filesRead);
            }
            System.out.println("      всего файлов: " + filesRead);
        }

        List<Map.Entry<String, Integer>> sorted = new ArrayList<>(counts.entrySet());
        sorted.removeIf(e -> e.getValue() < MIN_COUNT);
        sorted.sort((x, y) -> {
            int cmp = Integer.compare(y.getValue(), x.getValue());
            return cmp != 0 ? cmp : x.getKey().compareTo(y.getKey());
        });
        if (sorted.size() > MAX_VOCAB) sorted = sorted.subList(0, MAX_VOCAB);

        Vocab v = new Vocab();
        long totalCount = 0;
        for (Map.Entry<String, Integer> e : sorted) {
            v.wordToIdx.put(e.getKey(), v.words.size());
            v.words.add(e.getKey());
            v.freq.add(e.getValue());
            totalCount += e.getValue();
        }
        for (int f : v.freq) v.negSamplingTotal += Math.pow(f, NEG_SAMPLING_POWER);

        // Subsampling: keepProb = (sqrt(f/t) + 1) * t / f, ограничено [0,1].
        // Частые слова выпадают, редкие остаются. Стандарт word2vec.
        for (int f : v.freq) {
            double rel = (double) f / totalCount;
            double keep = (Math.sqrt(rel / SUBSAMPLE_T) + 1.0) * SUBSAMPLE_T / rel;
            v.keepProb.add(Math.min(1.0, keep));
        }
        return v;
    }

    /** Токенизация с понижением регистра. */
    private static List<String> tokenize(String text) {
        List<String> out = new ArrayList<>();
        Matcher m = TOKEN.matcher(text);
        while (m.find()) out.add(m.group().toLowerCase(Locale.ROOT));
        return out;
    }

    // ==================== граф ====================

    /**
     * Собирает {@link Functional} с двумя входами {@code ctx}/{@code tgt},
     * двумя {@link EmbeddingLayer} и {@link NegativeSamplingLayer}, и
     * обучает через {@link ModelTraining}.
     *
     * <p><b>Почему два входа.</b> В word2vec две таблицы: {@code E_in}
     * (вектор контекста) и {@code E_out} (вектор target-слова). Скалярное
     * произведение между ними даёт скор пары. {@code NegativeSamplingLayer}
     * реализует {@code NInputLayer} (2 входа), поэтому
     * {@link Functional.Builder#layerN} собирает его из двух узлов.
     */
    private static Functional trainFunctional(Vocab vocab, Path corpusDir) throws IOException {
        Functional.Builder b = Functional.builder(
                TensorShape.flat(1),   // ctx — один индекс
                TensorShape.flat(1));  // tgt — один индекс

        Functional.Node ctx = b.input();
        Functional.Node tgt = b.kvInput();

        Functional.Node eCtx = b.layer(new EmbeddingLayer.Dto(vocab.size(), D_MODEL), ctx);
        Functional.Node eTgt = b.layer(new EmbeddingLayer.Dto(vocab.size(), D_MODEL), tgt);

        Functional.Node p = b.layerN(new NegativeSamplingLayer.Dto(D_MODEL), eCtx, eTgt);

        Functional model = b.output(p);

        // Тест — небольшая выборка из стрима, чтобы метрики считались быстро.
        List<NInputSample> testSamples = takeTestPairs(vocab, corpusDir, TEST_PAIRS);
        System.out.printf(Locale.US, "      test-пар: %d%n", testSamples.size());

        SampleStream<NInputSample> trainStream = streamPairs(vocab, corpusDir);

        BinaryCrossEntropy loss = new BinaryCrossEntropy();
        AdamW opt = new AdamW(LR);
        List<Metric> metrics = List.of(new LossMetric(loss));

        new ModelTraining<NInputModel, NInputSample>()
                .setModel((NInputModel) model)
                .setDataset(trainStream)
                .setLoss(loss)
                .setOptimizer(opt)
                .setMaxEpoch(EPOCHS)
                .setBatchSize(BATCH)
                .setMetrics(metrics)
                .setClipNorm(CLIP_NORM)
                .setShuffleBuffer(SHUFFLE_BUFFER)
                .train();

        return model;
    }

    // ==================== стрим ====================

    /** Все {@code *.txt} рекурсивно, один раз (чтобы не ходить в FS каждую эпоху). */
    private static List<Path> listCorpusFiles(Path dir) throws IOException {
        try (Stream<Path> s = Files.walk(dir)) {
            return s.filter(Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".txt"))
                    .toList();
        }
    }

    /**
     * Ленивый {@link SampleStream} пар {@code (ctx, tgt, label)}.
     * {@link SampleStream#open()} вызывается {@link ModelTraining} на каждую
     * эпоху и должен возвращать новый независимый итератор.
     */
    private static SampleStream<NInputSample> streamPairs(Vocab vocab, Path corpusDir) throws IOException {
        List<Path> files = listCorpusFiles(corpusDir);
        return new SampleStream<>() {
            @Override public Iterator<NInputSample> open() { return new PairIterator(files, vocab); }
            @Override public int size() { return -1; }
            @Override public List<NInputSample> getTestSamples() { return List.of(); }
        };
    }

    /** Первые {@code max} пар из стрима — для теста. */
    private static List<NInputSample> takeTestPairs(Vocab vocab, Path corpusDir, int max)
            throws IOException {
        List<Path> files = listCorpusFiles(corpusDir);
        Iterator<NInputSample> it = new PairIterator(files, vocab);
        List<NInputSample> out = new ArrayList<>(max);
        while (out.size() < max && it.hasNext()) out.add(it.next());
        return out;
    }

    /**
     * Итератор пар по корпусу. Для каждого слова {@code t} и соседа
     * {@code t+d} в окне {@code |d| <= WINDOW} генерирует один positive
     * {@code (ctx=t, tgt=t+d, label=1)} и {@link #NEGATIVES} negative
     * {@code (ctx=t, tgt=случайное, label=0)}.
     *
     * <p>Subsampling: слово выбрасывается с вероятностью {@code 1 - keepProb}
     * ещё до генерации пар. Это режет объём и улучшает качество: очень
     * частые слова («и», «в») дают мало полезного сигнала.
     */
    private static final class PairIterator implements Iterator<NInputSample> {
        private final Iterator<Path> files;
        private final Vocab vocab;
        private final Random rnd = ThreadLocalRandom.current();
        private final ArrayList<NInputSample> buffer = new ArrayList<>();
        private int pos = 0;

        PairIterator(List<Path> files, Vocab vocab) {
            this.files = files.iterator();
            this.vocab = vocab;
        }

        @Override public boolean hasNext() { fill(); return pos < buffer.size(); }

        @Override public NInputSample next() {
            fill();
            if (pos >= buffer.size()) throw new NoSuchElementException();
            return buffer.get(pos++);
        }

        private void fill() {
            while (pos >= buffer.size() && files.hasNext()) {
                Path p = files.next();
                System.out.println("\t processing "+p.getFileName());
                try {
                    String text = Files.readString(p, StandardCharsets.UTF_8);
                    List<Integer> idx = new ArrayList<>();
                    for (String s : tokenize(text)) {
                        Integer v = vocab.wordToIdx.get(s);
                        if (v == null) continue;
                        // Subsampling: выбрасываем слово с вероятностью 1 - keepProb.
                        if (rnd.nextDouble() > vocab.keepProb.get(v)) continue;
                        idx.add(v);
                    }
                    int n = idx.size();
                    buffer.clear();
                    pos = 0;
                    for (int t = 0; t < n; t++) {
                        int ctx = idx.get(t);
                        int lo = Math.max(0, t - WINDOW);
                        int hi = Math.min(n - 1, t + WINDOW);
                        for (int j = lo; j <= hi; j++) {
                            if (j == t) continue;
                            int tgt = idx.get(j);
                            // positive
                            buffer.add(new NInputSample(
                                    new float[][]{ { ctx }, { tgt } },
                                    new float[]{ 1f }));
                            // negatives
                            for (int k = 0; k < NEGATIVES; k++) {
                                int neg = sampleNegative(vocab, rnd);
                                if (neg == tgt) continue;
                                buffer.add(new NInputSample(
                                        new float[][]{ { ctx }, { neg } },
                                        new float[]{ 0f }));
                            }
                        }
                    }
                } catch (IOException e) {
                    throw new RuntimeException("Не удалось прочитать " + p, e);
                }
            }
        }
    }

    /**
     * Negative sampling по {@code freq^NEG_SAMPLING_POWER} линейным поиском.
     * Просто и наглядно; для больших словарей в реальном коде — alias table
     * (O(1)) или префиксные суммы + binary search.
     */
    private static int sampleNegative(Vocab vocab, Random rnd) {
        int n = vocab.size();
        double r = rnd.nextDouble() * vocab.negSamplingTotal;
        double acc = 0.0;
        for (int i = 0; i < n; i++) {
            acc += Math.pow(vocab.freq.get(i), NEG_SAMPLING_POWER);
            if (acc >= r) return i;
        }
        return n - 1;
    }

    // ==================== эмбеддинги и аналогии ====================

    /**
     * Достаёт {@link EmbeddingLayer} из графа по порядку в
     * {@link Functional#debugTopo()}: {@code 0} — {@code E_in} (ctx),
     * {@code 1} — {@code E_out} (tgt).
     */
    private static EmbeddingLayer findEmbedding(Functional model, int occurrence) {
        int seen = 0;
        for (Functional.Node n : model.debugTopo()) {
            if (n.layer instanceof EmbeddingLayer e) {
                if (seen == occurrence) return e;
                seen++;
            }
        }
        return null;
    }

    /** Аналогия {@code a − b + c = ?} по {@code E_in}. */
    private static void analogies(EmbeddingLayer emb, Vocab vocab,
                                  String a, String b, String c, int topN) {
        Integer ia = vocab.wordToIdx.get(a.toLowerCase(Locale.ROOT));
        Integer ib = vocab.wordToIdx.get(b.toLowerCase(Locale.ROOT));
        Integer ic = vocab.wordToIdx.get(c.toLowerCase(Locale.ROOT));
        if (ia == null || ib == null || ic == null) {
            System.out.println("      слова нет в словаре: " + a + "/" + b + "/" + c);
            return;
        }
        float[] va = emb.embeddingOf(ia);
        float[] vb = emb.embeddingOf(ib);
        float[] vc = emb.embeddingOf(ic);
        float[] q = new float[D_MODEL];
        for (int i = 0; i < D_MODEL; i++) q[i] = va[i] - vb[i] + vc[i];
        printTopN(vocab, q, ia, ib, ic, topN, emb::embeddingOf);
    }

    /** Аналогия по усреднённым {@code (E_in + E_out)/2}. */
    private static void analogiesAveraged(EmbeddingLayer eIn, EmbeddingLayer eOut, Vocab vocab,
                                          String a, String b, String c, int topN) {
        Integer ia = vocab.wordToIdx.get(a.toLowerCase(Locale.ROOT));
        Integer ib = vocab.wordToIdx.get(b.toLowerCase(Locale.ROOT));
        Integer ic = vocab.wordToIdx.get(c.toLowerCase(Locale.ROOT));
        if (ia == null || ib == null || ic == null) {
            System.out.println("      слова нет в словаре: " + a + "/" + b + "/" + c);
            return;
        }
        float[] va = avg(eIn, eOut, ia);
        float[] vb = avg(eIn, eOut, ib);
        float[] vc = avg(eIn, eOut, ic);
        float[] q = new float[D_MODEL];
        for (int i = 0; i < D_MODEL; i++) q[i] = va[i] - vb[i] + vc[i];
        printTopN(vocab, q, ia, ib, ic, topN, i -> avg(eIn, eOut, i));
    }

    /** Top-N по косинусу к {@code query}, исключая три индекса. */
    private static void printTopN(Vocab vocab, float[] query, int ia, int ib, int ic,
                                  int topN, IntFunction<float[]> vecOf) {
        double qn = norm(query);
        int[] best = new int[topN];
        double[] bestSim = new double[topN];
        Arrays.fill(best, -1);
        Arrays.fill(bestSim, Double.NEGATIVE_INFINITY);

        for (int i = 0; i < vocab.size(); i++) {
            if (i == ia || i == ib || i == ic) continue;
            float[] v = vecOf.apply(i);
            double vn = norm(v);
            if (vn < 1e-12 || qn < 1e-12) continue;
            double dot = 0.0;
            for (int k = 0; k < D_MODEL; k++) dot += v[k] * query[k];
            double sim = dot / (vn * qn);
            for (int k = 0; k < topN; k++) {
                if (sim > bestSim[k]) {
                    System.arraycopy(bestSim, k, bestSim, k + 1, topN - k - 1);
                    System.arraycopy(best,   k, best,   k + 1, topN - k - 1);
                    bestSim[k] = sim;
                    best[k] = i;
                    break;
                }
            }
        }
        for (int k = 0; k < topN; k++) {
            if (best[k] < 0) break;
            System.out.printf(Locale.US, "      %2d. %-20s cos=%.4f%n",
                    k + 1, vocab.words.get(best[k]), bestSim[k]);
        }
    }

    /** Усреднённый вектор {@code (E_in[i] + E_out[i]) / 2}. */
    private static float[] avg(EmbeddingLayer eIn, EmbeddingLayer eOut, int i) {
        float[] a = eIn.embeddingOf(i);
        float[] b = eOut.embeddingOf(i);
        float[] r = new float[D_MODEL];
        for (int k = 0; k < D_MODEL; k++) r[k] = 0.5f * (a[k] + b[k]);
        return r;
    }

    private static double norm(float[] v) {
        double s = 0.0;
        for (float x : v) s += x * x;
        return Math.sqrt(s);
    }

    // ==================== демонстрация ====================

    /** Печатает первые {@code limit} слов словаря с частотами. */
    private static void printVocab(Vocab vocab, int limit) {
        System.out.printf(Locale.US, "Словарь (всего %d, показываю первые %d):%n",
                vocab.size(), Math.min(limit, vocab.size()));
        for (int i = 0; i < Math.min(limit, vocab.size()); i++) {
            System.out.printf(Locale.US, "  %4d. %-25s freq=%d%n",
                    i, vocab.words.get(i), vocab.freq.get(i));
        }
    }

    /** Проверяет, есть ли слова в словаре, и печатает idx/freq. */
    private static void checkWords(Vocab vocab, String... words) {
        for (String w : words) {
            Integer idx = vocab.wordToIdx.get(w.toLowerCase(Locale.ROOT));
            if (idx == null) System.out.printf(Locale.US, "  '%s' — НЕТ в словаре%n", w);
            else             System.out.printf(Locale.US, "  '%s' — idx=%d, freq=%d%n",
                    w, idx, vocab.freq.get(idx));
        }
    }

    /** Top-N ближайших к {@code word} по косинусу на {@code E_in}. */
    private static void nearest(EmbeddingLayer emb, Vocab vocab, String word, int topN) {
        Integer iw = vocab.wordToIdx.get(word.toLowerCase(Locale.ROOT));
        if (iw == null) {
            System.out.printf(Locale.US, "  '%s' — нет в словаре%n", word);
            return;
        }
        float[] q = emb.embeddingOf(iw);
        double qn = norm(q);
        int[] best = new int[topN];
        double[] sims = new double[topN];
        Arrays.fill(best, -1);
        Arrays.fill(sims, Double.NEGATIVE_INFINITY);

        for (int i = 0; i < vocab.size(); i++) {
            if (i == iw) continue;
            float[] v = emb.embeddingOf(i);
            double vn = norm(v);
            if (vn < 1e-12 || qn < 1e-12) continue;
            double dot = 0.0;
            for (int k = 0; k < D_MODEL; k++) dot += v[k] * q[k];
            double sim = dot / (vn * qn);
            for (int k = 0; k < topN; k++) {
                if (sim > sims[k]) {
                    System.arraycopy(sims, k, sims, k + 1, topN - k - 1);
                    System.arraycopy(best, k, best, k + 1, topN - k - 1);
                    sims[k] = sim;
                    best[k] = i;
                    break;
                }
            }
        }
        System.out.printf(Locale.US, "  ближайшие к '%s':%n", word);
        for (int k = 0; k < topN; k++) {
            if (best[k] < 0) break;
            System.out.printf(Locale.US, "    %2d. %-20s cos=%.4f%n",
                    k + 1, vocab.words.get(best[k]), sims[k]);
        }
    }
}