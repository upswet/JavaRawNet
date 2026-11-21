package rawnet.docs.ext.word2vec;

import rawnet.code.net.TensorShape;
import rawnet.code.net.core.loss.BinaryCrossEntropy;
import rawnet.code.net.layers.attention.NegativeSamplingLayer;
import rawnet.code.net.layers.core.EmbeddingLayer;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
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
import java.util.Random;
import java.util.function.IntFunction;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Word2vec-NS с ручным batched-циклом: один forward ctx на K+1 tgt,
 * sparse AdamW в {@link EmbeddingLayer} (обновляются только задействованные
 * строки), alias table для negative sampling.
 *
 * <h2>Почему не {@code Functional}</h2>
 * Он делает много накладных на каждый пример (обход топологии, клоны
 * дельт, аллокации {@code deltaAccum}). Для word2vec, где пример — это
 * пара индексов, накладные доминируют над полезной работой. Ручной цикл
 * убирает их и позволяет считать {@code ctx} один раз на K+1 negative.
 *
 * <h2>Ключевые оптимизации</h2>
 * <ol>
 *   <li><b>1 forward ctx на группу</b> вместо K+1 — и 1 backward ctx
 *       (с суммой {@code dL/dctx} по всем парам группы).</li>
 *   <li><b>Sparse AdamW</b> ({@link EmbeddingLayer#updateSparseAdamW}) —
 *       обновляются только задействованные строки {@code E}. Без этого
 *       dense {@code update} шёл бы по всему словарю на каждый батч:
 *       для {@code vocab=30k}, {@code dModel=64} это ~2M элементов ×
 *       ~5000 батчей/эпоху. Sparse — ×10–15.</li>
 *   <li><b>Alias table</b> для negative sampling: O(1) на сэмпл вместо
 *       O(V) линейного поиска по {@code freq^0.75}.</li>
 *   <li><b>Никаких {@code NInputSample}</b> в горячем пути: int-массивы
 *       на батч, переиспользуемые {@code float[][]}.</li>
 *   <li><b>Нет {@code zeroGrad} в горячем цикле</b> — {@code updateSparseAdamW}
 *       сам обнуляет {@code EG} обновлённых строк. Это убирает O(vocab·dModel)
 *       на батч.</li>
 * </ol>
 *
 * <h2>Совместимость файлов модели</h2>
 * Формат {@code embeddings_*.bin} изменился (добавлены поля sparse).
 * <b>Старые файлы несовместимы</b> — удалите {@code embeddings_in.bin} и
 * {@code embeddings_out.bin} при первом запуске.
 */
public final class Word2VecNegativeSamplingDemoManual {

    // ==================== гиперпараметры ====================

    /** Размерность эмбеддинга. */
    private static final int D_MODEL = 64;
    /** Половина окна контекста: пары {@code (t, t+d)} для {@code |d| <= WINDOW}. */
    private static final int WINDOW = 4;
    /** Сколько negative-примеров на один positive. */
    private static final int NEGATIVES = 3;
    /** Минимальная частота слова для словаря. */
    private static final int MIN_COUNT = 3;
    /**
     * Минимальная длина слова (в символах). При {@code 4} слова
     * «муж»/«жен» (3 символа) выпадают. Если для демонстрации аналогии
     * нужны именно они — поставьте {@code 3}.
     */
    private static final int MIN_WORD_LENGTH = 4;
    /** Батч в «группах»: 1 группа = 1 ctx + (1 + NEGATIVES) tgt. */
    private static final int BATCH_GROUPS = 512;
    /** Эпох обучения. */
    private static final int EPOCHS = 5;
    /** Learning rate. */
    private static final float LR = 0.01f;
    /** Максимум слов в словаре. */
    private static final int MAX_VOCAB = 30_000;
    /** Степень для negative sampling: {@code freq^0.75} — стандарт word2vec. */
    private static final double NEG_SAMPLING_POWER = 0.75;
    /** Subsampling threshold: чем больше, тем сильнее режем частые слова. */
    private static final double SUBSAMPLE_T = 1e-2;

    /** Файлы сохранения. */
    private static final String IN_FILE  = "embeddings_in.bin";
    private static final String OUT_FILE = "embeddings_out.bin";

    /** Токен = последовательность букв/цифр (Unicode). */
    private static final Pattern TOKEN = Pattern.compile(
            "[\\p{L}\\p{N}]+", Pattern.UNICODE_CHARACTER_CLASS);

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
                "      слов: %d (minCount=%d, minLen=%d, max=%d)%n",
                vocab.size(), MIN_COUNT, MIN_WORD_LENGTH, MAX_VOCAB);
        if (vocab.size() < 10) { System.err.println("маленький словарь"); return; }

        // ---------- 2. Слои ----------
        // Sparse AdamW: обновляются только задействованные строки.
        // Гиперпараметры задаются здесь (явно), их же берёт
        // updateSparseAdamW(touched, batch) в processBatch.
        EmbeddingLayer embIn  = createEmbedding(vocab);
        EmbeddingLayer embOut = createEmbedding(vocab);

        // ---------- 3. Загрузка или обучение ----------
        boolean loaded = tryLoad(embIn, IN_FILE) & tryLoad(embOut, OUT_FILE);
        if (loaded) {
            System.out.println("[2/4] Загружены " + IN_FILE + ", " + OUT_FILE);
        } else {
            System.out.println("[2/4] Обучаю.");
            long t0 = System.nanoTime();
            train(vocab, embIn, embOut, corpusDir);
            System.out.printf(Locale.US, "      обучение: %.1f c%n",
                    (System.nanoTime() - t0) / 1e9);
            System.out.println("[3/4] Сохраняю " + IN_FILE + ", " + OUT_FILE);
            save(embIn,  IN_FILE);
            save(embOut, OUT_FILE);
        }

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
            nearest(embIn, vocab, vocab.words.get(i), 5);
        }

        // 4.5 Соседи для слов запроса.
        System.out.println();
        nearest(embIn, vocab, "гарри", 5);
        nearest(embIn, vocab, "гермиона", 5);
    }

    // ==================== словарь ====================

    /**
     * Словарь: слово → индекс, частоты, вероятности сохранения для
     * subsampling, alias table для negative sampling.
     *
     * <p>Порядок слов определяется сортировкой по частоте в
     * {@link #buildVocab}. При смене {@link #MIN_COUNT}/{@link #MAX_VOCAB}/
     * {@link #MIN_WORD_LENGTH} удаляйте {@code embeddings_*.bin} —
     * иначе индексы разъедутся.
     */
    private static final class Vocab {
        final Map<String, Integer> wordToIdx = new HashMap<>();
        final List<String> words = new ArrayList<>();
        final List<Integer> freq  = new ArrayList<>();
        /** Вероятность оставить слово при subsampling. */
        final List<Double> keepProb = new ArrayList<>();
        /** Alias table для negative sampling. */
        AliasTable negTable;

        int size() { return words.size(); }
    }

    /**
     * Строит словарь: рекурсивно читаем {@code *.txt}, токенизируем,
     * фильтруем по длине и частоте, сортируем по убыванию частоты (при
     * равенстве — по слову), обрезаем до {@link #MAX_VOCAB}. Считаем
     * {@code keepProb} (subsampling) и строим alias table.
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
                    // Фильтр по длине: короткие слова («и», «в», «на») не нужны.
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

        // Subsampling: keepProb = (sqrt(f/t) + 1) * t / f, ограничено [0,1].
        for (int f : v.freq) {
            double rel = (double) f / totalCount;
            double keep = (Math.sqrt(rel / SUBSAMPLE_T) + 1.0) * SUBSAMPLE_T / rel;
            v.keepProb.add(Math.min(1.0, keep));
        }

        // Alias table по freq^power: O(V) на построение, O(1) на сэмпл.
        double[] probs = new double[v.size()];
        double total = 0.0;
        for (int i = 0; i < v.size(); i++) {
            probs[i] = Math.pow(v.freq.get(i), NEG_SAMPLING_POWER);
            total += probs[i];
        }
        for (int i = 0; i < v.size(); i++) probs[i] /= total;
        v.negTable = new AliasTable(probs);
        return v;
    }

    // ==================== alias table ====================

    /**
     * Alias table (Walker–Vose): O(1) на сэмпл, O(n) памяти.
     *
     * <p>Единичный отрезок разбит на {@code n} столбцов ширины {@code 1/n}.
     * Для каждого {@code i}: {@code prob[i]} — вероятность «взять i» внутри
     * столбца, {@code alias[i]} — альтернатива с вероятностью {@code 1-prob[i]}.
     * Сэмпл: {@code c = random(n)}; если {@code random() < prob[c]} → {@code c},
     * иначе → {@code alias[c]}.
     */
    private static final class AliasTable {
        private final double[] prob;
        private final int[] alias;

        AliasTable(double[] p) {
            int n = p.length;
            this.prob  = new double[n];
            this.alias = new int[n];

            double[] scaled = new double[n];
            for (int i = 0; i < n; i++) scaled[i] = p[i] * n;

            int[] small = new int[n];
            int[] large = new int[n];
            int nSmall = 0, nLarge = 0;
            for (int i = 0; i < n; i++) {
                if (scaled[i] < 1.0) small[nSmall++] = i;
                else                 large[nLarge++] = i;
            }

            while (nSmall > 0 && nLarge > 0) {
                int s = small[--nSmall];
                int l = large[--nLarge];
                prob[s]  = scaled[s];
                alias[s] = l;
                scaled[l] -= (1.0 - scaled[s]);
                if (scaled[l] < 1.0) small[nSmall++] = l;
                else                 large[nLarge++] = l;
            }
            while (nLarge > 0) { int l = large[--nLarge]; prob[l] = 1.0; alias[l] = l; }
            while (nSmall > 0) { int s = small[--nSmall]; prob[s] = 1.0; alias[s] = s; }
        }

        /** O(1) сэмпл. */
        int sample(Random rnd) {
            int c = rnd.nextInt(prob.length);
            return rnd.nextDouble() < prob[c] ? c : alias[c];
        }
    }

    // ==================== слои ====================

    /**
     * Создаёт {@link EmbeddingLayer} c sparse AdamW.
     *
     * <p>Гиперпараметры заданы явно — их берёт
     * {@link EmbeddingLayer#updateSparseAdamW(int[], int)} при вызове из
     * {@code processBatch}. {@code Dto.withSparseAdamW(...)} здесь не
     * используется, потому что слой создаётся вручную (без графа
     * {@code Functional}).
     */
    private static EmbeddingLayer createEmbedding(Vocab vocab) {
        EmbeddingLayer e = (EmbeddingLayer) ((EmbeddingLayer) new EmbeddingLayer()
                .setInputShape(TensorShape.flat(1))
                .setOutputShape(TensorShape.seq(1, D_MODEL)))
                .setVocab(vocab.size())
                .setDModel(D_MODEL)
                .setSparseAdamW(true)
                .setSparseHyperparams(LR, 0.9f, 0.999f, 1e-8f, 0.01f)
                .allocateBuffers();
        e.initWeights(null);
        return e;
    }

    /** Слой {@link NegativeSamplingLayer} — общий для всех пар. */
    private static NegativeSamplingLayer createNegSampling() {
        return (NegativeSamplingLayer) new NegativeSamplingLayer()
                .setInputShape(TensorShape.seq(1, D_MODEL))
                .setOutputShape(TensorShape.flat(1))
                .allocateBuffers();
    }

    // ==================== обучение ====================

    /**
     * Ручной цикл: 1 forward ctx на группу ({@code ctx} + {@code tgt} +
     * {@link #NEGATIVES} negatives). Sparse AdamW обновляет только
     * задействованные строки.
     *
     * <p><b>Без {@code zeroGrad}.</b> {@code updateSparseAdamW} сам обнуляет
     * {@code EG} обновлённых строк, поэтому вызывать {@code zeroGrad}
     * каждый батч (O(vocab·dModel)) не нужно.
     */
    private static void train(Vocab vocab, EmbeddingLayer embIn, EmbeddingLayer embOut,
                              Path corpusDir) throws IOException {
        NegativeSamplingLayer neg = createNegSampling();
        BinaryCrossEntropy bce = new BinaryCrossEntropy();

        List<Path> files = listCorpusFiles(corpusDir);
        Random rnd = new Random(42L);

        // ===== переиспользуемые буферы =====
        float[] dOut      = new float[1];
        float[] oneLabel  = new float[1];
        float[] sumDCtx   = new float[D_MODEL];
        float[] ctxInput  = new float[1];
        float[] tgtInput  = new float[1];
        float[][] negInputs = new float[2][];

        // Флаги для touched-индексов батча (переиспользуем).
        boolean[] touchedInFlag  = new boolean[vocab.size()];
        boolean[] touchedOutFlag = new boolean[vocab.size()];

        long totalGroups = 0;
        long totalPairs  = 0;

        // Буферы батча аллоцируются один раз.
        int[]   gCtx    = new int[BATCH_GROUPS];
        int[][] gTgts   = new int[BATCH_GROUPS][NEGATIVES + 1];
        float[][] gLabs = new float[BATCH_GROUPS][NEGATIVES + 1];

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            long t0 = System.nanoTime();
            int groupsInBatch = 0;
            double epochLoss = 0.0;
            long epGroups = 0;
            long epPairs  = 0;

            for (Path p : files) {
                System.out.println("\t processing "+p.getFileName());
                String text = Files.readString(p, StandardCharsets.UTF_8);

                // Индексы после subsampling.
                int[] idx = new int[1024];
                int n = 0;
                Matcher m = TOKEN.matcher(text);
                while (m.find()) {
                    Integer v = vocab.wordToIdx.get(m.group().toLowerCase(Locale.ROOT));
                    if (v == null) continue;
                    if (rnd.nextDouble() > vocab.keepProb.get(v)) continue;
                    if (n == idx.length) idx = Arrays.copyOf(idx, n * 2);
                    idx[n++] = v;
                }

                for (int t = 0; t < n; t++) {
                    int ctx = idx[t];
                    int lo = Math.max(0, t - WINDOW);
                    int hi = Math.min(n - 1, t + WINDOW);
                    for (int j = lo; j <= hi; j++) {
                        if (j == t) continue;

                        int tgt = idx[j];
                        int gi = groupsInBatch;
                        gCtx[gi] = ctx;
                        int[] tgts = gTgts[gi];
                        float[] labs = gLabs[gi];

                        tgts[0] = tgt; labs[0] = 1f;
                        for (int k = 1; k <= NEGATIVES; k++) {
                            int negIdx;
                            do { negIdx = vocab.negTable.sample(rnd); }
                            while (negIdx == tgt);
                            tgts[k] = negIdx; labs[k] = 0f;
                        }
                        groupsInBatch++;
                        epGroups++;
                        epPairs += NEGATIVES + 1;

                        if (groupsInBatch == BATCH_GROUPS) {
                            double loss = processBatch(embIn, embOut, neg, bce,
                                    gCtx, gTgts, gLabs, groupsInBatch,
                                    dOut, oneLabel, sumDCtx,
                                    ctxInput, tgtInput, negInputs,
                                    touchedInFlag, touchedOutFlag);
                            epochLoss += loss;
                            totalGroups += groupsInBatch;
                            totalPairs  += (long) groupsInBatch * (NEGATIVES + 1);
                            groupsInBatch = 0;
                        }
                    }
                }
            }

            if (groupsInBatch > 0) {
                double loss = processBatch(embIn, embOut, neg, bce,
                        gCtx, gTgts, gLabs, groupsInBatch,
                        dOut, oneLabel, sumDCtx,
                        ctxInput, tgtInput, negInputs,
                        touchedInFlag, touchedOutFlag);
                epochLoss += loss;
                totalGroups += groupsInBatch;
                totalPairs  += (long) groupsInBatch * (NEGATIVES + 1);
            }

            long dtMs = (System.nanoTime() - t0) / 1_000_000;
            double avgLoss = epochLoss / Math.max(1, epPairs);
            System.out.printf(Locale.US,
                    "  эпоха %d/%d: групп=%d, пар=%d, loss≈%.4f, время=%d мс%n",
                    epoch + 1, EPOCHS, epGroups, epPairs, avgLoss, dtMs);
        }

        System.out.printf(Locale.US, "  всего: групп=%d, пар=%d%n", totalGroups, totalPairs);
    }

    /**
     * Один батч «групп». Для каждой группы:
     * <ol>
     *   <li>1 {@code embIn.forward(ctx)};</li>
     *   <li>K+1 раз {@code embOut.forward(tgt_k)} +
     *       {@code neg.forwardN} + {@code bce.derivative} +
     *       {@code neg.backwardN} + {@code embOut.backward};</li>
     *   <li>суммирование {@code dL/dctx} по всем K+1 парам;</li>
     *   <li>1 {@code embIn.backward(sumDCtx)}.</li>
     * </ol>
     *
     * <p>После цикла собираются {@code touchedIn}/{@code touchedOut} и
     * вызывается {@code updateSparseAdamW(touched, pairs)}. Метод сам
     * обнуляет {@code EG} обновлённых строк, поэтому {@code zeroGrad}
     * здесь не нужен.
     */
    private static double processBatch(EmbeddingLayer embIn, EmbeddingLayer embOut,
                                       NegativeSamplingLayer neg, BinaryCrossEntropy bce,
                                       int[] gCtx, int[][] gTgts, float[][] gLabs,
                                       int groups,
                                       float[] dOut, float[] oneLabel, float[] sumDCtx,
                                       float[] ctxInput, float[] tgtInput,
                                       float[][] negInputs,
                                       boolean[] touchedInFlag, boolean[] touchedOutFlag) {
        double lossSum = 0.0;
        int pairs = 0;

        // Сброс флагов touched. Альтернатива — «поколения» (int[] lastUsedGen):
        // без O(vocab) на батч. Здесь O(vocab) на батч — дёшево по сравнению
        // с тем, что мы сэкономили, убрав zeroGrad/update.
        Arrays.fill(touchedInFlag, false);
        Arrays.fill(touchedOutFlag, false);

        for (int gi = 0; gi < groups; gi++) {
            int ctx = gCtx[gi];
            int[] tgts = gTgts[gi];
            float[] labs = gLabs[gi];

            ctxInput[0] = ctx;
            float[] eCtx = embIn.forward(ctxInput, true);
            touchedInFlag[ctx] = true;
            Arrays.fill(sumDCtx, 0f);

            for (int k = 0; k < tgts.length; k++) {
                oneLabel[0] = labs[k];

                tgtInput[0] = tgts[k];
                float[] eTgt = embOut.forward(tgtInput, true);
                touchedOutFlag[tgts[k]] = true;

                negInputs[0] = eCtx;
                negInputs[1] = eTgt;
                float[] p = neg.forwardN(negInputs, true);

                bce.derivative(p, oneLabel, dOut);

                float[][] dEGrad = neg.backwardN(dOut);
                embOut.backward(dEGrad[1]);
                for (int d = 0; d < D_MODEL; d++) sumDCtx[d] += dEGrad[0][d];

                lossSum += bce.loss(p, oneLabel);
                pairs++;
            }
            embIn.backward(sumDCtx);
        }

        // Собираем touched-индексы из флагов.
        int[] touchedIn  = collectTouched(touchedInFlag, pairs);
        int[] touchedOut = collectTouched(touchedOutFlag, pairs);

        // Sparse AdamW: обновляет только touched-строки и обнуляет их EG.
        // Гиперпараметры берутся из полей слоя (заданы в createEmbedding).
        embIn.updateSparseAdamW(touchedIn,  pairs);
        embOut.updateSparseAdamW(touchedOut, pairs);

        return lossSum;
    }

    /** Собирает индексы, у которых флаг {@code true}, в новый {@code int[]}. */
    private static int[] collectTouched(boolean[] flags, int hint) {
        int[] buf = new int[Math.max(16, hint)];
        int n = 0;
        for (int i = 0; i < flags.length; i++) {
            if (!flags[i]) continue;
            if (n == buf.length) buf = Arrays.copyOf(buf, n * 2);
            buf[n++] = i;
        }
        return Arrays.copyOf(buf, n);
    }

    // ==================== файлы ====================

    private static List<Path> listCorpusFiles(Path dir) throws IOException {
        try (Stream<Path> s = Files.walk(dir)) {
            return s.filter(Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".txt"))
                    .toList();
        }
    }

    // ==================== сохранение/загрузка ====================

    private static void save(EmbeddingLayer e, String file) throws IOException {
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(Files.newOutputStream(Path.of(file))))) {
            e.save(out);
        }
        System.out.println("      сохранено: " + file);
    }

    private static boolean tryLoad(EmbeddingLayer e, String file) {
        Path p = Path.of(file);
        if (!Files.isRegularFile(p)) return false;
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(Files.newInputStream(p)))) {
            String className = in.readUTF();           // <-- пропустить className
            if (!className.equals(EmbeddingLayer.class.getName())) {
                System.err.println("      " + file + ": ожидался EmbeddingLayer, получено " + className);
                return false;
            }
            e.load(in);                                // <-- читает формы + поля + E
            return true;
        } catch (IOException | RuntimeException ex) {
            System.err.println("      не загрузилось " + file + ": " + ex.getMessage());
            return false;
        }
    }

    // ==================== аналогии ====================

    /** Аналогия {@code a − b + c = ?} по {@code E_in}. */
    private static void analogies(EmbeddingLayer emb, Vocab vocab,
                                  String a, String b, String c, int topN) {
        Integer ia = vocab.wordToIdx.get(a.toLowerCase(Locale.ROOT));
        Integer ib = vocab.wordToIdx.get(b.toLowerCase(Locale.ROOT));
        Integer ic = vocab.wordToIdx.get(c.toLowerCase(Locale.ROOT));
        if (ia == null || ib == null || ic == null) {
            System.out.println("      нет в словаре: " + a + "/" + b + "/" + c);
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
            System.out.println("      нет в словаре: " + a + "/" + b + "/" + c);
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
            if (idx == null) System.out.printf(Locale.US, "  '%s' — нет в словаре%n", w);
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