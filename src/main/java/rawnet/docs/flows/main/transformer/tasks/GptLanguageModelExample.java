package rawnet.docs.flows.main.transformer.tasks;

import rawnet.code.dataset.Sample;
import rawnet.code.dataset.SampleDataset;
import rawnet.code.net.TensorShape;
import rawnet.code.net.composition.TransformerEncoderBlock;
import rawnet.code.net.composition.TransformerEncoderStack;
import rawnet.code.net.core.loss.SequenceCrossEntropy;
import rawnet.code.net.core.metric.LossMetric;
import rawnet.code.net.core.metric.Metric;
import rawnet.code.net.core.metric.TokenAccuracy;
import rawnet.code.net.core.optimizator.AdamW;
import rawnet.code.net.layers.core.EmbeddingLayer;
import rawnet.code.net.layers.core.PositionalEncodingLayer;
import rawnet.code.net.layers.core.TimeDistributedDenseLayer;
import rawnet.code.net.layers.mask.MaskingLayer;
import rawnet.code.net.model.Model;
import rawnet.code.net.model.ModelPrintUtil;
import rawnet.code.net.model.ModelTraining;
import rawnet.code.net.model.functional.Functional;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.Charset;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.function.BiPredicate;
import java.util.stream.Stream;

/**
 * Учебная GPT-style языковая модель: предсказание следующего токена
 * на корпусе стихов / короткой прозы.
 *
 * <h2>Архитектура</h2>
 * <pre>
 * ВХОД:  flat(SEQ_LEN) индексов
 *   │
 *   ▼ MaskingLayer(PAD)
 *   ▼ EmbeddingLayer(bpe.size(), dModel)
 *   ▼ PositionalEncodingLayer
 *   ▼ TransformerEncoderStack(N, causal=true)   ← GPT-блок
 *   │
 *   ▼ seq(SEQ_LEN, dModel)
 *   ▼ TimeDistributedDense(bpe.size())
 *   ▼ logits: seq(SEQ_LEN, bpe.size())
 *   ▼ SequenceCrossEntropy (по позициям)
 * </pre>
 *
 * <h2>GPT-блок = энкодер с causal=true</h2>
 * GPT — это <b>только декодер</b> Transformer'а, без cross-attention.
 * Каждый блок — self-attention + FFN + residuals, как в энкодере,
 * но с {@code causal = true}: токен на позиции {@code t} видит только
 * {@code 0..t-1}, не подсматривает будущее. Это то же, что
 * {@link TransformerEncoderBlock} с {@code causal = true} —
 * {@link TransformerEncoderStack} собирает N таких блоков.
 *
 * <h2>Сдвиг target</h2>
 * Как и в seq2seq, target для loss сдвинут на 1: на позиции {@code t}
 * модель видит {@code tgtIn[0..t]} и предсказывает {@code tgtIn[t+1]}.
 * {@code causal=true} гарантирует, что позиция {@code t} не подсмотрит
 * {@code tgtIn[t+1]}.
 *
 * <h2>Файлы</h2>
 * <ul>
 *   <li>{@code bpe_<vocab>.bin} — BPE (загружается или обучается);</li>
 *   <li>{@code gpt_<vocab>.model} — модель (загружается
 *       {@link Model#loadOrCreate} или создаётся);</li>
 *   <li>{@code gpt_<vocab>.model.train} — параметры обучения
 *       (optimizer, loss, currentEpoch, maxEpoch, ...) — для resume
 *       при Ctrl+C. При успешном завершении удаляется.</li>
 * </ul>
 *
 * <h2>Прерывание и восстановление (Ctrl+C)</h2>
 * {@link ModelTraining#train()} в начале вызывает {@code load()}:
 * если {@code gpt_<vocab>.model.train} существует — читает параметры,
 * грузит модель, {@code currentEpoch = saved + 1}. Если нет — с нуля.
 * <ul>
 *   <li>Первый Ctrl+C — мягкая остановка после текущей эпохи,
 *       сохраняются модель и {@code .model.train}.</li>
 *   <li>Второй Ctrl+C — жёсткий {@code halt(130)}, без сохранения.</li>
 *   <li>Успешное завершение (исчерпание эпох / early-stop) —
 *       модель сохраняется, {@code .model.train} удаляется.</li>
 * </ul>
 */
public final class GptLanguageModelExample {

    // ==================== гиперпараметры ====================

    /** Размерность модели. */
    private static final int D_MODEL = 64;
    /** Число голов attention. {@code dModel % heads == 0}. */
    private static final int HEADS = 8;
    /** Ширина FFN. */
    private static final int D_FF = 4 * D_MODEL;
    /** Число блоков. */
    private static final int N_DEC = 1;
    /** Dropout в блоках. */
    private static final float DROPOUT = 0.1f;
    /** Длина контекста. */
    private static final int SEQ_LEN = 128;
    /** Размер батча. */
    private static final int BATCH = 16;
    /** Эпох обучения (всего; при resume — до этого числа). */
    private static final int EPOCHS = 50;
    /** Learning rate. */
    private static final float LR = 5e-4f;
    /** Предел L2-нормы градиента для клипа. */
    private static final float CLIP_NORM = 1f;
    /** Размер BPE-словаря. */
    private static final int BPE_VOCAB = 4096;

    /** Директория с корпусом. */
    private static final String CORPUS_DIR = "corpus";

    /** Специальные токены. */
    private static final int PAD = 0;
    private static final int BOS = 1;
    private static final int EOS = 2;
    private static final int UNK = 3;

// ==================== генерация ====================

    /** Seed для воспроизводимой генерации. */
    private static final long GENERATION_SEED = 42L;

    /** Temperature: < 1 — острее (увереннее), > 1 — площе (разнообразнее). */
    private static final float TEMPERATURE = 0.8f;

    /** Top-k: оставляем только k самых вероятных токенов. 0 — отключить. */
    private static final int TOP_K = 40;

    /** Top-p (nucleus): оставляем минимальный набор токенов с суммой вероятностей ≥ p. 1.0 — отключить. */
    private static final float TOP_P = 0.9f;

    /** Repetition penalty: > 1 понижает логиты уже сгенерированных токенов. 1.0 — отключить. */
    private static final float REPETITION_PENALTY = 1.2f;

    /** Окно repetition penalty: сколько последних токенов штрафовать. */
    private static final int REPETITION_WINDOW = 32;

    // ==================== точка входа ====================

    public static void main(String[] args) throws IOException {
        // 1) Корпус.
        List<String> texts = loadCorpus();
        System.out.printf(Locale.US, "Корпус: %d текстов, ~%d символов%n",
                texts.size(), texts.stream().mapToInt(String::length).sum());
        if (texts.isEmpty()) {
            System.err.println("Корпус пуст — проверьте CORPUS_DIR и кодировку файлов.");
            return;
        }

        // 2) BPE: загрузить или обучить+сохранить.
        Path bpeFile = Path.of("bpe_" + BPE_VOCAB + ".bin");
        Bpe bpe = Bpe.loadOrTrain(bpeFile, texts, BPE_VOCAB);
        System.out.println("BPE: " + bpe);

        // 3) Датасет и loss/optimizer.
        SampleDataset<Sample> ds = buildDataset(texts, bpe);
        SequenceCrossEntropy loss = new SequenceCrossEntropy();
        AdamW opt = new AdamW(LR);

        // 4) Модель: loadOrCreate — загрузить gpt_<vocab>.model, если есть;
        //    иначе — свежая.
        Path modelBase = Path.of("gpt_" + BPE_VOCAB);
        Path modelPath = modelBase.resolveSibling(modelBase.getFileName() + ".model");
        Functional model = Model.loadOrCreate(modelPath, buildModel(bpe.size()));

        // 5) Метрики и train-loss для лога.
        List<Metric> testMetrics = List.of(new TokenAccuracy());

        // Train-loss (диагностика переобучения): свой LossMetric, не пересекается
        // с test-метриками; печатает и всегда возвращает false (не стоп).
        List<Metric> trainMetrics = List.of(new LossMetric(loss));
        BiPredicate<Model, List<Metric>> printTrainLoss = ModelPrintUtil.createPrintLossOnSamples(ds.getTrainSamples(), trainMetrics, System.out::println, "train");

        // 6) Обучение: ModelTraining сам делает resume (через .model.train),
        //    сохраняет модель и .model.train при Ctrl+C, удаляет .model.train
        //    при успехе.
        /*model = new ModelTraining<Functional, Sample>()
                .setModel(model)
                .setDataset(ds)
                .setLoss(loss)
                .setOptimizer(opt)
                .setMaxEpoch(EPOCHS)
                .setBatchSize(BATCH)
                .setMetrics(testMetrics)
                .setEndPredicate(printTrainLoss)   // только печать, не стоп
                .setClipNorm(CLIP_NORM)
                .setPadIndex(PAD)
                .setPath(modelBase)                // gpt_1024.model / gpt_1024.model.train
                .train();
        */

        // Однократная проверка формы logits (убрать после подтверждения).
        {
            int[] probe = new int[SEQ_LEN];
            Arrays.fill(probe, PAD);
            probe[0] = BOS;
            float[] probeLogits = model.forward(toFloat(probe), false);
            int expected = SEQ_LEN * bpe.size();
            if (probeLogits.length != expected) {
                throw new IllegalStateException(
                        "logits.length=" + probeLogits.length
                                + " != SEQ_LEN*vocab=" + expected
                                + " — off в generate() будет неверным");
            }
            System.out.println("OK: logits.length=" + probeLogits.length);
        }

        // 7) Генерация.
        System.out.println();
        System.out.println();
        System.out.println("Генерация (top-k=" + TOP_K + ", top-p=" + TOP_P
                + ", T=" + TEMPERATURE + ", rep=" + REPETITION_PENALTY
                + ", seed=" + GENERATION_SEED + "):");
        String[] prompts = {"осень", "люблю тебя", "в лесу", "ночь была"};
        for (String prompt : prompts) {
            String generated = generate(model, bpe, prompt, 40);
            System.out.printf(Locale.US, "  prompt: \"%s\"%n", prompt);
            System.out.printf(Locale.US, "  → %s%n", generated);
            System.out.println();
        }
    }

    // ==================== корпус ====================

    /**
     * Загружает корпус: все {@code *.txt} из {@link #CORPUS_DIR}.
     *
     * <p><b>Кодировка.</b> Сначала пробуем UTF-8 (strict). Если файл
     * не UTF-8 (например, windows-1251 — типичная кодировка русских
     * текстов из интернета) — читаем как windows-1251. Если и это
     * не удалось (битый файл) — пропускаем с предупреждением.
     *
     * <p><b>Нормализация.</b> {@code \r\n} → {@code \n}, {@code \n\n\n+}
     * → {@code \n\n} (максимум две пустых строки — конец строфы),
     * обрезка пробелов.
     */
    private static List<String> loadCorpus() throws IOException {
        List<String> texts = new ArrayList<>();

        try (Stream<Path> files = Files.walk(Path.of(CORPUS_DIR))) {
            for (Path p : files.filter(Files::isRegularFile)
                    .filter(f -> f.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".txt"))
                    .toList()) {
                String text = readText(p);
                if (text != null && !text.isEmpty()) {
                    texts.add(text);
                }
            }
        }

        List<String> normalized = new ArrayList<>(texts.size());
        for (String t : texts) {
            String s = t.replace("\r\n", "\n").replace('\r', '\n');
            s = s.replaceAll("\n{3,}", "\n\n");
            s = s.trim();
            if (!s.isEmpty()) normalized.add(s);
        }
        return normalized;
    }

    /**
     * Читает файл: UTF-8 (strict), при ошибке — windows-1251,
     * при ошибке чтения — {@code null}.
     */
    private static String readText(Path p) {
        try {
            byte[] bytes = Files.readAllBytes(p);
            try {
                return StandardCharsets.UTF_8.newDecoder()
                        .onMalformedInput(CodingErrorAction.REPORT)
                        .onUnmappableCharacter(CodingErrorAction.REPORT)
                        .decode(ByteBuffer.wrap(bytes))
                        .toString();
            } catch (CharacterCodingException e) {
                System.err.println("WARNING: " + p.getFileName()
                        + " — не UTF-8, читаю как windows-1251");
                return new String(bytes, Charset.forName("windows-1251"));
            }
        } catch (IOException e) {
            System.err.println("WARNING: не удалось прочитать " + p + ": " + e.getMessage());
            return null;
        }
    }

    // ==================== примеры ====================

    /**
     * Строит датасет: нарезает тексты на окна длиной {@link #SEQ_LEN}
     * без перекрытия и делает {@code SampleDataset.split(..., 0.9f)}.
     *
     * <p>Вход {@code [BOS, w0, ..., w_{N-2}]}, target {@code [w0, ..., w_{N-1}, EOS]}.
     * Сдвиг на 1: на позиции {@code t} модель предсказывает {@code tgtIn[t+1]}.
     */
    private static SampleDataset<Sample> buildDataset(List<String> texts, Bpe bpe) {
        return SampleDataset.split(buildSamples(texts, bpe), 0.9f);
    }

    /**
     * Нарезает тексты на окна длиной {@link #SEQ_LEN} без перекрытия.
     */
    private static List<Sample> buildSamples(List<String> texts, Bpe bpe) {
        List<Sample> out = new ArrayList<>();
        for (String text : texts) {
            List<Integer> ids = bpe.encode(text);
            if (ids.size() < 2) continue;

            for (int start = 0; start < ids.size(); start += SEQ_LEN) {
                int end = Math.min(start + SEQ_LEN, ids.size());
                int len = end - start;
                if (len < 2) break;

                boolean isLastWindow = (end == ids.size());   // ← новое

                int[] tgtIn  = new int[SEQ_LEN];
                int[] tgtOut = new int[SEQ_LEN];
                Arrays.fill(tgtIn, PAD);
                Arrays.fill(tgtOut, PAD);
                tgtIn[0] = BOS;
                for (int i = 1; i < len; i++) tgtIn[i] = ids.get(start + i - 1);
                for (int i = 0; i < len - 1; i++) tgtOut[i] = ids.get(start + i);

                if (isLastWindow) {
                    tgtOut[len - 1] = EOS;                    // EOS только в последнем окне
                } else {
                    tgtOut[len - 1] = ids.get(end);           // продолжаем текст следующим токеном
                }

                out.add(new Sample(toFloat(tgtIn), toFloat(tgtOut)));
            }
        }
        return out;
    }

    private static float[] toFloat(int[] a) {
        float[] f = new float[a.length];
        for (int i = 0; i < a.length; i++) f[i] = a[i];
        return f;
    }

    // ==================== модель ====================

    /**
     * GPT-блок = {@link TransformerEncoderStack} с {@code causal = true}.
     * Тот же энкодер, только с causal-маской в self-attention: токен
     * {@code t} видит только {@code 0..t-1}. Это и есть GPT.
     */
    private static Functional buildModel(int vocabSize) {
        Functional.Builder b = Functional.builder(TensorShape.flat(SEQ_LEN));
        Functional.Node in = b.input();

        Functional.Node mask = b.layer(new MaskingLayer.Dto(PAD), in);
        Functional.Node emb  = b.layer(new EmbeddingLayer.Dto(vocabSize, D_MODEL), mask);
        Functional.Node pe   = b.layer(new PositionalEncodingLayer.Dto(), emb);

        Functional.Node dec = TransformerEncoderStack.build(
                b, pe,
                new TransformerEncoderBlock.Dto(HEADS, D_MODEL, D_FF, DROPOUT, /* causal */ true),
                N_DEC);

        Functional.Node logits = b.layer(
                new TimeDistributedDenseLayer.Dto(vocabSize), dec);

        return b.output(logits);
    }

    // ==================== генерация ====================

    /**
     * Авторегрессионная генерация с sampling: top-k + top-p + repetition penalty.
     *
     * <p><b>Пайплайн на каждом шаге:</b>
     * <ol>
     *   <li>Взять последние {@link #SEQ_LEN} токенов префикса (обрезаем слева, если длиннее).</li>
     *   <li>Дополнить PAD до {@code SEQ_LEN} справа — модель ждёт фиксированную длину.</li>
     *   <li>{@code forward(..., false)} — получить logits формы {@code [SEQ_LEN, vocab]}.</li>
     *   <li>Взять срез для <b>последней реальной</b> позиции: {@code off = (len-1) * vocab}.</li>
     *   <li>Применить <b>repetition penalty</b> к токенам из окна {@link #REPETITION_WINDOW}.</li>
     *   <li>Применить <b>top-k</b>: обнулить (в -inf) всё, кроме k максимумов.</li>
     *   <li>Softmax с {@link #TEMPERATURE} → вероятности.</li>
     *   <li>Применить <b>top-p</b>: отсортировать по убыванию, отрезать хвост, где
     *       кумулятивная сумма превышает {@link #TOP_P}, перенормировать.</li>
     *   <li>Сэмплировать один токен из распределения через {@link Random#nextFloat()}.</li>
     *   <li>Если {@code EOS} — остановиться. Иначе добавить токен в префикс.</li>
     * </ol>
     *
     * <p><b>Без KV-cache:</b> attention пересчитывается по всему префиксу на каждом шаге.
     * Для {@code SEQ_LEN=128} и {@code maxTokens=40} — миллисекунды.
     *
     * <p><b>Воспроизводимость:</b> {@link Random} инициализируется {@link #GENERATION_SEED}
     * <b>внутри</b> метода, поэтому <b>каждый вызов</b> {@code generate} даёт один и тот же
     * результат при одинаковых входах — независимо от предыдущих вызовов и от порядка
     * prompt'ов. Если нужна разная генерация для разных prompt'ов — используйте
     * {@code new Random(GENERATION_SEED + prompt.hashCode())}.
     *
     * @param model      обученная модель
     * @param bpe        BPE-кодек (для encode prompt и decode результата)
     * @param prompt     начальный текст
     * @param maxTokens  максимум новых токенов (не считая prompt)
     * @return сгенерированный текст (без prompt, без BOS)
     */
    private static String generate(Functional model, Bpe bpe, String prompt, int maxTokens) {
        // Фиксированный seed — воспроизводимость. Внутри метода, чтобы каждый вызов
        // generate() был независим от предыдущих.
        Random rnd = new Random(GENERATION_SEED);

        // Префикс: BOS + токены prompt.
        List<Integer> ids = new ArrayList<>(bpe.encode(prompt));
        ids.add(0, BOS);

        int vocab = bpe.size();

        for (int step = 0; step < maxTokens; step++) {
            // --- 1) Обрезаем префикс до SEQ_LEN (слева), дополняем PAD (справа). ---
            int len = Math.min(ids.size(), SEQ_LEN);
            int from = ids.size() - len;
            int[] input = new int[SEQ_LEN];
            Arrays.fill(input, PAD);
            for (int i = 0; i < len; i++) input[i] = ids.get(from + i);

            // --- 2) Forward. ---
            float[] logits = model.forward(toFloat(input), false);

            // --- 3) Срез для последней реальной позиции. ---
            int off = (len - 1) * vocab;

            // --- 4) Repetition penalty. ---
            // Понижаем логиты токенов, которые уже встречались в последних
            // REPETITION_WINDOW позициях. Это ломает циклы «не мог не мог не мог».
            //
            // Важно: работаем на копии среза, чтобы не портить logits модели
            // (в текущей реализации forward возвращает новый массив, но
            // защищаемся от будущих оптимизаций с переиспользованием буфера).
            float[] slice = new float[vocab];
            System.arraycopy(logits, off, slice, 0, vocab);

            if (REPETITION_PENALTY > 1.0f) {
                int windowStart = Math.max(0, ids.size() - REPETITION_WINDOW);
                // Используем Set, чтобы не штрафовать один и тот же токен дважды
                // (иначе редкий токен, встретившийся 3 раза, получит penalty^3
                // и потеряет шанс вообще). Стандартная практика — штрафовать
                // каждый уникальный токен один раз.
                java.util.Set<Integer> seen = new java.util.HashSet<>();
                for (int i = windowStart; i < ids.size(); i++) seen.add(ids.get(i));

                for (int id : seen) {
                    if (id == PAD || id == BOS) continue;  // спецтокены не штрафуем
                    if (slice[id] > 0) {
                        slice[id] /= REPETITION_PENALTY;   // положительный → уменьшаем
                    } else {
                        slice[id] *= REPETITION_PENALTY;   // отрицательный → делаем ещё меньше
                    }
                }
            }

            // --- 5) Top-k: оставить только k максимумов, остальное → -inf. ---
            // Сложность O(vocab * log vocab) из-за сортировки. Для vocab=4096 — ок.
            // Альтернатива — quickselect за O(vocab), но не нужно при таких размерах.
            if (TOP_K > 0 && TOP_K < vocab) {
                // Копируем и сортируем, чтобы найти k-й максимум.
                float[] sorted = slice.clone();
                Arrays.sort(sorted);                       // по возрастанию
                float kthMax = sorted[vocab - TOP_K];      // k-й с конца = k-й максимум

                for (int v = 0; v < vocab; v++) {
                    if (slice[v] < kthMax) slice[v] = Float.NEGATIVE_INFINITY;
                }
            }

            // --- 6) Softmax с temperature. ---
            // Вычитаем max для численной стабильности (иначе exp переполнится
            // при logits ~ 7 и temperature ~ 0.1).
            float max = Float.NEGATIVE_INFINITY;
            for (int v = 0; v < vocab; v++) {
                if (slice[v] > max) max = slice[v];
            }

            float[] probs = new float[vocab];
            float sum = 0f;
            for (int v = 0; v < vocab; v++) {
                if (slice[v] == Float.NEGATIVE_INFINITY) {
                    probs[v] = 0f;
                } else {
                    probs[v] = (float) Math.exp((slice[v] - max) / TEMPERATURE);
                    sum += probs[v];
                }
            }
            // sum > 0 гарантирован: top-k оставил ≥1 конечный logit
            // (TOP_K ≥ 1 проверяется в константах выше).
            for (int v = 0; v < vocab; v++) probs[v] /= sum;

            // --- 7) Top-p (nucleus): отсортировать по убыванию, отрезать хвост. ---
            int next;
            if (TOP_P < 1.0f) {
                next = sampleTopP(probs, TOP_P, rnd);
            } else {
                next = sampleFromProbs(probs, rnd);
            }

            // --- 8) EOS — стоп. ---
            if (next == EOS) break;
            ids.add(next);
        }

        // Убираем BOS из результата (ids[0] — всегда BOS после вставки).
        List<Integer> out = ids.subList(1, ids.size());
        return bpe.decode(out);
    }

    /**
     * Top-p (nucleus) sampling: оставляет минимальный набор токенов с
     * кумулятивной вероятностью ≥ {@code p}, перенормирует, сэмплирует.
     *
     * <p><b>Зачем:</b> top-k отрезает по <b>числу</b> токенов, top-p — по
     * <b>массе вероятности</b>. На «плоском» распределении (модель неуверена)
     * top-p оставит много токенов; на «остром» (модель уверена) — мало.
     * Это адаптивнее top-k.
     *
     * <p><b>Реализация:</b> сортируем индексы по убыванию вероятности, идём
     * по ним, накапливаем массу, пока не превысим {@code p}. Всё, что за
     * границей, отбрасываем, оставшиеся вероятности перенормируем.
     *
     * @param probs вероятности (сумма = 1, уже после top-k и softmax)
     * @param p     порог nucleus, {@code 0 < p ≤ 1}
     * @param rnd   источник случайности
     * @return индекс выбранного токена
     */
    private static int sampleTopP(float[] probs, float p, Random rnd) {
        int vocab = probs.length;

        // Индексы, отсортированные по убыванию вероятности.
        Integer[] order = new Integer[vocab];
        for (int i = 0; i < vocab; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Float.compare(probs[b], probs[a]));

        // Накапливаем массу, пока не достигнем p.
        float cum = 0f;
        int cutoff = vocab;
        for (int i = 0; i < vocab; i++) {
            cum += probs[order[i]];
            if (cum >= p) {
                cutoff = i + 1;   // включаем текущий токен
                break;
            }
        }

        // Перенормировка на отрезанном множестве.
        float sum = 0f;
        for (int i = 0; i < cutoff; i++) sum += probs[order[i]];

        // Сэмплирование.
        float r = rnd.nextFloat() * sum;
        float acc = 0f;
        for (int i = 0; i < cutoff; i++) {
            acc += probs[order[i]];
            if (r <= acc) return order[i];
        }
        return order[cutoff - 1];   // на случай погрешности float
    }

    /**
     * Обычное сэмплирование из распределения {@code probs} (сумма = 1).
     *
     * <p>Используется, когда {@link #TOP_P} отключён ({@code ≥ 1.0}).
     *
     * @param probs вероятности (сумма = 1)
     * @param rnd   источник случайности
     * @return индекс выбранного токена
     */
    private static int sampleFromProbs(float[] probs, Random rnd) {
        float r = rnd.nextFloat();
        float acc = 0f;
        for (int v = 0; v < probs.length; v++) {
            acc += probs[v];
            if (r <= acc) return v;
        }
        return probs.length - 1;   // на случай погрешности float
    }
}