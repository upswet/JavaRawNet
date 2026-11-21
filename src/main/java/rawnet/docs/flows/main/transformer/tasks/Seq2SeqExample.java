package rawnet.docs.flows.main.transformer.tasks;

import rawnet.code.dataset.NInputSample;
import rawnet.code.dataset.SampleDataset;
import rawnet.code.net.TensorShape;
import rawnet.code.net.composition.TransformerDecoderBlock;
import rawnet.code.net.composition.TransformerEncoderBlock;
import rawnet.code.net.composition.TransformerEncoderStack;
import rawnet.code.net.core.loss.SequenceCrossEntropy;
import rawnet.code.net.core.metric.Metric;
import rawnet.code.net.core.metric.TokenAccuracy;
import rawnet.code.net.core.optimizator.AdamW;
import rawnet.code.net.layers.core.EmbeddingLayer;
import rawnet.code.net.layers.core.PositionalEncodingLayer;
import rawnet.code.net.layers.core.TimeDistributedDenseLayer;
import rawnet.code.net.layers.mask.MaskingLayer;
import rawnet.code.net.model.ModelTraining;
import rawnet.code.net.model.NInputModel;
import rawnet.code.net.model.functional.Functional;

import java.util.*;
import java.util.function.Predicate;

/**
 * Учебный seq2seq: «перевод» английских фраз на русский на
 * игрушечном параллельном корпусе.
 *
 * <h2>Архитектура (как в главе 11)</h2>
 * <pre>
 * source: flat(SRC_LEN) → Masking → Embedding(vocab_src) → PE → EncoderStack(N)
 *         → encOut: seq(SRC_LEN, dModel)
 *
 * target: flat(TGT_LEN) → Masking → Embedding(vocab_tgt) → PE
 *         → DecoderBlock(M, causal self + cross на encOut)
 *         → TimeDistributedDense(vocab_tgt) → seq(TGT_LEN, vocab_tgt) → CE
 * </pre>
 *
 * <h2>Что важно понять</h2>
 * <ul>
 *   <li><b>Teacher forcing.</b> На позиции {@code t} декодер получает
 *       токены {@code 0..t-1} (сдвинутый target), а предсказывает токен
 *       {@code t}. {@code causal=true} в self-attention гарантирует,
 *       что позиция {@code t} не подсмотрит токен {@code t+1}.</li>
 *   <li><b>Два входа {@link Functional}.</b> {@code src} — вход энкодера,
 *       {@code tgt} — вход декодера. {@code encOut} идёт как KV в
 *       cross-attention; это единственная связь между ветвями.</li>
 *   <li><b>SequenceCrossEntropy</b> принимает <b>логиты</b> формы
 *       {@code seq(TGT_LEN, vocab_tgt)}, а не вероятности. Softmax
 *       считается внутри, по каждой позиции отдельно.</li>
 *   <li><b>TokenAccuracy</b> — метрика: argmax по каждой позиции,
 *       сравнение с target. {@code Accuracy} здесь не подходит — она
 *       делает один argmax по всему вектору.</li>
 * </ul>
 *
 * <h2>Генерация</h2>
 * Обученный декодер умеет предсказывать следующий токен по префиксу.
 * Генерация — цикл: подать префикс, взять argmax последней позиции,
 * добавить токен, повторить. Без KV-cache: {@code O(Tq³)} на всю
 * последовательность, но для {@code Tq=10} — миллисекунды.
 */
public final class Seq2SeqExample {

    // ==================== гиперпараметры ====================

    /** Размерность модели. Должна делиться на {@link #HEADS}. */
    private static final int D_MODEL = 64;
    /** Число голов attention. {@code dModel % heads == 0}. */
    private static final int HEADS   = 8;
    /** Ширина FFN внутри блоков Transformer; стандарт — {@code 4·dModel}. */
    private static final int D_FF    = 4 * D_MODEL;
    /** Число блоков в энкодере. */
    private static final int N_ENC   = 3;
    /** Число блоков в декодере. */
    private static final int N_DEC   = 3;
    /** Dropout в блоках. Для игрушечного примера — 0 (не замедлять сходимость). */
    private static final float DROPOUT = 0.2f;
    /** Длина source-последовательности (padding-обрезка). */
    private static final int SRC_LEN = 8;
    /** Длина target-последовательности (padding-обрезка). */
    private static final int TGT_LEN = 10;
    /** Размер батча. */
    private static final int BATCH = 8;
    /** Эпох обучения. */
    private static final int EPOCHS = 25;
    /** Learning rate. */
    private static final float LR = 3e-3f;
    /** Предел L2-нормы градиента для клипа. */
    private static final float CLIP_NORM = 1f;

    /** Специальные токены. Индексы 0..3 — зарезервированы. */
    private static final int PAD = 0;
    private static final int BOS = 1;   // beginning of sequence
    private static final int EOS = 2;   // end of sequence
    private static final int UNK = 3;   // unknown

    // ==================== точка входа ====================

    // ==================== точка входа ====================

    public static void main(String[] args) {
        // 1) Игрушечный параллельный корпус: английский → русский.
        //List<String[]> pairs = toyCorpus();

        List<String[]> pairs = generateCorpus(200, 42L, /* includeHeldOut */ false);

        // 1a) Held-out: фразы, которых НЕТ в обучающем корпусе, но слова
        //     в них те же. Модель должна перевести их, если выучила правила
        //     (кот→кот, видит→видит, ...), а не запомнила 20 пар.
        //
        //     Важно: held-out НЕ участвует в buildVocab — иначе вокабуляр
        //     «подсмотрит» новые слова. Но сами слова должны быть в корпусе.
        //     Если добавить слово вне корпуса — оно уйдёт в <unk>, и это
        //     тоже показательно (модель «не знает» его).
        List<String[]> heldOut = heldOutCorpus();

        // 1b) Проверка: held-out фразы не должны пересекаться с train.
        //     (На случай, если кто-то поменяет корпус и забудет про это.)
        java.util.Set<String> trainSet = new java.util.HashSet<>();
        for (String[] p : pairs) trainSet.add(p[0].toLowerCase(Locale.ROOT));
        for (String[] p : heldOut) {
            if (trainSet.contains(p[0].toLowerCase(Locale.ROOT))) {
                System.err.println("WARNING: held-out фраза '" + p[0]
                        + "' присутствует в train — проверка нечестная.");
            }
        }

        // 2) Словари для source и target. Held-out НЕ участвует.
        Vocab srcVocab = buildVocab(pairs, 0);
        Vocab tgtVocab = buildVocab(pairs, 1);
        System.out.printf(Locale.US, "source vocab: %d, target vocab: %d, пар: %d%n",
                srcVocab.size(), tgtVocab.size(), pairs.size());

        // 3) Сеть.
        Functional model = buildModel(srcVocab, tgtVocab);

        // 4) Примеры (только train-пары).
        List<NInputSample> samples = buildSamples(pairs, srcVocab, tgtVocab);

        // 5) Train/test split.
        SampleDataset<NInputSample> ds = SampleDataset.split(samples, 0.9f);

        // 6) Обучение.
        SequenceCrossEntropy loss = new SequenceCrossEntropy();
        AdamW opt = new AdamW(LR);
        List<Metric> metrics = List.of(new TokenAccuracy());

        Predicate<List<Metric>> printEachEpoch = ms -> {
            System.out.printf(Locale.US, "  %s%n", Metric.print(ms));
            return false;
        };

        new ModelTraining<NInputModel, NInputSample>()
                .setModel((NInputModel) model)
                .setDataset(ds)
                .setLoss(loss)
                .setOptimizer(opt)
                .setMaxEpoch(EPOCHS)
                .setBatchSize(BATCH)
                .setMetrics(metrics)
                .setClipNorm(CLIP_NORM)
                .setPadIndex(PAD)
                .train();

        // 7) Демонстрация: переводы из test.
        System.out.println();
        System.out.println("Перевод (test из train-корпуса):");
        int shown = 0;
        for (NInputSample s : ds.getTestSamples()) {
            if (shown++ >= 5) break;
            int[] src = ints(s.getInputs()[0]);
            System.out.printf(Locale.US, "  EN: %-30s  RU: %s%n",
                    srcVocab.decode(src),
                    tgtVocab.decode(greedyTranslate(model, src, tgtVocab)));
        }

        // 8) Held-out: фразы, которых НЕТ в обучении.
        System.out.println("\nHeld-out (жадная vs beam):");
        for (String[] p : heldOut) {
            int[] src = encode(p[0], srcVocab, SRC_LEN, false);
            String greedy = tgtVocab.decode(greedyTranslate(model, src, tgtVocab));
            String beam   = tgtVocab.decode(beamTranslate(model, src, tgtVocab, 3));
            System.out.printf(Locale.US, "  EN: %-30s%n", p[0]);
            System.out.printf(Locale.US, "     ожидалось: %s%n", p[1]);
            System.out.printf(Locale.US, "     greedy   : %s%n", greedy);
            System.out.printf(Locale.US, "     beam(3)  : %s%n", beam);
        }

        System.out.printf(Locale.US, "%nTrain: %d, Test: %d%n",
                ds.getTrainSamples().size(), ds.getTestSamples().size());
    }

    // ==================== корпус и словарь ====================

    /**
     * Генерирует параллельный корпус EN→RU комбинаторно:
     * {@code subject × verb × object}, с фильтром «невозможных» пар
     * (кот ест кота — возможно, но «cat eats cat» — странно; фильтруем
     * одинаковые subject и object).
     *
     * <p>Списки слов **согласованы** между EN и RU по индексу: {@code subjects[0]}
     * = «cat»/«кот», {@code subjects[1]} = «dog»/«собака», и т.д. Это
     * ключевое: генератор пар не «переводит», а берёт **готовые** пары
     * из согласованных списков.
     *
     * <p><b>Зачем комбинаторно.</b> На 20 парах модель не может выучить
     * различение объектов: «кот видит X» и «собака видит X» с разными X
     * дают сигнал, но если X встречается только с одним subject — модель
     * не обобщит. Комбинаторный корпус покрывает **все** комбинации —
     * модель выучит «subject из source, verb из source, object из source».
     *
     * <p><b>Как выбрать N.</b> Полный декартов квадрат: 6 субъектов ×
     * 4 глагола × 10 объектов = 240 пар (минус фильтр). Если N меньше —
     * берём случайную подвыборку (с seed для воспроизводимости). Если
     * N больше — возвращаем все (не дублируем).
     *
     * @param n       желаемое число пар
     * @param seed    seed для перемешивания (детерминизм)
     * @param includeHeldOut если {@code true}, held-out фразы могут попасть
     *                       в train (для отладки); если {@code false} —
     *                       фильтруем, чтобы held-out остался честным
     */
    private static List<String[]> generateCorpus(int n, long seed, boolean includeHeldOut) {
        String[][] subjects = {
                {"the cat",   "кот"},
                {"the dog",   "собака"},
                {"the man",   "человек"},
                {"the woman", "женщина"},
                {"the boy",   "мальчик"},
                {"the girl",  "девочка"}
        };
        String[][] verbs = {
                {"sees",   "видит"},
                {"eats",   "ест"},
                {"reads",  "читает"},
                {"drinks", "пьет"}
        };
        String[][] objects = {
                {"the dog",    "собаку"},
                {"the cat",    "кота"},
                {"the fish",   "рыбу"},
                {"the meat",   "мясо"},
                {"the book",   "книгу"},
                {"the letter", "письмо"},
                {"the bird",   "птицу"},
                {"the flower", "цветок"},
                {"the bread",  "хлеб"},
                {"the water",  "воду"},
                {"the milk",   "молоко"}
        };

        // Собираем все «разумные» комбинации.
        List<String[]> all = new ArrayList<>();
        for (String[] s : subjects) {
            for (String[] v : verbs) {
                for (String[] o : objects) {
                    // Фильтр: субъект и объект не совпадают.
                    // (cat sees cat — странно; пропускаем.)
                    if (s[0].equals(o[0])) continue;
                    String en = s[0] + " " + v[0] + " " + o[0];
                    String ru = s[1] + " " + v[1] + " " + o[1];
                    all.add(new String[]{en, ru});
                }
            }
        }

        // Фильтр held-out, если нужно.
        if (!includeHeldOut) {
            Set<String> heldOutSet = new HashSet<>();
            for (String[] h : heldOutCorpus()) heldOutSet.add(h[0].toLowerCase(Locale.ROOT));
            all.removeIf(p -> heldOutSet.contains(p[0].toLowerCase(Locale.ROOT)));
        }

        // Перемешиваем с seed.
        Collections.shuffle(all, new Random(seed));

        // Берём n (или все, если n больше).
        if (n >= all.size()) return all;
        return new ArrayList<>(all.subList(0, n));
    }

    /**
     * Игрушечный параллельный корпус EN→RU. 20 пар — достаточно, чтобы
     * модель выучила основные соответствия (the↔(нет артикля), cat↔кот,
     * sees↔видит, ...), но мало для «настоящего» перевода. Для учебного
     * примера — идеально: обучение идёт секунды, результат виден.
     *
     * <p>Почему без артикля в русском: в английском «the cat» → в русском
     * «кот». Модель должна выучить, что «the» не имеет прямого аналога.
     * Это хороший тест на attention-выравнивание.
     */
    private static List<String[]> toyCorpus() {
        return List.of(
                new String[]{"the cat sees the dog",        "кот видит собаку"},
                new String[]{"the dog sees the cat",        "собака видит кота"},
                new String[]{"the cat eats the fish",       "кот ест рыбу"},
                new String[]{"the dog eats the meat",       "собака ест мясо"},
                new String[]{"the man reads the book",      "человек читает книгу"},
                new String[]{"the woman reads the letter",  "женщина читает письмо"},
                new String[]{"the boy sees the bird",       "мальчик видит птицу"},
                new String[]{"the girl sees the flower",    "девочка видит цветок"},
                new String[]{"the man eats the bread",      "человек ест хлеб"},
                new String[]{"the woman drinks the water",  "женщина пьет воду"},
                new String[]{"the cat drinks the milk",     "кот пьет молоко"},
                new String[]{"the dog drinks the water",    "собака пьет воду"},
                new String[]{"the boy reads the letter",    "мальчик читает письмо"},
                new String[]{"the girl reads the book",     "девочка читает книгу"},
                new String[]{"the man sees the woman",      "человек видит женщину"},
                new String[]{"the woman sees the man",      "женщина видит человека"},
                new String[]{"the boy eats the bread",      "мальчик ест хлеб"},
                new String[]{"the girl drinks the water",   "девочка пьет воду"},
                new String[]{"the cat sees the bird",       "кот видит птицу"},
                new String[]{"the dog sees the flower",     "собака видит цветок"}
        );
    }

    /**
     * Held-out корпус: фразы, которых НЕТ в {@link #toyCorpus()}, но
     * слова в них — те же. Проверяет, что модель выучила <b>правила</b>
     * (кот→кот, видит→видит, ...), а не запомнила 20 пар.
     *
     * <p>Некоторые фразы намеренно «склеены» из знакомых кусков:
     * «the woman sees the cat» — woman/see/cat есть в корпусе, но в
     * такой комбинации не встречались. Хорошая модель переведёт.
     *
     * <p>Одна фраза содержит слово вне корпуса («sings») — оно уйдёт
     * в {@code <unk>}, и перевод будет неполным. Это тоже полезно
     * показать: модель не может перевести то, чего не видела.
     */
    private static List<String[]> heldOutCorpus() {
        return List.of(
                // Все слова — из train-корпуса, но комбинации новые.
                new String[]{"the woman sees the cat",   "женщина видит кота"},
                new String[]{"the man reads the letter", "человек читает письмо"},
                new String[]{"the boy drinks the milk",  "мальчик пьет молоко"},
                new String[]{"the cat reads the book",   "кот читает книгу"},
                new String[]{"the girl sees the boy",    "девочка видит мальчика"},
                // Слово вне корпуса: «sings» → <unk> → перевод будет неполным.
                new String[]{"the woman sings the song", "женщина поет песню"}
        );
    }

    /** Словарь: слово → индекс, плюс декодирование индексов обратно в текст. */
    private static final class Vocab {
        final Map<String, Integer> wordToIdx = new HashMap<>();
        final List<String> idxToWord = new ArrayList<>();

        Vocab() {
            // Резервируем спец-токены в фиксированном порядке.
            addRaw("<pad>");
            addRaw("<s>");
            addRaw("</s>");
            addRaw("<unk>");
        }

        void addRaw(String w) {
            wordToIdx.put(w, idxToWord.size());
            idxToWord.add(w);
        }

        void add(String w) {
            if (!wordToIdx.containsKey(w)) addRaw(w);
        }

        int size() { return idxToWord.size(); }

        int idx(String w) {
            Integer i = wordToIdx.get(w);
            return i != null ? i : UNK;
        }

        /** Декодирование: int[] → строка, спец-токены опускаются. */
        String decode(int[] idx) {
            StringBuilder sb = new StringBuilder();
            for (int i : idx) {
                if (i == PAD || i == BOS || i == EOS) break;
                if (sb.length() > 0) sb.append(' ');
                sb.append(idxToWord.get(i));
            }
            return sb.toString();
        }
    }

    /**
     * Строит словарь по корпусу.
     *
     * @param side 0 — source (английский), 1 — target (русский)
     */
    private static Vocab buildVocab(List<String[]> pairs, int side) {
        Vocab v = new Vocab();
        for (String[] p : pairs) {
            for (String tok : p[side].split(" ")) v.add(tok.toLowerCase(Locale.ROOT));
        }
        return v;
    }

    // ==================== примеры ====================

    /**
     * Собирает примеры для обучения.
     *
     * <p><b>Teacher forcing.</b> Декодер видит {@code tgtIn = [BOS, w0, ..., EOS, PAD...]},
     * а loss считается по {@code tgtOut = [w0, w1, ..., EOS, PAD...]} —
     * сдвигу на 1. То есть {@code logits[t]} (декодер видел {@code tgtIn[0..t]})
     * предсказывает {@code tgtOut[t] = tgtIn[t+1]} — следующий токен.
     * Именно это нужно и для обучения, и для генерации.
     */
    private static List<NInputSample> buildSamples(List<String[]> pairs, Vocab srcVocab, Vocab tgtVocab) {
        List<NInputSample> out = new ArrayList<>(pairs.size());
        for (String[] p : pairs) {
            int[] src    = encode(p[0], srcVocab, SRC_LEN, false);
            int[] tgtIn  = encode(p[1], tgtVocab, TGT_LEN, true);   // BOS + слова + EOS
            int[] tgtOut = shiftLeft(tgtIn);                        // слова + EOS + PAD
            out.add(new NInputSample(
                    new float[][]{ toFloat(src), toFloat(tgtIn) },
                    toFloat(tgtOut)));
        }
        return out;
    }

    /** Сдвиг влево на 1: {@code [a, b, c, d] → [b, c, d, PAD]}. */
    private static int[] shiftLeft(int[] a) {
        int[] out = new int[a.length];
        System.arraycopy(a, 1, out, 0, a.length - 1);
        out[a.length - 1] = PAD;
        return out;
    }

    /** Кодирует фразу в int[] нужной длины: padding + BOS/EOS при необходимости. */
    private static int[] encode(String text, Vocab vocab, int len, boolean addBosEos) {
        int[] out = new int[len];
        Arrays.fill(out, PAD);
        int pos = 0;
        if (addBosEos) out[pos++] = BOS;
        for (String tok : text.split(" ")) {
            if (pos >= len) break;
            out[pos++] = vocab.idx(tok.toLowerCase(Locale.ROOT));
        }
        if (addBosEos && pos < len) out[pos] = EOS;
        return out;
    }

    private static float[] toFloat(int[] a) {
        float[] f = new float[a.length];
        for (int i = 0; i < a.length; i++) f[i] = a[i];
        return f;
    }

    private static int[] ints(float[] a) {
        int[] r = new int[a.length];
        for (int i = 0; i < a.length; i++) r[i] = Math.round(a[i]);
        return r;
    }

    // ==================== сборка модели ====================

    /**
     * Собирает {@link Functional} с двумя входами: {@code src} и {@code tgt}.
     *
     * <p>Ветвь энкодера: {@code src → Masking → Embedding → PE → Stack}.
     * Ветвь декодера: {@code tgt → Masking → Embedding → PE → DecoderBlock(M)}.
     * {@code encOut} идёт как KV в cross-attention декодера. Выход —
     * {@code logits: seq(TGT_LEN, vocab_tgt)}.
     */
    private static Functional buildModel(Vocab srcVocab, Vocab tgtVocab) {
        Functional.Builder b = Functional.builder(
                TensorShape.flat(SRC_LEN),   // src — индексы
                TensorShape.flat(TGT_LEN));  // tgt — индексы (сдвинутые)

        Functional.Node srcIn = b.input();
        Functional.Node tgtIn = b.kvInput();

        // ---------- Ветвь энкодера ----------
        Functional.Node srcMask = b.layer(new MaskingLayer.Dto(PAD), srcIn);
        Functional.Node srcEmb  = b.layer(new EmbeddingLayer.Dto(srcVocab.size(), D_MODEL), srcMask);
        Functional.Node srcPe   = b.layer(new PositionalEncodingLayer.Dto(), srcEmb);
        Functional.Node encOut  = TransformerEncoderStack.build(
                b, srcPe,
                new TransformerEncoderBlock.Dto(HEADS, D_MODEL, D_FF, DROPOUT, false),
                N_ENC);

        // ---------- Ветвь декодера ----------
        Functional.Node tgtMask = b.layer(new MaskingLayer.Dto(PAD), tgtIn);
        Functional.Node tgtEmb  = b.layer(new EmbeddingLayer.Dto(tgtVocab.size(), D_MODEL), tgtMask);
        Functional.Node tgtPe   = b.layer(new PositionalEncodingLayer.Dto(), tgtEmb);

        // M блоков декодера. Один блок = self(causal) + cross + FFN + residuals.
        Functional.Node dec = tgtPe;
        for (int i = 0; i < N_DEC; i++) {
            dec = TransformerDecoderBlock.build(
                    b, dec, encOut,
                    new TransformerDecoderBlock.Dto(HEADS, D_MODEL, D_FF, DROPOUT, SRC_LEN));
        }

        // ---------- Голова ----------
        Functional.Node logits = b.layer(new TimeDistributedDenseLayer.Dto(tgtVocab.size()), dec);

        return b.output(logits);
    }

    // ==================== генерация ====================

    // ==================== генерация ====================

    /**
     * Жадная генерация: на каждом шаге подаём текущий префикс, берём
     * argmax <b>последней позиции</b> (это предсказание следующего токена
     * благодаря teacher forcing при обучении), добавляем, повторяем.
     *
     * <p><b>PAD и EOS.</b> {@code EOS} — стоп. {@code PAD} — тоже стоп:
     * если модель выдаёт PAD, значит она «не уверена», и продолжать
     * бессмысленно (PAD в префикс писать нельзя — декодер обучен на
     * «PAD = конец полезной части»). Раньше пустые переводы получались
     * именно из-за того, что PAD попадал в префикс и `decode` резал
     * всё на нём.
     *
     * <p><b>Диагностика.</b> При {@code DEBUG = true} печатает top-3
     * предсказания на каждом шаге — видно, что модель «думает».
     *
     * <p>Без KV-cache: attention пересчитывается по всему префиксу на
     * каждом шаге — {@code O(Tq³)} на всю последовательность. Для
     * {@code Tq=10} — миллисекунды.
     */
    private static int[] greedyTranslate(Functional model, int[] src, Vocab tgtVocab) {
        boolean DEBUG = false;   // включи true, чтобы увидеть top-3 на каждом шаге
        int[] srcPadded = Arrays.copyOf(src, SRC_LEN);

        int[] prefix = new int[TGT_LEN];
        Arrays.fill(prefix, PAD);
        prefix[0] = BOS;
        int len = 1;

        int vocabSize = tgtVocab.size();

        while (len < TGT_LEN) {
            int[] tgtPadded = Arrays.copyOf(prefix, TGT_LEN);
            float[] logits = model.forward2(toFloat(srcPadded), toFloat(tgtPadded), false);

            int off = (len - 1) * vocabSize;   // последняя заполненная позиция
            int best = 0;
            for (int v = 1; v < vocabSize; v++) {
                if (logits[off + v] > logits[off + best]) best = v;
            }

            if (DEBUG) {
                int[] top3 = topK(logits, off, vocabSize, 3);
                System.out.printf(Locale.US, "  [debug] step %d (len=%d): top3 = %s%n",
                        len, len,
                        Arrays.stream(top3)
                                .mapToObj(i -> i + "(" + tgtVocab.idxToWord.get(i) + ")")
                                .toList());
            }

            // EOS/PAD — стоп. PAD в префикс не пишем: декодер обучен на
            // «PAD = конец», и продолжение выдаст мусор.
            if (best == EOS || best == PAD) break;

            prefix[len++] = best;
        }

        // Собираем результат: prefix[1..len-1) без BOS/PAD/EOS.
        int outLen = len - 1;
        int[] out = new int[outLen];
        System.arraycopy(prefix, 1, out, 0, outLen);
        return out;
    }

    /**
     * Возвращает индексы {@code k} наибольших значений в диапазоне
     * {@code [off, off + n)}. При {@code k=3} — три прохода argmax
     * с исключением уже найденных (дёшево, без сортировки).
     *
     * <p>Порядок — по убыванию значения (top1, top2, top3, ...).
     * Если {@code n < k}, возвращает только {@code n} индексов.
     *
     * @param logits логиты
     * @param off    начало диапазона
     * @param n      длина диапазона ({@code vocab})
     * @param k      сколько индексов вернуть
     * @return индексы относительно {@code off} (0..n-1), по убыванию значения
     */
    private static int[] topK(float[] logits, int off, int n, int k) {
        if (k > n) k = n;
        int[] best = new int[k];
        boolean[] taken = new boolean[n];
        for (int rank = 0; rank < k; rank++) {
            int argmax = -1;
            float bestVal = Float.NEGATIVE_INFINITY;
            for (int v = 0; v < n; v++) {
                if (taken[v]) continue;
                if (logits[off + v] > bestVal) {
                    bestVal = logits[off + v];
                    argmax = v;
                }
            }
            best[rank] = argmax;
            taken[argmax] = true;
        }
        return best;
    }

    /**
     * Beam search: на каждом шаге держим {@code beamWidth} лучших гипотез
     * по <b>сумме log-prob</b> сгенерированных токенов. В конце выбираем
     * лучшую с нормировкой на длину (length penalty), чтобы не предпочитать
     * короткие последовательности.
     *
     * <p><b>Почему log-prob, а не вероятности.</b> Произведение вероятностей
     * уходит в 0 для длинных последовательностей; сумма логарифмов —
     * стабильна и аддитивна: добавление токена = прибавление {@code log p}.
     *
     * <p><b>Length penalty.</b> Без неё beam search всегда предпочитает
     * короткие гипотезы (у них больше средняя вероятность на токен).
     * Стандарт — деление на {@code len^α} с {@code α = 0.6..1.0}.
     * Здесь {@code α = 0.6} — компромисс для игрушечного корпуса.
     *
     * <p><b>EOS.</b> Гипотезы, завершившиеся на {@code EOS}, переходят в
     * «готовые» и больше не расширяются. Если все {@code beamWidth}
     * гипотез готовы — выходим.
     *
     * <p><b>PAD.</b> Токен {@code PAD} не расширяет гипотезу: декодер
     * обучен на «PAD = конец», и продолжение выдаст мусор. Если beam
     * выдаёт PAD — эта гипотеза отбрасывается.
     *
     * @param model   обученная модель (forward2: src, tgt → logits)
     * @param src     source-индексы (будут padding'нуты до {@link #SRC_LEN})
     * @param tgtVocab целевой словарь (для {@code size()} и {@code idxToWord})
     * @param beamWidth ширина луча; {@code 1} эквивалентно жадной генерации
     * @return сгенерированные индексы без BOS/EOS/PAD
     */
    private static int[] beamTranslate(Functional model, int[] src, Vocab tgtVocab, int beamWidth) {
        int[] srcPadded = Arrays.copyOf(src, SRC_LEN);
        int vocabSize = tgtVocab.size();

        // Гипотеза: префикс (без BOS), сумма log-prob, завершена ли.
        final class Hyp {
            final int[] tokens;     // сгенерированные токены без BOS/EOS
            final double logProb;   // сумма log p по всем шагам
            final boolean done;     // завершилась на EOS
            Hyp(int[] tokens, double logProb, boolean done) {
                this.tokens = tokens; this.logProb = logProb; this.done = done;
            }
        }

        List<Hyp> beam = new ArrayList<>();
        beam.add(new Hyp(new int[0], 0.0, false));

        for (int step = 0; step < TGT_LEN - 1; step++) {
            List<Hyp> candidates = new ArrayList<>();

            for (Hyp h : beam) {
                if (h.done) {
                    candidates.add(h);   // готовые не расширяем
                    continue;
                }

                // Готовим вход декодера: [BOS, tokens..., PAD...]
                int[] tgtPadded = new int[TGT_LEN];
                Arrays.fill(tgtPadded, PAD);
                tgtPadded[0] = BOS;
                System.arraycopy(h.tokens, 0, tgtPadded, 1, h.tokens.length);

                float[] logits = model.forward2(
                        toFloat(srcPadded),
                        toFloat(tgtPadded),
                        false);

                int off = h.tokens.length * vocabSize;   // позиция нового токена
                // softmax по позиции off, чтобы получить log p
                double maxLogit = Double.NEGATIVE_INFINITY;
                for (int v = 0; v < vocabSize; v++) {
                    double l = logits[off + v];
                    if (l > maxLogit) maxLogit = l;
                }
                double sumExp = 0.0;
                for (int v = 0; v < vocabSize; v++) {
                    sumExp += Math.exp(logits[off + v] - maxLogit);
                }
                double logZ = Math.log(sumExp) + maxLogit;

                for (int v = 0; v < vocabSize; v++) {
                    if (v == PAD) continue;   // PAD не расширяет
                    double logP = logits[off + v] - logZ;

                    int[] newTokens;
                    boolean done;
                    if (v == EOS) {
                        newTokens = h.tokens;         // EOS в токены не пишем
                        done = true;
                    } else {
                        newTokens = Arrays.copyOf(h.tokens, h.tokens.length + 1);
                        newTokens[h.tokens.length] = v;
                        done = false;
                    }
                    candidates.add(new Hyp(newTokens, h.logProb + logP, done));
                }
            }

            // Сортируем по длине-нормированному log-prob (по убыванию),
            // берём top-beamWidth.
            final double ALPHA = 0.6;   // length penalty
            candidates.sort((a, b) -> {
                double sa = a.logProb / Math.pow(a.tokens.length + 1, ALPHA);
                double sb = b.logProb / Math.pow(b.tokens.length + 1, ALPHA);
                return Double.compare(sb, sa);
            });
            beam = new ArrayList<>(candidates.subList(0, Math.min(beamWidth, candidates.size())));

            // Если все готовы — дальше не идём.
            boolean allDone = true;
            for (Hyp h : beam) if (!h.done) { allDone = false; break; }
            if (allDone) break;
        }

        // Финальный выбор: лучшая по нормированному log-prob.
        final double ALPHA = 0.6;
        Hyp best = beam.get(0);
        double bestScore = best.logProb / Math.pow(best.tokens.length + 1, ALPHA);
        for (Hyp h : beam) {
            double s = h.logProb / Math.pow(h.tokens.length + 1, ALPHA);
            if (s > bestScore) { bestScore = s; best = h; }
        }
        return best.tokens;
    }
}