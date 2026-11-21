package rawnet.docs.flows.main.transformer.tasks;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Самопальный BPE (Byte Pair Encoding) — учебная реализация
 * субсловной токенизации.
 *
 * <h2>Зачем</h2>
 * В русском (и любом флективном) языке одно слово имеет много форм:
 * «хороший», «хорошая», «хорошее», «хорошие». Если каждая форма —
 * отдельный токен, словарь раздувается, а модель не обобщает сигнал
 * с одной формы на другую. BPE решает это, разбивая слова на
 * <b>общие куски</b>: «хорош» + «ий» / «ая» / «ее» / «ие».
 *
 * <h2>Как работает</h2>
 * <ol>
 *   <li><b>Инициализация.</b> Каждое слово разбивается на символы +
 *       маркер конца слова {@code </w>} (чтобы различать «хорош» в
 *       конце и в середине).</li>
 *   <li><b>Подсчёт пар.</b> По всему корпусу считаются частоты пар
 *       соседних символов.</li>
 *   <li><b>Слияние.</b> Самая частая пара сливается в новый токен.</li>
 *   <li><b>Повтор.</b> Шаги 2–3 повторяются, пока словарь не достигнет
 *       заданного размера.</li>
 * </ol>
 *
 * <p>Результат — список токенов + правила слияния ({@code merges}).
 * Правила применяются к новому тексту <b>в том же порядке</b>, в котором
 * появились при обучении — иначе результат будет другим.
 *
 * <h2>Отличия от SentencePiece</h2>
 * <ul>
 *   <li>Нет нормализации Unicode (NFC/NFKC). Для русского — почти не
 *       критично; для языков с диакритикой — сломается.</li>
 *   <li>Нет работы с байтами. Java {@code char} — 16 бит; для эмодзи
 *       (surrogate pairs) нужно работать с codepoints или байтами.
 *       Для русского — не нужно.</li>
 *   <li>Наивное обучение {@code O(N · V · L)}. Для игрушечного корпуса
 *       (сотни фраз) — секунды; для 1M фраз — часы. SentencePiece
 *       использует priority queue и работает за секунды.</li>
 *   <li>Нет {@code byte fallback}. Если символ не встречался в train —
 *       вернёт {@code <unk>}. SentencePiece разбивает на байты
 *       ({@code <0xAB>}) и обрабатывает.</li>
 * </ul>
 *
 * <h2>Пример использования</h2>
 * <pre>{@code
 * List<String> corpus = ...;                    // тексты
 * Bpe bpe = Bpe.train(corpus, /* vocabSize *\/ 200);
 *
 * // Кодирование
 * List<Integer> ids = bpe.encode("фильм был хороший");
 *
 * // Декодирование (обратно в текст)
 * String text = bpe.decode(ids);
 * }</pre>
 *
 * <h2>Регистр и нормализация</h2>
 * Все слова приводятся к нижнему регистру через
 * {@link Locale#ROOT}. Для русского это корректно; для турецкого
 * (где {@code i/İ} особые) — нужен {@link Locale#forLanguageTag}.
 * Регистр теряется — это стандарт для BPE-токенизаторов (BERT, GPT
 * тоже работают с lowercase-токенами).
 */
public final class Bpe {

    /** Маркер конца слова. */
    private static final String EOW = "</w>";

    /** Разделитель пар в {@code merges}. */
    private static final String PAIR_SEP = "\u0000";

    /**
     * Токенизация для подсчёта слов: буквы (включая кириллицу) и
     * апострофы. {@link Pattern#UNICODE_CHARACTER_CLASS} критичен —
     * без него {@code \p{L}} = {@code [A-Za-z]}, и кириллица выпадет.
     */
    private static final Pattern WORD = Pattern.compile(
            "[\\p{L}']+", Pattern.UNICODE_CHARACTER_CLASS);

    /** Итоговый словарь токенов (символы + слитые пары). */
    private final List<String> vocab = new ArrayList<>();

    /** Индекс токена в {@link #vocab}. */
    private final Map<String, Integer> tokenToIdx = new HashMap<>();

    /**
     * Правила слияния в порядке появления: {@code merges.get(i) = [left, right]}.
     * При кодировании применяются по порядку — как при обучении.
     */
    private final List<String[]> merges = new ArrayList<>();

    private Bpe() {}

    // ==================== обучение ====================

    /**
     * Обучает BPE на корпусе.
     *
     * @param corpus   тексты (каждая строка — фраза)
     * @param vocabSize желаемый размер словаря; {@code >= 10}
     * @return обученный BPE
     */
    public static Bpe train(List<String> corpus, int vocabSize) {
        if (corpus == null || corpus.isEmpty())
            throw new IllegalArgumentException("corpus пуст");
        if (vocabSize < 10)
            throw new IllegalArgumentException("vocabSize >= 10, получено " + vocabSize);

        Bpe bpe = new Bpe();

        // 1) Считаем частоты слов.
        Map<String, Integer> wordFreq = new HashMap<>();
        for (String line : corpus) {
            Matcher m = WORD.matcher(line);
            while (m.find()) {
                String w = m.group().toLowerCase(Locale.ROOT);
                if (w.isEmpty()) continue;
                wordFreq.merge(w, 1, Integer::sum);
            }
        }

        // 2) Каждое слово — список символов + EOW. Храним вместе с частотой.
        Map<List<String>, Integer> splits = new HashMap<>();
        for (Map.Entry<String, Integer> e : wordFreq.entrySet()) {
            List<String> chars = new ArrayList<>();
            for (int i = 0; i < e.getKey().length(); i++) {
                chars.add(String.valueOf(e.getKey().charAt(i)));
            }
            chars.add(EOW);
            splits.put(chars, e.getValue());
        }

        // 3) Стартовый словарь — все уникальные символы + EOW.
        Set<String> alphabet = new HashSet<>();
        for (List<String> chars : splits.keySet()) alphabet.addAll(chars);
        bpe.vocab.addAll(alphabet);
        for (int i = 0; i < bpe.vocab.size(); i++) bpe.tokenToIdx.put(bpe.vocab.get(i), i);

        // 4) Итеративно сливаем самые частые пары.
        while (bpe.vocab.size() < vocabSize) {
            // 4a) Частоты пар.
            Map<String, Integer> pairFreq = new HashMap<>();
            for (Map.Entry<List<String>, Integer> e : splits.entrySet()) {
                List<String> chars = e.getKey();
                int freq = e.getValue();
                for (int i = 0; i < chars.size() - 1; i++) {
                    String pair = chars.get(i) + PAIR_SEP + chars.get(i + 1);
                    pairFreq.merge(pair, freq, Integer::sum);
                }
            }
            if (pairFreq.isEmpty()) break;

            // 4b) Самая частая пара.
            String bestPair = null;
            int bestFreq = -1;
            for (Map.Entry<String, Integer> e : pairFreq.entrySet()) {
                if (e.getValue() > bestFreq) {
                    bestFreq = e.getValue();
                    bestPair = e.getKey();
                }
            }
            String[] parts = bestPair.split(PAIR_SEP, -1);
            String merged = parts[0] + parts[1];

            // 4c) Добавляем в словарь, если ещё нет.
            if (!bpe.tokenToIdx.containsKey(merged)) {
                bpe.tokenToIdx.put(merged, bpe.vocab.size());
                bpe.vocab.add(merged);
            }
            bpe.merges.add(parts);

            // 4d) Применяем слияние ко всем словам.
            Map<List<String>, Integer> newSplits = new HashMap<>();
            for (Map.Entry<List<String>, Integer> e : splits.entrySet()) {
                List<String> chars = e.getKey();
                List<String> out = new ArrayList<>();
                int i = 0;
                while (i < chars.size()) {
                    if (i < chars.size() - 1
                            && chars.get(i).equals(parts[0])
                            && chars.get(i + 1).equals(parts[1])) {
                        out.add(merged);
                        i += 2;
                    } else {
                        out.add(chars.get(i));
                        i++;
                    }
                }
                newSplits.put(out, e.getValue());
            }
            splits = newSplits;
        }

        return bpe;
    }

    // ==================== кодирование ====================

    /**
     * Кодирует текст в плоский список индексов сабвордов.
     *
     * <p><b>Порядок {@code merges} критичен.</b> Применяем правила
     * в том же порядке, в котором они появились при обучении. Если
     * применить в другом порядке — получится другой результат.
     *
     * <p>Символы, которых не было в train, вернут {@code <unk>}
     * (индекс {@code 0}, если {@code <unk>} не задан явно).
     *
     * @param text текст (любой регистр)
     * @return список индексов сабвордов
     */
    public List<Integer> encode(String text) {
        List<Integer> out = new ArrayList<>();
        Matcher m = WORD.matcher(text);
        while (m.find()) {
            String w = m.group().toLowerCase(Locale.ROOT);
            for (String piece : encodeWord(w)) {
                Integer idx = tokenToIdx.get(piece);
                out.add(idx != null ? idx : tokenToIdx.getOrDefault("<unk>", 0));
            }
        }
        return out;
    }

    /**
     * Кодирует одно слово в последовательность сабвордов.
     *
     * <p>Не возвращает индексы, а возвращает сами строки — полезно
     * для отладки: видно, на что разбилось слово.
     */
    public List<String> encodeWord(String word) {
        List<String> chars = new ArrayList<>();
        String w = word.toLowerCase(Locale.ROOT);
        for (int i = 0; i < w.length(); i++) {
            chars.add(String.valueOf(w.charAt(i)));
        }
        chars.add(EOW);

        // Применяем merges по порядку.
        for (String[] merge : merges) {
            List<String> out = new ArrayList<>();
            int i = 0;
            while (i < chars.size()) {
                if (i < chars.size() - 1
                        && chars.get(i).equals(merge[0])
                        && chars.get(i + 1).equals(merge[1])) {
                    out.add(merge[0] + merge[1]);
                    i += 2;
                } else {
                    out.add(chars.get(i));
                    i++;
                }
            }
            chars = out;
        }
        return chars;
    }

    /**
     * Декодирует список индексов обратно в текст.
     *
     * <p>Склеивает сабворды, убирает {@code </w>}, восстанавливает
     * пробелы между словами по {@code </w>}. {@code <unk>} заменяется
     * на символ {@code ?}.
     */
    public String decode(List<Integer> ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            if (id < 0 || id >= vocab.size()) continue;
            String piece = vocab.get(id);
            if (piece.equals(EOW)) {
                sb.append(' ');
            } else if (piece.endsWith(EOW)) {
                sb.append(piece, 0, piece.length() - EOW.length());
                sb.append(' ');
            } else {
                sb.append(piece);
            }
        }
        return sb.toString().trim();
    }

    // ==================== доступ ====================

    /** Размер словаря (символы + слитые пары). */
    public int size() { return vocab.size(); }

    /** Индекс токена или {@code -1}, если токена нет. */
    public int indexOf(String token) {
        Integer i = tokenToIdx.get(token);
        return i != null ? i : -1;
    }

    /** Токен по индексу или {@code null}, если индекса нет. */
    public String tokenAt(int idx) {
        return (idx >= 0 && idx < vocab.size()) ? vocab.get(idx) : null;
    }

    /** Правила слияния (для отладки/сериализации). */
    public List<String[]> merges() {
        List<String[]> copy = new ArrayList<>(merges.size());
        for (String[] m : merges) copy.add(m.clone());
        return copy;
    }

    @Override
    public String toString() {
        return "Bpe{vocab=" + vocab.size() + ", merges=" + merges.size() + "}";
    }

    // ==================== сохранение/загрузка ====================

    /** Версия формата — чтобы при изменении можно было отличить. */
    private static final String FORMAT_HEADER = "BPE_V1";

    /**
     * Сохраняет BPE в файл.
     *
     * <p><b>Формат</b> (текстовый, UTF-8):
     * <pre>
     *   BPE_V1
     *   vocab &lt;N&gt;
     *   &lt;token_0&gt;
     *   &lt;token_1&gt;
     *   ...
     *   merges &lt;M&gt;
     *   &lt;left_0&gt; &lt;right_0&gt;
     *   &lt;left_1&gt; &lt;right_1&gt;
     *   ...
     * </pre>
     * Токены могут содержать пробелы (например, `</w>` — нет, но
     * «a b» — может), поэтому пишем их **построчно**, а не через
     * пробел. Для merges — `left` и `right` разделены табуляцией
     * ({@code \t}), потому что `left`/`right` могут содержать пробелы.
     *
     * @param path путь к файлу; родительские директории создаются
     * @throws IOException при ошибке записи
     */
    public void save(Path path) throws IOException {
        Path parent = path.toAbsolutePath().getParent();
        if (parent != null) Files.createDirectories(parent);

        try (BufferedWriter w = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            w.write(FORMAT_HEADER);
            w.newLine();

            w.write("vocab " + vocab.size());
            w.newLine();
            for (String tok : vocab) {
                w.write(tok);
                w.newLine();
            }

            w.write("merges " + merges.size());
            w.newLine();
            for (String[] m : merges) {
                w.write(m[0]);
                w.write('\t');
                w.write(m[1]);
                w.newLine();
            }
        }
    }

    /**
     * Загружает BPE из файла, сохранённого {@link #save(Path)}.
     *
     * @param path путь к файлу
     * @return загруженный BPE
     * @throws IOException при ошибке чтения или неверном формате
     */
    public static Bpe load(Path path) throws IOException {
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        if (lines.isEmpty())
            throw new IOException("BPE: пустой файл " + path);

        int line = 0;
        String header = lines.get(line++);
        if (!FORMAT_HEADER.equals(header))
            throw new IOException("BPE: ожидался заголовок " + FORMAT_HEADER
                    + ", получено: " + header);

        Bpe bpe = new Bpe();

        // --- vocab ---
        if (line >= lines.size())
            throw new IOException("BPE: не хватает строки 'vocab N'");
        String vocabHeader = lines.get(line++);
        String[] vocabParts = vocabHeader.split(" ");
        if (vocabParts.length != 2 || !"vocab".equals(vocabParts[0]))
            throw new IOException("BPE: ожидалось 'vocab N', получено: " + vocabHeader);
        int vocabSize;
        try { vocabSize = Integer.parseInt(vocabParts[1]); }
        catch (NumberFormatException e) {
            throw new IOException("BPE: некорректный размер vocab: " + vocabParts[1], e);
        }
        if (vocabSize < 0)
            throw new IOException("BPE: vocab size < 0: " + vocabSize);

        for (int i = 0; i < vocabSize; i++) {
            if (line >= lines.size())
                throw new IOException("BPE: не хватает строк для vocab (нужно " + vocabSize
                        + ", есть " + i + ")");
            String tok = lines.get(line++);
            bpe.vocab.add(tok);
            bpe.tokenToIdx.put(tok, i);
        }

        // --- merges ---
        if (line >= lines.size())
            throw new IOException("BPE: не хватает строки 'merges M'");
        String mergesHeader = lines.get(line++);
        String[] mergesParts = mergesHeader.split(" ");
        if (mergesParts.length != 2 || !"merges".equals(mergesParts[0]))
            throw new IOException("BPE: ожидалось 'merges M', получено: " + mergesHeader);
        int mergesSize;
        try { mergesSize = Integer.parseInt(mergesParts[1]); }
        catch (NumberFormatException e) {
            throw new IOException("BPE: некорректный размер merges: " + mergesParts[1], e);
        }
        if (mergesSize < 0)
            throw new IOException("BPE: merges size < 0: " + mergesSize);

        for (int i = 0; i < mergesSize; i++) {
            if (line >= lines.size())
                throw new IOException("BPE: не хватает строк для merges (нужно " + mergesSize
                        + ", есть " + i + ")");
            String mline = lines.get(line++);
            int tab = mline.indexOf('\t');
            if (tab < 0)
                throw new IOException("BPE: ожидался 'left\\tright', получено: " + mline);
            String left  = mline.substring(0, tab);
            String right = mline.substring(tab + 1);
            bpe.merges.add(new String[]{left, right});
        }

        return bpe;
    }

    // ==================== «загрузи или обучи» ====================

    /**
     * Загружает BPE из файла, если он есть; иначе обучает на корпусе и
     * сохраняет по тому же пути.
     *
     * <p><b>Зачем.</b> Один вызов вместо ручного {@code if (Files.exists)}:
     * <pre>{@code
     * Bpe bpe = Bpe.loadOrTrain(Path.of("bpe.bin"), corpus, 500);
     * }</pre>
     *
     * <p><b>Что проверяется.</b> Если файл есть, но загруженный BPE
     * имеет {@code size() > vocabSize} — значит, он обучен с другим
     * {@code vocabSize} (или на другом корпусе) и <b>несовместим</b>
     * с ожиданиями вызывающего. В этом случае BPE <b>переобучается</b>
     * и перезаписывается. Это защищает от «загрузился старый BPE,
     * модель сломалась».
     *
     * <p><b>Кэш по пути.</b> Метод <b>не</b> кэширует BPE в памяти —
     * каждый вызов читает файл (или обучает). Для учебного примера
     * это ок; для продакшена — кэш в вызывающем коде.
     *
     * @param path      путь к файлу BPE; родительские директории создаются
     * @param corpus    тексты для обучения (используются, только если
     *                  файла нет или он несовместим)
     * @param vocabSize желаемый размер словаря (используется для проверки
     *                  совместимости и для обучения)
     * @return загруженный или обученный BPE
     * @throws IOException при ошибке чтения/записи
     */
    public static Bpe loadOrTrain(Path path, List<String> corpus, int vocabSize) throws IOException {
        if (path == null)
            throw new IllegalArgumentException("path == null");
        if (corpus == null || corpus.isEmpty())
            throw new IllegalArgumentException("corpus пуст");
        if (vocabSize < 10)
            throw new IllegalArgumentException("vocabSize >= 10, получено " + vocabSize);

        if (Files.isRegularFile(path)) {
            // Пытаемся загрузить. Если формат битый — падаем с понятной
            // ошибкой, чтобы пользователь удалил файл (а не «тихо»
            // переобучал).
            Bpe bpe = load(path);
            if (bpe.size() > vocabSize) {
                // Несовместим: обучен с большим словарём. Переобучаем.
                System.err.println("BPE: файл " + path + " имеет size=" + bpe.size()
                        + " > ожидаемого " + vocabSize + " — переобучаю...");
                Bpe retrained = train(corpus, vocabSize);
                retrained.save(path);
                return retrained;
            }
            return bpe;
        }

        // Файла нет — обучаем и сохраняем.
        Bpe bpe = train(corpus, vocabSize);
        bpe.save(path);
        return bpe;
    }
}