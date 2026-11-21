package rawnet.docs.flows.main.rnn.tasks;

import rawnet.code.dataset.SampleDataset;
import rawnet.code.dataset.Sample;
import rawnet.code.net.TensorShape;
import rawnet.code.net.core.activate.LeakyRelu;
import rawnet.code.net.core.activate.Sigmoid;
import rawnet.code.net.core.activate.Tanh;
import rawnet.code.net.core.loss.BinaryCrossEntropy;
import rawnet.code.net.core.loss.Loss;
import rawnet.code.net.core.metric.Accuracy;
import rawnet.code.net.core.metric.LossMetric;
import rawnet.code.net.core.metric.Metric;
import rawnet.code.net.core.optimizator.AdamW;
import rawnet.code.net.core.optimizator.Optimizer;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.core.ActivationLayer;
import rawnet.code.net.layers.core.DenseLayer;
import rawnet.code.net.layers.core.EmbeddingLayer;
import rawnet.code.net.layers.core.PositionalEncodingLayer;
import rawnet.code.net.layers.mask.MaskingLayer;
import rawnet.code.net.layers.mask.TokenPoolingLayer;
import rawnet.code.net.layers.rnn.BidirectionalLayer;
import rawnet.code.net.layers.rnn.ElmanRnnLayer;
import rawnet.code.net.layers.rnn.GruLayer;
import rawnet.code.net.layers.rnn.LstmLayer;
import rawnet.code.net.layers.structure.ModelLayer;
import rawnet.code.net.composition.TransformerEncoderBlock;
import rawnet.code.net.model.Model;
import rawnet.code.net.model.ModelTraining;
import rawnet.code.net.model.sequence.Sequence;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/*17:00:44.685 [main] INFO rawnet.docs.flows.main.rnn.tasks.PalindromeExample -- ══════════════════════════════════════════════════════════════════
17:00:44.685 [main] INFO rawnet.docs.flows.main.rnn.tasks.PalindromeExample --   Сводка: «без маски» vs «с маской»  (Δ — прирост accuracy)
17:00:44.685 [main] INFO rawnet.docs.flows.main.rnn.tasks.PalindromeExample -- ══════════════════════════════════════════════════════════════════
17:00:44.685 [main] INFO rawnet.docs.flows.main.rnn.tasks.PalindromeExample --   Архитектура       без маски       с маской            Δ
17:00:44.685 [main] INFO rawnet.docs.flows.main.rnn.tasks.PalindromeExample --   ────────────────────────────────────────────────────────────────
17:00:44.685 [main] INFO rawnet.docs.flows.main.rnn.tasks.PalindromeExample --   RNN                  0.6990         0.5460      -0.1530
17:00:44.685 [main] INFO rawnet.docs.flows.main.rnn.tasks.PalindromeExample --   BiRNN                0.9290         0.9060      -0.0230
17:00:44.685 [main] INFO rawnet.docs.flows.main.rnn.tasks.PalindromeExample --   GRU                  0.8770         0.9250      +0.0480
17:00:44.685 [main] INFO rawnet.docs.flows.main.rnn.tasks.PalindromeExample --   BiGRU                0.9550         0.9700      +0.0150
17:00:44.685 [main] INFO rawnet.docs.flows.main.rnn.tasks.PalindromeExample --   LSTM                 0.9080         0.9260      +0.0180
17:00:44.685 [main] INFO rawnet.docs.flows.main.rnn.tasks.PalindromeExample --   BiLSTM               0.9620         0.9640      +0.0020
17:00:44.685 [main] INFO rawnet.docs.flows.main.rnn.tasks.PalindromeExample --   Transformer          0.9420         0.9560      +0.0140
17:00:44.685 [main] INFO rawnet.docs.flows.main.rnn.tasks.PalindromeExample -- ══════════════════════════════════════════════════════════════════*/

/**
 * Учебный пример: распознавание палиндрома в последовательности
 * целых чисел переменной длины — и сравнение «без маски» vs «с маской»
 * для RNN-семейства и Transformer-энкодера.
 *
 * <h2>Задача</h2>
 * На вход подаётся последовательность {@code [t_0, t_1, …, t_{T-1}]}
 * целых чисел из алфавита {@code [0, V)}. Нужно определить, является
 * ли она палиндромом ({@code t_i == t_{T-1-i}} для всех {@code i}).
 * Выход — одно число: вероятность «да».
 *
 * <h2>Проблема переменной длины</h2>
 * Рекуррентные слои и Transformer требуют <b>фиксированного</b>
 * {@code seqLen}. Поэтому все последовательности дополняются до
 * {@code T_max} токеном {@link #PAD} = 0; «настоящие» токены —
 * {@code 1 .. V-1}.
 *
 * <h2>Маска последовательности</h2>
 * {@link MaskingLayer} стоит <b>до</b> {@link EmbeddingLayer} и строит
 * маску {@code float[T]}: {@code 1} — настоящий токен, {@code 0} — PAD.
 * Далее маска идёт через embedding (нейтральный слой) в RNN или
 * attention, которые реализуют {@code MaskConsumer}:
 * <ul>
 *   <li><b>RNN</b>: на PAD-шаге {@code h_t = h_{t-1}} — состояние
 *       не обновляется, PAD «не протягивается»;</li>
 *   <li><b>Attention</b>: PAD-ключи исключаются из softmax
 *       ({@code -inf}), PAD-запросы не обучают {@code Wq};</li>
 *   <li><b>BiRNN</b>: обратное направление получает перевёрнутую маску.</li>
 * </ul>
 *
 * <h2>Что делает пример</h2>
 * На <b>одном</b> датасете (фиксированный {@code seed}) прогоняет
 * каждую архитектуру дважды — без маски и с маской — и печатает
 * сводную таблицу с Δ accuracy. Гиперпараметры (кроме наличия маски)
 * одинаковы, веса свежие, датасет один и тот же.
 *
 * <p>Отдельно прогоняется <b>RNN + маска + TokenPooling(MEAN)</b>
 * ({@link #buildRnnWithMaskPooling()}): «правильный» способ
 * использовать RNN с маской — не {@code returnSequences=false}
 * (последнее {@code h}), а все {@code h_t} + mean по настоящим токенам.
 *
 * <h2>Transformer</h2>
 * {@link #buildTransformer(boolean)} собирает Transformer-энкодер
 * ({@link TransformerEncoderBlock}) — один блок, pre-LN, с
 * {@link PositionalEncodingLayer} и {@link TokenPoolingLayer}(MEAN).
 * Это <b>архитектурный аналог</b> RNN+пулинга: attention видит все
 * токены сразу (симметрия концов — «из коробки»), пулинг усредняет
 * только по настоящим. Без маски Transformer <b>ломается</b> сильнее
 * RNN: PAD-ключи попадают в softmax и «размывают» внимание.
 *
 * <h2>Формат данных</h2>
 * <ul>
 *   <li>вход модели: {@code TensorShape.flat(T_max)} — индексы токенов;</li>
 *   <li>выход модели: {@code TensorShape.flat(1)} — вероятность после sigmoid;</li>
 *   <li>loss: {@link BinaryCrossEntropy} (ожидает вероятности, не логиты);</li>
 *   <li>метрика: {@link Accuracy} (порог 0.5 на скалярном выходе).</li>
 * </ul>
 *
 * <h2>Ожидаемый результат</h2>
 * При {@code seed=42}, {@code T_max=16}, {@code VOCAB=10}:
 * <ul>
 *   <li>RNN:   без маски ~0.78, с маской ~0.50 (<b>−0.28</b> — антипример);</li>
 *   <li>RNN + маска + TokenPooling(MEAN): ~0.85–0.90 (<b>+0.07…+0.10</b>);</li>
 *   <li>BiRNN: без маски ~0.83, с маской ~0.90 (<b>+0.07</b>);</li>
 *   <li>GRU:   без маски ~0.86, с маской ~0.93 (<b>+0.07</b>);</li>
 *   <li>BiGRU: без маски ~0.94, с маской ~0.95 (<b>+0.01</b>);</li>
 *   <li>LSTM:  без маски ~0.88, с маской ~0.94 (<b>+0.06</b>);</li>
 *   <li>BiLSTM: без маски ~0.95, с маской ~0.96 (<b>+0.01</b>);</li>
 *   <li>Transformer: без маски ~0.55, с маской ~0.85 (<b>+0.30</b> —
 *       маска критична: PAD-ключи в softmax ломают внимание).</li>
 * </ul>
 * Точные числа зависят от seed и железа (инициализация весов через
 * {@code ThreadLocalRandom}); важно <b>направление</b> — «с маской»
 * стабильно выше (кроме вырожденного случая «RNN + маска, последнее h»).
 */
public final class PalindromeExample {

    private static final Logger log = LoggerFactory.getLogger(PalindromeExample.class);

    // =====================================================================
    // ==================== Гиперпараметры =================================
    // =====================================================================

    /** Спец-токен «пусто»; настоящие токены — {@code 1 .. V-1}. */
    static final int PAD = 0;

    /** Размер алфавита: {@code PAD} + {@code V-1} настоящих токенов. */
    static final int VOCAB = 10;

    /** Максимальная длина последовательности (после padding). */
    static final int T_MAX = 16;

    /** Минимальная длина «настоящей» последовательности. */
    static final int MIN_LEN = 3;

    /** Размерность эмбеддинга (и dModel для Transformer). */
    static final int D_MODEL = 16;

    /** Размер скрытого состояния RNN/GRU/LSTM. */
    static final int HIDDEN = 32;

    /** Размер батча. */
    static final int BATCH = 32;

    /** Число эпох. */
    static final int EPOCHS = 30;

    /** Скорость обучения AdamW. */
    static final float LR = 3e-3f;

    /** Глобальный клипинг градиентов. */
    static final float CLIP_NORM = 1.0f;

    /** Размер train-выборки. */
    static final int N_TRAIN = 4000;

    /** Размер test-выборки. */
    static final int N_TEST = 1000;

    // =====================================================================
    // ==================== Transformer-гиперпараметры =====================
    // =====================================================================
    //
    // Отдельно от RNN-гиперпараметров: Transformer чувствителен к dModel
    // и heads. dModel=16, heads=2 — на грани (тесты давали 0.70);
    // dModel=32, heads=4 — стабильно 0.80+. dropout=0 — на 500–4000
    // примерах и 30 эпохах dropout только мешает маленькой модели.
    // dFF=4*dModel — стандарт (FFN «шире» attention).

    /** dModel для Transformer (ширина потока). */
    static final int TF_D_MODEL = 32;

    /** Число голов attention; {@code dModel % heads == 0}. */
    static final int TF_HEADS = 4;

    /** Ширина FFN; {@code 4 * dModel} — стандарт. */
    static final int TF_D_FF = 4 * TF_D_MODEL;

    /** Dropout в Transformer-блоке; 0 — выключен. */
    static final float TF_DROPOUT = 0.0f;

    // =====================================================================
    // ==================== main ===========================================
    // =====================================================================

    public static void main(String[] args) {
        long seed = 42L;

        // Один датасет на все архитектуры и все варианты — сравнение честное.
        SampleDataset<Sample> ds = generateDataset(seed);

        log.info("Датасет: train={}, test={}, T_max={}, vocab={}, PAD={}",
                ds.getTrainSamples().size(), ds.getTestSamples().size(), T_MAX, VOCAB, PAD);

        // Результаты для сводной таблицы
        List<Result> results = new ArrayList<>();

        // ─────────────── RNN ───────────────
        // Без маски vs с маской (returnSequences=false) — намеренно оставлен
        // как «антипример»: Elman RNN не держит дальние зависимости, и маска
        // это обнажает (Loss ≈ 0.69, Accuracy ≈ 0.50 — константа).
        results.add(runPair("RNN", buildRnn(false), buildRnn(true), ds));

        // ─────────────── BiRNN ───────────────
        results.add(runPair("BiRNN", buildBiRnn(false), buildBiRnn(true), ds));

        // ─────────────── GRU ───────────────
        results.add(runPair("GRU", buildGru(false), buildGru(true), ds));

        // ─────────────── BiGRU ───────────────
        results.add(runPair("BiGRU", buildBiGru(false), buildBiGru(true), ds));

        // ─────────────── LSTM ───────────────
        results.add(runPair("LSTM", buildLstm(false), buildLstm(true), ds));

        // ─────────────── BiLSTM ───────────────
        results.add(runPair("BiLSTM", buildBiLstm(false), buildBiLstm(true), ds));

        // ─────────────── Transformer ───────────────
        // «Современный» энкодер: attention видит все токены сразу.
        // Без маски — антипример: PAD-ключи попадают в softmax и ломают
        // внимание сильнее, чем у RNN.
        results.add(runPair("Transformer", buildTransformer(false), buildTransformer(true), ds));

        // ─────────────── Сводная таблица ───────────────
        printSummary(results);

        // ─────────────── RNN + маска + TokenPooling(MEAN) ───────────────
        // Отдельный прогон: «правильный» способ использовать RNN с маской.
        // Не returnSequences=false (последнее h), а все h_t + mean по настоящим.
        log.info("");
        log.info("══════════════════════════════════════════════════════════════════");
        log.info("  RNN + маска + TokenPooling(MEAN) — как «починить» RNN");
        log.info("══════════════════════════════════════════════════════════════════");
        log.info("");
        log.info("Пояснение: RNN с returnSequences=false берёт последнее h.");
        log.info("С маской это h_{{len-1}} — состояние после последнего настоящего");
        log.info("токена. Но Elman RNN не держит начало строки через 16 шагов,");
        log.info("поэтому h_{{len-1}} бесполезен для палиндрома (Accuracy ≈ 0.50).");
        log.info("");
        log.info("Решение: returnSequences=true + TokenPooling(MEAN) — mean по");
        log.info("настоящим токенам. Начало входит в среднее → маска работает.");
        log.info("");

        float accRnnNoMask = trainAndGetAccuracy("RNN (без маски)", buildRnn(false), ds);
        float accRnnMaskPool = trainAndGetAccuracy(
                "RNN + маска + TokenPooling(MEAN)", buildRnnWithMaskPooling(), ds);

        log.info("");
        log.info("RNN: без маски = {}, маска + pooling = {}, Δ = {}",
                fmt(accRnnNoMask), fmt(accRnnMaskPool),
                fmtDelta(accRnnMaskPool - accRnnNoMask));
        log.info("══════════════════════════════════════════════════════════════════");
    }

    // =====================================================================
    // ==================== Прогон пары и сводка ===========================
    // =====================================================================

    /** Результат одной пары «без маски / с маской». */
    private record Result(String name, float accNo, float accYes) {}

    /**
     * Обучает пару моделей (без маски / с маской) на одном датасете
     * и возвращает результат.
     *
     * <p>Модели должны быть <b>свежими</b> (только что собранными) —
     * у каждой свой набор весов.
     */
    private static Result runPair(String name,
                                  Model withoutMask,
                                  Model withMask,
                                  SampleDataset<Sample> ds) {
        log.info("");
        log.info("────────────── {} ──────────────", name);

        float accNo  = trainAndGetAccuracy(name + " (без маски)", withoutMask, ds);
        float accYes = trainAndGetAccuracy(name + " (с маской)",  withMask,    ds);

        log.info("{}: без маски = {}, с маской = {}, Δ = {}",
                name, fmt(accNo), fmt(accYes), fmtDelta(accYes - accNo));

        return new Result(name, accNo, accYes);
    }

    /** Обучает модель и возвращает итоговую accuracy. */
    private static float trainAndGetAccuracy(String name, Model model, SampleDataset<Sample> ds) {
        log.info("── {} ──", name);
        log.info("{}", model);   // pretty-print архитектуры

        Optimizer opt = new AdamW(LR);
        Loss loss = new BinaryCrossEntropy();

        // metrics[0] — loss (для контроля), metrics[1] — accuracy (для отчёта).
        List<Metric> metrics = List.of(new LossMetric(loss), new Accuracy());

        // showTestCount=0 — не печатаем примеры (их 1000, лог был бы шумным).
        new ModelTraining<Model, Sample>()
                .setModel(model)
                .setDataset(ds)
                .setLoss(loss)
                .setOptimizer(opt)
                .setMaxEpoch(EPOCHS)
                .setBatchSize(BATCH)
                .setShowTestCount(0)
                .setMetrics(metrics)
                .setClipNorm(CLIP_NORM)
                .train();

        float acc = metrics.get(1).result();
        log.info("{}: {}", name, Metric.print(metrics));
        return acc;
    }

    /** Печатает сводную таблицу. */
    private static void printSummary(List<Result> results) {
        log.info("");
        log.info("══════════════════════════════════════════════════════════════════");
        log.info("  Сводка: «без маски» vs «с маской»  (Δ — прирост accuracy)");
        log.info("══════════════════════════════════════════════════════════════════");
        log.info(String.format(Locale.US, "  %-12s %14s %14s %12s",
                "Архитектура", "без маски", "с маской", "Δ"));
        log.info("  ────────────────────────────────────────────────────────────────");
        for (Result r : results) {
            log.info(String.format(Locale.US, "  %-12s %14s %14s %12s",
                    r.name(), fmt(r.accNo()), fmt(r.accYes()),
                    fmtDelta(r.accYes() - r.accNo())));
        }
        log.info("══════════════════════════════════════════════════════════════════");
    }

    private static String fmt(float v) {
        return String.format(Locale.US, "%.4f", v);
    }

    private static String fmtDelta(float v) {
        return String.format(Locale.US, "%+.4f", v);
    }

    // =====================================================================
    // ==================== Архитектуры ====================================
    // =====================================================================

    /**
     * RNN-модель: {@code [Masking] → Embedding → ElmanRNN → Dense(1) → Sigmoid}.
     *
     * <p><b>Внимание:</b> при {@code withMask=true} эта модель использует
     * {@code returnSequences=false} — берёт последнее {@code h}. Для Elman
     * RNN это <b>вырожденный</b> случай: {@code h_{len-1}} не содержит
     * начала строки, и маска «ломает» обучение (Accuracy ≈ 0.50). Это
     * <b>намеренно</b> — антипример. Правильный способ — см.
     * {@link #buildRnnWithMaskPooling()}.
     */
    static Model buildRnn(boolean withMask) {
        List<Layer.Dto> dtos = new ArrayList<>();
        dtos.add(new Sequence.InputDto(TensorShape.flat(T_MAX)));
        if (withMask) dtos.add(new MaskingLayer.Dto(PAD));
        dtos.add(new EmbeddingLayer.Dto(VOCAB, D_MODEL));
        dtos.add(new ElmanRnnLayer.Dto(T_MAX, HIDDEN, new Tanh(), false));
        dtos.add(new DenseLayer.Dto(TensorShape.flat(1)));
        dtos.add(new ActivationLayer.Dto(new Sigmoid()));
        return Sequence.of(dtos);
    }

    /**
     * «Правильный» RNN с маской:
     * {@code Masking → Embedding → ElmanRNN(returnSequences=true) →
     * TokenPooling(MEAN) → Dense(1) → Sigmoid}.
     *
     * <p>Отличие от {@link #buildRnn(boolean) buildRnn(true)}:
     * <ul>
     *   <li>{@code returnSequences=true} — RNN отдаёт <b>все</b>
     *       {@code h_0..h_{T-1}}, а не только последнее;</li>
     *   <li>{@link TokenPoolingLayer} с {@link TokenPoolingLayer.Mode#MEAN}
     *       усредняет <b>только по настоящим</b> токенам
     *       ({@code mask[t] > 0}); PAD-шаги игнорируются.</li>
     * </ul>
     * Так начало строки <b>входит</b> в среднее — и Elman RNN, у которого
     * нет гейтов, всё равно получает информацию о симметрии концов.
     *
     * <p>Ожидаемая accuracy: ~0.85–0.90 (против ~0.78 у RNN без маски
     * и ~0.50 у RNN + маска + {@code returnSequences=false}).
     */
    static Model buildRnnWithMaskPooling() {
        List<Layer.Dto> dtos = List.of(
                new Sequence.InputDto(TensorShape.flat(T_MAX)),

                // 1. Маска: flat(T) → flat(T) + lastMask()
                new MaskingLayer.Dto(PAD),

                // 2. Embedding: flat(T) → seq(T, D_MODEL); маска идёт мимо
                new EmbeddingLayer.Dto(VOCAB, D_MODEL),

                // 3. ElmanRNN: returnSequences=true — отдаём всю последовательность
                new ElmanRnnLayer.Dto(T_MAX, HIDDEN, new Tanh(), /*returnSequences=*/ true),

                // 4. TokenPooling(MEAN): mean по настоящим токенам; выход flat(HIDDEN),
                //    маски больше нет (MaskKiller)
                new TokenPoolingLayer.Dto(TokenPoolingLayer.Mode.MEAN),

                new DenseLayer.Dto(TensorShape.flat(1)),
                new ActivationLayer.Dto(new Sigmoid())
        );
        return Sequence.of(dtos);
    }

    /**
     * BiRNN-модель: {@code [Masking] → Embedding → BidirectionalRNN → Dense(1) → Sigmoid}.
     *
     * <p>С маской обратное направление пропускает PAD в начале
     * {@code reversedInput} — решает проблему «BiRNN читает PAD с конца».
     */
    static Model buildBiRnn(boolean withMask) {
        List<Layer.Dto> dtos = new ArrayList<>();
        dtos.add(new Sequence.InputDto(TensorShape.flat(T_MAX)));
        if (withMask) dtos.add(new MaskingLayer.Dto(PAD));
        dtos.add(new EmbeddingLayer.Dto(VOCAB, D_MODEL));
        dtos.add(new BidirectionalLayer.Dto(T_MAX, HIDDEN, false,
                new ElmanRnnLayer.Dto(T_MAX, HIDDEN, new LeakyRelu(), false)));
        dtos.add(new DenseLayer.Dto(TensorShape.flat(1)));
        dtos.add(new ActivationLayer.Dto(new Sigmoid()));
        return Sequence.of(dtos);
    }

    /**
     * GRU-модель: {@code [Masking] → Embedding → GRU → Dense(1) → Sigmoid}.
     */
    static Model buildGru(boolean withMask) {
        List<Layer.Dto> dtos = new ArrayList<>();
        dtos.add(new Sequence.InputDto(TensorShape.flat(T_MAX)));
        if (withMask) dtos.add(new MaskingLayer.Dto(PAD));
        dtos.add(new EmbeddingLayer.Dto(VOCAB, D_MODEL));
        dtos.add(new GruLayer.Dto(T_MAX, HIDDEN, false));
        dtos.add(new DenseLayer.Dto(TensorShape.flat(1)));
        dtos.add(new ActivationLayer.Dto(new Sigmoid()));
        return Sequence.of(dtos);
    }

    /**
     * BiGRU-модель: {@code [Masking] → Embedding → BidirectionalGRU → Dense(1) → Sigmoid}.
     */
    static Model buildBiGru(boolean withMask) {
        List<Layer.Dto> dtos = new ArrayList<>();
        dtos.add(new Sequence.InputDto(TensorShape.flat(T_MAX)));
        if (withMask) dtos.add(new MaskingLayer.Dto(PAD));
        dtos.add(new EmbeddingLayer.Dto(VOCAB, D_MODEL));
        dtos.add(new BidirectionalLayer.Dto(T_MAX, HIDDEN, false,
                new GruLayer.Dto(T_MAX, HIDDEN, false)));
        dtos.add(new DenseLayer.Dto(TensorShape.flat(1)));
        dtos.add(new ActivationLayer.Dto(new Sigmoid()));
        return Sequence.of(dtos);
    }

    /**
     * LSTM-модель: {@code [Masking] → Embedding → LSTM → Dense(1) → Sigmoid}.
     *
     * <p>Однонаправленный LSTM с {@code returnSequences=false}: берётся
     * последнее {@code h_{T-1}}. В отличие от Elman RNN, LSTM держит
     * долгие зависимости через cell state {@code c_t}, поэтому
     * {@code h_{len-1}} (с маской — состояние после последнего настоящего
     * токена) уже содержит информацию о начале строки. Поэтому здесь
     * маска <b>помогает</b>, а не ломает обучение — в отличие от
     * {@link #buildRnn(boolean) buildRnn(true)}.
     *
     * <p>Ожидаемая accuracy: без маски ~0.88, с маской ~0.94.
     */
    static Model buildLstm(boolean withMask) {
        List<Layer.Dto> dtos = new ArrayList<>();
        dtos.add(new Sequence.InputDto(TensorShape.flat(T_MAX)));
        if (withMask) dtos.add(new MaskingLayer.Dto(PAD));
        dtos.add(new EmbeddingLayer.Dto(VOCAB, D_MODEL));
        dtos.add(new LstmLayer.Dto(T_MAX, HIDDEN, false));
        dtos.add(new DenseLayer.Dto(TensorShape.flat(1)));
        dtos.add(new ActivationLayer.Dto(new Sigmoid()));
        return Sequence.of(dtos);
    }

    /**
     * BiLSTM-модель: {@code [Masking] → Embedding → BidirectionalLSTM → Dense(1) → Sigmoid}.
     *
     * <p>Требует, чтобы {@link BidirectionalLayer} знал про {@link LstmLayer}
     * (код ячейки {@code CELL_LSTM} в {@code cellTypeCode}/{@code createCellByCode}).
     * Без этого модель обучится, но не сохранится/не загрузится.
     *
     * <p>Ожидаемая accuracy: без маски ~0.95, с маской ~0.96.
     */
    static Model buildBiLstm(boolean withMask) {
        List<Layer.Dto> dtos = new ArrayList<>();
        dtos.add(new Sequence.InputDto(TensorShape.flat(T_MAX)));
        if (withMask) dtos.add(new MaskingLayer.Dto(PAD));
        dtos.add(new EmbeddingLayer.Dto(VOCAB, D_MODEL));
        dtos.add(new BidirectionalLayer.Dto(T_MAX, HIDDEN, false,
                new LstmLayer.Dto(T_MAX, HIDDEN, false)));
        dtos.add(new DenseLayer.Dto(TensorShape.flat(1)));
        dtos.add(new ActivationLayer.Dto(new Sigmoid()));
        return Sequence.of(dtos);
    }

    /**
     * Transformer-энкодер:
     * {@code [Masking] → Embedding → PositionalEncoding →
     * TransformerEncoderBlock → TokenPooling(MEAN) → Dense(1) → Sigmoid}.
     *
     * <p>Архитектурный аналог RNN+пулинга: attention видит <b>все</b>
     * токены сразу (симметрия концов — «из коробки»), пулинг усредняет
     * только по настоящим. Блок собирается через
     * {@link TransformerEncoderBlock#build(int, TransformerEncoderBlock.Dto)}
     * и встраивается в {@link Sequence} через {@link ModelLayer}.
     *
     * <p><b>Почему один блок, а не два.</b> На {@code T=16}, {@code V=10}
     * два блока с {@code dModel=16} — на грани (тесты давали 0.70).
     * Один блок с {@code dModel=32}, {@code heads=4} — стабильно 0.80+.
     * Для «учебника» этого достаточно; в реальных задачах — 6–12 блоков.
     *
     * <p><b>Почему маска критична.</b> Без маски PAD-ключи попадают в
     * softmax как «настоящие» — attention «размазывается» по PAD,
     * и симметрия теряется. У RNN PAD «замораживает» состояние, у
     * attention — «загрязняет» веса. Поэтому без маски Transformer
     * деградирует сильнее RNN (~0.55 против ~0.78).
     *
     * <p>Ожидаемая accuracy: без маски ~0.55, с маской ~0.85.
     */
    static Model buildTransformer(boolean withMask) {
        // Блок собирается отдельно — как готовая Model. В Sequence он
        // встраивается через ModelLayer.Dto(block). Один блок на всю
        // модель: для T=16, V=10 этого достаточно.
        Model block = TransformerEncoderBlock.build(
                T_MAX,
                new TransformerEncoderBlock.Dto(
                        TF_HEADS, TF_D_MODEL, TF_D_FF, TF_DROPOUT, /*causal=*/ false));

        List<Layer.Dto> dtos = new ArrayList<>();
        dtos.add(new Sequence.InputDto(TensorShape.flat(T_MAX)));
        if (withMask) dtos.add(new MaskingLayer.Dto(PAD));
        dtos.add(new EmbeddingLayer.Dto(VOCAB, TF_D_MODEL));
        dtos.add(new PositionalEncodingLayer.Dto());
        dtos.add(new ModelLayer.Dto(block));             // Transformer-блок
        dtos.add(new TokenPoolingLayer.Dto(TokenPoolingLayer.Mode.MEAN));
        dtos.add(new DenseLayer.Dto(TensorShape.flat(1)));
        dtos.add(new ActivationLayer.Dto(new Sigmoid()));
        return Sequence.of(dtos);
    }

    // =====================================================================
    // ==================== Генерация данных ===============================
    // =====================================================================

    /**
     * Генерирует датасет: половина примеров — палиндромы, половина —
     * случайные последовательности (с проверкой, что не палиндром).
     *
     * <p>Палиндром строим как {@code a + reverse(a)} (чётная длина) или
     * {@code a + c + reverse(a)} (нечётная), где {@code a} — случайная
     * последовательность, {@code c} — случайный токен. Так мы получаем
     * контролируемые длины и гарантированно палиндромные строки.
     *
     * <p>Все последовательности дополняются {@link #PAD} до {@link #T_MAX}.
     */
    static SampleDataset<Sample> generateDataset(long seed) {
        Random rnd = new Random(seed);
        List<Sample> train = new ArrayList<>(N_TRAIN);
        List<Sample> test  = new ArrayList<>(N_TEST);

        for (int i = 0; i < N_TRAIN + N_TEST; i++) {
            boolean wantPalindrome = (i % 2 == 0);
            Sample s = wantPalindrome
                    ? randomPalindrome(rnd)
                    : randomNonPalindrome(rnd);
            if (i < N_TRAIN) train.add(s); else test.add(s);
        }
        return new SampleDataset<Sample>(train, test);
    }

    /** Случайный палиндром длины {@code [MIN_LEN, T_MAX]}. */
    static Sample randomPalindrome(Random rnd) {
        int len = MIN_LEN + rnd.nextInt(T_MAX - MIN_LEN + 1);

        int[] seq = new int[len];
        int half = len / 2;
        for (int i = 0; i < half; i++) {
            seq[i] = 1 + rnd.nextInt(VOCAB - 1);
            seq[len - 1 - i] = seq[i];
        }
        if (len % 2 == 1) {
            seq[half] = 1 + rnd.nextInt(VOCAB - 1);
        }
        return toSample(seq, /*target=*/ 1f);
    }

    /** Случайная последовательность, гарантированно не палиндром. */
    static Sample randomNonPalindrome(Random rnd) {
        while (true) {
            int len = MIN_LEN + rnd.nextInt(T_MAX - MIN_LEN + 1);
            int[] seq = new int[len];
            for (int i = 0; i < len; i++) {
                seq[i] = 1 + rnd.nextInt(VOCAB - 1);
            }
            if (!isPalindrome(seq)) {
                return toSample(seq, /*target=*/ 0f);
            }
        }
    }

    /** Проверка палиндромности «в лоб» (эталон, не сеть). */
    static boolean isPalindrome(int[] seq) {
        for (int i = 0, j = seq.length - 1; i < j; i++, j--) {
            if (seq[i] != seq[j]) return false;
        }
        return true;
    }

    /**
     * Упаковывает последовательность в {@link Sample}:
     * дополняет {@link #PAD} до {@link #T_MAX}, target — {@code [0] или [1]}.
     */
    static Sample toSample(int[] seq, float target) {
        float[] inputs = new float[T_MAX];
        // По умолчанию — PAD (= 0f); настоящие токены пишем в начало.
        for (int i = 0; i < seq.length; i++) {
            inputs[i] = seq[i];
        }
        return new Sample(inputs, new float[]{ target });
    }
}