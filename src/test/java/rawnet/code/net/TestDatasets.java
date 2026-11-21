package rawnet.code.net;

import rawnet.code.dataset.SampleDataset;
import rawnet.code.dataset.Sample;
import rawnet.code.net.core.loss.BinaryCrossEntropy;
import rawnet.code.net.core.metric.Accuracy;
import rawnet.code.net.core.metric.LossMetric;
import rawnet.code.net.core.metric.Metric;
import rawnet.code.net.core.optimizator.AdamW;
import rawnet.code.net.model.Model;
import rawnet.code.net.model.ModelTraining;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Общие хелперы для «учебных» тестов: маленькие датасеты и быстрое обучение.
 *
 * <p>Гиперпараметры подобраны эмпирически так, чтобы XOR и мини-палиндром
 * сходились за 15–20 эпох на 200–300 примерах. Меньше эпох / меньше lr —
 * сеть не успевает (это не баг библиотеки, а бюджет теста).
 *
 * <p><b>Новый API.</b> {@link ModelTraining} — объект-тренер с сеттерами
 * (chain) и {@code train()} без аргументов. {@code trainSmall} и
 * {@code trainSmallBestAcc} создают тренер, настраивают и вызывают
 * {@code train()}.
 */
public final class TestDatasets {

    private TestDatasets() {}

    // =====================================================================
    // ==================== Маленькое обучение =============================
    // =====================================================================

    /** Дефолтный lr для «учебных» тестов: 1e-2. */
    public static final float DEFAULT_LR = 1e-2f;

    /** Дефолтное число эпох для «учебных» тестов: 20. */
    public static final int DEFAULT_EPOCHS = 20;

    /** Эквивалентно {@code trainSmall(net, ds, DEFAULT_EPOCHS, DEFAULT_LR)}. */
    public static List<Metric> trainSmall(Model net, SampleDataset<Sample> ds) {
        return trainSmall(net, ds, DEFAULT_EPOCHS, DEFAULT_LR);
    }

    /**
     * Маленькое обучение: AdamW(lr), BCE, батч 16, глобальный клип 1.0.
     *
     * <p>Возвращает метрики после обучения: {@code [0]} — loss,
     * {@code [1]} — accuracy. Тесты проверяют {@code [1]}.
     *
     * @param net    сеть (свежая)
     * @param ds     датасет (train/test)
     * @param epochs число эпох
     * @param lr     learning rate
     */
    public static List<Metric> trainSmall(Model net, SampleDataset<Sample> ds, int epochs, float lr) {
        var loss = new BinaryCrossEntropy();
        var opt = new AdamW(lr);
        List<Metric> metrics = List.of(new LossMetric(loss), new Accuracy());

        new ModelTraining<Model, Sample>()
                .setModel(net)
                .setDataset(ds)
                .setLoss(loss)
                .setOptimizer(opt)
                .setMaxEpoch(epochs)
                .setBatchSize(16)
                .setMetrics(metrics)
                .setEndPredicate(null)
                .setClipNorm(1.0f)
                .train();

        return metrics;
    }

    /**
     * Обучает и возвращает ЛУЧШУЮ accuracy за все эпохи (защита от переобучения).
     */
    public static float trainSmallBestAcc(Model net, SampleDataset<Sample> ds, int epochs, float lr) {
        var loss = new BinaryCrossEntropy();
        var opt = new AdamW(lr);
        var acc = new Accuracy();
        List<Metric> metrics = List.of(new LossMetric(loss), acc);

        float[] best = {0f};

        new ModelTraining<Model, Sample>()
                .setModel(net)
                .setDataset(ds)
                .setLoss(loss)
                .setOptimizer(opt)
                .setMaxEpoch(epochs)
                .setBatchSize(16)
                .setMetrics(metrics)
                .setEndPredicate((model, m) -> {
                    best[0] = Math.max(best[0], acc.result());
                    return false;
                })
                .setClipNorm(1.0f)
                .train();

        return best[0];
    }

    // =====================================================================
    // ==================== XOR ============================================
    // =====================================================================

    /** XOR с шумом ±0.05, разбиение 80/20. */
    public static SampleDataset<Sample> xorDataset(int n, Random rnd) {
        List<Sample> all = new ArrayList<>(n);
        float[][] xs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        float[] ys = {0, 1, 1, 0};
        for (int i = 0; i < n; i++) {
            int k = i % 4;
            float[] x = xs[k].clone();
            x[0] += (rnd.nextFloat() - 0.5f) * 0.1f;
            x[1] += (rnd.nextFloat() - 0.5f) * 0.1f;
            all.add(new Sample(x, new float[]{ys[k]}));
        }
        int split = (int) (n * 0.8);
        return new SampleDataset<Sample>(all.subList(0, split), all.subList(split, n));
    }

    // =====================================================================
    // ==================== Палиндром ======================================
    // =====================================================================

    /** Мини-палиндром: длины 4–8, PAD=0, алфавит 1..5, T=8, разбиение 80/20. */
    public static SampleDataset<Sample> palindromeDataset(int n, Random rnd) {
        List<Sample> all = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            boolean wantPal = (i % 2 == 0);
            int len = 4 + rnd.nextInt(5);
            int[] seq = new int[len];
            if (wantPal) {
                int half = len / 2;
                for (int j = 0; j < half; j++) {
                    seq[j] = 1 + rnd.nextInt(5);
                    seq[len - 1 - j] = seq[j];
                }
                if (len % 2 == 1) seq[half] = 1 + rnd.nextInt(5);
            } else {
                do {
                    for (int j = 0; j < len; j++) seq[j] = 1 + rnd.nextInt(5);
                } while (isPalindrome(seq));
            }
            float[] x = new float[8];
            for (int j = 0; j < len; j++) x[j] = seq[j];
            all.add(new Sample(x, new float[]{wantPal ? 1f : 0f}));
        }
        int split = (int) (n * 0.8);
        return new SampleDataset<Sample>(all.subList(0, split), all.subList(split, n));
    }

    public static boolean isPalindrome(int[] s) {
        for (int i = 0, j = s.length - 1; i < j; i++, j--)
            if (s[i] != s[j]) return false;
        return true;
    }
}