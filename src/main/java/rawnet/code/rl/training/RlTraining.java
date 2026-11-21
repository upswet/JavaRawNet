package rawnet.code.rl.training;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Locale;

/**
 * Обёртка над {@link RlTrainer}: цикл обучения, логи, early stopping.
 *
 * <h2>Early stopping</h2>
 * <ul>
 *   <li>Средняя награда считается по окну {@link #window} итераций.</li>
 *   <li>Если средняя не улучшилась на {@link #minDelta} относительно
 *       лучшей за {@link #patience} итераций подряд — останавливаемся.</li>
 *   <li>«Улучшение» — рост средней награды на {@code minDelta}
 *       (абсолютный, для CartPole {@code 10.0}, для GridWorld {@code 0.05}).</li>
 *   <li>Дополнительно: если {@code policyLoss}, {@code entropy} или
 *       {@code avgReward} стали {@code NaN}/{@code Inf} — останавливаемся
 *       (расходимость).</li>
 * </ul>
 *
 * <h2>Почему valueLoss == NaN — это НЕ расходимость</h2>
 * У REINFORCE нет critic, поэтому {@link RlMetrics#reinforce} кладёт
 * в поле {@code valueLoss} константу {@link Float#NaN} — это маркер
 * «не применимо». Если проверять {@code Float.isNaN(m.valueLoss())},
 * early stopping сработает на <b>первой же</b> итерации REINFORCE,
 * хотя никакой расходимости нет. Поэтому для {@code valueLoss}
 * проверяем только {@code isInfinite} (реальный NaN в loss critic —
 * это уже {@code Inf}, а не {@code NaN}, из-за {@code dV*dV}).
 *
 * <h2>Логи</h2>
 * Каждые {@link #logEvery} итераций — средняя награда, длина эпизода,
 * policy loss, value loss (если есть), энтропия.
 */
public final class RlTraining {

    private static final Logger log = LoggerFactory.getLogger(RlTraining.class);

    private final RlTrainer trainer;
    private final int maxIterations;
    private final int window;
    private final int patience;
    private final float minDelta;
    private final int logEvery;

    public RlTraining(RlTrainer trainer, int maxIterations,
                      int window, int patience, float minDelta, int logEvery) {
        this.trainer = trainer;
        this.maxIterations = maxIterations;
        this.window = Math.max(1, window);
        this.patience = patience;
        this.minDelta = minDelta;
        this.logEvery = Math.max(1, logEvery);
    }

    /** Упрощённый конструктор: window=50, patience=20, minDelta=0.05, logEvery=10. */
    public RlTraining(RlTrainer trainer, int maxIterations) {
        this(trainer, maxIterations, 50, 20, 0.05f, 10);
    }

    /**
     * Запускает цикл обучения до {@link #maxIterations} или early stop.
     *
     * @return лучшая средняя награда за всё обучение
     */
    public float train() {
        Deque<Float> rewardWindow = new ArrayDeque<>(window);
        float bestAvg = Float.NEGATIVE_INFINITY;
        int stale = 0;

        for (int it = 0; it < maxIterations; it++) {
            trainer.trainIteration();
            RlMetrics m = trainer.lastMetrics();

            // ==================== NaN-защита ====================
            // policyLoss, entropy, avgReward — обязательные поля, NaN/Inf
            // означает расходимость. valueLoss — опциональное (NaN =
            // «не применимо» для REINFORCE), проверяем только Inf.
            boolean diverged =
                    Float.isNaN(m.policyLoss())    || Float.isInfinite(m.policyLoss())
                            || Float.isNaN(m.entropy())       || Float.isInfinite(m.entropy())
                            || Float.isNaN(m.avgReward())     || Float.isInfinite(m.avgReward())
                            || Float.isInfinite(m.valueLoss());   // NaN здесь — норма

            if (diverged) {
                log.warn("[early stop] NaN/Inf в loss на итерации {}. "
                                + "Metrics: reward={}, epLen={}, πLoss={}, VLoss={}, H={}",
                        it + 1, m.avgReward(), m.avgEpisodeLength(),
                        m.policyLoss(), m.valueLoss(), m.entropy());
                break;
            }

            // Окно средней награды.
            rewardWindow.addLast(m.avgReward());
            if (rewardWindow.size() > window) rewardWindow.removeFirst();
            float avg = mean(rewardWindow);

            // Early stopping: сравнение с лучшей средней.
            if (avg > bestAvg + minDelta) {
                bestAvg = avg;
                stale = 0;
            } else {
                stale++;
                if (stale >= patience) {
                    log.info("[early stop] {} итераций без улучшения (best={}, current={})",
                            patience, fmt(bestAvg), fmt(avg));
                    break;
                }
            }

            // Логи.
            if ((it + 1) % logEvery == 0 || it == 0) {
                logMetrics(it + 1, m, avg, bestAvg);
            }
        }

        log.info("Обучение завершено. Лучшая средняя награда: {}", fmt(bestAvg));
        return bestAvg;
    }

    private static void logMetrics(int iter, RlMetrics m, float windowAvg, float bestAvg) {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(Locale.US,
                "итерация %4d | reward=%.2f | epLen=%.1f | avg=%.2f | best=%.2f",
                iter, m.avgReward(), m.avgEpisodeLength(), windowAvg, bestAvg));
        sb.append(String.format(Locale.US, " | πLoss=%.4f", m.policyLoss()));
        if (!Float.isNaN(m.valueLoss())) {
            sb.append(String.format(Locale.US, " | VLoss=%.4f", m.valueLoss()));
        }
        sb.append(String.format(Locale.US, " | H=%.3f", m.entropy()));
        log.info(sb.toString());
    }

    private static float mean(Deque<Float> d) {
        if (d.isEmpty()) return 0f;
        float s = 0f;
        for (float v : d) s += v;
        return s / d.size();
    }

    private static String fmt(float v) {
        return String.format(Locale.US, "%.2f", v);
    }
}