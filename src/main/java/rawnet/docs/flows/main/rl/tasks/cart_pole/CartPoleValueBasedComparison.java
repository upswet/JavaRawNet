package rawnet.docs.flows.main.rl.tasks.cart_pole;

import rawnet.code.net.TensorShape;
import rawnet.code.net.core.activate.LeakyRelu;
import rawnet.code.net.core.optimizator.AdamW;
import rawnet.code.net.layers.core.DenseLayer;
import rawnet.code.net.model.functional.Functional;
import rawnet.code.rl.Environment;
import rawnet.code.rl.training.DoubleDqnTrainer;
import rawnet.code.rl.training.DqnTrainer;
import rawnet.code.rl.training.QLearningTrainer;
import rawnet.code.rl.training.RlTraining;

import java.util.Locale;

/**
 * CartPole: tabular Q-learning vs DQN vs Double DQN.
 *
 * <h2>Ключевой pedagogical момент</h2>
 * <b>Q-learning (tabular) НЕ ПРИМЕНИМ к CartPole.</b> Наблюдение —
 * {@code float[4]} (непрерывное: {@code x, x', θ, θ'}), таблица
 * {@code Q[state][action]} требует <b>дискретного</b> индекса
 * состояния. {@link QLearningTrainer#stateIndex} бросает
 * {@code IllegalStateException("observation not one-hot")} —
 * это <b>честная</b> иллюстрация ограничения: tabular работает только
 * на средах с конечным числом состояний.
 *
 * <p><b>DQN решает</b>: нейросеть {@code Q(s,a; θ)} принимает
 * {@code float[4]} на вход — <b>любое</b> непрерывное наблюдение.
 * Это <b>почему</b> DQN нужен: обобщение на непрерывные состояния.
 *
 * <h2>Гиперпараметры</h2>
 * Подобраны так, чтобы DQN сходился на CartPole:
 * <ul>
 *   <li>{@code LR = 1e-3} — стандарт для CartPole;</li>
 *   <li>{@code MAX_EPISODES = 3000} — DQN сходится за 500–2000;</li>
 *   <li>{@code EPS_DECAY = 5000} — ε падает за ~200 эпизодов;</li>
 *   <li>{@code WARMUP = 100} — обучение с 4-го эпизода;</li>
 *   <li>{@code SYNC = 500} — target стабилен ~20 эпизодов;</li>
 *   <li>{@code HIDDEN = 256}, {@code BATCH = 128} — больше ёмкости;</li>
 *   <li>{@code PATIENCE = 500} — дать шанс «пересидеть» плато.</li>
 * </ul>
 *
 * <h2>Что показывает запуск</h2>
 * <ul>
 *   <li>Q-learning: <b>падает</b> с {@code IllegalStateException}
 *       (наблюдение не one-hot).</li>
 *   <li>DQN: <b>сходится</b> к {@code avgReward ≈ 200–500}
 *       (500 — максимум) за ~300–1000 эпизодов.</li>
 *   <li>Double DQN: похожий результат; {@code πLoss} меньше
 *       (меньше overestimation), но на 2 действиях разница
 *       в пределах шума.</li>
 * </ul>
 */
public final class CartPoleValueBasedComparison {

    private static final int OBS_SIZE = 4;
    private static final int NUM_ACTIONS = 2;

    private static final float GAMMA = 0.99f;
    private static final long[] SEEDS = {42L, 43L, 44L};

    // Q-learning — параметры есть, но запуск упадёт (см. javadoc).
    private static final float QL_ALPHA = 0.1f;
    private static final float QL_EPS_START = 1.0f;
    private static final float QL_EPS_END = 0.01f;
    private static final int QL_EPS_DECAY_STEPS = 5_000;
    private static final int QL_MAX_EPISODES = 500;

    // DQN / Double DQN — общие гиперпараметры
    private static final int HIDDEN = 256;                   // было 128
    private static final float DQN_LR = 1e-3f;               // было 3e-4
    private static final float DQN_EPS_START = 1.0f;
    private static final float DQN_EPS_END = 0.01f;
    private static final int DQN_EPS_DECAY_STEPS = 5_000;    // было 10_000
    private static final int DQN_MAX_EPISODES = 3_000;       // было 1_000

    // DQN / Double DQN — параметры тренера
    private static final int DQN_BUFFER_SIZE = 100_000;      // было 50_000
    private static final int DQN_BATCH_SIZE = 128;           // было 64
    private static final int DQN_WARMUP_STEPS = 500;         // было 200
    private static final int DQN_SYNC_STEPS = 500;           // было 200
    private static final float DQN_HUBER_DELTA = 1.0f;

    // RlTraining для DQN / Double DQN
    private static final int DQN_WINDOW = 100;               // было 50
    private static final int DQN_PATIENCE = 500;             // было 200
    private static final float DQN_MIN_DELTA = 10.0f;        // без изменений
    private static final int DQN_LOG_EVERY = 100;

    public static void main(String[] args) {
        System.out.println("=== CartPole: Q-learning (tabular) vs DQN vs Double DQN ===\n");

        // Q-learning — демонстративно падает.
        System.out.println("--- Q-learning (tabular) ---");
        try {
            runQLearning(SEEDS[0]);
            System.out.println("  (неожиданно: Q-learning не упал)");
        } catch (IllegalStateException | IllegalArgumentException e) {
            System.out.println("  ОЖИДАЕМО упал: " + e.getMessage());
            System.out.println("  → tabular Q-learning не применим к непрерывному наблюдению.");
            System.out.println("  → нужен DQN (нейросеть принимает float[4]).\n");
        }

        // DQN.
        System.out.println("--- DQN ---");
        RunResult[] dqn = new RunResult[SEEDS.length];
        for (int i = 0; i < SEEDS.length; i++) dqn[i] = runDqn(SEEDS[i]);

        // Double DQN.
        System.out.println("\n--- Double DQN ---");
        RunResult[] ddqn = new RunResult[SEEDS.length];
        for (int i = 0; i < SEEDS.length; i++) ddqn[i] = runDoubleDqn(SEEDS[i]);

        System.out.println();
        printTable("DQN", dqn);
        System.out.println();
        printTable("Double DQN", ddqn);

        System.out.println();
        System.out.println("=== Итог ===");
        System.out.println("Q-learning (tabular): НЕ ПРИМЕНИМ (непрерывное наблюдение).");
        System.out.printf(Locale.US, "DQN:        средняя награда %.2f±%.2f%n",
                averageReward(dqn), stdReward(dqn));
        System.out.printf(Locale.US, "Double DQN: средняя награда %.2f±%.2f%n",
                averageReward(ddqn), stdReward(ddqn));
    }

    // ==================== Q-learning (tabular) — упадёт ====================

    private static RunResult runQLearning(long seed) {
        Environment env = new CartPole();
        // Конструктор пройдёт: observationSize=4 ≤ MAX_STATES=1024.
        // Но stateIndex() упадёт на первом же шаге:
        // наблюдение CartPole = [x, x', θ, θ'] — не one-hot.
        QLearningTrainer trainer = new QLearningTrainer(
                env, QL_ALPHA, GAMMA,
                QL_EPS_START, QL_EPS_END, QL_EPS_DECAY_STEPS, seed);

        RlTraining training = new RlTraining(
                trainer, QL_MAX_EPISODES, 100, 100, 0.05f, 200);
        float bestAvg = training.train();
        return new RunResult(seed, bestAvg, QL_MAX_EPISODES);
    }

    // ==================== DQN ====================

    private static RunResult runDqn(long seed) {
        Environment env = new CartPole();
        Functional qNet = buildQNet();
        Functional targetNet = buildQNet();      // веса могут быть любыми —
        // DqnTrainer перезапишет их в конструкторе
        AdamW opt = new AdamW(DQN_LR);

        DqnTrainer trainer = new DqnTrainer(
                env, qNet, targetNet, opt, GAMMA,
                DQN_EPS_START, DQN_EPS_END, DQN_EPS_DECAY_STEPS, seed,
                DQN_BUFFER_SIZE, DQN_BATCH_SIZE,
                DQN_WARMUP_STEPS, DQN_SYNC_STEPS,
                DQN_HUBER_DELTA);

        RlTraining training = new RlTraining(
                trainer, DQN_MAX_EPISODES,
                DQN_WINDOW, DQN_PATIENCE, DQN_MIN_DELTA, DQN_LOG_EVERY);
        float bestAvg = training.train();
        return new RunResult(seed, bestAvg, DQN_MAX_EPISODES);
    }

    // ==================== Double DQN ====================

    private static RunResult runDoubleDqn(long seed) {
        Environment env = new CartPole();
        Functional qNet = buildQNet();
        Functional targetNet = buildQNet();
        AdamW opt = new AdamW(DQN_LR);

        DoubleDqnTrainer trainer = new DoubleDqnTrainer(
                env, qNet, targetNet, opt, GAMMA,
                DQN_EPS_START, DQN_EPS_END, DQN_EPS_DECAY_STEPS, seed,
                DQN_BUFFER_SIZE, DQN_BATCH_SIZE,
                DQN_WARMUP_STEPS, DQN_SYNC_STEPS,
                DQN_HUBER_DELTA);

        RlTraining training = new RlTraining(
                trainer, DQN_MAX_EPISODES,
                DQN_WINDOW, DQN_PATIENCE, DQN_MIN_DELTA, DQN_LOG_EVERY);
        float bestAvg = training.train();
        return new RunResult(seed, bestAvg, DQN_MAX_EPISODES);
    }

    // ==================== Общие хелперы ====================

    private static Functional buildQNet() {
        Functional.Builder b = Functional.builder(TensorShape.flat(OBS_SIZE));
        Functional.Node in = b.input();
        Functional.Node h = b.layer(new DenseLayer.Dto(TensorShape.flat(HIDDEN)), in);
        Functional.Node act = b.activation(h, new LeakyRelu());
        Functional.Node out = b.layer(new DenseLayer.Dto(TensorShape.flat(NUM_ACTIONS)), act);
        return b.output(out);
    }

    // ==================== Статистика ====================

    private static float averageReward(RunResult[] results) {
        float sum = 0f;
        for (RunResult r : results) sum += r.avgReward();
        return sum / results.length;
    }

    private static float stdReward(RunResult[] results) {
        float avg = averageReward(results);
        float sum = 0f;
        for (RunResult r : results) {
            float d = r.avgReward() - avg;
            sum += d * d;
        }
        return (float) Math.sqrt(sum / results.length);
    }

    // ==================== Таблица ====================

    private record RunResult(long seed, float avgReward, int iterations) {}

    private static void printTable(String name, RunResult[] results) {
        System.out.printf(Locale.US, "=== %s ===%n", name);
        System.out.printf(Locale.US, "%-10s | %-14s | %-12s%n",
                "seed", "avgReward", "iterations");
        System.out.println("-----------|----------------|-------------");
        for (RunResult r : results) {
            System.out.printf(Locale.US, "%-10d | %-14.2f | %-12d%n",
                    r.seed(), r.avgReward(), r.iterations());
        }
    }
}