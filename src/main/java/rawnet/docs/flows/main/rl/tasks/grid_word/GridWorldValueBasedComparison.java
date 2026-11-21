package rawnet.docs.flows.main.rl.tasks.grid_word;

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
 * GridWorld 5×5: tabular Q-learning vs DQN vs Double DQN.
 *
 * <h2>Что сравниваем</h2>
 * <ul>
 *   <li><b>Q-learning (tabular)</b>: таблица {@code Q[25][4]},
 *       update по Беллману. Точечное обновление без обобщения —
 *       для 25 состояний это точно.</li>
 *   <li><b>DQN</b>: нейросеть {@code Q(s,a; θ)}, replay buffer,
 *       target network, Huber loss. Обобщает через сеть — но one-hot
 *       входы <b>не</b> кодируют соседство клеток, и обобщение
 *       между {@code s=0} и {@code s=1} случайно.</li>
 *   <li><b>Double DQN</b>: DQN с разделением выбора и оценки target
 *       ({@code a* = argmax Q_online}, {@code target = Q_target(a*)}).
 *       Меньше overestimation bias — Q-значения стабильнее.</li>
 * </ul>
 *
 * <h2>Почему GridWorld подходит для всех трёх</h2>
 * Наблюдение — one-hot 25 чисел. Для Q-learning это индекс состояния
 * (первый {@code 1}). Для DQN/Double DQN — вход нейросети. Одна и
 * та же среда, три подхода.
 *
 * <h2>Ожидаемый результат</h2>
 * <ul>
 *   <li><b>Q-learning (tabular)</b>: доходит до цели на части seed'ов
 *       (точечное обновление).</li>
 *   <li><b>DQN</b>: часто застревает в локальном оптимуме «бродить,
 *       не падая в яму» ({@code epLen → 50}, {@code reward → 0}).
 *       Это не баг, а свойство DQN на one-hot наблюдениях.</li>
 *   <li><b>Double DQN</b>: стабильнее, Q-значения не уезжают вверх,
 *       но на GridWorld 5×5 разница с DQN может быть небольшой —
 *       overestimation не катастрофический при 4 действиях.</li>
 * </ul>
 */
public final class GridWorldValueBasedComparison {

    private static final int OBS_SIZE = GridWorld.OBS_SIZE;      // 25
    private static final int NUM_ACTIONS = GridWorld.NUM_ACTIONS; // 4
    private static final int HIDDEN = 128;

    private static final float GAMMA = 0.95f;
    private static final long[] SEEDS = {42L, 43L, 44L};

    // Q-learning (tabular)
    private static final float QL_ALPHA = 0.1f;
    private static final float QL_EPS_START = 1.0f;
    private static final float QL_EPS_END = 0.01f;
    private static final int QL_EPS_DECAY_STEPS = 5_000;
    private static final int QL_MAX_EPISODES = 2_000;

    // DQN / Double DQN — общие гиперпараметры
    private static final float DQN_LR = 3e-4f;
    private static final float DQN_EPS_START = 1.0f;
    private static final float DQN_EPS_END = 0.01f;
    private static final int DQN_EPS_DECAY_STEPS = 2_000;
    private static final int DQN_MAX_EPISODES = 2_000;

    // DQN / Double DQN — параметры тренера
    private static final int DQN_BUFFER_SIZE = 20_000;
    private static final int DQN_BATCH_SIZE = 64;
    private static final int DQN_WARMUP_STEPS = 64;      // минимум = batchSize
    private static final int DQN_SYNC_STEPS = 200;       // чаще, чем 1000
    private static final float DQN_HUBER_DELTA = 1.0f;

    // RlTraining для DQN / Double DQN
    private static final int DQN_WINDOW = 100;
    private static final int DQN_PATIENCE = 500;         // не 50
    private static final float DQN_MIN_DELTA = 0.05f;    // не 10.0f
    private static final int DQN_LOG_EVERY = 200;

    public static void main(String[] args) {
        System.out.println("=== GridWorld 5×5: Q-learning (tabular) vs DQN vs Double DQN ===\n");

        RunResult[] ql = new RunResult[SEEDS.length];
        for (int i = 0; i < SEEDS.length; i++) ql[i] = runQLearning(SEEDS[i]);

        RunResult[] dqn = new RunResult[SEEDS.length];
        for (int i = 0; i < SEEDS.length; i++) dqn[i] = runDqn(SEEDS[i]);

        RunResult[] ddqn = new RunResult[SEEDS.length];
        for (int i = 0; i < SEEDS.length; i++) ddqn[i] = runDoubleDqn(SEEDS[i]);

        printTable("Q-learning (tabular)", ql);
        System.out.println();
        printTable("DQN", dqn);
        System.out.println();
        printTable("Double DQN", ddqn);

        System.out.println();
        System.out.println("=== Сравнение (средняя по " + SEEDS.length + " запускам) ===");
        System.out.printf(Locale.US, "%-22s | %-16s | %-12s%n",
                "Алгоритм", "reward", "iterations");
        System.out.println("----------------------|------------------|-------------");
        printAverage("Q-learning (tabular)", ql);
        printAverage("DQN", dqn);
        printAverage("Double DQN", ddqn);
    }

    // ==================== Q-learning (tabular) ====================

    private static RunResult runQLearning(long seed) {
        Environment env = new GridWorld(seed);
        QLearningTrainer trainer = new QLearningTrainer(
                env, QL_ALPHA, GAMMA,
                QL_EPS_START, QL_EPS_END, QL_EPS_DECAY_STEPS, seed);

        // Q-learning: minDelta = 0.05 (награда от -1 до +1),
        // patience = 500 (не 100 — дать шанс выучить).
        RlTraining training = new RlTraining(
                trainer, QL_MAX_EPISODES, 100, 500, 0.05f, 200);
        float bestAvg = training.train();
        return new RunResult(seed, bestAvg, QL_MAX_EPISODES);
    }

    // ==================== DQN ====================

    private static RunResult runDqn(long seed) {
        Environment env = new GridWorld(seed);
        Functional qNet = buildQNet();
        Functional targetNet = buildQNet();
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
        Environment env = new GridWorld(seed);
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

    // ==================== Таблица ====================

    private record RunResult(long seed, float avgReward, int iterations) {}

    private static void printTable(String name, RunResult[] results) {
        System.out.printf(Locale.US, "=== %s ===%n", name);
        System.out.printf(Locale.US, "%-10s | %-14s | %-12s%n",
                "seed", "avgReward", "iterations");
        System.out.println("-----------|----------------|-------------");
        for (RunResult r : results) {
            System.out.printf(Locale.US, "%-10d | %-14.4f | %-12d%n",
                    r.seed(), r.avgReward(), r.iterations());
        }
    }

    private static void printAverage(String name, RunResult[] results) {
        float avgR = 0f;
        int avgIt = 0;
        for (RunResult r : results) {
            avgR += r.avgReward();
            avgIt += r.iterations();
        }
        avgR /= results.length;
        avgIt /= results.length;

        float std = 0f;
        for (RunResult r : results) {
            float d = r.avgReward() - avgR;
            std += d * d;
        }
        std = (float) Math.sqrt(std / results.length);

        System.out.printf(Locale.US, "%-22s | %.4f±%.4f | %-12d%n",
                name, avgR, std, avgIt);
    }
}