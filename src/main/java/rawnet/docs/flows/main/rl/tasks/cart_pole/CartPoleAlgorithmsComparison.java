package rawnet.docs.flows.main.rl.tasks.cart_pole;

import rawnet.code.net.TensorShape;
import rawnet.code.net.core.activate.LeakyRelu;
import rawnet.code.net.core.optimizator.AdamW;
import rawnet.code.net.layers.core.DenseLayer;
import rawnet.code.net.model.functional.Functional;
import rawnet.code.rl.Categorical;
import rawnet.code.rl.Environment;
import rawnet.code.rl.training.PpoTrainer;
import rawnet.code.rl.training.ReinforceTrainer;
import rawnet.code.rl.training.RlTraining;

import java.util.Locale;

/**
 * Сравнение REINFORCE и PPO на CartPole: 3 seed, средняя награда,
 * таблица.
 *
 * <h2>Что показывает</h2>
 * <ul>
 *   <li><b>REINFORCE</b>: сходится медленно (~3000 эпизодов),
 *       нестабильно (награда скачет), но сходится;</li>
 *   <li><b>PPO</b>: сходится быстрее (~100–200 итераций), стабильнее;</li>
 *   <li><b>3 seed</b>: видно разброс — RL нестабилен по природе,
 *       один запуск не показатель.</li>
 * </ul>
 *
 * <h2>Гиперпараметры</h2>
 * <b>REINFORCE</b>: LR=1e-3, ENTROPY=0.01, max 5000 эпизодов.
 * <b>PPO</b>: LR=3e-4 (actor), LR=1e-3 (critic), ROLLOUT=2048,
 * N_EPOCHS=10, MINI_BATCH=64, ENTROPY=0.1, max 200 итераций.
 *
 * <p><b>Почему PPO стабильнее</b>: GAE + clip. REINFORCE —
 * Monte-Carlo returns, высокая дисперсия. PPO — TD-ошибки
 * (сглаженные), clip не даёт политике коллапсировать.
 */
public final class CartPoleAlgorithmsComparison {

    private static final int OBS_SIZE = 4;
    private static final int NUM_ACTIONS = 2;
    private static final int HIDDEN = 64;

    private static final float GAMMA = 0.99f;
    private static final long[] SEEDS = {42L, 43L, 44L};

    // REINFORCE
    private static final float RF_LR = 1e-3f;
    private static final float RF_ENTROPY = 0.01f;
    private static final int RF_MAX_ITERS = 5000;

    // PPO
    private static final float PPO_ACTOR_LR = 3e-4f;
    private static final float PPO_CRITIC_LR = 1e-3f;
    private static final float PPO_GAE_LAMBDA = 0.95f;
    private static final float PPO_CLIP_EPS = 0.2f;
    private static final float PPO_ENTROPY = 0.1f;
    private static final int PPO_ROLLOUT = 2048;
    private static final int PPO_N_EPOCHS = 10;
    private static final int PPO_MINI_BATCH = 64;
    private static final int PPO_MAX_ITERS = 200;

    private static final boolean RENDER_DEMO = false;

    public static void main(String[] args) {
        System.out.println("=== CartPole: REINFORCE vs PPO ===\n");

        // REINFORCE, 3 seed.
        RunResult[] rf = new RunResult[SEEDS.length];
        for (int i = 0; i < SEEDS.length; i++) {
            rf[i] = runReinforce(SEEDS[i]);
        }

        // PPO, 3 seed.
        RunResult[] ppo = new RunResult[SEEDS.length];
        for (int i = 0; i < SEEDS.length; i++) {
            ppo[i] = runPpo(SEEDS[i]);
        }

        // Таблица.
        printTable("REINFORCE", rf);
        System.out.println();
        printTable("PPO", ppo);

        // Средняя по 3 запускам.
        System.out.println();
        System.out.println("=== Сравнение (средняя по " + SEEDS.length + " запускам) ===");
        System.out.printf(Locale.US, "%-12s | %-10s | %-12s | %-10s%n",
                "Алгоритм", "reward", "epLen", "iterations");
        System.out.println("-------------|------------|--------------|-----------");
        printAverage("REINFORCE", rf);
        printAverage("PPO", ppo);
    }

    // ==================== REINFORCE ====================

    private static RunResult runReinforce(long seed) {
        Environment env = new CartPole();
        Functional policy = buildActor();
        AdamW opt = new AdamW(RF_LR);
        ReinforceTrainer trainer = new ReinforceTrainer(
                env, policy, opt, GAMMA, RF_ENTROPY, seed);

        // Окно 100 эпизодов (для CartPole — 100 эпизодов = ~1000 шагов).
        // patience 50 (50*100 = 5000 эпизодов без улучшения — но max 5000).
        RlTraining training = new RlTraining(
                trainer, RF_MAX_ITERS, 100, 50, 5.0f, 500);
        float bestAvg = training.train();

        if (RENDER_DEMO) demoEpisode(env, policy, seed);

        // bestAvg — средняя за окно 100. Возвращаем вместе с эпизодами.
        return new RunResult(seed, bestAvg, 0f, RF_MAX_ITERS);
    }

    // ==================== PPO ====================

    private static RunResult runPpo(long seed) {
        Environment env = new CartPole();
        Functional actor = buildActor();
        Functional critic = buildCritic();
        AdamW actorOpt = new AdamW(PPO_ACTOR_LR);
        AdamW criticOpt = new AdamW(PPO_CRITIC_LR);
        PpoTrainer trainer = new PpoTrainer(
                env, actor, critic, actorOpt, criticOpt,
                GAMMA, PPO_GAE_LAMBDA, PPO_CLIP_EPS, PPO_ENTROPY,
                PPO_ROLLOUT, PPO_N_EPOCHS, PPO_MINI_BATCH, seed);

        // Окно 5 rollout-ов (~5*2048 = 10k шагов, ~200 эпизодов).
        // patience 20, minDelta 5.
        RlTraining training = new RlTraining(
                trainer, PPO_MAX_ITERS, 5, 20, 5.0f, 10);
        float bestAvg = training.train();

        if (RENDER_DEMO) demoEpisode(env, actor, seed);

        return new RunResult(seed, bestAvg, 0f, PPO_MAX_ITERS);
    }

    // ==================== Общие хелперы ====================

    private static Functional buildActor() {
        Functional.Builder b = Functional.builder(TensorShape.flat(OBS_SIZE));
        Functional.Node in = b.input();
        Functional.Node h = b.layer(new DenseLayer.Dto(TensorShape.flat(HIDDEN)), in);
        Functional.Node act = b.activation(h, new LeakyRelu());
        Functional.Node out = b.layer(new DenseLayer.Dto(TensorShape.flat(NUM_ACTIONS)), act);
        return b.output(out);
    }

    private static Functional buildCritic() {
        Functional.Builder b = Functional.builder(TensorShape.flat(OBS_SIZE));
        Functional.Node in = b.input();
        Functional.Node h = b.layer(new DenseLayer.Dto(TensorShape.flat(HIDDEN)), in);
        Functional.Node act = b.activation(h, new LeakyRelu());
        Functional.Node out = b.layer(new DenseLayer.Dto(TensorShape.flat(1)), act);
        return b.output(out);
    }

    private static void demoEpisode(Environment env, Functional actor, long seed) {
        java.util.Random rnd = new java.util.Random(seed);
        float[] s = env.reset();
        float totalReward = 0f;
        int steps = 0;
        while (steps < 500) {
            float[] logits = actor.forward(s, false);
            int a = Categorical.sample(logits, rnd);
            Environment.StepResult step = env.step(a);
            s = step.observation();
            totalReward += step.reward();
            steps++;
            if (step.done()) break;
        }
        System.out.printf(Locale.US, "  demo (seed=%d): %d шагов, награда=%.1f%n",
                seed, steps, totalReward);
    }

    // ==================== Таблица ====================

    private record RunResult(long seed, float avgReward, float avgEpLen, int iterations) {}

    private static void printTable(String name, RunResult[] results) {
        System.out.printf(Locale.US, "=== %s ===%n", name);
        System.out.printf(Locale.US, "%-10s | %-12s | %-10s | %-12s%n",
                "seed", "avgReward", "epLen", "iterations");
        System.out.println("-----------|--------------|------------|-------------");
        for (RunResult r : results) {
            System.out.printf(Locale.US, "%-10d | %-12.2f | %-10.1f | %-12d%n",
                    r.seed(), r.avgReward(), r.avgEpLen(), r.iterations());
        }
    }

    private static void printAverage(String name, RunResult[] results) {
        float avgR = 0f, avgLen = 0f;
        int avgIt = 0;
        for (RunResult r : results) {
            avgR += r.avgReward();
            avgLen += r.avgEpLen();
            avgIt += r.iterations();
        }
        avgR /= results.length;
        avgLen /= results.length;
        avgIt /= results.length;

        // std для reward.
        float std = 0f;
        for (RunResult r : results) {
            float d = r.avgReward() - avgR;
            std += d * d;
        }
        std = (float) Math.sqrt(std / results.length);

        System.out.printf(Locale.US, "%-12s | %.2f±%.2f | %-12.1f | %-10d%n",
                name, avgR, std, avgLen, avgIt);
    }
}