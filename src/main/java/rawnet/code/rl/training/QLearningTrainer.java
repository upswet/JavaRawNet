package rawnet.code.rl.training;

import rawnet.code.rl.Environment;

import java.util.Random;

/**
 * Tabular Q-learning для маленьких дискретных сред.
 *
 * <h2>Идея</h2>
 * Таблица {@code Q[s][a]} — ценность действия {@code a} в состоянии
 * {@code s}. Политика — {@code argmax_a Q(s,a)} (с ε-greedy для
 * исследования). Update по Беллману:
 * {@code Q(s,a) += α·(r + γ·max Q(s',a') − Q(s,a))}.
 *
 * <h2>Ограничения</h2>
 * Только для сред с {@code observationSize()} ≤ {@link #MAX_STATES}
 * (иначе таблица не влезет в память). Для CartPole (непрерывное
 * наблюдение) нужен DQN — см. главу 14.3.
 */
public final class QLearningTrainer implements RlTrainer {

    private static final int MAX_STATES = 1024;

    private final Environment env;
    private final float alpha;
    private final float gamma;
    private final float epsilonStart;
    private final float epsilonEnd;
    private final int epsilonDecaySteps;
    private final Random rnd;

    /** Q-таблица: [state][action]. */
    private final float[][] q;
    private int steps;

    private RlMetrics lastMetrics = new RlMetrics(0f, 0f, 0f, Float.NaN, 0f, 0);

    public QLearningTrainer(Environment env, float alpha, float gamma,
                            float epsilonStart, float epsilonEnd,
                            int epsilonDecaySteps, long seed) {
        if (env.observationSize() > MAX_STATES) {
            throw new IllegalArgumentException(
                    "observationSize=" + env.observationSize()
                            + " > MAX_STATES=" + MAX_STATES
                            + " — нужен DQN, не tabular Q-learning");
        }
        this.env = env;
        this.alpha = alpha;
        this.gamma = gamma;
        this.epsilonStart = epsilonStart;
        this.epsilonEnd = epsilonEnd;
        this.epsilonDecaySteps = epsilonDecaySteps;
        this.rnd = new Random(seed);
        this.q = new float[env.observationSize()][env.numActions()];
    }

    @Override
    public RlMetrics lastMetrics() { return lastMetrics; }

    @Override
    public void trainIteration() {
        // Один эпизод.
        float[] s = env.reset();
        int stateIdx = stateIndex(s);
        boolean done = false;
        float totalReward = 0f;
        int len = 0;
        float lossSum = 0f;
        int lossCount = 0;

        while (!done) {
            int a = chooseAction(stateIdx);
            Environment.StepResult step = env.step(a);
            int nextIdx = stateIndex(step.observation());
            float r = step.reward();

            // TD-target: r + γ·max Q(s',·), или r, если done.
            float target = r;
            if (!step.done()) {
                float maxNext = q[nextIdx][0];
                for (int i = 1; i < q[nextIdx].length; i++) {
                    if (q[nextIdx][i] > maxNext) maxNext = q[nextIdx][i];
                }
                target += gamma * maxNext;
            }

            // TD-ошибка и update.
            float tdError = target - q[stateIdx][a];
            q[stateIdx][a] += alpha * tdError;
            lossSum += tdError * tdError;
            lossCount++;

            totalReward += r;
            len++;
            s = step.observation();
            stateIdx = nextIdx;
            done = step.done();
            steps++;
        }

        float loss = lossCount > 0 ? lossSum / lossCount : 0f;
        lastMetrics = RlMetrics.reinforce(totalReward, len, loss, 0f, 1);
    }

    private int chooseAction(int stateIdx) {
        float eps = currentEpsilon();
        if (rnd.nextDouble() < eps) {
            return rnd.nextInt(env.numActions());
        }
        int best = 0;
        for (int i = 1; i < q[stateIdx].length; i++) {
            if (q[stateIdx][i] > q[stateIdx][best]) best = i;
        }
        return best;
    }

    private float currentEpsilon() {
        if (steps >= epsilonDecaySteps) return epsilonEnd;
        float t = (float) steps / epsilonDecaySteps;
        return epsilonStart + t * (epsilonEnd - epsilonStart);
    }

    /** One-hot наблюдение → индекс состояния. */
    private int stateIndex(float[] obs) {
        for (int i = 0; i < obs.length; i++) {
            if (obs[i] > 0.5f) return i;
        }
        throw new IllegalStateException("observation not one-hot");
    }
}