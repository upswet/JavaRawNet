package rawnet.code.rl.training;

import rawnet.code.net.core.optimizator.Optimizer;
import rawnet.code.net.model.functional.Functional;
import rawnet.code.rl.Categorical;
import rawnet.code.rl.Environment;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * PPO (Proximal Policy Optimization) с GAE — упрощённая версия.
 *
 * <h2>Отличия от Actor-Critic</h2>
 * <ul>
 *   <li><b>GAE</b>: advantage = экспоненциально сглаженная сумма
 *       TD-ошибок: {@code A_t = δ_t + (γλ)·δ_{t+1} + ...}, где
 *       {@code δ_t = r_t + γ·V(s_{t+1}) − V(s_t)}. Снижает дисперсию
 *       сильнее, чем Monte-Carlo {@code G_t − V(s_t)}.</li>
 *   <li><b>Clip</b>: ограничивает шаг обновления политики:
 *       {@code ratio = π_new / π_old}, {@code clip(ratio, 1−ε, 1+ε)}.
 *       Не даёт политике коллапсировать в локальный оптимум.</li>
 *   <li><b>Несколько эпох на rollout</b>: собираем N шагов, потом
 *       N_EPOCHS раз проходим мини-батчами. Больше обновлений на одни
 *       данные — быстрее сходится.</li>
 * </ul>
 *
 * <h2>Один rollout = одна итерация</h2>
 * Собираем {@code rolloutSteps} переходов, обновляем actor (PPO clip)
 * и critic (MSE), заполняем {@link #lastMetrics}.
 *
 * <h2>Метрики</h2>
 * После итерации: средняя награда за rollout, средняя длина эпизода,
 * policy loss (clipped objective), value loss (MSE), энтропия.
 */
public final class PpoTrainer implements RlTrainer {

    private final Environment env;
    private final Functional actor;
    private final Functional critic;
    private final Optimizer actorOpt;
    private final Optimizer criticOpt;

    private final float gamma;
    private final float gaeLambda;
    private final float clipEps;
    private final float entropyBonus;
    private final int rolloutSteps;
    private final int nEpochs;
    private final int miniBatchSize;
    private final Random rnd;

    /** Записанный переход rollout. */
    private record Step(float[] state, int action, float reward,
                        float value, boolean done) {}

    /** Метрики последней итерации; заполняются в {@link #trainIteration()}. */
    private RlMetrics lastMetrics = new RlMetrics(0f, 0f, 0f, 0f, 0f, 0);

    public PpoTrainer(Environment env,
                      Functional actor, Functional critic,
                      Optimizer actorOpt, Optimizer criticOpt,
                      float gamma, float gaeLambda, float clipEps,
                      float entropyBonus, int rolloutSteps, int nEpochs,
                      int miniBatchSize, long seed) {
        this.env = env;
        this.actor = actor;
        this.critic = critic;
        this.actorOpt = actorOpt;
        this.criticOpt = criticOpt;
        this.gamma = gamma;
        this.gaeLambda = gaeLambda;
        this.clipEps = clipEps;
        this.entropyBonus = entropyBonus;
        this.rolloutSteps = rolloutSteps;
        this.nEpochs = nEpochs;
        this.miniBatchSize = miniBatchSize;
        this.rnd = new Random(seed);
    }

    @Override
    public RlMetrics lastMetrics() { return lastMetrics; }

    /**
     * Один rollout: собрать {@code rolloutSteps} переходов, посчитать
     * GAE и returns, обновить actor (PPO clip) и critic (MSE),
     * заполнить {@link #lastMetrics}.
     */
    @Override
    public void trainIteration() {
        // ==================== 1. Собираем rollout ====================
        List<Step> rollout = new ArrayList<>(rolloutSteps);
        float[] s = env.reset();
        boolean done = true;
        float totalReward = 0f;
        int episodes = 0;

        for (int t = 0; t < rolloutSteps; t++) {
            if (done) {
                s = env.reset();
                episodes++;
            }

            float[] logits = actor.forward(s, true);
            int a = Categorical.sample(logits, rnd);
            float value = critic.forward(s, true)[0];

            Environment.StepResult step = env.step(a);
            rollout.add(new Step(s.clone(), a, step.reward(), value, step.done()));
            totalReward += step.reward();

            s = step.observation();
            done = step.done();
        }

        // ==================== 2. GAE и returns ====================
        int n = rollout.size();
        float[] advantages = new float[n];
        float[] returns = new float[n];

        float lastValue = 0f;
        if (!rollout.get(n - 1).done()) {
            lastValue = critic.forward(s, false)[0];
        }

        float gae = 0f;
        for (int t = n - 1; t >= 0; t--) {
            Step st = rollout.get(t);
            float nextValue = st.done() ? 0f : (t == n - 1 ? lastValue : rollout.get(t + 1).value());
            float delta = st.reward() + gamma * nextValue - st.value();
            gae = delta + gamma * gaeLambda * gae;
            advantages[t] = gae;
            returns[t] = gae + st.value();
        }

        // Нормализация advantage (стандарт PPO).
        float mean = 0f;
        for (float a : advantages) mean += a;
        mean /= n;
        float std = 0f;
        for (float a : advantages) std += (a - mean) * (a - mean);
        std = (float) Math.sqrt(std / n) + 1e-8f;
        for (int t = 0; t < n; t++) advantages[t] = (advantages[t] - mean) / std;

        // ==================== 3. Старые logits для clip ====================
        // Восстанавливаем old logits детерминированно (training = false).
        float[][] oldLogits = new float[n][];
        for (int t = 0; t < n; t++) {
            oldLogits[t] = actor.forward(rollout.get(t).state(), false).clone();
        }

        List<Integer> indices = new ArrayList<>(n);
        for (int i = 0; i < n; i++) indices.add(i);

        // ==================== 4. Actor update + метрики ====================
        float policyLossSum = 0f;
        float entropySum = 0f;
        int policyCount = 0;

        for (int epoch = 0; epoch < nEpochs; epoch++) {
            Collections.shuffle(indices, rnd);
            for (int start = 0; start < n; start += miniBatchSize) {
                int end = Math.min(start + miniBatchSize, n);
                int batch = end - start;

                for (int k = start; k < end; k++) {
                    int t = indices.get(k);
                    Step st = rollout.get(t);

                    // ---- Actor: PPO clip ----
                    float[] newLogits = actor.forward(st.state(), true);
                    float[] pNew = Categorical.softmax(newLogits);
                    float[] pOld = Categorical.softmax(oldLogits[t]);

                    int a = st.action();
                    float logpNew = (float) Math.log(pNew[a] + 1e-10f);
                    float logpOld = (float) Math.log(pOld[a] + 1e-10f);
                    float ratio = (float) Math.exp(logpNew - logpOld);
                    float adv = advantages[t];

                    // Метрики: clipped policy loss + энтропия.
                    float clipped = Math.max(1f - clipEps, Math.min(1f + clipEps, ratio));
                    policyLossSum += -(float) Math.min(ratio * adv, clipped * adv);
                    float ent = 0f;
                    for (float pi : pNew) ent -= pi * (float) Math.log(pi + 1e-10f);
                    entropySum += ent;
                    policyCount++;

                    // Градиент: 0, если clip сработал; иначе -adv·ratio.
                    float dRatio;
                    if ((adv > 0 && ratio > 1f + clipEps) || (adv < 0 && ratio < 1f - clipEps)) {
                        dRatio = 0f;
                    } else {
                        dRatio = -adv * ratio;
                    }

                    float[] dLogits = new float[newLogits.length];
                    for (int i = 0; i < dLogits.length; i++) {
                        float onehot = (i == a) ? 1f : 0f;
                        dLogits[i] = dRatio * (onehot - pNew[i]);
                    }

                    if (entropyBonus > 0f) {
                        float sumPlogPPlus1 = 0f;
                        float[] dEntropy = new float[dLogits.length];
                        for (int i = 0; i < dLogits.length; i++) {
                            float logP = (float) Math.log(pNew[i] + 1e-10f);
                            dEntropy[i] = -pNew[i] * (logP + 1f);
                            sumPlogPPlus1 += pNew[i] * (logP + 1f);
                        }
                        for (int i = 0; i < dLogits.length; i++) {
                            dLogits[i] += entropyBonus * (dEntropy[i] - pNew[i] * sumPlogPPlus1);
                        }
                    }

                    actor.backward(dLogits);
                }
                actor.update(actorOpt, batch);
            }
        }

        // ==================== 5. Critic update + метрики ====================
        float valueLossSum = 0f;
        int valueCount = 0;

        for (int epoch = 0; epoch < nEpochs; epoch++) {
            Collections.shuffle(indices, rnd);
            for (int start = 0; start < n; start += miniBatchSize) {
                int end = Math.min(start + miniBatchSize, n);
                int batch = end - start;

                for (int k = start; k < end; k++) {
                    int t = indices.get(k);
                    Step st = rollout.get(t);
                    float v = critic.forward(st.state(), true)[0];
                    float dV = v - returns[t];
                    valueLossSum += dV * dV;
                    valueCount++;
                    critic.backward(new float[]{ dV });
                }
                critic.update(criticOpt, batch);
            }
        }

        // ==================== 6. Метрики итерации ====================
        float policyLoss = policyCount > 0 ? policyLossSum / policyCount : 0f;
        float entropy = policyCount > 0 ? entropySum / policyCount : 0f;
        float valueLoss = valueCount > 0 ? valueLossSum / valueCount : 0f;

        lastMetrics = new RlMetrics(
                totalReward,                // суммарная награда за rollout
                episodes > 0 ? (float) n / episodes : 0f,  // средняя длина эпизода
                policyLoss,
                valueLoss,
                entropy,
                episodes);
    }
}