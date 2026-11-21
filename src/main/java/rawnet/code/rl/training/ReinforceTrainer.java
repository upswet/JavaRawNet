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
 * REINFORCE с clip (PPO-style, но без critic и GAE) и gradient clipping.
 *
 * <h2>Идея</h2>
 * Политика {@code π(a|s) = softmax(π_net(s))}. Агент играет эпизод,
 * собирает траекторию {@code (s_t, a_t, r_t)}. Returns
 * {@code G_t = Σ_{t'>=t} γ^{t'-t} r_{t'}}. Loss:
 * {@code −min(r_t·A_t, clip(r_t, 1−ε, 1+ε)·A_t)}, где
 * {@code r_t = π_new(a_t|s_t) / π_old(a_t|s_t)} — как в PPO.
 *
 * <h2>Численная устойчивость энтропии</h2>
 * Энтропия {@code −Σ p_i · log(p_i)} при насыщении softmax даёт
 * {@code 0 · log(0) = NaN} в IEEE 754 (предел равен 0, но float
 * его не берёт). Поэтому при подсчёте энтропии и её градиента
 * <b>пропускаем</b> компоненты с {@code p_i < 1e-10}. Это классическая
 * ловушка RL — без неё REINFORCE падает в NaN на первой же итерации,
 * как только политика насыщается.
 *
 * <h2>Gradient clipping</h2>
 * После всех backward и перед update — clip градиента по L2-норме:
 * если {@code sqrt(gradNormSq) > GRAD_CLIP}, масштабируем градиент
 * через {@code scaleGrad}. Стандарт в RL (PPO, A2C).
 *
 * <h2>Несколько эпох на траекторию</h2>
 * Как в PPO: собрали траекторию, потом {@code nEpochs} раз проходим
 * по ней мини-батчами.
 *
 * <h2>Диагностика NaN</h2>
 * Перед обновлением проверяем advantage, oldLogits, newLogits, dLogits.
 * После update — проверяем forward. Если что-то NaN/Inf — бросаем
 * {@link IllegalStateException} с диагностикой.
 */
public final class ReinforceTrainer implements RlTrainer {

    /** Порог L2-нормы градиента для gradient clipping. */
    private static final double GRAD_CLIP_NORM = 1.0;

    /** Порог, ниже которого p_i считается нулём при подсчёте энтропии. */
    private static final float ENTROPY_EPS = 1e-10f;

    private final Environment env;
    private final Functional policy;
    private final Optimizer optimizer;
    private final float gamma;
    private final float entropyBonus;
    private final float clipEps;
    private final int nEpochs;
    private final int miniBatchSize;
    private final Random rnd;

    /** Записанный переход эпизода: state, action, reward. */
    private record Step(float[] state, int action, float reward) {}

    /** Метрики последней итерации; заполняются в {@link #trainIteration()}. */
    private RlMetrics lastMetrics = new RlMetrics(0f, 0f, 0f, Float.NaN, 0f, 0);

    public ReinforceTrainer(Environment env, Functional policy,
                            Optimizer optimizer, float gamma,
                            float entropyBonus, float clipEps,
                            int nEpochs, int miniBatchSize, long seed) {
        this.env = env;
        this.policy = policy;
        this.optimizer = optimizer;
        this.gamma = gamma;
        this.entropyBonus = entropyBonus;
        this.clipEps = clipEps;
        this.nEpochs = nEpochs;
        this.miniBatchSize = miniBatchSize;
        this.rnd = new Random(seed);
    }

    /** Упрощённый конструктор: clipEps=0.2, nEpochs=4, miniBatch=64. */
    public ReinforceTrainer(Environment env, Functional policy,
                            Optimizer optimizer, float gamma,
                            float entropyBonus, long seed) {
        this(env, policy, optimizer, gamma, entropyBonus, 0.2f, 4, 64, seed);
    }

    @Override
    public RlMetrics lastMetrics() { return lastMetrics; }

    @Override
    public void trainIteration() {
        // ==================== 1. Собираем траекторию ====================
        List<Step> trajectory = new ArrayList<>();
        float[] s = env.reset();
        boolean done = false;
        float totalReward = 0f;

        while (!done) {
            float[] logits = policy.forward(s, true);
            int a = Categorical.sample(logits, rnd);
            Environment.StepResult step = env.step(a);

            // state.clone() — s может быть внутренним буфером среды.
            trajectory.add(new Step(s.clone(), a, step.reward()));
            totalReward += step.reward();
            s = step.observation();
            done = step.done();
        }

        // ==================== 2. Returns G_t ====================
        int T = trajectory.size();
        float[] returns = new float[T];
        float G = 0f;
        for (int t = T - 1; t >= 0; t--) {
            G = trajectory.get(t).reward() + gamma * G;
            returns[t] = G;
        }

        // ==================== 3. Baseline + нормализация advantage ====================
        // Baseline = среднее returns. advantage = returns - baseline.
        // Затем нормализация по mean/std (стандарт PPO/REINFORCE).
        float sumReturns = 0f;
        for (float r : returns) sumReturns += r;
        float baseline = sumReturns / T;

        float[] advantages = new float[T];
        float mean = 0f;
        for (int t = 0; t < T; t++) {
            advantages[t] = returns[t] - baseline;
            mean += advantages[t];
        }
        mean /= T;
        float std = 0f;
        for (int t = 0; t < T; t++) {
            float d = advantages[t] - mean;
            std += d * d;
        }
        std = (float) Math.sqrt(std / T) + 1e-8f;
        for (int t = 0; t < T; t++) advantages[t] = (advantages[t] - mean) / std;

        // Диагностика advantage.
        for (int t = 0; t < T; t++) {
            if (Float.isNaN(advantages[t]) || Float.isInfinite(advantages[t])) {
                throw new IllegalStateException(
                        "NaN/Inf в advantage на шаге " + t
                                + ": returns=" + returns[t]
                                + ", baseline=" + baseline
                                + ", mean=" + mean + ", std=" + std);
            }
        }

        // ==================== 4. Старые logits для clip ====================
        float[][] oldLogits = new float[T][];
        for (int t = 0; t < T; t++) {
            oldLogits[t] = policy.forward(trajectory.get(t).state(), false).clone();
            for (float z : oldLogits[t]) {
                if (Float.isNaN(z) || Float.isInfinite(z)) {
                    throw new IllegalStateException(
                            "NaN/Inf в oldLogits на шаге " + t
                                    + " — веса сети уже взорвались");
                }
            }
        }

        // ==================== 5. Обновление с clip (nEpochs раз) ====================
        float policyLossSum = 0f;
        float entropySum = 0f;
        int policyCount = 0;

        List<Integer> indices = new ArrayList<>(T);
        for (int i = 0; i < T; i++) indices.add(i);

        for (int epoch = 0; epoch < nEpochs; epoch++) {
            Collections.shuffle(indices, rnd);
            for (int start = 0; start < T; start += miniBatchSize) {
                int end = Math.min(start + miniBatchSize, T);
                int batch = end - start;

                for (int k = start; k < end; k++) {
                    int t = indices.get(k);
                    Step st = trajectory.get(t);

                    float[] newLogits = policy.forward(st.state(), true);
                    for (float z : newLogits) {
                        if (Float.isNaN(z) || Float.isInfinite(z)) {
                            throw new IllegalStateException(
                                    "NaN/Inf в newLogits на шаге " + t + " — веса взорвались");
                        }
                    }

                    float[] pNew = Categorical.softmax(newLogits);
                    float[] pOld = Categorical.softmax(oldLogits[t]);

                    int a = st.action();
                    float logpNew = (float) Math.log(pNew[a] + 1e-10f);
                    float logpOld = (float) Math.log(pOld[a] + 1e-10f);
                    float ratio = (float) Math.exp(logpNew - logpOld);
                    float adv = advantages[t];

                    // Метрики: clipped policy loss.
                    float clipped = Math.max(1f - clipEps, Math.min(1f + clipEps, ratio));
                    policyLossSum += -(float) Math.min(ratio * adv, clipped * adv);

                    // Энтропия с защитой 0·log(0):
                    // пропускаем pi < ENTROPY_EPS, чтобы не получить NaN.
                    float ent = 0f;
                    for (float pi : pNew) {
                        if (pi > ENTROPY_EPS) {
                            ent -= pi * (float) Math.log(pi);
                        }
                    }
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

                    // Entropy bonus с защитой 0·log(0) в градиенте.
                    if (entropyBonus > 0f) {
                        float sumPlogPPlus1 = 0f;
                        float[] dEntropy = new float[dLogits.length];
                        for (int i = 0; i < dLogits.length; i++) {
                            if (pNew[i] > ENTROPY_EPS) {
                                float logP = (float) Math.log(pNew[i]);
                                dEntropy[i] = -pNew[i] * (logP + 1f);
                                sumPlogPPlus1 += pNew[i] * (logP + 1f);
                            } else {
                                dEntropy[i] = 0f;
                            }
                        }
                        for (int i = 0; i < dLogits.length; i++) {
                            dLogits[i] += entropyBonus * (dEntropy[i] - pNew[i] * sumPlogPPlus1);
                        }
                    }

                    // Диагностика dLogits.
                    for (float d : dLogits) {
                        if (Float.isNaN(d) || Float.isInfinite(d)) {
                            throw new IllegalStateException(
                                    "NaN/Inf в dLogits на шаге " + t
                                            + " (ratio=" + ratio + ", adv=" + adv
                                            + ", pNew[" + a + "]=" + pNew[a] + ")");
                        }
                    }

                    policy.backward(dLogits);
                }

                // ==================== 6. Gradient clipping ====================
                double gradNormSq = policy.gradNormSq();
                double gradNorm = Math.sqrt(gradNormSq);
                if (gradNorm > GRAD_CLIP_NORM) {
                    float scale = (float) (GRAD_CLIP_NORM / gradNorm);
                    policy.scaleGrad(scale);
                }

                policy.update(optimizer, batch);
            }
        }

        // ==================== 7. Проверка после обновления ====================
        float[] check = policy.forward(trajectory.get(0).state(), false);
        for (float z : check) {
            if (Float.isNaN(z) || Float.isInfinite(z)) {
                throw new IllegalStateException(
                        "NaN/Inf в logits после update — веса взорвались "
                                + "(grad clip не помог, уменьшите LR)");
            }
        }

        // ==================== 8. Метрики итерации ====================
        float policyLoss = policyCount > 0 ? policyLossSum / policyCount : 0f;
        float entropy = policyCount > 0 ? entropySum / policyCount : 0f;
        lastMetrics = RlMetrics.reinforce(
                totalReward, T, policyLoss, entropy, 1);
    }
}