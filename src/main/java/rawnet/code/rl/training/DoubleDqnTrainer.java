package rawnet.code.rl.training;

import rawnet.code.net.core.optimizator.Optimizer;
import rawnet.code.net.model.functional.Functional;
import rawnet.code.rl.Environment;

import java.util.List;

/**
 * Double DQN: DQN с разделением выбора действия и оценки target.
 *
 * <h2>Проблема DQN: overestimation bias</h2>
 * В обычном {@link DqnTrainer} target считается так:
 * <pre>{@code
 * a* = argmax_a Q_target(s', a)        // выбор — target
 * target = r + γ · Q_target(s', a*)    // оценка — target
 * }</pre>
 * Одна и та же {@code targetNet} и <b>выбирает</b> лучшее действие,
 * и <b>оценивает</b> его. Но {@code max} по шумным оценкам
 * <b>систематически завышает</b> Q: если у Q-сети есть шум
 * {@code ε}, то {@code max(Q + ε) ≥ max(Q)}, и смещение растёт
 * с числом действий. Это <b>overestimation bias</b>.
 *
 * <h2>Решение Double DQN: разделить выбор и оценку</h2>
 * <pre>{@code
 * a* = argmax_a Q_online(s', a)        // выбор — online
 * target = r + γ · Q_target(s', a*)    // оценка — target
 * }</pre>
 * Online выбирает действие (её шум «свежий» и не коррелирует
 * с шумом target), target оценивает. Смещение меньше, Q-значения
 * не уезжают вверх, политика стабильнее.
 *
 * <h2>Что меняется в коде</h2>
 * Только {@link #trainBatch()}: вместо
 * {@code max(targetNet.forward(sNext))} —
 * {@code targetNet.forward(sNext)[argmax(qNet.forward(sNext))]}.
 * Всё остальное (replay buffer, sync, ε-greedy, Huber) — как в
 * {@link DqnTrainer}.
 *
 * <h2>Ссылка</h2>
 * Van Hasselt, Guez, Silver. <i>Deep Reinforcement Learning with
 * Double Q-learning</i>. AAAI 2016.
 *
 * <h2>Использование</h2>
 * <pre>{@code
 * DoubleDqnTrainer trainer = new DoubleDqnTrainer(
 *         env, qNet, targetNet, opt, gamma,
 *         epsStart, epsEnd, epsDecaySteps, seed,
 *         bufferSize, batchSize, warmupSteps, syncSteps, huberDelta);
 * new RlTraining(trainer, maxEpisodes, ...).train();
 * }</pre>
 */
public class DoubleDqnTrainer extends DqnTrainer {

    /**
     * Упрощённый конструктор с дефолтами — те же, что у
     * {@link DqnTrainer#DqnTrainer(Environment, Functional, Functional,
     * Optimizer, float, float, float, int, long)}.
     */
    public DoubleDqnTrainer(Environment env, Functional qNet, Functional targetNet,
                            Optimizer optimizer, float gamma,
                            float epsilonStart, float epsilonEnd,
                            int epsilonDecaySteps, long seed) {
        super(env, qNet, targetNet, optimizer, gamma,
                epsilonStart, epsilonEnd, epsilonDecaySteps, seed);
    }

    /**
     * Полный конструктор — те же параметры, что у
     * {@link DqnTrainer#DqnTrainer(Environment, Functional, Functional,
     * Optimizer, float, float, float, int, long, int, int, int, int, float)}.
     */
    public DoubleDqnTrainer(Environment env, Functional qNet, Functional targetNet,
                            Optimizer optimizer, float gamma,
                            float epsilonStart, float epsilonEnd,
                            int epsilonDecaySteps, long seed,
                            int bufferSize, int batchSize,
                            int warmupSteps, int targetSyncSteps,
                            float huberDelta) {
        super(env, qNet, targetNet, optimizer, gamma,
                epsilonStart, epsilonEnd, epsilonDecaySteps, seed,
                bufferSize, batchSize, warmupSteps, targetSyncSteps, huberDelta);
    }

    /**
     * Переопределяем <b>только</b> target: Double DQN.
     *
     * <p>Сравнение с {@link DqnTrainer#trainBatch()}:
     * <pre>{@code
     * // DQN:
     * float maxNext = max(targetNet.forward(sNext));
     * target = r + γ · maxNext;
     *
     * // Double DQN:
     * int aStar = argmax(qNet.forward(sNext));         // online выбирает
     * float maxNext = targetNet.forward(sNext)[aStar]; // target оценивает
     * target = r + γ · maxNext;
     * }</pre>
     *
     * <p><b>Accessor'ы record'а.</b> {@link Transition} — record,
     * его поля {@code private}, доступ — только через accessor'ы
     * {@code s()}, {@code a()}, {@code r()}, {@code sNext()},
     * {@code done()}. Прямой доступ {@code tr.s} не компилируется
     * даже из наследника — это контракт record'а.
     */
    @Override
    protected BatchStats trainBatch() {
        List<Transition> batch = sampleBatch(batchSize);
        float lossSum = 0f;
        float absTDSum = 0f;
        float qSum = 0f;
        int n = batch.size();

        qNet.zeroGrad();

        for (Transition tr : batch) {
            // 1. target: forward(sNext) для qNet и targetNet.
            //    ВАЖНО: forward(sNext) для qNet перезапишет его буфер,
            //    поэтому forward(s) делаем ПОСЛЕ.
            float target = tr.r();
            if (!tr.done()) {
                float[] qNextOnline = qNet.forward(tr.sNext(), false);
                int aStar = 0;
                for (int i = 1; i < qNextOnline.length; i++) {
                    if (qNextOnline[i] > qNextOnline[aStar]) aStar = i;
                }
                float[] qNextTarget = targetNet.forward(tr.sNext(), false);
                target += gamma * qNextTarget[aStar];
            }

            // 2. forward(s) — последний forward для qNet.
            //    backward пойдёт именно по нему.
            float[] qAll = qNet.forward(tr.s(), true);
            float qPred = qAll[tr.a()];

            // 3. loss
            float delta = qPred - target;
            float huber, dHuber;
            if (Math.abs(delta) < huberDelta) {
                huber = 0.5f * delta * delta;
                dHuber = delta;
            } else {
                huber = huberDelta * (Math.abs(delta) - 0.5f * huberDelta);
                dHuber = (float) Math.signum(delta) * huberDelta;
            }
            lossSum += huber;
            absTDSum += Math.abs(delta);
            qSum += qPred;

            // 4. backward по s
            float[] dQ = new float[qAll.length];
            dQ[tr.a()] = dHuber;
            qNet.backward(dQ);
        }

        qNet.update(optimizer, n);
        return new BatchStats(lossSum / n, absTDSum / n, qSum / n);
    }
}