package rawnet.code.rl.training;

import rawnet.code.net.core.optimizator.Optimizer;
import rawnet.code.net.model.functional.Functional;
import rawnet.code.rl.Environment;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * DQN: Deep Q-Network. Q(s,a; θ) — нейросеть.
 *
 * <h2>Три трюка для стабильности</h2>
 * <ul>
 *   <li><b>Replay buffer</b>: декоррелирует переходы.</li>
 *   <li><b>Target network</b>: стабильный target, обновляется редко.</li>
 *   <li><b>Huber loss</b>: устойчив к выбросам.</li>
 * </ul>
 *
 * <h2>Почему без энтропии</h2>
 * Исследование — через ε-greedy (внешний механизм). Энтропия
 * относится к политике, а DQN учит ценность. Энтропийный бонус
 * был бы «ценностью с бонусом за неопределённость» — это уже
 * другое (Soft Q-learning), за рамками книги.
 *
 * <h2>Sync target network</h2>
 * {@link Functional} умеет сохраняться в {@link DataOutputStream}
 * и загружаться из {@link DataInputStream}. Sync прогоняет online
 * через {@code ByteArrayOutputStream} и загружает обратно —
 * это копирует <b>только топологию и параметры слоёв</b>
 * (буферы активаций {@code node.output}/{@code node.deltaAccum} —
 * transient, они не сериализуются).
 *
 * <p><b>Почему {@code targetNet} не {@code final}.</b>
 * {@link Functional#load} возвращает <b>новый</b> объект, а не пишет
 * в существующий. Поэтому sync подменяет ссылку. Это безопасно:
 * {@code targetNet} — приватное поле {@link DqnTrainer}, снаружи
 * на него никто не смотрит.
 *
 * <p><b>Проверка sync.</b> После каждой синхронизации делается
 * probe-forward на фиксированном входе: выходы online и target
 * должны совпасть с точностью {@code 1e-6}. Если не совпали —
 * {@link IllegalStateException}: значит, {@code save}/{@code load}
 * не копирует веса (баг в {@code Layer.save}/{@code Layer.load}),
 * и учиться на таком target бессмысленно.
 *
 * <h2>Параметры</h2>
 * <ul>
 *   <li>{@code bufferSize} — размер replay buffer ({@code 1..1_000_000});</li>
 *   <li>{@code batchSize} — размер мини-батча ({@code 1..bufferSize});</li>
 *   <li>{@code warmupSteps} — сколько шагов собрать <b>до</b> первого
 *       обучения ({@code >= batchSize});</li>
 *   <li>{@code targetSyncSteps} — как часто синхронизировать target
 *       ({@code >= 1}).</li>
 * </ul>
 * Дефолты: {@code bufferSize=100_000}, {@code batchSize=64},
 * {@code warmupSteps=1000}, {@code targetSyncSteps=1000}.
 *
 * <p><b>Для маленьких сред</b> (CartPole, GridWorld, где эпизод —
 * десятки шагов) дефолты слишком большие: {@code warmupSteps=1000}
 * при эпизоде 5–24 шага — это половина бюджета обучения до первого
 * батча. Уменьшайте: {@code warmupSteps=100}, {@code targetSyncSteps=200}.
 *
 * <h2>Наследование</h2>
 * Класс не {@code final}, поля и ключевые методы — {@code protected},
 * чтобы {@link DoubleDqnTrainer} мог переопределить только
 * {@link #trainBatch()}. Всё остальное (replay, sync, ε-greedy)
 * остаётся общим.
 */
public class DqnTrainer implements RlTrainer {

    // ==================== дефолты ====================

    protected static final int DEFAULT_BUFFER_SIZE = 100_000;
    protected static final int DEFAULT_BATCH_SIZE = 64;
    protected static final int DEFAULT_WARMUP_STEPS = 1000;
    protected static final int DEFAULT_TARGET_SYNC_STEPS = 1000;
    protected static final float DEFAULT_HUBER_DELTA = 1.0f;

    /** Допуск для проверки sync: forward online vs target. */
    protected static final float SYNC_PROBE_EPS = 1e-6f;

    // ==================== поля ====================

    protected final Environment env;
    protected final Functional qNet;       // online
    /**
     * Target network. <b>Не final</b>: {@link #syncTarget()} подменяет
     * ссылку, потому что {@link Functional#load} возвращает новый объект.
     */
    protected Functional targetNet;
    protected final Optimizer optimizer;
    protected final float gamma;
    protected final float epsilonStart;
    protected final float epsilonEnd;
    protected final int epsilonDecaySteps;
    protected final Random rnd;

    // ==================== replay buffer ====================

    /**
     * Буфер хранится как {@link ArrayList} с кольцевым указателем
     * {@link #bufferPos}. Это даёт O(1) random access по индексу —
     * в отличие от {@code ArrayDeque}, который не поддерживает
     * {@code get(int)}.
     */
    protected final ArrayList<Transition> buffer;
    /** Куда писать следующий переход, когда буфер полон. */
    protected int bufferPos = 0;

    protected final int bufferSize;
    protected final int batchSize;
    protected final int warmupSteps;
    protected final int targetSyncSteps;
    protected final float huberDelta;

    protected int steps;

    protected record Transition(float[] s, int a, float r, float[] sNext, boolean done) {}

    // ==================== метрики ====================

    protected RlMetrics lastMetrics = new RlMetrics(0f, 0f, 0f, Float.NaN, 0f, 0);
    /** Сумма |TD| за последнюю итерацию (для диагностики). */
    protected float lastAvgAbsTD = 0f;
    /** Сумма Q(s,a) за последнюю итерацию (для диагностики). */
    protected float lastAvgQ = 0f;

    // ==================== конструкторы ====================

    /**
     * Упрощённый конструктор с дефолтами:
     * {@code bufferSize=100_000}, {@code batchSize=64},
     * {@code warmupSteps=1000}, {@code targetSyncSteps=1000},
     * {@code huberDelta=1.0}.
     *
     * <p>Сразу синхронизирует target network с online.
     */
    public DqnTrainer(Environment env, Functional qNet, Functional targetNet,
                      Optimizer optimizer, float gamma,
                      float epsilonStart, float epsilonEnd,
                      int epsilonDecaySteps, long seed) {
        this(env, qNet, targetNet, optimizer, gamma,
                epsilonStart, epsilonEnd, epsilonDecaySteps, seed,
                DEFAULT_BUFFER_SIZE, DEFAULT_BATCH_SIZE,
                DEFAULT_WARMUP_STEPS, DEFAULT_TARGET_SYNC_STEPS,
                DEFAULT_HUBER_DELTA);
    }

    /**
     * Полный конструктор.
     *
     * @param env               среда
     * @param qNet              online Q-сеть
     * @param targetNet         target Q-сеть (будет заменена в
     *                          {@link #syncTarget()} сразу в конструкторе)
     * @param optimizer         оптимизатор для online
     * @param gamma             дисконт
     * @param epsilonStart      начальный ε
     * @param epsilonEnd        конечный ε
     * @param epsilonDecaySteps за сколько шагов ε дойдёт от start до end
     * @param seed              seed для {@link Random}
     * @param bufferSize        размер replay buffer
     * @param batchSize         размер мини-батча
     * @param warmupSteps       сколько шагов собрать до первого обучения
     *                          ({@code >= batchSize})
     * @param targetSyncSteps   как часто синхронизировать target
     * @param huberDelta        порог Huber loss
     * @throws IllegalArgumentException если параметры некорректны
     */
    public DqnTrainer(Environment env, Functional qNet, Functional targetNet,
                      Optimizer optimizer, float gamma,
                      float epsilonStart, float epsilonEnd,
                      int epsilonDecaySteps, long seed,
                      int bufferSize, int batchSize,
                      int warmupSteps, int targetSyncSteps,
                      float huberDelta) {
        if (bufferSize < 1)
            throw new IllegalArgumentException("bufferSize >= 1, got " + bufferSize);
        if (batchSize < 1 || batchSize > bufferSize)
            throw new IllegalArgumentException(
                    "batchSize in [1, " + bufferSize + "], got " + batchSize);
        if (warmupSteps < batchSize)
            throw new IllegalArgumentException(
                    "warmupSteps >= batchSize (" + batchSize + "), got " + warmupSteps);
        if (targetSyncSteps < 1)
            throw new IllegalArgumentException("targetSyncSteps >= 1, got " + targetSyncSteps);
        if (huberDelta <= 0f)
            throw new IllegalArgumentException("huberDelta > 0, got " + huberDelta);

        this.env = env;
        this.qNet = qNet;
        this.targetNet = targetNet;
        this.optimizer = optimizer;
        this.gamma = gamma;
        this.epsilonStart = epsilonStart;
        this.epsilonEnd = epsilonEnd;
        this.epsilonDecaySteps = epsilonDecaySteps;
        this.rnd = new Random(seed);
        this.bufferSize = bufferSize;
        this.batchSize = batchSize;
        this.warmupSteps = warmupSteps;
        this.targetSyncSteps = targetSyncSteps;
        this.huberDelta = huberDelta;
        this.buffer = new ArrayList<>(bufferSize);

        // Начальная синхронизация: target = копия online.
        syncTarget();
    }

    // ==================== API ====================

    @Override
    public RlMetrics lastMetrics() { return lastMetrics; }

    /** Средний |TD| за последнюю итерацию (диагностика). */
    public float lastAvgAbsTD() { return lastAvgAbsTD; }

    /** Средний Q(s,a) за последнюю итерацию (диагностика). */
    public float lastAvgQ() { return lastAvgQ; }

    /** Текущий ε (для логов / отладки). */
    public float currentEpsilon() {
        if (steps >= epsilonDecaySteps) return epsilonEnd;
        float t = (float) steps / epsilonDecaySteps;
        return epsilonStart + t * (epsilonEnd - epsilonStart);
    }

    /** Текущий размер буфера (для логов / отладки). */
    public int bufferFill() { return buffer.size(); }

    /** Текущий шаг (для логов / отладки). */
    public int steps() { return steps; }

    // ==================== trainIteration ====================

    @Override
    public void trainIteration() {
        // Один эпизод: собираем переходы в буфер + обучаем.
        float[] s = env.reset();
        boolean done = false;
        float totalReward = 0f;
        int len = 0;
        float lossSum = 0f;
        int lossCount = 0;
        float absTDSum = 0f;
        float qSum = 0f;
        int qCount = 0;

        while (!done) {
            int a = chooseAction(s);
            Environment.StepResult step = env.step(a);

            addTransition(new Transition(
                    s.clone(), a, step.reward(),
                    step.observation().clone(), step.done()));

            totalReward += step.reward();
            len++;
            s = step.observation();
            done = step.done();
            steps++;

            // Sync target network — до обучения, чтобы первый батч
            // после sync учился на свежем target.
            if (steps % targetSyncSteps == 0) {
                syncTarget();
            }

            // Обучаем, если буфер достаточно заполнен.
            if (buffer.size() >= warmupSteps) {
                BatchStats st = trainBatch();
                lossSum += st.loss();
                absTDSum += st.avgAbsTD();
                qSum += st.avgQ();
                lossCount++;
                qCount++;
            }
        }

        float loss = lossCount > 0 ? lossSum / lossCount : 0f;
        lastAvgAbsTD = lossCount > 0 ? absTDSum / lossCount : 0f;
        lastAvgQ = qCount > 0 ? qSum / qCount : 0f;

        // Защита от расходимости: если loss/Q/TD — NaN/Inf,
        // RlTraining увидит NaN в policyLoss и остановится.
        if (Float.isNaN(loss) || Float.isInfinite(loss)
                || Float.isNaN(lastAvgQ) || Float.isInfinite(lastAvgQ)) {
            lastMetrics = new RlMetrics(
                    totalReward, len, Float.NaN, Float.NaN, 0f, 1);
        } else {
            lastMetrics = RlMetrics.reinforce(totalReward, len, loss, 0f, 1);
        }
    }

    // ==================== replay buffer ====================

    /**
     * Кольцевая запись в буфер: пока не полон — {@code add},
     * когда полон — перезапись по {@link #bufferPos}.
     */
    protected void addTransition(Transition t) {
        if (buffer.size() < bufferSize) {
            buffer.add(t);
        } else {
            buffer.set(bufferPos, t);
            bufferPos = (bufferPos + 1) % bufferSize;
        }
    }

    /**
     * Сэмплирует {@code k} случайных переходов из буфера.
     *
     * <p><b>O(k)</b>, а не O(N): берём {@code k} случайных индексов
     * и читаем по ним. Старая реализация делала
     * {@code new ArrayList<>(buffer)} + {@code shuffle} — это O(N)
     * копирование на каждый батч, что для {@code bufferSize=100_000}
     * и {@code batchSize=64} давало 100_000 копирований вместо 64.
     *
     * <p><b>Стратегия.</b> Сэмплирование <b>с возвращением</b>
     * ({@code rnd.nextInt(size)} на каждую позицию). Стандарт для DQN:
     * переходы независимы, дубликаты в одном батче не вредят
     * (gradient по ним просто складывается).
     */
    protected List<Transition> sampleBatch(int k) {
        int size = buffer.size();
        int n = Math.min(k, size);
        List<Transition> batch = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            batch.add(buffer.get(rnd.nextInt(size)));
        }
        return batch;
    }

    // ==================== trainBatch ====================

    protected record BatchStats(float loss, float avgAbsTD, float avgQ) {}

    /**
     * Один шаг обучения: сэмплировать батч, посчитать target и loss,
     * сделать backward и update.
     *
     * <p>В {@link DqnTrainer} target считается как
     * {@code r + γ·max_a Q_target(s', a)} (обычный DQN).
     * {@link DoubleDqnTrainer} переопределяет этот метод и меняет
     * только способ расчёта target — см. его javadoc.
     */
    protected BatchStats trainBatch() {
        List<Transition> batch = sampleBatch(batchSize);
        float lossSum = 0f;
        float absTDSum = 0f;
        float qSum = 0f;
        int n = batch.size();

        // СБРОС градиентов перед накоплением.
        // Без этого градиенты накапливаются между батчами,
        // эффективный LR растёт с каждым батчем → расходимость.
        qNet.zeroGrad();

        for (Transition tr : batch) {
            float[] qAll = qNet.forward(tr.s(), true);
            float qPred = qAll[tr.a()];
            float target = tr.r();
            if (!tr.done()) {
                float[] qNext = targetNet.forward(tr.sNext(), false);
                float maxNext = qNext[0];
                for (int i = 1; i < qNext.length; i++) {
                    if (qNext[i] > maxNext) maxNext = qNext[i];
                }
                target += gamma * maxNext;
            }
            float delta = qPred - target;
            float huber;
            float dHuber;
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

            float[] dQ = new float[qAll.length];
            dQ[tr.a()] = dHuber;
            qNet.backward(dQ);
        }

        qNet.update(optimizer, n);
        return new BatchStats(lossSum / n, absTDSum / n, qSum / n);
    }

    // ==================== action selection ====================

    protected int chooseAction(float[] s) {
        if (rnd.nextDouble() < currentEpsilon()) {
            return rnd.nextInt(env.numActions());
        }
        float[] q = qNet.forward(s, false);
        int best = 0;
        for (int i = 1; i < q.length; i++) {
            if (q[i] > q[best]) best = i;
        }
        return best;
    }

    // ==================== sync target ====================

    /**
     * Sync target network: скопировать веса online → target.
     *
     * <p><b>Как.</b> {@link Functional} умеет сохраняться в
     * {@link DataOutputStream} и загружаться из
     * {@link DataInputStream}. Прогоняем online через
     * {@code ByteArrayOutputStream} и загружаем обратно:
     * <ol>
     *   <li>{@code qNet.save(out)} — пишет топологию + параметры
     *       каждого слоя в буфер;</li>
     *   <li>{@link Functional#load} — читает и создаёт <b>новый</b>
     *       {@code Functional}; буферы активаций не сериализуются
     *       (они {@code transient}), так что target получает только
     *       «знание» (структура + веса);</li>
     *   <li>ссылка {@code this.targetNet} подменяется на загруженный
     *       объект;</li>
     *   <li>проверка: forward на probe-входе должен совпасть
     *       (см. {@link #SYNC_PROBE_EPS}).</li>
     * </ol>
     *
     * @throws IllegalStateException если save/load не удался или
     *                               forward после sync не совпал
     */
    protected void syncTarget() {
        try {
            ByteArrayOutputStream buf = new ByteArrayOutputStream();
            try (DataOutputStream out = new DataOutputStream(buf)) {
                qNet.save(out);
            }
            try (DataInputStream in = new DataInputStream(
                    new ByteArrayInputStream(buf.toByteArray()))) {
                this.targetNet = Functional.load(in);
            }
        } catch (IOException e) {
            throw new IllegalStateException(
                    "syncTarget: save/load не удался", e);
        }
        verifySync();
    }

    /**
     * Проверка: forward на фиксированном probe-входе должен дать
     * одинаковый выход у online и target.
     *
     * <p>Если не совпал — значит, {@code Layer.save}/{@code Layer.load}
     * не копирует веса (баг в слое). Учиться на таком target
     * бессмысленно: target «не тот», и Q-значения уезжают.
     */
    protected void verifySync() {
        float[] probe = new float[env.observationSize()];
        java.util.Arrays.fill(probe, 0.5f);

        float[] q1 = qNet.forward(probe, false);
        float[] q2 = targetNet.forward(probe, false);

        if (q1.length != q2.length) {
            throw new IllegalStateException(
                    "syncTarget: разные размеры выходов online="
                            + q1.length + ", target=" + q2.length);
        }
        for (int i = 0; i < q1.length; i++) {
            float diff = Math.abs(q1[i] - q2[i]);
            if (diff > SYNC_PROBE_EPS) {
                throw new IllegalStateException(
                        "syncTarget: веса не скопировались. probe[" + i + "]: "
                                + "online=" + q1[i] + ", target=" + q2[i]
                                + " (|diff|=" + diff + " > " + SYNC_PROBE_EPS + "). "
                                + "Проверь Layer.save/Layer.load в твоих слоях.");
            }
        }
    }
}