package rawnet.code.net.layers.core;

import lombok.Setter;
import lombok.experimental.Accessors;
import rawnet.code.net.TensorShape;
import rawnet.code.net.core.activate.Activate;
import rawnet.code.net.core.optimizator.AdamW;
import rawnet.code.net.core.optimizator.Optimizer;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.contract.IndexConsumer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Таблица эмбеддингов: {@code y[t] = E[idx[t]]}.
 *
 * <pre>
 *   вход:  flat(T)          — T целых индексов
 *   выход: seq(T, dModel)   — T векторов длины dModel
 * </pre>
 *
 * <h2>Два режима обновления весов</h2>
 * <ul>
 *   <li><b>Dense</b> ({@code sparseAdamW = false}, по умолчанию):
 *       {@link #update(Optimizer, int)} передаёт весь {@code E} в
 *       {@code optimizer.updateWeights(E, EG, batchSize)} — обновляются
 *       все строки, включая те, у которых градиент нулевой. Это стандарт
 *       для маленьких словарей и точной эквивалентности dense-AdamW.</li>
 *   <li><b>Sparse</b> ({@code sparseAdamW = true}):
 *       {@link #update(Optimizer, int)} использует <b>своё</b> состояние
 *       AdamW по строкам ({@code m}, {@code v}, {@code step} — для каждой
 *       строки отдельно) и обновляет только строки с ненулевым градиентом
 *       (или явно переданные через
 *       {@link #updateSparseAdamW(int[], int)}). Для word2vec при
 *       {@code vocab=30k} и батче 512 это даёт ×10–15 по скорости.</li>
 * </ul>
 *
 * <h2>Гиперпараметры sparse AdamW</h2>
 * Если в DTO задан {@code sparseAdamW = true}:
 * <ul>
 *   <li>значения {@code sparseLr}, {@code sparseBeta1}, {@code sparseBeta2},
 *       {@code sparseEps}, {@code sparseWeightDecay} {@code >= 0} — используются
 *       напрямую;</li>
 *   <li>значения {@code < 0} — берутся из переданного {@link AdamW}
 *       (через геттеры). Если переданный оптимизатор — не {@link AdamW}
 *       и поле {@code < 0} — будет исключение.</li>
 * </ul>
 *
 * <p>Удобные фабрики в {@link Dto}:
 * <ul>
 *   <li>{@code withSparseAdamW()} — всё из {@link AdamW};</li>
 *   <li>{@code withSparseAdamW(float lr)} — только {@code lr}, остальное
 *       из {@link AdamW};</li>
 *   <li>{@code withSparseAdamW(lr, beta1, beta2, eps, weightDecay)} — всё явно.</li>
 * </ul>
 *
 * <h2>Weight decay</h2>
 * Dense-AdamW регуляризует <b>все</b> строки; sparse — <b>только
 * задействованные</b>. Это стандартное поведение sparse word2vec. Если
 * нужна точная эквивалентность dense-AdamW — оставьте
 * {@code sparseAdamW = false}.
 *
 * <h2>zeroGrad в sparse-режиме</h2>
 * В sparse-режиме {@link #updateSparseAdamW(int[], int)} сам обнуляет
 * градиенты обновлённых строк. Вызывать {@link #zeroGrad()} в горячем
 * цикле <b>не нужно</b> — это лишняя O(vocab·dModel) работа на батч.
 * Если в одном батче часть строк не была обновлена (например, из-за
 * {@link #freezeRows}), их градиенты останутся, но на следующем батче
 * {@code backward} будет копить поверх — если это нежелательно, вызывайте
 * {@link #zeroGrad()} вручную или не используйте заморозку.
 *
 * <h2>Расширение словаря и transfer learning</h2>
 * <ul>
 *   <li>{@link #getWeights()} / {@link #setWeights(float[][])} — вся таблица;</li>
 *   <li>{@link #getRow(int)} / {@link #setRow(int, float[])} — одно слово;</li>
 *   <li>{@link #expand(int, float[][])} — расширить, сохранив индексы;</li>
 *   <li>{@link #expandAndRemap(int, int[], float[][])} — расширить с
 *       перестановкой (если словарь пересортировали);</li>
 *   <li>{@link #freezeRows(int)} — заморозить префикс строк.</li>
 * </ul>
 */
@Accessors(chain = true)
public class EmbeddingLayer extends Layer implements IndexConsumer {

    /**
     * DTO эмбеддинг-слоя.
     *
     * <p>Если {@link #sparseAdamW} == {@code true}, слой использует
     * собственный sparse AdamW. Гиперпараметры берутся из полей
     * {@code sparseLr}, {@code sparseBeta1}, {@code sparseBeta2},
     * {@code sparseEps}, {@code sparseWeightDecay}. Отрицательное
     * значение поля означает «взять из переданного {@link AdamW}».
     */
    public static class Dto extends Layer.Dto implements IndexConsumer {
        public final int vocab;
        public final int dModel;

        /** Использовать sparse AdamW вместо dense update. */
        public boolean sparseAdamW = false;

        /** Learning rate для sparse AdamW; {@code < 0} — взять из {@link AdamW}. */
        public float sparseLr = -1f;
        /** beta1; {@code < 0} — взять из {@link AdamW}. */
        public float sparseBeta1 = -1f;
        /** beta2; {@code < 0} — взять из {@link AdamW}. */
        public float sparseBeta2 = -1f;
        /** eps; {@code < 0} — взять из {@link AdamW}. */
        public float sparseEps = -1f;
        /** weight decay; {@code < 0} — взять из {@link AdamW}. */
        public float sparseWeightDecay = -1f;

        public Dto(int vocab, int dModel) {
            if (vocab < 1)  throw new IllegalArgumentException("vocab >= 1, получено " + vocab);
            if (dModel < 1) throw new IllegalArgumentException("dModel >= 1, получено " + dModel);
            this.vocab = vocab;
            this.dModel = dModel;
        }

        /**
         * Включить sparse AdamW; все гиперпараметры — из переданного
         * {@link AdamW}. Если переданный оптимизатор — не {@link AdamW},
         * будет исключение при первом {@code update}.
         */
        public Dto withSparseAdamW() {
            this.sparseAdamW = true;
            return this;
        }

        /**
         * Включить sparse AdamW; задан только {@code lr}, остальные
         * гиперпараметры — из переданного {@link AdamW} (или дефолты,
         * если {@link AdamW} не передан).
         *
         * @param lr learning rate для sparse AdamW
         */
        public Dto withSparseAdamW(float lr) {
            if (lr <= 0f) throw new IllegalArgumentException("lr > 0, получено " + lr);
            this.sparseAdamW = true;
            this.sparseLr = lr;
            return this;
        }

        /**
         * Включить sparse AdamW со всеми явными гиперпараметрами.
         *
         * @param lr          learning rate
         * @param beta1       экспоненциальное затухание первого момента
         * @param beta2       экспоненциальное затухание второго момента
         * @param eps         малое число для численной устойчивости
         * @param weightDecay коэффициент L2
         */
        public Dto withSparseAdamW(float lr, float beta1, float beta2,
                                   float eps, float weightDecay) {
            if (lr <= 0f)  throw new IllegalArgumentException("lr > 0, получено " + lr);
            if (beta1 < 0f || beta1 >= 1f)
                throw new IllegalArgumentException("beta1 in [0, 1), получено " + beta1);
            if (beta2 < 0f || beta2 >= 1f)
                throw new IllegalArgumentException("beta2 in [0, 1), получено " + beta2);
            if (eps <= 0f) throw new IllegalArgumentException("eps > 0, получено " + eps);
            if (weightDecay < 0f)
                throw new IllegalArgumentException("weightDecay >= 0, получено " + weightDecay);
            this.sparseAdamW       = true;
            this.sparseLr          = lr;
            this.sparseBeta1       = beta1;
            this.sparseBeta2       = beta2;
            this.sparseEps         = eps;
            this.sparseWeightDecay = weightDecay;
            return this;
        }

        @Override
        public Layer create(TensorShape inputShape) {
            return new EmbeddingLayer()
                    .setVocab(vocab)
                    .setDModel(dModel)
                    .setSparseAdamW(sparseAdamW)
                    .setSparseHyperparams(sparseLr, sparseBeta1, sparseBeta2, sparseEps, sparseWeightDecay)
                    .setInputShape(inputShape)
                    .setOutputShape(computeOutputShape(inputShape, vocab, dModel))
                    .allocateBuffers().initWeights(null);
        }
    }

    @Setter private int vocab;
    @Setter private int dModel;

    /** Использовать sparse AdamW в {@link #update(Optimizer, int)}. */
    @Setter private boolean sparseAdamW = false;

    // Гиперпараметры sparse AdamW; < 0 — «взять из AdamW».
    private float sparseLr          = -1f;
    private float sparseBeta1       = -1f;
    private float sparseBeta2       = -1f;
    private float sparseEps         = -1f;
    private float sparseWeightDecay = -1f;

    private int T;

    /** {@code [vocab][dModel]} */
    float[][] E;
    private float[][] EG;

    /** Последние индексы (для backward). */
    private int[] lastIdx;

    private float[] output;
    private float[] prevDelta;

    /**
     * С какого индекса {@link #update(Optimizer, int)} /
     * {@link #updateSparseAdamW} НЕ обновляют строки.
     * {@code 0} — обучаем всё (по умолчанию).
     */
    private int frozenUpTo = 0;

    // ==================== sparse AdamW состояние ====================
    //
    // Ленивая аллокация: только при первом sparse-вызове. Dense-пользователи
    // не платят памятью. Формат — по строкам: mW[i], vW[i] для слова i.
    //
    // step[i] — счётчик обновлений строки i (для bias correction). Разные
    // строки могут иметь разные step, если обновлялись разное число раз.
    // Это правильная семантика для sparse: bias correction считается по
    // «своему» step, а не глобальному.

    private float[][] mW;
    private float[][] vW;
    private int[]     step;

    public EmbeddingLayer() {}

    /**
     * Установить гиперпараметры sparse AdamW.
     *
     * @param lr          learning rate; {@code < 0} — взять из {@link AdamW}
     * @param beta1       beta1; {@code < 0} — взять из {@link AdamW}
     * @param beta2       beta2; {@code < 0} — взять из {@link AdamW}
     * @param eps         eps; {@code < 0} — взять из {@link AdamW}
     * @param weightDecay weight decay; {@code < 0} — взять из {@link AdamW}
     */
    public EmbeddingLayer setSparseHyperparams(float lr, float beta1, float beta2,
                                               float eps, float weightDecay) {
        this.sparseLr          = lr;
        this.sparseBeta1       = beta1;
        this.sparseBeta2       = beta2;
        this.sparseEps         = eps;
        this.sparseWeightDecay = weightDecay;
        return this;
    }

    public boolean isSparseAdamW() { return sparseAdamW; }

    public static TensorShape computeOutputShape(TensorShape in, int vocab, int dModel) {
        if (in == null) throw new IllegalArgumentException("EmbeddingLayer: inputShape == null");
        if (!in.isFlat())
            throw new IllegalArgumentException("EmbeddingLayer: ожидался flat-вход, получено " + in);
        if (vocab < 1)  throw new IllegalArgumentException("EmbeddingLayer: vocab >= 1, получено " + vocab);
        if (dModel < 1) throw new IllegalArgumentException("EmbeddingLayer: dModel >= 1, получено " + dModel);
        return TensorShape.seq(in.size(), dModel);
    }

    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("EmbeddingLayer.create: inputShape и outputShape должны быть заданы");
        if (!inputShape.isFlat())
            throw new IllegalArgumentException("EmbeddingLayer.create: ожидался flat-вход, получено " + inputShape);
        if (vocab < 1 || dModel < 1)
            throw new IllegalArgumentException(
                    "EmbeddingLayer.create: vocab >= 1 и dModel >= 1, получено vocab=" + vocab + ", dModel=" + dModel);
        if (outputShape.seqLen() != inputShape.size() || outputShape.channels() != dModel)
            throw new IllegalArgumentException(
                    "EmbeddingLayer.create: outputShape " + outputShape
                            + " не соответствует flat(" + inputShape.size() + ") → seq("
                            + inputShape.size() + ", " + dModel + ")");

        this.T = inputShape.size();
        this.E = new float[vocab][dModel];
        this.EG = new float[vocab][dModel];
        this.lastIdx = new int[T];
        this.output = new float[T * dModel];
        this.prevDelta = new float[T];

        // Sparse-состояние аллоцируется лениво в ensureSparseState().
        this.mW = null;
        this.vW = null;
        this.step = null;
        return this;
    }

    @Override
    public Layer initWeights(Activate activate) {
        float limit = Layer.initLimit(null, dModel, dModel);
        Random rnd = ThreadLocalRandom.current();
        for (int v = 0; v < vocab; v++)
            for (int j = 0; j < dModel; j++)
                E[v][j] = rnd.nextFloat() * 2 * limit - limit;
        return this;
    }

    // ==================== forward / backward ====================

    @Override
    public float[] forward(float[] input, boolean training) {
        if (input.length != inputShape.size())
            throw new IllegalArgumentException(
                    "EmbeddingLayer.forward: ожидалось " + inputShape.size() + ", получено " + input.length);

        for (int t = 0; t < T; t++) {
            float raw = input[t];
            int idx = Math.round(raw);
            if (Math.abs(raw - idx) > 1e-6f)
                throw new IllegalArgumentException(
                        "EmbeddingLayer.forward: индекс #" + t + " не целый: " + raw);
            if (idx < 0 || idx >= vocab)
                throw new IllegalArgumentException(
                        "EmbeddingLayer.forward: индекс #" + t + " вне [0, " + vocab + "): " + idx);
            lastIdx[t] = idx;
            System.arraycopy(E[idx], 0, output, t * dModel, dModel);
        }
        return output;
    }

    @Override
    public float[] backward(float[] deltaNext) {
        if (deltaNext.length != outputShape.size())
            throw new IllegalArgumentException(
                    "EmbeddingLayer.backward: ожидалось " + outputShape.size() + ", получено " + deltaNext.length);
        if (lastIdx == null)
            throw new IllegalStateException("EmbeddingLayer.backward: forward не был вызван");

        for (int t = 0; t < T; t++) {
            int idx = lastIdx[t];
            int off = t * dModel;
            float[] eg = EG[idx];
            for (int j = 0; j < dModel; j++) eg[j] += deltaNext[off + j];
        }
        Arrays.fill(prevDelta, 0f);
        return prevDelta;
    }

    // ==================== обновление весов ====================

    /**
     * Обновить веса.
     *
     * <ul>
     *   <li>Если {@link #isSparseAdamW()} == {@code false} — dense update
     *       через {@code optimizer.updateWeights(E, EG, batchSize)}.</li>
     *   <li>Если {@code true} — sparse AdamW со своим состоянием; параметры
     *       берутся из полей {@code sparse*} (если {@code >= 0}) или из
     *       переданного {@link AdamW}. Список {@code touched} собирается
     *       из {@code EG} (ненулевые градиенты).</li>
     * </ul>
     *
     * <p>В sparse-режиме {@link #zeroGrad()} вызывать не нужно —
     * {@link #updateSparseAdamW} сам обнуляет градиенты обновлённых строк.
     */
    @Override
    public void update(Optimizer optimizer, int batchSize) {
        if (batchSize <= 0)
            throw new IllegalArgumentException("batchSize должен быть > 0, получено " + batchSize);

        if (!sparseAdamW) {
            denseUpdate(optimizer, batchSize);
            return;
        }

        int[] touched = collectTouchedFromEG();
        Hyperparams hp = resolveHyperparams(optimizer);
        sparseAdamWUpdate(hp.lr, hp.beta1, hp.beta2, hp.eps, hp.weightDecay, touched, batchSize);
    }

    /**
     * Sparse AdamW с явными гиперпараметрами. Используется, когда известен
     * список задействованных индексов (например, из батча word2vec).
     *
     * @param lr          learning rate
     * @param beta1       beta1
     * @param beta2       beta2
     * @param eps         eps
     * @param weightDecay weight decay
     * @param touched     индексы строк; повторы игнорируются
     * @param batchSize   число примеров, по которым накоплен градиент в {@code EG}
     */
    public void updateSparseAdamW(float lr, float beta1, float beta2, float eps,
                                  float weightDecay, int[] touched, int batchSize) {
        if (batchSize <= 0)
            throw new IllegalArgumentException("batchSize должен быть > 0, получено " + batchSize);
        sparseAdamWUpdate(lr, beta1, beta2, eps, weightDecay, touched, batchSize);
    }

    /**
     * Sparse AdamW с гиперпараметрами из полей слоя. Если какое-то поле
     * не задано ({@code < 0}), будет исключение.
     */
    public void updateSparseAdamW(int[] touched, int batchSize) {
        if (sparseLr <= 0f || sparseBeta1 < 0f || sparseBeta2 < 0f
                || sparseEps < 0f || sparseWeightDecay < 0f)
            throw new IllegalStateException(
                    "updateSparseAdamW(touched, batchSize): гиперпараметры не заданы. "
                            + "Задайте их через setSparseHyperparams(...) "
                            + "или используйте перегрузку с явными аргументами.");
        sparseAdamWUpdate(sparseLr, sparseBeta1, sparseBeta2, sparseEps,
                sparseWeightDecay, touched, batchSize);
    }

    /** Внутренняя реализация. Все параметры уже валидные. */
    private void sparseAdamWUpdate(float lr, float beta1, float beta2, float eps,
                                   float weightDecay, int[] touched, int batchSize) {
        if (touched == null || touched.length == 0) return;
        ensureSparseState();

        int[] uniq = dedup(touched);
        float scale = 1f / batchSize;

        for (int idx : uniq) {
            if (idx < frozenUpTo) continue;
            if (idx < 0 || idx >= vocab)
                throw new IllegalArgumentException("sparseAdamW: индекс " + idx + " вне [0, " + vocab + ")");

            int s = ++step[idx];
            float bc1 = (float) (1.0 - Math.pow(beta1, s));
            float bc2 = (float) (1.0 - Math.pow(beta2, s));

            float[] w = E[idx];
            float[] g = EG[idx];
            float[] m = mW[idx];
            float[] v = vW[idx];

            for (int j = 0; j < dModel; j++) {
                float grad = g[j] * scale;

                m[j] = beta1 * m[j] + (1f - beta1) * grad;
                v[j] = beta2 * v[j] + (1f - beta2) * grad * grad;

                float mHat = m[j] / bc1;
                float vHat = v[j] / bc2;
                float upd  = mHat / (float) (Math.sqrt(vHat) + eps);

                w[j] -= lr * (upd + weightDecay * w[j]);
            }
            // Обнуляем градиент строки — следующий батч начнёт с чистого листа.
            Arrays.fill(g, 0f);
        }
    }

    /** Dense update (старая семантика). */
    private void denseUpdate(Optimizer optimizer, int batchSize) {
        if (frozenUpTo <= 0) {
            optimizer.updateWeights(E, EG, batchSize);
            return;
        }
        int n = vocab - frozenUpTo;
        if (n <= 0) return;
        float[][] w = new float[n][];
        float[][] g = new float[n][];
        System.arraycopy(E,  frozenUpTo, w, 0, n);
        System.arraycopy(EG, frozenUpTo, g, 0, n);
        optimizer.updateWeights(w, g, batchSize);
        System.arraycopy(w, 0, E, frozenUpTo, n);
    }

    /** Собирает индексы, у которых {@code EG} не полностью нулевой. */
    private int[] collectTouchedFromEG() {
        int[] buf = new int[64];
        int n = 0;
        for (int i = 0; i < vocab; i++) {
            float[] g = EG[i];
            boolean nonzero = false;
            for (int j = 0; j < dModel; j++) if (g[j] != 0f) { nonzero = true; break; }
            if (!nonzero) continue;
            if (n == buf.length) buf = Arrays.copyOf(buf, n * 2);
            buf[n++] = i;
        }
        return Arrays.copyOf(buf, n);
    }

    private void ensureSparseState() {
        if (mW != null) return;
        mW = new float[vocab][dModel];
        vW = new float[vocab][dModel];
        step = new int[vocab];
    }

    /** Гиперпараметры, разрешённые из полей слоя или переданного оптимизатора. */
    private Hyperparams resolveHyperparams(Optimizer opt) {
        float lr  = sparseLr          >= 0f ? sparseLr          : Float.NaN;
        float b1  = sparseBeta1       >= 0f ? sparseBeta1       : Float.NaN;
        float b2  = sparseBeta2       >= 0f ? sparseBeta2       : Float.NaN;
        float eps = sparseEps         >= 0f ? sparseEps         : Float.NaN;
        float wd  = sparseWeightDecay >= 0f ? sparseWeightDecay : Float.NaN;

        if (opt instanceof AdamW a) {
            if (Float.isNaN(lr))  lr  = a.getLr();
            if (Float.isNaN(b1))  b1  = a.getBeta1();
            if (Float.isNaN(b2))  b2  = a.getBeta2();
            if (Float.isNaN(eps)) eps = a.getEps();
            if (Float.isNaN(wd))  wd  = a.getWeightDecay();
        }

        if (Float.isNaN(lr) || Float.isNaN(b1) || Float.isNaN(b2)
                || Float.isNaN(eps) || Float.isNaN(wd)) {
            throw new IllegalStateException(
                    "sparseAdamW: не заданы гиперпараметры. "
                            + "Задайте их через Dto.withSparseAdamW(...) / setSparseHyperparams(...), "
                            + "либо передайте AdamW (у него есть геттеры), "
                            + "либо используйте updateSparseAdamW(...) с явными аргументами.");
        }
        return new Hyperparams(lr, b1, b2, eps, wd);
    }

    private record Hyperparams(float lr, float beta1, float beta2, float eps, float weightDecay) {}

    @Override
    public void zeroGrad() {
        for (float[] row : EG) Arrays.fill(row, 0f);
    }

    @Override
    public double gradNormSq() {
        double sum = 0;
        for (float[] row : EG) for (float v : row) sum += v * v;
        return sum;
    }

    @Override
    public void scaleGrad(float scale) {
        for (float[] row : EG)
            for (int j = 0; j < row.length; j++) row[j] *= scale;
    }

    // ==================== заморозка и доступ к E ====================

    /** Заморозить строки {@code [0, fromIdx)}. */
    public EmbeddingLayer freezeRows(int fromIdx) {
        if (fromIdx < 0 || fromIdx > vocab)
            throw new IllegalArgumentException("freezeRows: fromIdx in [0, " + vocab + "], получено " + fromIdx);
        this.frozenUpTo = fromIdx;
        return this;
    }

    public int frozenUpTo() { return frozenUpTo; }

    /** @return копия всей таблицы {@code E} */
    public float[][] getWeights() {
        float[][] copy = new float[vocab][dModel];
        for (int i = 0; i < vocab; i++) System.arraycopy(E[i], 0, copy[i], 0, dModel);
        return copy;
    }

    /** Заменить всю таблицу {@code E}. */
    public EmbeddingLayer setWeights(float[][] newE) {
        if (newE == null) throw new IllegalArgumentException("setWeights: newE == null");
        if (newE.length != vocab)
            throw new IllegalArgumentException(
                    "setWeights: newE.length=" + newE.length + " != vocab=" + vocab);
        for (int i = 0; i < vocab; i++) {
            if (newE[i] == null || newE[i].length != dModel)
                throw new IllegalArgumentException("setWeights: newE[" + i + "] неверной длины");
            System.arraycopy(newE[i], 0, E[i], 0, dModel);
        }
        return this;
    }

    public float[] getRow(int i) {
        checkIdx(i);
        return E[i].clone();
    }

    public EmbeddingLayer setRow(int i, float[] vec) {
        checkIdx(i);
        if (vec == null || vec.length != dModel)
            throw new IllegalArgumentException("setRow: vec.length != dModel");
        System.arraycopy(vec, 0, E[i], 0, dModel);
        return this;
    }

    private void checkIdx(int i) {
        if (i < 0 || i >= vocab)
            throw new IllegalArgumentException("индекс " + i + " вне [0, " + vocab + ")");
    }

    // ==================== расширение словаря ====================

    /**
     * Расширить словарь до {@code newVocab}, сохранив старые индексы.
     * Старые строки {@code E[0..vocab)} остаются на своих местах; новые
     * инициализируются из {@code initForNew} (если передан) или случайно.
     *
     * <p>Sparse-состояние сбрасывается и переаллоцируется лениво.
     */
    public EmbeddingLayer expand(int newVocab, float[][] initForNew) {
        if (newVocab < vocab)
            throw new IllegalArgumentException("expand: newVocab=" + newVocab + " < vocab=" + vocab);
        if (initForNew != null && initForNew.length != newVocab - vocab)
            throw new IllegalArgumentException(
                    "expand: initForNew.length=" + initForNew.length
                            + " != newVocab-vocab=" + (newVocab - vocab));

        int oldVocab = vocab;
        float[][] newE  = new float[newVocab][dModel];
        float[][] newEG = new float[newVocab][dModel];
        for (int i = 0; i < oldVocab; i++) {
            System.arraycopy(E[i],  0, newE[i],  0, dModel);
            System.arraycopy(EG[i], 0, newEG[i], 0, dModel);
        }

        float limit = Layer.initLimit(null, dModel, dModel);
        Random rnd = ThreadLocalRandom.current();
        for (int i = oldVocab; i < newVocab; i++) {
            if (initForNew != null && initForNew[i - oldVocab] != null) {
                float[] src = initForNew[i - oldVocab];
                if (src.length != dModel)
                    throw new IllegalArgumentException(
                            "expand: initForNew[" + (i - oldVocab) + "].length=" + src.length + " != dModel");
                System.arraycopy(src, 0, newE[i], 0, dModel);
            } else {
                for (int j = 0; j < dModel; j++)
                    newE[i][j] = rnd.nextFloat() * 2 * limit - limit;
            }
        }

        this.E = newE;
        this.EG = newEG;
        this.vocab = newVocab;
        // Sparse-состояние невалидно: индексы строк остались, но vocab
        // изменился. Проще переаллоцировать лениво.
        this.mW = null; this.vW = null; this.step = null;
        return this;
    }

    /**
     * Расширить словарь с <b>перестановкой</b> старых индексов.
     *
     * <p>{@code oldToNew[oldIdx] = newIdx} или {@code -1}, если слово выпало.
     * Новые строки (не покрытые ни одним {@code oldToNew[i] >= 0})
     * инициализируются из {@code initForNew} (если передан) или случайно.
     * Значения {@code oldToNew[i] >= 0} должны быть уникальны.
     */
    public EmbeddingLayer expandAndRemap(int newVocab, int[] oldToNew, float[][] initForNew) {
        int oldVocab = vocab;
        if (newVocab < oldVocab)
            throw new IllegalArgumentException("expandAndRemap: newVocab=" + newVocab + " < oldVocab=" + oldVocab);
        if (oldToNew == null || oldToNew.length != oldVocab)
            throw new IllegalArgumentException(
                    "expandAndRemap: oldToNew.length=" + (oldToNew == null ? "null" : oldToNew.length)
                            + " != oldVocab=" + oldVocab);

        float[][] newE  = new float[newVocab][dModel];
        float[][] newEG = new float[newVocab][dModel];
        boolean[] filled = new boolean[newVocab];

        for (int oldIdx = 0; oldIdx < oldVocab; oldIdx++) {
            int newIdx = oldToNew[oldIdx];
            if (newIdx < 0) continue;
            if (newIdx >= newVocab)
                throw new IllegalArgumentException(
                        "expandAndRemap: oldToNew[" + oldIdx + "]=" + newIdx + " >= newVocab=" + newVocab);
            if (filled[newIdx])
                throw new IllegalArgumentException(
                        "expandAndRemap: два старых индекса попали в newIdx=" + newIdx);
            System.arraycopy(E[oldIdx],  0, newE[newIdx],  0, dModel);
            System.arraycopy(EG[oldIdx], 0, newEG[newIdx], 0, dModel);
            filled[newIdx] = true;
        }

        int freshCount = newVocab - oldVocab;
        if (initForNew != null && initForNew.length != freshCount)
            throw new IllegalArgumentException(
                    "expandAndRemap: initForNew.length=" + initForNew.length + " != " + freshCount);

        float limit = Layer.initLimit(null, dModel, dModel);
        Random rnd = ThreadLocalRandom.current();
        int fresh = 0;
        for (int i = 0; i < newVocab; i++) {
            if (filled[i]) continue;
            if (initForNew != null && initForNew[fresh] != null) {
                float[] src = initForNew[fresh];
                if (src.length != dModel)
                    throw new IllegalArgumentException(
                            "expandAndRemap: initForNew[" + fresh + "].length=" + src.length + " != dModel");
                System.arraycopy(src, 0, newE[i], 0, dModel);
            } else {
                for (int j = 0; j < dModel; j++)
                    newE[i][j] = rnd.nextFloat() * 2 * limit - limit;
            }
            fresh++;
        }
        if (fresh != freshCount)
            throw new IllegalStateException(
                    "expandAndRemap: заполнено " + fresh + " новых строк, ожидалось " + freshCount
                            + " (проверьте oldToNew: возможно, есть пропуски)");

        this.E = newE;
        this.EG = newEG;
        this.vocab = newVocab;
        // Sparse-состояние переаллоцируется лениво: индексы строк изменились,
        // старые m/v/step привязаны к старым индексам и больше не валидны.
        this.mW = null; this.vW = null; this.step = null;
        return this;
    }

    // ==================== сериализация ====================

    /**
     * Сохраняет: формы (через {@code super.save}), архитектурные поля,
     * флаг sparse и гиперпараметры, затем всю таблицу {@code E}.
     *
     * <p>Sparse-состояние ({@code mW}, {@code vW}, {@code step}) <b>не
     * сохраняется</b> — как и состояние {@code AdamW} в модели. После
     * загрузки sparse начнётся «с чистого листа». Это стандартное
     * поведение: для инференса состояние оптимизатора не нужно.
     */
    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);
        out.writeInt(vocab);
        out.writeInt(dModel);
        out.writeInt(frozenUpTo);
        out.writeBoolean(sparseAdamW);
        out.writeFloat(sparseLr);
        out.writeFloat(sparseBeta1);
        out.writeFloat(sparseBeta2);
        out.writeFloat(sparseEps);
        out.writeFloat(sparseWeightDecay);
        for (float[] row : E) for (float v : row) out.writeFloat(v);
    }

    /**
     * Загружает: формы (через {@code super.load}), архитектурные поля,
     * флаг sparse, гиперпараметры, затем пересоздаёт буферы и читает
     * {@code E}.
     */
    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);
        this.vocab             = in.readInt();
        this.dModel            = in.readInt();
        this.frozenUpTo        = in.readInt();
        this.sparseAdamW       = in.readBoolean();
        this.sparseLr          = in.readFloat();
        this.sparseBeta1       = in.readFloat();
        this.sparseBeta2       = in.readFloat();
        this.sparseEps         = in.readFloat();
        this.sparseWeightDecay = in.readFloat();
        allocateBuffers();
        for (float[] row : E)
            for (int j = 0; j < row.length; j++) row[j] = in.readFloat();
    }

    // ==================== утилиты ====================

    /** Размер словаря (число строк E). */
    public int vocabSize() { return vocab; }

    /** Размерность эмбеддинга (число столбцов E). */
    public int dModel() { return dModel; }

    /**
     * Копия вектора токена {@code tokenIndex}.
     *
     * @throws IllegalArgumentException если индекс вне диапазона
     */
    public float[] embeddingOf(int tokenIndex) {
        checkIdx(tokenIndex);
        return E[tokenIndex].clone();
    }

    /**
     * Дедупликация {@code int[]}. Для маленьких массивов — O(n²), для
     * больших — сортировка. Возвращает новый массив, вход не модифицируется.
     */
    private static int[] dedup(int[] a) {
        if (a.length <= 16) {
            int[] out = new int[a.length];
            int n = 0;
            outer:
            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < n; j++) if (out[j] == a[i]) continue outer;
                out[n++] = a[i];
            }
            return Arrays.copyOf(out, n);
        }
        int[] sorted = a.clone();
        Arrays.sort(sorted);
        int n = 1;
        for (int i = 1; i < sorted.length; i++)
            if (sorted[i] != sorted[n - 1]) sorted[n++] = sorted[i];
        return Arrays.copyOf(sorted, n);
    }
}