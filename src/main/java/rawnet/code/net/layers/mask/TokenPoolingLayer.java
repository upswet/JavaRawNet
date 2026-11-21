package rawnet.code.net.layers.mask;

import lombok.RequiredArgsConstructor;
import rawnet.code.net.TensorShape;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.core.activate.Activate;
import rawnet.code.net.layers.contract.MaskConsumer;
import rawnet.code.net.layers.contract.MaskKiller;
import rawnet.code.net.core.optimizator.Optimizer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * TokenPoolingLayer — «схлопывает» последовательность векторов в один вектор:
 * {@code seq(T, d) → flat(d)}.
 *
 * <p>Простыми словами: на вход приходит T векторов длины d (например, выход
 * RNN/Transformer по токенам), а на выходе нужен один вектор длины d —
 * «представление всей последовательности». Слой решает, <b>какие токены
 * важнее</b>, и собирает из них один вектор согласно выбранному режиму.
 *
 * <p>Заменяет связку {@code SliceLayer(SEQ, T-1, 1) + ReshapeLayer}
 * («взять последний токен и выпрямить») и умеет корректно обрабатывать
 * маску: PAD-токены ({@code mask[t] == 0}) в пулинге не участвуют.
 * Если маски нет, все токены считаются настоящими.
 *
 * <h2>Режимы ({@link Mode})</h2>
 * <ul>
 *   <li>{@link Mode#LAST_VALID} — <b>последний настоящий</b> токен
 *       ({@code mask[t] > 0}). Классический выбор для «прочитал
 *       предложение — взял финальное состояние». Если все PAD — нулевой
 *       вектор.</li>
 *
 *   <li>{@link Mode#MEAN} — <b>среднее</b> по настоящим токенам (делим
 *       на их число, а не на {@code T}). Простой, устойчивый, без
 *       параметров. Хорош, когда все токены примерно равнозначны.</li>
 *
 *   <li>{@link Mode#MAX} — <b>поэлементный максимум</b> по настоящим
 *       токенам. Берёт самые «сильные» признаки по каждому измерению
 *       независимо. Полезен в задачах, где важен факт наличия сигнала
 *       (например, детекция).</li>
 *
 *   <li>{@link Mode#FIRST} — вектор на <b>позиции 0
 *  Если mask[0] > 0 (токен 0 — настоящий) → выход = input[0] (вектор первого токена как есть).
  * Если mask[0] == 0 (токен 0 — PAD) → выход = нулевой вектор длины D.
 *       </li>
 *
 *   <li>{@link Mode#ATTENTION} — <b>обучаемый взвешенный пулинг</b>.
 *       Для каждого токена считается скор
 *       {@code score[t] = <w, h_t> + bias}, по настоящим токенам берётся
 *       softmax, результат — взвешенная сумма векторов. Сеть сама учится,
 *       какие токены важны для задачи. Это единственный режим с
 *       <b>параметрами</b> ({@code w} длины {@code d} и скалярный
 *       {@code bias}).</li>
 * </ul>
 *
 * <h2>Маска</h2>
 * Реализует {@link MaskConsumer} (читает маску, но не меняет её) и
 * {@link MaskKiller} (выход — плоский вектор, последовательности больше
 * нет, маска «съедается»).
 *
 * <p>Соглашение: {@code mask[t] > 0} — настоящий токен,
 * {@code mask[t] == 0} — PAD. Без маски все токены считаются настоящими.
 *
 * <h2>Backward</h2>
 * Распределяет {@code deltaNext} обратно по токенам согласно режиму:
 * <ul>
 *   <li>{@code LAST}/{@code CLS} — весь градиент в выбранную позицию;</li>
 *   <li>{@code MEAN} — поровну на все настоящие токены;</li>
 *   <li>{@code MAX} — в позицию максимума по каждому измерению отдельно;</li>
 *   <li>{@code ATTENTION} — стандартный backward softmax-взвешенной суммы
 *       плюс накопление градиентов {@code w}/{@code bias}. PAD-токены
 *       получают нулевой градиент и не влияют на {@code w}/{@code bias}.</li>
 * </ul>
 *
 * <h2>Когда что выбирать</h2>
 * <ul>
 *   <li>Классификация текста RNN-энкодером — {@code LAST} (или {@code MEAN}
 *       как более устойчивый baseline).</li>
 *   <li>Transformer с CLS-токеном — {@code CLS}.</li>
 *   <li>Детекция/наличие сигнала — {@code MAX}.</li>
 *   <li>Хочется «умный» пулинг без отдельной attention-головы —
 *       {@code ATTENTION}.</li>
 * </ul>
 */
public class TokenPoolingLayer extends Layer implements MaskConsumer, MaskKiller {

    /**
     * Режим пулинга.
     *
     * <p><b>Внимание:</b> порядок констант участвует в сериализации
     * (пишется {@code ordinal()}). Не переставляйте существующие значения
     * без миграции формата — добавляйте новые <b>в конец</b>.
     */
    public enum Mode {
        /** Последний настоящий токен. */
        LAST_VALID,
        /** Среднее по настоящим токенам. */
        MEAN,
        /** Поэлементный максимум по настоящим токенам. */
        MAX,
        /** Вектор на позиции 0 (CLS-токен). */
        FIRST,
        /** Обучаемый взвешенный пулинг (softmax по скорам). */
        ATTENTION
    }

    /** DTO слоя пулинга. */
    @RequiredArgsConstructor
    public static class Dto extends Layer.Dto {
        /**
         * Режим пулинга; {@code null} трактуется как {@link Mode#MEAN}
         * (историческое поведение по умолчанию).
         */
        public final Mode mode;

        @Override
        public Layer create(TensorShape inputShape) {
            if (!inputShape.isSequence())
                throw new IllegalArgumentException(
                        "TokenPoolingLayer.Dto: ожидался sequence-вход, получено " + inputShape);
            return new TokenPoolingLayer()
                    .setMode(mode)
                    .setInputShape(inputShape)
                    .setOutputShape(TensorShape.flat(inputShape.channels()))
                    .allocateBuffers()
                    .initWeights(null);
        }
    }

    private Mode mode = Mode.MEAN;

    private int T, D;
    private float[] out;
    private float[] prevDelta;
    private float[] lastInput;
    private float[] lastMask;

    // ==================== параметры режима ATTENTION ====================
    //
    // Единственный режим с обучаемыми параметрами. w — «вес важности»
    // каждого измерения вектора токена (длина D); bias — общий сдвиг скора.
    // Для остальных режимов эти поля null и весь параметрический контракт
    // (zeroGrad/update/gradNormSq/scaleGrad) — no-op.

    /** Веса скоринга для ATTENTION: {@code score[t] = <w, h_t> + bias}. */
    float[] w;
    /** Смещение скоринга для ATTENTION. */
    float bias;
    private float[] wGrad;
    private float biasGrad;

    /** Скоры последнего forward (для backward). null вне ATTENTION. */
    private float[] scores;
    /** Softmax-веса последнего forward (для backward). null вне ATTENTION. */
    private float[] attn;

    public TokenPoolingLayer() {}

    public TokenPoolingLayer setMode(Mode mode) {
        this.mode = mode != null ? mode : Mode.MEAN;
        return this;
    }

    public Mode mode() { return mode; }

    /**
     * Валидирует формы и выделяет буферы. Для {@link Mode#ATTENTION}
     * дополнительно создаёт {@code w}/{@code bias} и их накопители.
     *
     * @throws IllegalStateException    если формы не заданы
     * @throws IllegalArgumentException если вход не sequence
     */
    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("TokenPoolingLayer.create: inputShape и outputShape должны быть заданы");
        if (!inputShape.isSequence())
            throw new IllegalArgumentException("TokenPoolingLayer: ожидался sequence-вход, получено " + inputShape);
        if (outputShape.size() != inputShape.channels())
            throw new IllegalArgumentException(
                    "TokenPoolingLayer: outputShape.size=" + outputShape.size()
                            + " != dModel=" + inputShape.channels());

        this.T = inputShape.seqLen();
        this.D = inputShape.channels();
        this.out = new float[D];
        this.prevDelta = new float[T * D];
        this.lastInput = new float[T * D];
        this.lastMask = new float[T];

        // Параметры ATTENTION — только для этого режима. Для остальных
        // оставляем поля null, чтобы параметрический контракт оставался no-op
        // и не тратил память на неиспользуемые массивы.
        if (mode == Mode.ATTENTION) {
            this.w = new float[D];
            this.bias = 0f;
            this.wGrad = new float[D];
            this.biasGrad = 0f;
            this.scores = new float[T];
            this.attn = new float[T];
        } else {
            this.w = null;
            this.bias = 0f;
            this.wGrad = null;
            this.biasGrad = 0f;
            this.scores = null;
            this.attn = null;
        }
        return this;
    }

    /**
     * Инициализация параметров ATTENTION. Для остальных режимов — no-op.
     *
     * <p>{@code w} инициализируется маленькими случайными значениями
     * (Xavier-подобный масштаб от {@code D}), {@code bias = 0}. Маленький
     * {@code w} важен: при больших весах softmax «насыщается» на первых
     * шагах и обучение застревает.
     */
    @Override
    public Layer initWeights(Activate activate) {
        if (mode != Mode.ATTENTION) return this;
        float limit = (float) Math.sqrt(6.0 / (D + 1));
        Random rnd = ThreadLocalRandom.current();
        for (int i = 0; i < D; i++) w[i] = rnd.nextFloat() * 2 * limit - limit;
        bias = 0f;
        return this;
    }

    // ==================== forward ====================

    /**
     * {@inheritDoc}
     *
     * <p>Без маски: все токены считаются настоящими.
     */
    @Override
    public float[] forward(float[] input, boolean training) {
        if (input.length != T * D)
            throw new IllegalArgumentException(
                    "TokenPoolingLayer.forward: ожидалось " + (T * D) + ", получено " + input.length);
        float[] allOnes = new float[T];
        Arrays.fill(allOnes, 1f);
        return forwardMasked(input, allOnes, training);
    }

    /**
     * Прямой проход с маской.
     *
     * @param input {@code seq(T, d)} — плоская раскладка {@code [t*D + d]}
     * @param mask  {@code float[T]} — {@code 1} настоящий, {@code 0} PAD
     * @param training не используется (у слоя нет dropout/BatchNorm)
     * @return {@code flat(D)}
     */
    @Override
    public float[] forwardMasked(float[] input, float[] mask, boolean training) {
        if (input.length != T * D)
            throw new IllegalArgumentException(
                    "TokenPoolingLayer.forwardMasked: ожидалось " + (T * D) + ", получено " + input.length);
        if (mask.length != T)
            throw new IllegalArgumentException(
                    "TokenPoolingLayer.forwardMasked: маска длины " + T + ", получено " + mask.length);

        System.arraycopy(input, 0, lastInput, 0, T * D);
        System.arraycopy(mask, 0, lastMask, 0, T);

        switch (mode) {
            case LAST_VALID -> {
                int last = lastValidIndex(mask);
                if (last < 0) Arrays.fill(out, 0f);
                else System.arraycopy(input, last * D, out, 0, D);
            }
            case FIRST -> {
                // Позиция фиксирована: t = 0. Если это PAD — выход нулевой
                // (модель должна обеспечить настоящий CLS-токен на позиции 0).
                if (mask[0] > 0f) System.arraycopy(input, 0, out, 0, D);
                else Arrays.fill(out, 0f);
            }
            case MEAN -> {
                Arrays.fill(out, 0f);
                int count = 0;
                for (int t = 0; t < T; t++) if (mask[t] > 0f) {
                    int off = t * D;
                    for (int d = 0; d < D; d++) out[d] += input[off + d];
                    count++;
                }
                if (count > 0) {
                    float inv = 1f / count;
                    for (int d = 0; d < D; d++) out[d] *= inv;
                }
            }
            case MAX -> {
                Arrays.fill(out, Float.NEGATIVE_INFINITY);
                for (int t = 0; t < T; t++) if (mask[t] > 0f) {
                    int off = t * D;
                    for (int d = 0; d < D; d++)
                        if (input[off + d] > out[d]) out[d] = input[off + d];
                }
                for (int d = 0; d < D; d++)
                    if (out[d] == Float.NEGATIVE_INFINITY) out[d] = 0f;
            }
            case ATTENTION -> forwardAttention(input, mask);
        }
        return out;
    }

    /**
     * ATTENTION: {@code score[t] = <w, h_t> + bias}, softmax по настоящим,
     * взвешенная сумма. PAD-токены получают {@code -inf} в скоре и нулевой
     * вес в softmax. Если все PAD — выход нулевой, {@code attn} тоже нули
     * (backward тогда ничего не накопит).
     */
    private void forwardAttention(float[] input, float[] mask) {
        Arrays.fill(out, 0f);

        // 1) Скоры и max-trick для численной устойчивости softmax.
        //    PAD → -inf, чтобы softmax дал им ровно 0 (а не «немного»).
        float maxScore = Float.NEGATIVE_INFINITY;
        for (int t = 0; t < T; t++) {
            if (mask[t] <= 0f) {
                scores[t] = Float.NEGATIVE_INFINITY;
                continue;
            }
            int off = t * D;
            float s = bias;
            for (int d = 0; d < D; d++) s += w[d] * input[off + d];
            scores[t] = s;
            if (s > maxScore) maxScore = s;
        }

        // 2) softmax по настоящим токенам.
        if (maxScore == Float.NEGATIVE_INFINITY) {
            // Все PAD — вырожденный случай: выход нулевой, веса нулевые.
            Arrays.fill(attn, 0f);
            return;
        }
        float sum = 0f;
        for (int t = 0; t < T; t++) {
            if (scores[t] == Float.NEGATIVE_INFINITY) {
                attn[t] = 0f;
            } else {
                attn[t] = (float) Math.exp(scores[t] - maxScore);
                sum += attn[t];
            }
        }
        if (sum > 0f) {
            float inv = 1f / sum;
            for (int t = 0; t < T; t++) attn[t] *= inv;
        }

        // 3) Взвешенная сумма векторов.
        for (int t = 0; t < T; t++) {
            float a = attn[t];
            if (a == 0f) continue;
            int off = t * D;
            for (int d = 0; d < D; d++) out[d] += a * input[off + d];
        }
    }

    /** Индекс последнего настоящего токена или {@code -1}, если все PAD. */
    private static int lastValidIndex(float[] mask) {
        for (int t = mask.length - 1; t >= 0; t--) if (mask[t] > 0f) return t;
        return -1;
    }

    // ==================== backward ====================

    /**
     * {@inheritDoc}
     *
     * <p>Без маски: все токены считаются настоящими.
     */
    @Override
    public float[] backward(float[] deltaNext) {
        if (deltaNext.length != D)
            throw new IllegalArgumentException(
                    "TokenPoolingLayer.backward: ожидалось " + D + ", получено " + deltaNext.length);
        return backwardMasked(deltaNext, lastMask);
    }

    /**
     * Обратный проход с маской.
     *
     * @param deltaNext {@code flat(D)} — градиент по выходу
     * @param mask      {@code float[T]} — та же маска, что в forward
     * @return {@code seq(T, d)} — градиент по входу
     */
    @Override
    public float[] backwardMasked(float[] deltaNext, float[] mask) {
        if (deltaNext.length != D)
            throw new IllegalArgumentException(
                    "TokenPoolingLayer.backwardMasked: ожидалось " + D + ", получено " + deltaNext.length);
        if (mask.length != T)
            throw new IllegalArgumentException(
                    "TokenPoolingLayer.backwardMasked: маска длины " + T + ", получено " + mask.length);

        Arrays.fill(prevDelta, 0f);
        switch (mode) {
            case LAST_VALID -> {
                int last = lastValidIndex(mask);
                if (last >= 0) System.arraycopy(deltaNext, 0, prevDelta, last * D, D);
            }
            case FIRST -> {
                if (mask[0] > 0f) System.arraycopy(deltaNext, 0, prevDelta, 0, D);
            }
            case MEAN -> {
                int count = 0;
                for (int t = 0; t < T; t++) if (mask[t] > 0f) count++;
                if (count > 0) {
                    float inv = 1f / count;
                    for (int t = 0; t < T; t++) if (mask[t] > 0f) {
                        int off = t * D;
                        for (int d = 0; d < D; d++) prevDelta[off + d] = deltaNext[d] * inv;
                    }
                }
            }
            case MAX -> {
                for (int d = 0; d < D; d++) {
                    int bestT = -1;
                    float best = Float.NEGATIVE_INFINITY;
                    for (int t = 0; t < T; t++) if (mask[t] > 0f) {
                        float v = lastInput[t * D + d];
                        if (v > best) { best = v; bestT = t; }
                    }
                    if (bestT >= 0) prevDelta[bestT * D + d] = deltaNext[d];
                }
            }
            case ATTENTION -> backwardAttention(deltaNext, mask);
        }
        return prevDelta;
    }

    /**
     * ATTENTION backward.
     *
     * <p>Пусть {@code a_t = softmax(score)_t}, {@code h_t} — входной вектор
     * токена {@code t}, {@code o = Σ_t a_t h_t}. Тогда:
     * <pre>
     *   dL/dh_t = a_t · dL/do
     *   dL/da_t = &lt;dL/do, h_t&gt;
     *   dL/dscore_t = a_t · (dL/da_t − Σ_j a_j · dL/da_j)
     *   dL/dw    += Σ_t dL/dscore_t · h_t
     *   dL/dbias += Σ_t dL/dscore_t
     * </pre>
     * PAD-токены имеют {@code a_t = 0} и {@code dL/dscore_t = 0}, поэтому
     * не влияют ни на вход, ни на {@code w}/{@code bias}.
     */
    private void backwardAttention(float[] deltaNext, float[] mask) {
        // Если forward не оставил весов (все PAD) — ничего не накапливаем.
        if (attn == null) return;

        // 1) dL/da_t = <dL/do, h_t>; dL/dh_t = a_t · dL/do (для настоящих)
        float[] dA = new float[T];
        float dot = 0f;
        for (int t = 0; t < T; t++) {
            if (mask[t] <= 0f) continue;
            int off = t * D;
            float s = 0f;
            for (int d = 0; d < D; d++) s += deltaNext[d] * lastInput[off + d];
            dA[t] = s;
            dot += attn[t] * s;

            float a = attn[t];
            if (a != 0f) {
                for (int d = 0; d < D; d++) prevDelta[off + d] = a * deltaNext[d];
            }
        }

        // 2) dL/dscore_t = a_t · (dA_t − dot); накапливаем dW и dbias.
        for (int t = 0; t < T; t++) {
            if (mask[t] <= 0f) continue;
            float a = attn[t];
            if (a == 0f) continue;
            float ds = a * (dA[t] - dot);
            int off = t * D;
            for (int d = 0; d < D; d++) wGrad[d] += ds * lastInput[off + d];
            biasGrad += ds;
        }
    }

    // ==================== параметры / градиенты ====================
    //
    // Все четыре метода имеют смысл только для ATTENTION; для остальных
    // режимов поля w/wGrad == null и методы — no-op (как и раньше).

    /** Обнуляет накопители градиентов ATTENTION. Для остальных режимов — no-op. */
    @Override
    public void zeroGrad() {
        if (mode != Mode.ATTENTION) return;
        Arrays.fill(wGrad, 0f);
        biasGrad = 0f;
    }

    /**
     * Шаг оптимизатора для ATTENTION. {@code w} обновляется через
     * матричную версию {@link Optimizer#updateWeights} (обёртка
     * {@code [1][D]} — тот же массив, не копия), {@code bias} — через
     * {@link Optimizer#updateBiases}.
     *
     * <p><b>Замечание.</b> Два отдельных вызова (а не {@code update(...)})
     * использованы намеренно: у {@code bias} здесь скаляр, а не массив,
     * и оборачивать его в {@code float[1]} ради единого вызова дороже,
     * чем вызвать {@code updateBiases} с временным массивом. Оптимизаторы
     * с общим счётчиком шага (AdamW) в этом месте увидят два шага вместо
     * одного — приемлемо, т.к. параметров мало и они независимы.
     */
    @Override
    public void update(Optimizer optimizer, int batchSize) {
        if (mode != Mode.ATTENTION) return;
        if (batchSize <= 0) throw new IllegalArgumentException("batchSize должен быть > 0, получено " + batchSize);
        optimizer.updateWeights(new float[][]{ w }, new float[][]{ wGrad }, batchSize);
        optimizer.updateBiases(new float[]{ bias }, new float[]{ biasGrad }, batchSize);
    }

    /** Сумма квадратов градиентов ATTENTION. Для остальных режимов — {@code 0}. */
    @Override
    public double gradNormSq() {
        if (mode != Mode.ATTENTION) return 0.0;
        double s = 0;
        for (float v : wGrad) s += v * v;
        s += biasGrad * biasGrad;
        return s;
    }

    /** Масштабирует градиенты ATTENTION. Для остальных режимов — no-op. */
    @Override
    public void scaleGrad(float scale) {
        if (mode != Mode.ATTENTION) return;
        for (int i = 0; i < wGrad.length; i++) wGrad[i] *= scale;
        biasGrad *= scale;
    }

    // ==================== сериализация ====================

    /**
     * Пишет формы (через {@code super.save}), затем {@code mode.ordinal()},
     * и — только для ATTENTION — {@code w} и {@code bias}.
     *
     * <p>Порядок чтения в {@link #load(DataInputStream)} симметричен:
     * сначала {@code mode}, затем (если ATTENTION) параметры. Старые файлы
     * {@code LAST}/{@code MEAN}/{@code MAX} содержат только {@code mode}
     * и читаются корректно.
     */
    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);
        out.writeByte(mode.ordinal());
        if (mode == Mode.ATTENTION) {
            out.writeFloat(bias);
            for (float v : w) out.writeFloat(v);
        }
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);
        int ord = in.readByte();
        if (ord < 0 || ord >= Mode.values().length)
            throw new IOException("TokenPoolingLayer: некорректный код режима " + ord);
        mode = Mode.values()[ord];
        allocateBuffers();
        if (mode == Mode.ATTENTION) {
            bias = in.readFloat();
            for (int i = 0; i < w.length; i++) w[i] = in.readFloat();
        }
    }
}