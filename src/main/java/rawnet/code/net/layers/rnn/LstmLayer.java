package rawnet.code.net.layers.rnn;

import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;
import rawnet.code.net.TensorShape;
import rawnet.code.net.core.activate.Activate;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.contract.MaskConsumer;
import rawnet.code.net.core.optimizator.Optimizer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * LSTM (Long Short-Term Memory, Hochreiter & Schmidhuber, 1997) с BPTT.
 *
 * <pre>
 *   i_t = σ(Wi·x_t + Ui·h_{t-1} + bi)            // input gate
 *   f_t = σ(Wf·x_t + Uf·h_{t-1} + bf)            // forget gate
 *   o_t = σ(Wo·x_t + Uo·h_{t-1} + bo)            // output gate
 *   g_t = tanh(Wg·x_t + Ug·h_{t-1} + bg)         // candidate (cell input)
 *   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t              // cell state
 *   h_t = o_t ⊙ tanh(c_t)                        // hidden state
 * </pre>
 *
 * <p>Простыми словами: у LSTM два состояния — {@code c_t} («долгая память»,
 * cell state) и {@code h_t} («короткая память», hidden state). Три гейта
 * управляют тем, что <b>записать</b> в память ({@code i}), что <b>забыть</b>
 * ({@code f}) и что <b>выдать</b> наружу ({@code o}). Ключевое отличие от
 * {@link ElmanRnnLayer}/{@link GruLayer} — аддитивное обновление памяти
 * {@code c_t = f ⊙ c + i ⊙ g}: градиент по {@code c} не проходит через
 * нелинейность, поэтому LSTM лучше держит долгие зависимости и меньше
 * страдает от затухания градиента.
 *
 * <h2>Маска</h2>
 * Реализует {@link MaskConsumer}: на PAD-шагах ({@code mask[t] == 0})
 * <b>оба</b> состояния не обновляются ({@code h_t = h_{t-1}},
 * {@code c_t = c_{t-1}}), гейты не считаются, {@code weightGrad}/
 * {@code biasGrad}/{@code prevDelta} не накапливаются, а градиент проходит
 * сквозь шаг ({@code dhPrev = dhCurr}, {@code dcPrev = dcCurr}). Это
 * позволяет использовать слой внутри {@link BidirectionalLayer}: обратное
 * направление пропускает PAD с конца.
 *
 * <h2>Формат входа/выхода</h2>
 * <ul>
 *   <li>вход: {@code seq(seqLen, features)};</li>
 *   <li>выход при {@code returnSequences = true}:
 *       {@code seq(seqLen, hidden)} — {@code h_t} на каждом шаге;</li>
 *   <li>выход при {@code returnSequences = false}:
 *       {@code flat(hidden)} — {@code h_{T-1}} (последнее состояние).</li>
 * </ul>
 *
 * <h2>Параметры</h2>
 * Одна матрица {@code [4·hidden][features + hidden]} (блоки
 * {@code i, f, o, g}; в каждой строке сначала {@code features} входных
 * весов, затем {@code hidden} рекуррентных) и {@code biases[4·hidden]}.
 * Инициализация — Xavier/Glorot. По аналогии с известным трюком
 * (Gers et al., 2000) {@code bf} инициализируется в {@code 1}, чтобы в
 * начале обучения сеть «помнила» и не забывала сразу — это заметно
 * ускоряет сходимость; остальные смещения — {@code 0}.
 *
 * <h2>Градиентный клипинг</h2>
 * {@code clipGradNorm} ограничивает L2-норму {@code dhPrev} на каждом
 * шаге BPTT. Это не полный clip по норме всех градиентов, а локальный
 * (как в {@link ElmanRnnLayer} и {@link GruLayer}); для глобального
 * используйте {@code Model.train(..., clipNorm, ...)}.
 */
@Accessors(chain = true)
public class LstmLayer extends Layer implements MaskConsumer {

    @RequiredArgsConstructor
    public static class Dto extends Layer.Dto {
        public final int seqLen;
        public final int hidden;
        public final boolean returnSequences;
        public float clipGradNorm = 1f;

        public Dto setClipGradNorm(float clipGradNorm) {
            this.clipGradNorm = clipGradNorm;
            return this;
        }

        @Override
        public Layer create(TensorShape inputShape) {
            return new LstmLayer()
                    .setSeqLen(seqLen)
                    .setHidden(hidden)
                    .setReturnSequences(returnSequences)
                    .setClipGradNorm(clipGradNorm)
                    .setInputShape(inputShape)
                    .setOutputShape(computeOutputShape(inputShape, seqLen, hidden, returnSequences))
                    .allocateBuffers().initWeights(null);
        }
    }

    @Setter private int seqLen;
    @Setter private int hidden;
    @Setter private boolean returnSequences;
    @Setter private float clipGradNorm = 1f;

    private int features;

    // ==================== параметры ====================
    //
    // Одна матрица weights[4·hidden][features + hidden]. Строки разбиты на
    // 4 блока по hidden: [0, H) — i, [H, 2H) — f, [2H, 3H) — o,
    // [3H, 4H) — g. В каждой строке первые features столбцов — веса входа,
    // следующие hidden — рекуррентные веса. Такая упаковка позволяет
    // обновлять всё одним вызовом Optimizer.update(...).

    /** {@code [4·hidden][features + hidden]} */
    float[][] weights;
    /** {@code [4·hidden]} */
    float[] biases;

    private float[][] weightGrad;
    private float[] biasGrad;

    // ==================== кэши forward ====================

    private transient float[] inputRef;

    /** {@code hCache[t]} — {@code h_{t-1}} (hCache[0] = 0). */
    private float[][] hCache;
    /** {@code cCache[t]} — {@code c_{t-1}} (cCache[0] = 0). */
    private float[][] cCache;
    /** {@code tanhCCache[t]} — {@code tanh(c_t)} (нужен backward для dh). */
    private float[][] tanhCCache;
    /** {@code gateCache[t]} — {@code [i | f | o | g]} длины {@code 4·hidden}. */
    private float[][] gateCache;
    /** Валидность шага: {@code true} — настоящий, {@code false} — PAD. */
    private boolean[] validCache;

    // ==================== буферы backward ====================

    private float[] output;
    private float[] prevDelta;
    private float[] dhCurr;
    private float[] dhPrev;
    private float[] dcCurr;
    private float[] dcPrev;
    private float[] dGates;    // [4·hidden], переиспользуется на каждом шаге
    private float[] dxBuf;     // [features]

    public LstmLayer() {}

    public static TensorShape computeOutputShape(TensorShape in, int seqLen,
                                                 int hidden, boolean returnSequences) {
        if (in == null) throw new IllegalArgumentException("LstmLayer: inputShape == null");
        if (!in.isSequence())
            throw new IllegalArgumentException("LstmLayer: ожидался sequence-вход, получено " + in);
        if (seqLen < 1) throw new IllegalArgumentException("LstmLayer: seqLen >= 1, получено " + seqLen);
        if (hidden < 1) throw new IllegalArgumentException("LstmLayer: hidden >= 1, получено " + hidden);
        if (in.seqLen() != seqLen)
            throw new IllegalArgumentException(
                    "LstmLayer: seqLen слоя (" + seqLen + ") != seqLen формы (" + in.seqLen() + ")");
        return returnSequences
                ? TensorShape.seq(seqLen, hidden)
                : TensorShape.flat(hidden);
    }

    public Layer allocateBuffers() {
        if (inputShape == null || outputShape == null)
            throw new IllegalStateException("LstmLayer.create: inputShape и outputShape должны быть заданы");
        if (!inputShape.isSequence())
            throw new IllegalArgumentException("LstmLayer.create: ожидался sequence-вход, получено " + inputShape);
        if (inputShape.seqLen() != seqLen)
            throw new IllegalArgumentException(
                    "LstmLayer.create: inputShape.seqLen=" + inputShape.seqLen()
                            + " != seqLen=" + seqLen);
        if (hidden < 1)
            throw new IllegalArgumentException("LstmLayer.create: hidden >= 1, получено " + hidden);

        this.features = inputShape.channels();

        final int rows = 4 * hidden;
        final int cols = features + hidden;

        this.weights    = new float[rows][cols];
        this.biases     = new float[rows];
        this.weightGrad = new float[rows][cols];
        this.biasGrad   = new float[rows];

        this.hCache     = new float[seqLen + 1][hidden];
        this.cCache     = new float[seqLen + 1][hidden];
        this.tanhCCache = new float[seqLen][hidden];
        this.gateCache  = new float[seqLen][rows];
        this.validCache = new boolean[seqLen];

        this.output    = new float[outputShape.size()];
        this.prevDelta = new float[seqLen * features];
        this.dhCurr    = new float[hidden];
        this.dhPrev    = new float[hidden];
        this.dcCurr    = new float[hidden];
        this.dcPrev    = new float[hidden];
        this.dGates    = new float[rows];
        this.dxBuf     = new float[features];
        return this;
    }

    @Override
    public Layer initWeights(Activate activate) {
        final int rows = 4 * hidden;
        final int cols = features + hidden;
        float limit = (float) Math.sqrt(6.0 / (cols + hidden));
        Random rnd = ThreadLocalRandom.current();
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                weights[i][j] = rnd.nextFloat() * 2 * limit - limit;

        // Forget-gate bias = 1 (Gers et al., 2000): в начале обучения «помнить»
        // дешевле, чем «забывать», иначе LSTM часто застревает в «забыл всё».
        Arrays.fill(biases, 0f);
        for (int k = 0; k < hidden; k++) biases[hidden + k] = 1f;   // блок f

        return this;
    }

    // ==================== forward ====================

    @Override
    public float[] forward(float[] input, boolean training) {
        return forwardMasked(input, null, training);
    }

    @Override
    public float[] forwardMasked(float[] input, float[] mask, boolean training) {
        if (input.length != inputShape.size())
            throw new IllegalArgumentException(
                    "LstmLayer.forwardMasked: ожидалось " + inputShape.size() + ", получено " + input.length);
        if (mask != null && mask.length != seqLen)
            throw new IllegalArgumentException(
                    "LstmLayer.forwardMasked: маска длины " + seqLen + ", получено " + mask.length);

        this.inputRef = input;
        Arrays.fill(hCache[0], 0f);
        Arrays.fill(cCache[0], 0f);

        final int H = hidden;

        for (int t = 0; t < seqLen; t++) {
            boolean valid = (mask == null) || (mask[t] > 0f);
            validCache[t] = valid;

            float[] hPrev = hCache[t];
            float[] cPrev = cCache[t];
            float[] hCurr = hCache[t + 1];
            float[] cCurr = cCache[t + 1];

            if (!valid) {
                // PAD: оба состояния «замораживаются» (копия, не алиас).
                System.arraycopy(hPrev, 0, hCurr, 0, H);
                System.arraycopy(cPrev, 0, cCurr, 0, H);
                if (returnSequences) System.arraycopy(hCurr, 0, output, t * H, H);
                continue;
            }

            float[] gates = gateCache[t];
            final int xOff = t * features;

            // Считаем все 4 блока гейтов одним проходом.
            for (int k = 0; k < H; k++) {
                float ai = biases[k]           + dot(weights[k],           input, xOff, hPrev);
                float af = biases[H + k]       + dot(weights[H + k],       input, xOff, hPrev);
                float ao = biases[2 * H + k]   + dot(weights[2 * H + k],   input, xOff, hPrev);
                float ag = biases[3 * H + k]   + dot(weights[3 * H + k],   input, xOff, hPrev);

                float i = sigmoid(ai);
                float f = sigmoid(af);
                float o = sigmoid(ao);
                float g = tanh(ag);

                gates[k]           = i;
                gates[H + k]       = f;
                gates[2 * H + k]   = o;
                gates[3 * H + k]   = g;

                float c = f * cPrev[k] + i * g;
                cCurr[k] = c;
                float tc = tanh(c);
                tanhCCache[t][k] = tc;
                hCurr[k] = o * tc;
            }

            if (returnSequences) System.arraycopy(hCurr, 0, output, t * H, H);
        }

        if (!returnSequences) System.arraycopy(hCache[seqLen], 0, output, 0, H);
        return output;
    }

    /** Скалярное произведение: {@code Σ_j w[j]·input[xOff + j] + Σ_j w[features+j]·hPrev[j]}. */
    private float dot(float[] w, float[] input, int xOff, float[] hPrev) {
        float s = 0f;
        for (int j = 0; j < features; j++) s += w[j] * input[xOff + j];
        for (int j = 0; j < hidden;   j++) s += w[features + j] * hPrev[j];
        return s;
    }

    // ==================== backward ====================

    @Override
    public float[] backward(float[] deltaNext) {
        return backwardMasked(deltaNext, null);
    }

    @Override
    public float[] backwardMasked(float[] deltaNext, float[] mask) {
        if (deltaNext.length != outputShape.size())
            throw new IllegalArgumentException(
                    "LstmLayer.backwardMasked: ожидалось " + outputShape.size()
                            + ", получено " + deltaNext.length);
        if (mask != null && mask.length != seqLen)
            throw new IllegalArgumentException(
                    "LstmLayer.backwardMasked: маска длины " + seqLen + ", получено " + mask.length);
        if (inputRef == null)
            throw new IllegalStateException("LstmLayer.backwardMasked: forward не был вызван");

        final int H = hidden;

        Arrays.fill(prevDelta, 0f);
        Arrays.fill(dhCurr, 0f);
        Arrays.fill(dcCurr, 0f);

        for (int t = seqLen - 1; t >= 0; t--) {
            boolean valid = (mask == null) || (mask[t] > 0f);
            final boolean last = (t == seqLen - 1);
            final int dOff = returnSequences ? t * H : 0;

            // Внешний градиент по h_t.
            if (returnSequences) {
                for (int k = 0; k < H; k++) dhCurr[k] += deltaNext[dOff + k];
            } else if (last) {
                for (int k = 0; k < H; k++) dhCurr[k] += deltaNext[k];
            }

            if (!valid) {
                // PAD: градиент по h и c проходит сквозь без изменений.
                // dhCurr и dcCurr уже содержат накопленный градиент от t+1;
                // на следующем (более раннем) шаге он должен дойти до h_{t-1}, c_{t-1}.
                float[] tmp;
                tmp = dhCurr; dhCurr = dhPrev; dhPrev = tmp;
                tmp = dcCurr; dcCurr = dcPrev; dcPrev = tmp;
                System.arraycopy(dhPrev, 0, dhCurr, 0, H);
                System.arraycopy(dcPrev, 0, dcCurr, 0, H);
                continue;
            }

            float[] gates = gateCache[t];
            float[] hPrev = hCache[t];
            float[] cPrev = cCache[t];
            float[] tc    = tanhCCache[t];

            // Восстанавливаем i, f, o, g из кэша.
            Arrays.fill(dGates, 0f);
            for (int k = 0; k < H; k++) {
                float i = gates[k];
                float f = gates[H + k];
                float o = gates[2 * H + k];
                float g = gates[3 * H + k];

                float dh = dhCurr[k];
                float dc = dcCurr[k];

                // h = o * tanh(c)
                float dO  = dh * tc[k];
                float dTc = dh * o;
                // d(tanh(c)) / dc = 1 - tanh^2(c)
                float dC  = dc + dTc * (1f - tc[k] * tc[k]);

                // c = f * cPrev + i * g
                float dF = dC * cPrev[k];
                float dI = dC * g;
                float dG = dC * i;

                // Через сигмоиды/танх: dz = dPreAct * a * (1 - a) для σ;
                // dz = dPreAct * (1 - a^2) для tanh.
                dGates[k]         = dI * i * (1f - i);
                dGates[H + k]     = dF * f * (1f - f);
                dGates[2 * H + k] = dO * o * (1f - o);
                dGates[3 * H + k] = dG * (1f - g * g);

                // Градиент по предыдущему c и h.
                dcPrev[k] = dC * f;
                // hPrev получает вклад от всех 4 блоков через U-часть.
                dhPrev[k] = 0f;
            }

            // Градиенты по U-части (рекуррентные) и по hPrev.
            for (int k = 0; k < H; k++) {
                float dzI = dGates[k];
                float dzF = dGates[H + k];
                float dzO = dGates[2 * H + k];
                float dzG = dGates[3 * H + k];
                for (int j = 0; j < H; j++) {
                    dhPrev[j] += weights[k][features + j]         * dzI;
                    dhPrev[j] += weights[H + k][features + j]     * dzF;
                    dhPrev[j] += weights[2 * H + k][features + j] * dzO;
                    dhPrev[j] += weights[3 * H + k][features + j] * dzG;
                }
            }

            // Накопление градиентов параметров + dx_t.
            final int xOff = t * features;
            Arrays.fill(dxBuf, 0f);
            for (int k = 0; k < H; k++) {
                float dzI = dGates[k];
                float dzF = dGates[H + k];
                float dzO = dGates[2 * H + k];
                float dzG = dGates[3 * H + k];

                // bias
                biasGrad[k]         += dzI;
                biasGrad[H + k]     += dzF;
                biasGrad[2 * H + k] += dzO;
                biasGrad[3 * H + k] += dzG;

                // weights (W-часть по x_t и U-часть по h_{t-1})
                float[] wI = weightGrad[k];
                float[] wF = weightGrad[H + k];
                float[] wO = weightGrad[2 * H + k];
                float[] wG = weightGrad[3 * H + k];

                for (int j = 0; j < features; j++) {
                    float xj = inputRef[xOff + j];
                    wI[j] += dzI * xj;
                    wF[j] += dzF * xj;
                    wO[j] += dzO * xj;
                    wG[j] += dzG * xj;
                }
                for (int j = 0; j < H; j++) {
                    float hj = hPrev[j];
                    wI[features + j] += dzI * hj;
                    wF[features + j] += dzF * hj;
                    wO[features + j] += dzO * hj;
                    wG[features + j] += dzG * hj;
                }

                // dx_t: вклад W-части от всех 4 блоков
                for (int j = 0; j < features; j++) {
                    dxBuf[j] += weights[k][j]           * dzI
                            + weights[H + k][j]       * dzF
                            + weights[2 * H + k][j]   * dzO
                            + weights[3 * H + k][j]   * dzG;
                }
            }
            System.arraycopy(dxBuf, 0, prevDelta, xOff, features);

            clipByNorm(dhPrev, clipGradNorm);

            // Свап буферов: dhPrev/dcPrev → dhCurr/dcCurr для следующего шага.
            float[] tmp;
            tmp = dhCurr; dhCurr = dhPrev; dhPrev = tmp;
            tmp = dcCurr; dcCurr = dcPrev; dcPrev = tmp;
        }
        return prevDelta;
    }

    // ==================== параметры ====================

    @Override
    public void zeroGrad() {
        for (float[] row : weightGrad) Arrays.fill(row, 0f);
        Arrays.fill(biasGrad, 0f);
    }

    @Override
    public void update(Optimizer optimizer, int batchSize) {
        if (batchSize <= 0)
            throw new IllegalArgumentException("batchSize должен быть > 0, получено " + batchSize);
        optimizer.update(weights, weightGrad, biases, biasGrad, batchSize);
    }

    @Override
    public double gradNormSq() {
        double sum = 0;
        for (float[] row : weightGrad) for (float v : row) sum += v * v;
        for (float v : biasGrad) sum += v * v;
        return sum;
    }

    @Override
    public void scaleGrad(float scale) {
        for (float[] row : weightGrad)
            for (int j = 0; j < row.length; j++) row[j] *= scale;
        for (int i = 0; i < biasGrad.length; i++) biasGrad[i] *= scale;
    }

    // ==================== сериализация ====================

    @Override
    public void save(DataOutputStream out) throws IOException {
        super.save(out);
        out.writeInt(seqLen);
        out.writeInt(hidden);
        out.writeBoolean(returnSequences);
        out.writeFloat(clipGradNorm);
        for (float[] row : weights) for (float v : row) out.writeFloat(v);
        for (float b : biases) out.writeFloat(b);
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        super.load(in);
        this.seqLen          = in.readInt();
        this.hidden          = in.readInt();
        this.returnSequences = in.readBoolean();
        this.clipGradNorm    = in.readFloat();
        allocateBuffers();
        for (float[] row : weights)
            for (int j = 0; j < row.length; j++) row[j] = in.readFloat();
        for (int i = 0; i < biases.length; i++) biases[i] = in.readFloat();
    }

    // ==================== хелперы ====================

    private static void clipByNorm(float[] v, float clipNorm) {
        if (clipNorm <= 0f) return;
        double sq = 0;
        for (float x : v) sq += x * x;
        double norm = Math.sqrt(sq);
        if (norm > clipNorm) {
            float scale = (float) (clipNorm / norm);
            for (int i = 0; i < v.length; i++) v[i] *= scale;
        }
    }

    // Быстрые аппроксимации, как в GruLayer (единый стиль по библиотеке).
    // Если нужна точность Math.tanh / Math.exp — заменить здесь.

    private static float tanh(float x) {
        if (x <= -3f) return -1f;
        if (x >=  3f) return  1f;
        final float x2 = x * x;
        return x * (27f + x2) / (27f + 9f * x2);
    }

    private static float sigmoid(float x) {
        if (x <= -8f) return 0f;
        if (x >=  8f) return 1f;
        return 0.5f + 0.5f * tanh(0.5f * x);
    }
}