package rawnet.code.net;

import rawnet.code.net.core.activate.Sigmoid;
import rawnet.code.net.core.loss.BinaryCrossEntropy;
import rawnet.code.net.core.loss.Loss;
import rawnet.code.net.core.loss.Mse;
import rawnet.code.net.core.loss.SequenceCrossEntropy;
import rawnet.code.net.core.loss.SoftmaxCrossEntropy;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Юнит-тесты {@link Loss}: формулы, композиция с активацией, masked-версии.
 *
 * <p><b>Почему это важно.</b> Ошибка в {@code derivative} не роняет
 * обучение — оно «сходится», но не туда. Тесты ловят опечатки в
 * формулах и несовпадение {@code loss}/{@code derivative}.
 */
class LossTest {

    // =====================================================================
    // ==================== BCE + Sigmoid ==================================
    // =====================================================================

    /**
     * Стандарт: {@code BCE.derivative(p, t) * Sigmoid.derivative(z) = p - t},
     * где {@code p = sigmoid(z)}. Это гарантирует, что в паре
     * {@code Sigmoid + BCE} градиент по {@code z} — {@code p - t}
     * (быстро и численно устойчиво).
     */
    @Test
    void bceDerivativeComposesWithSigmoid() {
        BinaryCrossEntropy bce = new BinaryCrossEntropy();
        Sigmoid sigmoid = new Sigmoid();

        float[] z = {-2f, -1f, 0f, 1f, 2f};
        float[] p = new float[z.length];
        sigmoid.apply(z, p);

        float[] t = {0f, 1f, 1f, 0f, 1f};
        float[] dLdp = new float[p.length];
        float[] dLdz = new float[p.length];
        bce.derivative(p, t, dLdp);

        // Sigmoid.derivative принимает (input=z, output=p, deriv)
        float[] sigDeriv = new float[p.length];
        sigmoid.derivative(z, p, sigDeriv);

        for (int i = 0; i < z.length; i++) {
            dLdz[i] = dLdp[i] * sigDeriv[i];
            assertEquals(p[i] - t[i], dLdz[i], 1e-4f,
                    "BCE+Sigmoid: dL/dz должен быть (p - t) в z=" + z[i]);
        }
    }

    /** BCE loss: при {@code p = t} loss ≈ 0; при {@code p = 1-t} — большое. */
    @Test
    void bceLossValues() {
        BinaryCrossEntropy bce = new BinaryCrossEntropy();

        float[] p = {1f, 0f};
        float[] t = {1f, 0f};
        assertTrue(bce.loss(p, t) < 1e-6f, "BCE при точном попадании ≈ 0");

        float[] p2 = {0f, 1f};
        assertTrue(bce.loss(p2, t) > 10f, "BCE при полной ошибке — большое");
    }

    // =====================================================================
    // ==================== SoftmaxCrossEntropy ============================
    // =====================================================================

    /**
     * {@code dL/dz = p - t}, где {@code p = softmax(z)}. Никакого
     * деления на {@code p} — численно устойчиво.
     */
    @Test
    void softmaxCrossEntropyGradientIsPminusT() {
        SoftmaxCrossEntropy loss = new SoftmaxCrossEntropy();

        float[] logits = {1.0f, 2.0f, 0.5f};
        float[] target = {0f, 1f, 0f};

        float[] p = SoftmaxCrossEntropy.softmax(logits);

        float[] dLdz = new float[logits.length];
        loss.derivative(logits, target, dLdz);

        for (int i = 0; i < logits.length; i++) {
            assertEquals(p[i] - target[i], dLdz[i], 1e-6f,
                    "SoftmaxCE: dL/dz = p - t в позиции " + i);
        }
    }

    /** Sum of softmax = 1, все в (0, 1). */
    @Test
    void softmaxProperties() {
        float[] logits = {1f, 2f, 3f, 4f};
        float[] p = SoftmaxCrossEntropy.softmax(logits);

        float sum = 0f;
        for (float v : p) {
            assertTrue(v > 0f && v < 1f, "softmax в (0, 1)");
            sum += v;
        }
        assertEquals(1f, sum, 1e-5f, "сумма softmax = 1");
    }

    // =====================================================================
    // ==================== Mse ============================================
    // =====================================================================

    /**
     * {@code Mse.loss = mean((p-t)^2)}, {@code Mse.derivative = p - t}
     * (без {@code 2/n} — стандарт для библиотек: градиент «уже усреднён»
     * по батчу в {@code Optimizer.update(..., batchSize)}).
     */
    @Test
    void mseLossAndDerivative() {
        Mse mse = new Mse();

        float[] p = {1f, 2f, 3f};
        float[] t = {0f, 0f, 0f};

        // loss = (1+4+9)/3 = 14/3
        assertEquals(14f / 3f, mse.loss(p, t), 1e-5f);

        float[] d = new float[3];
        mse.derivative(p, t, d);
        for (int i = 0; i < p.length; i++) {
            assertEquals(p[i] - t[i], d[i], 1e-6f,
                    "Mse.derivative = p - t в позиции " + i);
        }
    }

    // =====================================================================
    // ==================== MaskAcceptLoss (SequenceCrossEntropy) ==========
    // =====================================================================

    /**
     * {@link SequenceCrossEntropy#derivativeMasked} с маской:
     * PAD-позиции (mask=0) дают нулевой градиент, настоящие — обычный
     * {@code softmax - onehot} (делённый на число валидных).
     */
    @Test
    void sequenceCrossEntropyMasked() {
        SequenceCrossEntropy loss = new SequenceCrossEntropy();

        int T = 3;
        int vocab = 4;
        float[] logits = new float[T * vocab];
        float[] target = new float[T];
        float[] mask   = new float[T];

        // t=0: класс 1, t=1: PAD (mask=0), t=2: класс 3
        target[0] = 1; mask[0] = 1;
        target[1] = 0; mask[1] = 0;   // PAD
        target[2] = 3; mask[2] = 1;

        // Произвольные логиты
        for (int i = 0; i < logits.length; i++) logits[i] = 0.1f * i;

        float[] d = new float[logits.length];
        loss.derivativeMasked(logits, target, d, mask);

        // t=1 — PAD: все градиенты 0
        for (int v = 0; v < vocab; v++) {
            assertEquals(0f, d[1 * vocab + v], 1e-6f,
                    "PAD-позиция: градиент = 0 в v=" + v);
        }

        // t=0 и t=2 — не PAD: сумма по v = 0 (softmax - onehot, сумма = 0)
        for (int t : new int[]{0, 2}) {
            float sum = 0f;
            for (int v = 0; v < vocab; v++) sum += d[t * vocab + v];
            assertEquals(0f, sum, 1e-5f,
                    "не-PAD позиция t=" + t + ": сумма градиентов = 0");
        }
    }

    /** {@link SequenceCrossEntropy} с {@code mask = null} = без маски. */
    @Test
    void sequenceCrossEntropyWithoutMask() {
        SequenceCrossEntropy loss = new SequenceCrossEntropy();

        int T = 2, vocab = 3;
        float[] logits = new float[T * vocab];
        float[] target = {0, 2};
        float[] mask = null;

        for (int i = 0; i < logits.length; i++) logits[i] = 0.1f * i;

        float[] dWith = new float[logits.length];
        float[] dWithout = new float[logits.length];
        loss.derivativeMasked(logits, target, dWith, mask);
        loss.derivative(logits, target, dWithout);

        assertArrayEquals(dWithout, dWith, 1e-6f,
                "mask = null = без маски");
    }

    // =====================================================================
    // ==================== Общие проверки =================================
    // =====================================================================

    @Test
    void numericGradientMseSigmoid() {
        Mse mse = new Mse();
        Sigmoid sigmoid = new Sigmoid();

        float[] z = {-1.5f, -0.5f, 0.5f, 1.5f};
        float[] t = {0f, 1f, 1f, 0f};

        float[] p = new float[z.length];
        sigmoid.apply(z, p);

        // dL/dz аналитически: (2/N) * (p - t) * sigmoid'(z).
        // Mse.derivative возвращает (p - t) — БЕЗ 2/N (соглашение: loss усреднён,
        // derivative — «сумма без 2»). Учитываем коэффициент 2/N.
        float[] dLdp = new float[z.length];
        float[] sigDeriv = new float[z.length];
        mse.derivative(p, t, dLdp);
        sigmoid.derivative(z, p, sigDeriv);

        float scale = 2f / z.length;
        float[] analytic = new float[z.length];
        for (int i = 0; i < z.length; i++)
            analytic[i] = dLdp[i] * sigDeriv[i] * scale;

        // Численно — по mse.loss (он усреднён, поэтому численный = d(loss)/dp)
        float eps = 1e-3f;
        for (int i = 0; i < z.length; i++) {
            float[] zPlus = z.clone(); zPlus[i] += eps;
            float[] zMinus = z.clone(); zMinus[i] -= eps;
            float[] pPlus = new float[z.length];
            float[] pMinus = new float[z.length];
            sigmoid.apply(zPlus, pPlus);
            sigmoid.apply(zMinus, pMinus);
            float numeric = (mse.loss(pPlus, t) - mse.loss(pMinus, t)) / (2 * eps);
            assertEquals(numeric, analytic[i], 1e-3f,
                    "численный градиент не совпал с аналитическим в i=" + i);
        }
    }
}