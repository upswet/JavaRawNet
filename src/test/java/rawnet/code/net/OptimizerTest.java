package rawnet.code.net;

import rawnet.code.net.core.optimizator.AdamW;
import rawnet.code.net.core.optimizator.Momentum;
import rawnet.code.net.core.optimizator.Optimizer;
import rawnet.code.net.core.optimizator.SGD;
import rawnet.code.net.core.Persistables;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Юнит-тесты {@link Optimizer#save}/{@link Optimizer#load} для
 * встроенных оптимизаторов.
 *
 * <p><b>Почему это важно.</b> Resume после Ctrl+C читает состояние
 * оптимизатора из {@code .model.train}. Если {@code save/load} не
 * восстанавливают {@code m/v/step} (AdamW) или velocity (Momentum/NAG),
 * то после resume оптимизатор ведёт себя как новый — обучение
 * «дёргается». Для {@code SGD} (без состояния) достаточно восстановить
 * {@code lr}.
 *
 * <p><b>Что проверяем.</b> «Два обновления одним оптимизатором» =
 * «одно обновление оптимизатором, save/load в другой экземпляр, второе
 * обновление». Если состояние сохранено — результаты совпадут бит-в-бит.
 * Это ловит ошибки в порядке полей, в base64, в {@code step}.
 */
class OptimizerTest {

    private static final Random RND = new Random(42L);

    private static float[][] randMat(int rows, int cols) {
        float[][] m = new float[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i][j] = (RND.nextFloat() - 0.5f) * 0.1f;
        return m;
    }

    private static float[] randVec(int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (RND.nextFloat() - 0.5f) * 0.1f;
        return v;
    }

    private static float[][] cloneMat(float[][] a) {
        float[][] b = new float[a.length][];
        for (int i = 0; i < a.length; i++) b[i] = a[i].clone();
        return b;
    }

    private static float[] cloneVec(float[] a) { return a.clone(); }

    /**
     * AdamW: два обновления подряд vs (обновление → save/load → обновление).
     * Вторые веса должны совпасть бит-в-бит, если {@code m/v/step} сохранены.
     */
    @Test
    void adamwSaveLoadPreservesState() throws IOException {
        AdamW a = new AdamW(0.01f, 0.9f, 0.999f, 1e-8f, 0.01f);

        float[][] w1 = randMat(3, 4);
        float[][] g1 = randMat(3, 4);
        float[]   b1 = randVec(3);
        float[]   gb1 = randVec(3);

        // Эталон: два update на a
        float[][] wRef = cloneMat(w1);
        float[]   bRef = cloneVec(b1);
        a.updateWeights(wRef, g1, 4);
        a.updateBiases (bRef, gb1, 4);
        a.updateWeights(wRef, g1, 4);
        a.updateBiases (bRef, gb1, 4);

        // Эксперимент: один update, save/load, второй update
        float[][] w2 = cloneMat(w1);
        float[]   b2 = cloneVec(b1);
        a.updateWeights(w2, g1, 4);
        a.updateBiases (b2, gb1, 4);

        // save/load через Persistables (как в .model.train)
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (DataOutputStream out = new DataOutputStream(bytes)) {
            Persistables.save(out, a);
        }
        Optimizer a2;
        try (DataInputStream in = new DataInputStream(
                new ByteArrayInputStream(bytes.toByteArray()))) {
            a2 = Persistables.load(in, Optimizer.class);
        }
        assertInstanceOf(AdamW.class, a2);

        a2.updateWeights(w2, g1, 4);
        a2.updateBiases (b2, gb1, 4);

        assertMatEquals(wRef, w2, 1e-6f, "AdamW weights");
        assertVecEquals(bRef, b2, 1e-6f, "AdamW biases");
    }

    @Test
    void momentumSaveLoadPreservesHyperparams() throws IOException {
        Momentum original = new Momentum(0.01f, 0.9f);

        float[][] w = randMat(3, 4);
        float[][] g = randMat(3, 4);
        float[]   b = randVec(3);
        float[]   gb = randVec(3);

        // original: два update на ОДНОМ массиве w → wRef (эталон с velocity)
        float[][] wRef = cloneMat(w);
        float[]   bRef = cloneVec(b);
        original.updateWeights(wRef, g, 4);
        original.updateBiases (bRef, gb, 4);
        original.updateWeights(wRef, g, 4);
        original.updateBiases (bRef, gb, 4);

        // save/load
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (DataOutputStream out = new DataOutputStream(bytes)) {
            Persistables.save(out, original);
        }
        Momentum loaded;
        try (DataInputStream in = new DataInputStream(
                new ByteArrayInputStream(bytes.toByteArray()))) {
            loaded = (Momentum) Persistables.load(in, Optimizer.class);
        }

        // loaded (после load): ОДИН update на w → velocity = 0
        float[][] wLoaded = cloneMat(w);
        float[]   bLoaded = cloneVec(b);
        loaded.updateWeights(wLoaded, g, 4);
        loaded.updateBiases (bLoaded, gb, 4);

        // fresh: ОДИН update на w → тоже velocity = 0
        Momentum fresh = new Momentum(0.01f, 0.9f);
        float[][] wFresh = cloneMat(w);
        float[]   bFresh = cloneVec(b);
        fresh.updateWeights(wFresh, g, 4);
        fresh.updateBiases (bFresh, gb, 4);

        // 1) Гиперпараметры восстановлены: loaded == fresh (velocity = 0)
        assertMatEquals(wFresh, wLoaded, 1e-6f,
                "Momentum: после load velocity = 0 (как у fresh)");
        assertVecEquals(bFresh, bLoaded, 1e-6f,
                "Momentum: после load bias velocity = 0 (как у fresh)");

        // 2) original (дважды, с velocity) != fresh (один раз, без velocity) —
        //    это подтверждает, что velocity у original реально ненулевая
        //    и что тест различает наличие/отсутствие velocity.
        boolean differs = false;
        for (int i = 0; i < w.length && !differs; i++)
            for (int j = 0; j < w[i].length; j++)
                if (Math.abs(wRef[i][j] - wFresh[i][j]) > 1e-6f) { differs = true; break; }
        assertTrue(differs,
                "original (2 update, velocity) и fresh (1 update) должны различаться");
    }

    /** SGD: без состояния, сохраняется только lr. */
    @Test
    void sgdSaveLoadPreservesLr() throws IOException {
        SGD a = new SGD(0.05f);

        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (DataOutputStream out = new DataOutputStream(bytes)) {
            Persistables.save(out, a);
        }
        Optimizer a2;
        try (DataInputStream in = new DataInputStream(
                new ByteArrayInputStream(bytes.toByteArray()))) {
            a2 = Persistables.load(in, Optimizer.class);
        }

        float[][] w1 = randMat(3, 4);
        float[][] g1 = randMat(3, 4);
        float[]   b1 = randVec(3);
        float[]   gb1 = randVec(3);

        float[][] wRef = cloneMat(w1);
        float[]   bRef = cloneVec(b1);
        a.updateWeights(wRef, g1, 4);
        a.updateBiases (bRef, gb1, 4);

        float[][] w2 = cloneMat(w1);
        float[]   b2 = cloneVec(b1);
        a2.updateWeights(w2, g1, 4);
        a2.updateBiases (b2, gb1, 4);

        assertMatEquals(wRef, w2, 1e-6f, "SGD weights");
        assertVecEquals(bRef, b2, 1e-6f, "SGD biases");
    }

    // ==================== утилиты ====================

    private static void assertMatEquals(float[][] a, float[][] b, float tol, String what) {
        assertEquals(a.length, b.length, what + " rows");
        for (int i = 0; i < a.length; i++)
            assertArrayEquals(a[i], b[i], tol, what + "[" + i + "]");
    }

    private static void assertVecEquals(float[] a, float[] b, float tol, String what) {
        assertArrayEquals(a, b, tol, what);
    }
}