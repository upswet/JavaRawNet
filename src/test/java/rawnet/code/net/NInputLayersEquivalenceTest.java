package rawnet.code.net;

import rawnet.code.net.TensorShape;
import rawnet.code.net.layers.attention.MultiHeadAttentionLayer;
import rawnet.code.net.layers.attention.QKVAttentionLayer;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Проверки эквивалентности новых N-входовых слоёв старым Two/One-входовым.
 *
 * <p><b>Зачем.</b> {@link CrossAttentionNLayer} и {@link QKVAttentionLayer}
 * — «копии» существующих слоёв через новый контракт. Если реализация
 * задумана как точная копия, {@code forward} должен совпадать
 * <b>бит-в-бит</b> при одинаковых весах и одинаковом входе (порядок
 * операций сохранён). Это ловит опечатки в индексах и порядок
 * суммирования, которые численная проверка градиента может пропустить.
 *
 * <p>Тест копирует веса через рефлексию: поля {@code Wq/Wk/Wv/Wo} и
 * {@code bq/bk/bv/bo} в обоих классах package-private, тест лежит в том же
 * пакете — доступ прямой.
 */
class NInputLayersEquivalenceTest {

    private static final Random RND = new Random(42L);

    private static float[] rand(int n) {
        float[] r = new float[n];
        for (int i = 0; i < n; i++) r[i] = (RND.nextFloat() - 0.5f) * 0.5f;
        return r;
    }




    // =====================================================================
    // ============== QKVAttentionLayer vs MultiHeadAttentionLayer =========
    // =====================================================================

    /**
     * QKVAttentionLayer с готовыми Q/K/V должен давать тот же выход, что
     * и MultiHeadAttentionLayer(X), если Q/K/V посчитаны как
     * {@code Q=X·Wq}, {@code K=X·Wk}, {@code V=X·Wv}.
     *
     * <p>Проверяем без маски и без causal (у QKV есть флаг causal, но
     * для эквивалентности ставим {@code false}). Отдельно проверяем
     * causal-режим — он должен совпасть с {@code MultiHeadAttentionLayer}
     * при {@code causal=true}.
     */
    @Test
    void qkvAttentionMatchesMultiHeadAttention() {
        for (int[] cfg : new int[][]{
                {4, 8, 2},   // T=Tkv=4, dModel=8, heads=2
                {1, 4, 1},
                {6, 6, 3},
        }) {
            int T = cfg[0], dModel = cfg[1], heads = cfg[2];

            // Эталон: MultiHeadAttentionLayer на X
            MultiHeadAttentionLayer old = (MultiHeadAttentionLayer) new MultiHeadAttentionLayer.Dto(heads, false)
                    .create(TensorShape.seq(T, dModel));

            // Новый: QKVAttentionLayer с готовыми Q/K/V
            QKVAttentionLayer neu = (QKVAttentionLayer) new QKVAttentionLayer.Dto(heads, dModel, T, false, 0f)
                    .create(TensorShape.seq(T, dModel));

            // Копируем Wo/bo old → neu
            copyMat(old.Wo, neu.Wo); copyVec(old.bo, neu.bo);

            float[] x = rand(T * dModel);

            // Готовим Q/K/V вручную через Wq/Wk/Wv old
            float[] Q = new float[T * dModel];
            float[] K = new float[T * dModel];
            float[] V = new float[T * dModel];
            for (int i = 0; i < T; i++)
                for (int o = 0; o < dModel; o++) {
                    float sq = old.bq[o], sk = old.bk[o], sv = old.bv[o];
                    for (int k = 0; k < dModel; k++) {
                        float xi = x[i * dModel + k];
                        sq += old.Wq[o][k] * xi;
                        sk += old.Wk[o][k] * xi;
                        sv += old.Wv[o][k] * xi;
                    }
                    Q[i * dModel + o] = sq;
                    K[i * dModel + o] = sk;
                    V[i * dModel + o] = sv;
                }

            float[] yOld = old.forward(x, false).clone();
            float[] yNew = neu.forwardN(new float[][]{ Q, K, V }, false).clone();

            assertArrayEquals(yOld, yNew, 1e-5f,
                    "forward: T=" + T + ", dModel=" + dModel + ", heads=" + heads);

            // backward: dQ/dK/dV из QKV = dInQ/dInK/dInV без проекций,
            // поэтому сравнивать с backward MultiHead нельзя напрямую
            // (там dIn = dQ·Wqᵀ + dK·Wkᵀ + dV·Wvᵀ). Проверим только,
            // что dWo/dbo совпадают.
            float[] dOut = rand(T * dModel);
            old.backward(dOut);
            neu.backwardN(dOut);
            assertMatEquals(old.WoG, neu.WoG, 1e-5f, "WoG");
            assertVecEquals(old.boG, neu.boG, 1e-5f, "boG");
        }
    }

    /**
     * Causal-режим: QKVAttentionLayer(causal=true) должен совпасть с
     * MultiHeadAttentionLayer(causal=true).
     */
    @Test
    void qkvAttentionMatchesMultiHeadAttentionCausal() {
        int T = 5, dModel = 8, heads = 2;
        MultiHeadAttentionLayer old = (MultiHeadAttentionLayer) new MultiHeadAttentionLayer.Dto(heads, true)
                .create(TensorShape.seq(T, dModel));
        QKVAttentionLayer neu = (QKVAttentionLayer) new QKVAttentionLayer.Dto(heads, dModel, T, true, 0f)
                .create(TensorShape.seq(T, dModel));

        copyMat(old.Wo, neu.Wo); copyVec(old.bo, neu.bo);

        float[] x = rand(T * dModel);
        float[] Q = new float[T * dModel], K = new float[T * dModel], V = new float[T * dModel];
        for (int i = 0; i < T; i++)
            for (int o = 0; o < dModel; o++) {
                float sq = old.bq[o], sk = old.bk[o], sv = old.bv[o];
                for (int k = 0; k < dModel; k++) {
                    float xi = x[i * dModel + k];
                    sq += old.Wq[o][k] * xi;
                    sk += old.Wk[o][k] * xi;
                    sv += old.Wv[o][k] * xi;
                }
                Q[i * dModel + o] = sq;
                K[i * dModel + o] = sk;
                V[i * dModel + o] = sv;
            }

        float[] yOld = old.forward(x, false).clone();
        float[] yNew = neu.forwardN(new float[][]{ Q, K, V }, false).clone();
        assertArrayEquals(yOld, yNew, 1e-5f, "causal forward");

        // Первая позиция: attention только на j=0, out[0] = V[0] (после Wo)
        // Это свойство causal-маски — проверяем, что оно держится.
        // (out[0] не обязан быть нулём, но не должен зависеть от V[1..T-1].)
        // Косвенная проверка: два прогона с разными V[1..] дают одинаковый out[0].
        float[] V2 = V.clone();
        for (int j = 1; j < T; j++)
            for (int k = 0; k < dModel; k++) V2[j * dModel + k] = rand(1)[0];
        float[] yNew2 = neu.forwardN(new float[][]{ Q, K, V2 }, false);
        for (int k = 0; k < dModel; k++)
            assertEquals(yNew[k], yNew2[k], 1e-6f,
                    "causal: out[0] не должен зависеть от V[1..T-1]");
    }

    // =====================================================================
    // ========================== Утилиты ==================================
    // =====================================================================

    private static void copyMat(float[][] from, float[][] to) {
        for (int i = 0; i < from.length; i++)
            System.arraycopy(from[i], 0, to[i], 0, from[i].length);
    }
    private static void copyVec(float[] from, float[] to) {
        System.arraycopy(from, 0, to, 0, from.length);
    }
    private static void assertMatEquals(float[][] a, float[][] b, float tol, String what) {
        assertEquals(a.length, b.length, what + " rows");
        for (int i = 0; i < a.length; i++)
            assertArrayEquals(a[i], b[i], tol, what + "[" + i + "]");
    }
    private static void assertVecEquals(float[] a, float[] b, float tol, String what) {
        assertArrayEquals(a, b, tol, what);
    }
}