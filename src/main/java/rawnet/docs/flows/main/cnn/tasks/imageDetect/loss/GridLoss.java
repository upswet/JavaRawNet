package rawnet.docs.flows.main.cnn.tasks.imageDetect.loss;

import rawnet.code.net.core.loss.Loss;

/**
 * YOLO-grid loss в <b>channel-major</b> раскладке: {@code flat((5+C)·G·G)}.
 *
 * <p>Почему channel-major, а не pixel-major: выход головы —
 * {@code cnn(5+C, G, G)}, и в памяти он лежит именно как
 * {@code [c0(все клетки), c1(все клетки), ...]}. Если reshape'ить
 * в pixel-major, порядок ломается (теряется пространственная
 * структура, loss не учится). Поэтому loss и декодер работают
 * прямо в channel-major.
 *
 * <p>Каналы:
 * <ol start="0">
 *   <li>obj — BCE, нормируется на {@code nObj};</li>
 *   <li>cx_local ∈ [0,1) — MSE, только в занятых;</li>
 *   <li>cy_local ∈ [0,1) — MSE, только в занятых;</li>
 *   <li>w ∈ (0,1] — MSE на √σ, только в занятых;</li>
 *   <li>h ∈ (0,1] — MSE на √σ, только в занятых;</li>
 *   <li>...</li>
 *   <li>c_{C-1} — softmax-CE, только в занятых.</li>
 * </ol>
 *
 * <p>Нормировка: obj — на {@code max(1,nObj)}, box и class — на
 * {@code cells} (= G·G). Это даёт стабильный масштаб между примерами
 * и согласованный аналитический/численный градиент.
 */
public class GridLoss implements Loss {

    private static final float EPS = 1e-15f;

    private final int gridCells;
    private final int numClasses;
    private final int per;
    private final int cells;

    private final float lambdaObj;
    private final float lambdaBox;
    private final float lambdaCls;
    private final float lambdaNoObj;

    public GridLoss(int gridCells, int numClasses) {
        this(gridCells, numClasses, 1f, 5f, 1f, 0.5f);
    }

    public GridLoss(int gridCells, int numClasses,
                    float lambdaObj, float lambdaBox, float lambdaCls, float lambdaNoobj) {
        if (gridCells < 1) throw new IllegalArgumentException("gridCells >= 1");
        if (numClasses < 1) throw new IllegalArgumentException("numClasses >= 1");
        if (lambdaObj < 0f || lambdaBox < 0f || lambdaCls < 0f || lambdaNoobj < 0f)
            throw new IllegalArgumentException("λ >= 0");
        this.gridCells = gridCells;
        this.numClasses = numClasses;
        this.per = 5 + numClasses;
        this.cells = gridCells * gridCells;
        this.lambdaObj = lambdaObj;
        this.lambdaBox = lambdaBox;
        this.lambdaCls = lambdaCls;
        this.lambdaNoObj = lambdaNoobj;
    }

    @Override
    public float loss(float[] logits, float[] target) {
        checkLengths(logits, target);

        double sObj = 0, sBox = 0, sCls = 0;
        int nObj = 0;

        for (int i = 0; i < cells; i++) {
            float objT = target[i];
            boolean occupied = objT > 0.5f;
            if (occupied) nObj++;

            float objP = sigmoid(logits[i]);
            float wObj = occupied ? lambdaObj : lambdaNoObj;
            sObj += wObj * bce(objP, objT);

            if (!occupied) continue;

            float cxP = sigmoid(logits[1 * cells + i]);
            float cyP = sigmoid(logits[2 * cells + i]);
            float wP  = (float) Math.sqrt(Math.max(sigmoid(logits[3 * cells + i]), EPS));
            float hP  = (float) Math.sqrt(Math.max(sigmoid(logits[4 * cells + i]), EPS));
            float wT  = (float) Math.sqrt(Math.max(target[3 * cells + i], EPS));
            float hT  = (float) Math.sqrt(Math.max(target[4 * cells + i], EPS));

            float dCx = cxP - target[1 * cells + i];
            float dCy = cyP - target[2 * cells + i];
            float dW  = wP  - wT;
            float dH  = hP  - hT;
            sBox += lambdaBox * (dCx * dCx + dCy * dCy + dW * dW + dH * dH);

            // softmax-CE по каналам 5..5+C-1
            int clsOffset = 5 * cells;
            float maxZ = logits[clsOffset + i];
            for (int k = 1; k < numClasses; k++) {
                float v = logits[clsOffset + k * cells + i];
                if (v > maxZ) maxZ = v;
            }
            double sumExp = 0;
            for (int k = 0; k < numClasses; k++)
                sumExp += Math.exp(logits[clsOffset + k * cells + i] - maxZ);
            double logZ = Math.log(sumExp) + maxZ;

            double tz = 0, tSum = 0;
            for (int k = 0; k < numClasses; k++) {
                double tk = target[clsOffset + k * cells + i];
                tz   += tk * logits[clsOffset + k * cells + i];
                tSum += tk;
            }
            sCls += lambdaCls * (logZ * tSum - tz);
        }

        // нормировка: obj на nObj, box/class на cells
        return (float) (sObj / Math.max(1, nObj)
                + (sBox + sCls) / cells);
    }

    @Override
    public void derivative(float[] logits, float[] target, float[] deriv) {
        checkLengths(logits, target);
        if (deriv.length != logits.length)
            throw new IllegalArgumentException("GridLoss: deriv.length != logits.length");

        int nObj = 0;
        for (int i = 0; i < cells; i++)
            if (target[i] > 0.5f) nObj++;

        final float invObj = 1f / Math.max(1, nObj);
        final float invCells = 1f / cells;

        for (int i = 0; i < cells; i++) {
            float objT = target[i];
            boolean occupied = objT > 0.5f;

            // obj: BCE
            float objP = sigmoid(logits[i]);
            float wObj = occupied ? lambdaObj : lambdaNoObj;
            deriv[i] = wObj * (objP - objT) * invObj;

            // обнуляем остальные каналы, если ячейка пуста
            if (!occupied) {
                for (int k = 1; k < per; k++) deriv[k * cells + i] = 0f;
                continue;
            }

            // cx, cy
            float cxP = sigmoid(logits[1 * cells + i]);
            float cyP = sigmoid(logits[2 * cells + i]);
            deriv[1 * cells + i] = lambdaBox * 2f
                    * (cxP - target[1 * cells + i]) * cxP * (1f - cxP) * invCells;
            deriv[2 * cells + i] = lambdaBox * 2f
                    * (cyP - target[2 * cells + i]) * cyP * (1f - cyP) * invCells;

            // w, h на √σ
            float wSig = Math.max(sigmoid(logits[3 * cells + i]), EPS);
            float hSig = Math.max(sigmoid(logits[4 * cells + i]), EPS);
            float wS = (float) Math.sqrt(wSig);
            float hS = (float) Math.sqrt(hSig);
            float wT = (float) Math.sqrt(Math.max(target[3 * cells + i], EPS));
            float hT = (float) Math.sqrt(Math.max(target[4 * cells + i], EPS));

            deriv[3 * cells + i] = lambdaBox * (wS - wT) / wS * wSig * (1f - wSig) * invCells;
            deriv[4 * cells + i] = lambdaBox * (hS - hT) / hS * hSig * (1f - hSig) * invCells;

            // softmax-CE
            int clsOffset = 5 * cells;
            float maxZ = logits[clsOffset + i];
            for (int k = 1; k < numClasses; k++) {
                float v = logits[clsOffset + k * cells + i];
                if (v > maxZ) maxZ = v;
            }
            double sumExp = 0;
            for (int k = 0; k < numClasses; k++)
                sumExp += Math.exp(logits[clsOffset + k * cells + i] - maxZ);
            double invSum = 1.0 / sumExp;
            for (int k = 0; k < numClasses; k++) {
                float p = (float) (Math.exp(logits[clsOffset + k * cells + i] - maxZ) * invSum);
                deriv[clsOffset + k * cells + i] =
                        lambdaCls * (p - target[clsOffset + k * cells + i]) * invCells;
            }
        }
    }

    private void checkLengths(float[] logits, float[] target) {
        int expected = per * cells;
        if (logits.length != expected)
            throw new IllegalArgumentException(
                    "GridLoss: ожидалось " + expected + " логитов, получено " + logits.length);
        if (target.length != expected)
            throw new IllegalArgumentException(
                    "GridLoss: ожидалось " + expected + " целей, получено " + target.length);
    }

    private static float sigmoid(float z) {
        if (z <= -30f) return 0f;
        if (z >=  30f) return 1f;
        return 1f / (1f + (float) Math.exp(-z));
    }

    private static float bce(float p, float t) {
        float pc = Math.max(Math.min(p, 1f - EPS), EPS);
        return -(t * (float) Math.log(pc) + (1f - t) * (float) Math.log(1f - pc));
    }
}