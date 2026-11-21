package rawnet.docs.flows.main.cnn.tasks.styleTransform;

import rawnet.code.net.TensorShape;

/**
 * Вспомогательные операции для Style Transfer:
 * {@link #gram}, {@link #contentLoss}, {@link #gramLoss} + аналитические градиенты.
 */
public final class StyleOps {

    private StyleOps() {}

    private record Shape(int C, int HW) {
        static Shape of(TensorShape s) {
            if (!s.isCnn())
                throw new IllegalArgumentException("StyleOps: нужна CNN-форма, получено " + s);
            return new Shape(s.channels(), s.height() * s.width());
        }
    }

    /** {@code mean((F_o − F_c)^2)}. */
    public static float contentLoss(float[] out, float[] content) {
        if (out.length != content.length)
            throw new IllegalArgumentException("contentLoss: длины не совпадают");
        double s = 0;
        for (int i = 0; i < out.length; i++) {
            double d = out[i] - content[i];
            s += d * d;
        }
        return (float) (s / out.length);
    }

    /** {@code dL/dF_out = 2·(F_out − F_content) / N}. */
    public static void contentLossGrad(float[] out, float[] content, float[] gradOut) {
        if (out.length != content.length || out.length != gradOut.length)
            throw new IllegalArgumentException("contentLossGrad: длины не совпадают");
        float k = 2f / out.length;
        for (int i = 0; i < out.length; i++) gradOut[i] = k * (out[i] - content[i]);
    }

    /** Gram: {@code G[c1][c2] = (1/HW) Σ F[c1,h,w]·F[c2,h,w]}. */
    public static float[] gram(float[] F, TensorShape shape) {
        Shape sh = Shape.of(shape);
        int C = sh.C, HW = sh.HW;
        float[] G = new float[C * C];
        float inv = 1f / HW;
        for (int c1 = 0; c1 < C; c1++) {
            int off1 = c1 * HW;
            for (int c2 = c1; c2 < C; c2++) {
                int off2 = c2 * HW;
                float s = 0;
                for (int i = 0; i < HW; i++) s += F[off1 + i] * F[off2 + i];
                s *= inv;
                G[c1 * C + c2] = s;
                G[c2 * C + c1] = s;
            }
        }
        return G;
    }

    /** {@code mean((G_out − G_style)^2)}. */
    public static float gramLoss(float[] Gout, float[] Gstyle) {
        if (Gout.length != Gstyle.length)
            throw new IllegalArgumentException("gramLoss: размеры Gram не совпадают");
        double s = 0;
        for (int i = 0; i < Gout.length; i++) {
            double d = Gout[i] - Gstyle[i];
            s += d * d;
        }
        return (float) (s / Gout.length);
    }

    /** {@code dL/dF[c,h,w] = (4/(C·C·HW)) Σ_{c'} (G_out[c,c']−G_style[c,c'])·F_out[c',h,w]}. */
    public static void gramLossGrad(float[] Fout, float[] Gout, float[] Gstyle,
                                    TensorShape shape, float[] gradOut) {
        Shape sh = Shape.of(shape);
        int C = sh.C, HW = sh.HW;
        if (Fout.length != gradOut.length || Fout.length != C * HW)
            throw new IllegalArgumentException("gramLossGrad: неверная длина F_out/gradOut");
        if (Gout.length != C * C || Gstyle.length != C * C)
            throw new IllegalArgumentException("gramLossGrad: неверная длина Gram");

        float k = 4f / ((float) C * C * HW);
        for (int c = 0; c < C; c++) {
            int baseOut = c * HW;
            for (int i = 0; i < HW; i++) gradOut[baseOut + i] = 0f;
        }
        for (int c = 0; c < C; c++) {
            int baseOut = c * HW;
            int gRow = c * C;
            for (int cp = 0; cp < C; cp++) {
                float dG = Gout[gRow + cp] - Gstyle[gRow + cp];
                if (dG == 0f) continue;
                int baseIn = cp * HW;
                float kk = k * dG;
                for (int i = 0; i < HW; i++) gradOut[baseOut + i] += kk * Fout[baseIn + i];
            }
        }
    }
}