package rawnet.code.net.layers.contract;

/** Слой-источник маски: создаёт маску по входу. */
public interface MaskSource {
    /**
     * @return маска длины T (1 — настоящий токен, 0 — PAD) для последнего forward
     */
    float[] lastMask();
}