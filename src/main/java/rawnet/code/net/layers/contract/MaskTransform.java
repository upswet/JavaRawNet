package rawnet.code.net.layers.contract;

/** Слой, который меняет маску согласованно с данными (Reshape, Slice(SEQ), Merge(concat)). */
public interface MaskTransform {
    /**
     * @param inMask  маска входа (может быть null)
     * @return маска выхода (может быть null)
     */
    float[] transformMask(float[] inMask);
}
