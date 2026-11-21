package rawnet.code.net.layers.contract;

/** Слой-потребитель маски: читает её, но не меняет. */
public interface MaskConsumer {
    float[] forwardMasked(float[] input, float[] mask, boolean training);
    float[] backwardMasked(float[] deltaNext, float[] mask);
}
