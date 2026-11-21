package rawnet.docs.flows.main.cnn.tasks.styleTransform;

import rawnet.code.net.TensorShape;
import rawnet.code.net.core.activate.LeakyRelu;
import rawnet.code.net.layers.cnn.ConvLayer;
import rawnet.code.net.layers.cnn.UpsampleLayer;
import rawnet.code.net.layers.core.ActivationLayer;
import rawnet.code.net.layers.core.InstanceNormLayer;
import rawnet.code.net.model.sequence.Sequence;
import rawnet.docs.flows.main.cnn.tasks.imageDetect.ImageDatasetGenerator;

import java.awt.image.BufferedImage;
import java.util.List;

/**
 * Сеть-стилизатор {@code S: cnn(3,S,S) → cnn(3,S,S)}.
 *
 * <p>Энкодер-декодер (без skip-связей, для простоты):
 * <pre>
 * cnn(3,128,128)
 *  → Conv(16, k=3, s=2, p=1) → LeakyReLU → cnn(16,64,64)
 *  → Conv(32, k=3, s=2, p=1) → LeakyReLU → cnn(32,32,32)
 *  → Conv(64, k=3, s=1, p=1) → LeakyReLU → cnn(64,32,32)
 *  → Conv(64, k=3, s=1, p=1) → LeakyReLU → cnn(64,32,32)     (bottleneck)
 *  → Upsample(2) → Conv(32) → IN → LeakyReLU → cnn(32,64,64)
 *  → Upsample(2) → Conv(16) → IN → LeakyReLU → cnn(16,128,128)
 *  → Conv(3, k=1)                            → cnn(3,128,128)
 * </pre>
 *
 * <h3>Почему Conv(stride=2), а не MaxPool</h3>
 * {@code MaxPool.backward} пропускает градиент <b>только</b> в позицию
 * максимума (derivative = 0 вне argmax). В стилизаторе это «мёртвые зоны»:
 * фильтры, которые не выиграли argmax, <b>не учатся</b>. {@code Conv(stride=2)}
 * дифференцируема везде — все фильтры получают градиент.
 *
 * <h3>Почему InstanceNorm только в декодере</h3>
 * {@code IN} нормирует каждый канал (mean=0, std=1) — это <b>вымывает</b>
 * абсолютные значения (яркость, цвет). В энкодере это ломает content loss
 * (content-фичи становятся «центрированными», MSE всегда ≈ 0).
 * В декодере {@code IN} стоит <b>после</b> content loss — нормирует выход,
 * не трогая форму.
 *
 * <h3>Почему без Sigmoid</h3>
 * {@link StyleStylizerTrainer} клипует выход в {@code [0,1]} при подсчёте
 * лосса, а {@link #apply} — при инференсе. Sigmoid «насыщается» и теряет
 * градиент — клип дешевле.
 */
public final class StyleStylizer {

    private StyleStylizer() {}

    /** Собирает сеть-стилизатор для стороны {@code size} (кратно 4, >= 16). */
    public static Sequence build(int size) {
        if (size < 16 || size % 4 != 0)
            throw new IllegalArgumentException("StyleStylizer.build: size >= 16 и кратно 4");

        return Sequence.of(List.of(
                new Sequence.InputDto(TensorShape.cnn(3, size, size)),

                // ── энкодер: /2 → /4, БЕЗ InstanceNorm ──
                new ConvLayer.Dto(16, 3, 2, 1),                 // 64
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),

                new ConvLayer.Dto(32, 3, 2, 1),                 // 32
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),

                new ConvLayer.Dto(64, 3, 1, 1),                 // 32
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),

                new ConvLayer.Dto(64, 3, 1, 1),                 // 32
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),

                // ── декодер: ×2 → ×2, С InstanceNorm ──
                new UpsampleLayer.Dto(2),                       // 64
                new ConvLayer.Dto(32, 3, 1, 1),
                new InstanceNormLayer.Dto(),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),

                new UpsampleLayer.Dto(2),                       // 128
                new ConvLayer.Dto(16, 3, 1, 1),
                new InstanceNormLayer.Dto(),
                new ActivationLayer.Dto(new LeakyRelu(0.1f)),

                // ── голова: 3 канала, 1×1, без активации ──
                new ConvLayer.Dto(3, 1, 1, 0)
        ));
    }

    /** Применяет стилизатор к картинке (инференс). Выход клипуется в {@code [0,1]}. */
    public static BufferedImage apply(Sequence stylizer, BufferedImage img) {
        int w = img.getWidth(), h = img.getHeight();
        if (w != h)
            throw new IllegalArgumentException("StyleStylizer.apply: картинка должна быть квадратной");

        float[] in = ImageDatasetGenerator.toFlatRgb(img);
        float[] out = stylizer.forward(in, false);

        for (int i = 0; i < out.length; i++)
            out[i] = Math.max(0f, Math.min(1f, out[i]));

        return ImageDatasetGenerator.fromFlatRgb(out, w, h);
    }
}