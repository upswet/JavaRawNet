package rawnet.code.net;

import rawnet.code.dataset.SampleDataset;
import rawnet.code.dataset.Sample;
import rawnet.code.net.composition.TransformerEncoderStack;
import rawnet.code.net.core.activate.Sigmoid;
import rawnet.code.net.layers.Layer;
import rawnet.code.net.layers.core.ActivationLayer;
import rawnet.code.net.layers.core.DenseLayer;
import rawnet.code.net.layers.core.EmbeddingLayer;
import rawnet.code.net.layers.core.PositionalEncodingLayer;
import rawnet.code.net.layers.mask.MaskingLayer;
import rawnet.code.net.layers.mask.TokenPoolingLayer;
import rawnet.code.net.layers.structure.ModelLayer;
import rawnet.code.net.composition.TransformerEncoderBlock;
import rawnet.code.net.model.Model;
import rawnet.code.net.model.sequence.Sequence;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import static rawnet.code.net.TestDatasets.palindromeDataset;
import static rawnet.code.net.TestDatasets.trainSmall;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * «Учебник»: Transformer-энкодер на задаче распознавания палиндрома.
 *
 * <p>Палиндром требует сравнить начало и конец строки — именно то, что
 * делает attention. RNN для этого нужны гейты (GRU/LSTM), а Transformer
 * решает «из коробки». Поэтому эта задача хорошо показывает, что блок
 * работает.
 *
 * <p>Архитектура:
 * <pre>
 *   Masking → Embedding → PositionalEncoding
 *           → TransformerEncoderBlock × 2
 *           → TokenPooling(MEAN) → Dense → Sigmoid
 * </pre>
 *
 * <p>Ожидаемая accuracy: ~0.75–0.85 на 500 примерах за 40 эпох.
 * Порог 0.70 — с запасом.
 */
class TransformerTutorialTest {

    @Test
    void transformer_encoder_learns_palindrome() {
        final int T = 8;
        final int dModel = 32;
        final int heads = 4;
        final int dFF = 64;
        final int vocab = 6;   // PAD=0 + 1..5

        SampleDataset<Sample> ds = palindromeDataset(500, new Random(42L));

        // Собираем два блока Transformer (как готовые Model).
        var block1 = TransformerEncoderBlock.build(
                T, new TransformerEncoderBlock.Dto(heads, dModel, dFF, 0.0f, false));
        var block2 = TransformerEncoderBlock.build(
                T, new TransformerEncoderBlock.Dto(heads, dModel, dFF, 0.0f, false));

        List<Layer.Dto> dtos = List.of(
                new Sequence.InputDto(TensorShape.flat(T)),
                new MaskingLayer.Dto(0),
                new EmbeddingLayer.Dto(vocab, dModel),
                new PositionalEncodingLayer.Dto(),
                new ModelLayer.Dto(block1),          // блок 1 (trainable)
                new ModelLayer.Dto(block2),          // блок 2 (trainable)
                new TokenPoolingLayer.Dto(TokenPoolingLayer.Mode.MEAN),
                new DenseLayer.Dto(TensorShape.flat(1)),
                new ActivationLayer.Dto(new Sigmoid())
        );
        Sequence net = Sequence.of(dtos);

        float acc = trainSmall(net, ds, 100, 3e-4f).get(1).result();
        assertTrue(acc > 0.70f,
                "Transformer: ожидалось > 0.70, получено " + acc);
    }


    @Test
    void encoder_stack_builds_and_saves(@TempDir Path tmp) {
        var dto = new TransformerEncoderBlock.Dto(2, 16, 32, 0.0f, false);

        // Stack из 3 блоков + LN
        Model stack = TransformerEncoderStack.build(8, dto, 3);

        // forward: seq(8,16) → seq(8,16)
        float[] x = new float[8 * 16];
        float[] y = stack.forward(x, false);
        assertEquals(8 * 16, y.length);

        // save/load: Functional с ModelLayer внутри
        Path file = tmp.resolve("stack.bin");
        stack.save(file);
        Model loaded = Model.load(file);
        float[] y2 = loaded.forward(x, false);
        for (int i = 0; i < y.length; i++) {
            assertEquals(y[i], y2[i], 1e-6f);
        }
    }
}