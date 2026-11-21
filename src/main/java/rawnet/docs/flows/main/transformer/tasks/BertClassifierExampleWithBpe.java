package rawnet.docs.flows.main.transformer.tasks;

import rawnet.code.dataset.Sample;
import rawnet.code.dataset.SampleDataset;
import rawnet.code.net.TensorShape;
import rawnet.code.net.composition.TransformerEncoderBlock;
import rawnet.code.net.composition.TransformerEncoderStack;
import rawnet.code.net.core.loss.SoftmaxCrossEntropy;
import rawnet.code.net.core.metric.Accuracy;
import rawnet.code.net.core.metric.Metric;
import rawnet.code.net.core.optimizator.AdamW;
import rawnet.code.net.layers.core.DenseLayer;
import rawnet.code.net.layers.core.EmbeddingLayer;
import rawnet.code.net.layers.core.PositionalEncodingLayer;
import rawnet.code.net.layers.mask.MaskingLayer;
import rawnet.code.net.layers.mask.TokenPoolingLayer;
import rawnet.code.net.model.Model;
import rawnet.code.net.model.ModelTraining;
import rawnet.code.net.model.functional.Functional;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * BERT-style классификатор русской тональности — <b>с BPE</b>.
 *
 * <p>Токенизация — BPE: слова разбиваются на общие куски. «хороший»
 * и «хорошая» → общий префикс «хорош» + разные суффиксы. Модель
 * обобщает сигнал с одной формы на другую.
 *
 * <p>Сравни с {@code BertClassifierExample} — там без BPE, и на
 * held-out фразах со словами вне словаря («восхитительная»,
 * «кошмарная») модель не может классифицировать: слово → {@code <unk>}.
 * Здесь — может: «восхит» и «кошм» есть в train (из «восхитительный»
 * и «кошмарный»), и BPE их выделяет.
 */
public final class BertClassifierExampleWithBpe {

    // ==================== гиперпараметры ====================

    private static final int D_MODEL = 128;
    private static final int HEADS   = 8;
    private static final int D_FF    = 4 * D_MODEL;
    private static final int N_ENC   = 2;
    private static final float DROPOUT = 0f;
    /** BPE-токенов больше, чем слов: SEQ_LEN увеличен. */
    private static final int SEQ_LEN = 24;
    private static final int BATCH = 8;
    private static final int EPOCHS = 60;
    private static final float LR = 5e-4f;
    private static final float CLIP_NORM = 1f;
    private static final int NUM_CLASSES = 2;

    private static final int PAD = 0;
    private static final int CLS = 1;

    /** Размер BPE-словаря. */
    private static final int BPE_VOCAB = 500;

    // ==================== точка входа ====================

    public static void main(String[] args) {
        List<String[]> corpus = toyCorpus();
        System.out.printf(Locale.US, "[BertClassifierExampleWithBpe] Корпус: %d фраз%n", corpus.size());

        // 1) Обучаем BPE на текстах корпуса.
        List<String> texts = new ArrayList<>();
        for (String[] p : corpus) texts.add(p[0]);
        Bpe bpe = Bpe.train(texts, BPE_VOCAB);
        System.out.println("BPE: " + bpe);

        // 2) Сеть. EmbeddingLayer — размер BPE-словаря.
        Functional model = buildModel(bpe.size());

        // 3) Примеры: BPE-кодирование.
        List<Sample> samples = buildSamples(corpus, bpe);
        SampleDataset<Sample> ds = SampleDataset.split(samples, 0.8f);
        System.out.printf(Locale.US, "Train: %d, Test: %d%n",
                ds.getTrainSamples().size(), ds.getTestSamples().size());

        SoftmaxCrossEntropy loss = new SoftmaxCrossEntropy();
        AdamW opt = new AdamW(LR);
        List<Metric> metrics = List.of(new Accuracy());

        new ModelTraining<Model, Sample>()
                .setModel(model)
                .setDataset(ds)
                .setLoss(loss)
                .setOptimizer(opt)
                .setMaxEpoch(EPOCHS)
                .setBatchSize(BATCH)
                .setMetrics(metrics)
                .setClipNorm(CLIP_NORM)
                .train();

        System.out.println();
        System.out.println("Демонстрация (held-out фразы):");
        String[] heldOut = {
                "фильм был очень хороший",
                "фильм был очень плохой",
                "мне понравилась игра актеров",
                "мне не понравился сюжет",
                // Слова ВНЕ словаря слов, но BPE разбивает на знакомые куски.
                "картина была восхитительная",
                "картина была кошмарная"
        };
        for (String text : heldOut) {
            int[] idx = encode(text, bpe);
            float[] logits = model.forward(toFloat(idx), false);
            float[] probs = SoftmaxCrossEntropy.softmax(logits);
            int label = argmax(probs);
            // Показываем, на что разбилась фраза — видно, что «восхит» и «кошм» есть.
            System.out.printf(Locale.US, "  %-40s → %s (pos=%.2f, neg=%.2f)%n",
                    "\"" + text + "\"",
                    label == 1 ? "POSITIVE" : "NEGATIVE",
                    probs[1], probs[0]);
            System.out.printf(Locale.US, "     pieces: %s%n",
                    bpe.encode(text).stream().map(bpe::tokenAt).toList());
        }
    }

    // ==================== корпус ====================

    /** Тот же корпус, что в {@code BertClassifierExample} — для честного сравнения. */
    private static List<String[]> toyCorpus() {
        String[] positive = {
                "фильм был хороший", "фильм был отличный", "фильм был прекрасный",
                "фильм был замечательный", "фильм был великолепный", "фильм был интересный",
                "фильм был увлекательный", "фильм был потрясающий",
                "какой хороший фильм", "какой отличный фильм",
                "какой прекрасный фильм", "какой замечательный фильм",
                "мне понравился фильм", "мне понравилась игра",
                "мне понравился сюжет", "мне понравились актеры",
                "игра актеров была хорошая", "игра актеров была отличная",
                "сюжет был интересный", "сюжет был увлекательный",
                "этот фильм хороший", "этот фильм отличный",
                "это отличный фильм", "это прекрасный фильм",
                "это замечательный фильм", "просто отличный фильм",
                "просто прекрасный фильм", "очень хороший фильм",
                "очень интересный фильм", "очень увлекательный фильм",
                "лучший фильм года", "лучший фильм что я видел",
                "мне очень понравился фильм", "мне очень понравилась игра",
                "фильм подарил радость", "фильм заставил улыбаться",
                "актеры играли отлично", "сюжет был замечательный",
                "прекрасная игра актеров", "замечательная история",
                // Для BPE-сравнения: «восхитительный» (мужской род).
                "фильм был восхитительный", "это восхитительный фильм",
                // Для BPE-сравнения: в женском роде.
                "какой восхитительный фильм",
                "это восхитительная картина",
                "история была замечательная"
        };
        String[] negative = {
                "фильм был плохой", "фильм был ужасный", "фильм был скучный",
                "фильм был отвратительный", "фильм был скверный", "фильм был нудный",
                "фильм был медленный", "фильм был тоскливый",
                "какой плохой фильм", "какой ужасный фильм",
                "какой скучный фильм", "какой отвратительный фильм",
                "мне не понравился фильм", "мне не понравилась игра",
                "мне не понравился сюжет", "мне не понравились актеры",
                "игра актеров была плохая", "игра актеров была ужасная",
                "сюжет был скучный", "сюжет был нудный",
                "этот фильм плохой", "этот фильм ужасный",
                "это плохой фильм", "это ужасный фильм",
                "это скучный фильм", "просто ужасный фильм",
                "просто скучный фильм", "очень плохой фильм",
                "очень скучный фильм", "очень нудный фильм",
                "худший фильм года", "худший фильм что я видел",
                "мне очень не понравился фильм", "мне очень не понравилась игра",
                "фильм вызвал раздражение", "фильм заставил скучать",
                "актеры играли плохо", "сюжет был ужасный",
                "ужасная игра актеров", "скучная история",
                // Для BPE-сравнения: «кошмарный» (мужской род).
                "фильм был кошмарный", "это кошмарный фильм",
                "какой кошмарный фильм"
        };
        List<String[]> out = new ArrayList<>(positive.length + negative.length);
        for (String s : positive) out.add(new String[]{s, "1"});
        for (String s : negative) out.add(new String[]{s, "0"});
        return out;
    }

    // ==================== примеры ====================

    private static List<Sample> buildSamples(List<String[]> corpus, Bpe bpe) {
        List<Sample> out = new ArrayList<>(corpus.size());
        for (String[] pair : corpus) {
            int[] idx = encode(pair[0], bpe);
            float[] target = new float[NUM_CLASSES];
            target[Integer.parseInt(pair[1])] = 1f;
            out.add(new Sample(toFloat(idx), target));
        }
        return out;
    }

    private static int[] encode(String text, Bpe bpe) {
        int[] out = new int[SEQ_LEN];
        Arrays.fill(out, PAD);
        out[0] = CLS;
        List<Integer> pieces = bpe.encode(text);
        int pos = 1;
        for (int id : pieces) {
            if (pos >= SEQ_LEN) break;
            out[pos++] = id;
        }
        return out;
    }

    private static float[] toFloat(int[] a) {
        float[] f = new float[a.length];
        for (int i = 0; i < a.length; i++) f[i] = a[i];
        return f;
    }

    private static int argmax(float[] a) {
        int best = 0;
        for (int i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
        return best;
    }

    // ==================== модель ====================

    private static Functional buildModel(int vocabSize) {
        Functional.Builder b = Functional.builder(TensorShape.flat(SEQ_LEN));
        Functional.Node in = b.input();
        Functional.Node mask = b.layer(new MaskingLayer.Dto(PAD), in);
        Functional.Node emb  = b.layer(new EmbeddingLayer.Dto(vocabSize, D_MODEL), mask);
        Functional.Node pe   = b.layer(new PositionalEncodingLayer.Dto(), emb);
        Functional.Node enc  = TransformerEncoderStack.build(
                b, pe,
                new TransformerEncoderBlock.Dto(HEADS, D_MODEL, D_FF, DROPOUT, false),
                N_ENC);
        Functional.Node pooled = b.layer(
                new TokenPoolingLayer.Dto(TokenPoolingLayer.Mode.FIRST), enc);
        Functional.Node logits = b.layer(
                new DenseLayer.Dto(TensorShape.flat(NUM_CLASSES)), pooled);
        return b.output(logits);
    }
}