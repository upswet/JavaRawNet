package rawnet.docs.flows.main.mlp.tasks;

import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import rawnet.code.dataset.SampleDataset;
import rawnet.code.dataset.Sample;
import rawnet.code.net.TensorShape;
import rawnet.code.net.core.activate.LeakyRelu;
import rawnet.code.net.layers.cnn.ConvLayer;
import rawnet.code.net.layers.cnn.MaxPoolLayer;
import rawnet.code.net.layers.core.ActivationLayer;
import rawnet.code.net.layers.core.DenseLayer;
import rawnet.code.net.layers.core.DropoutLayer;
import rawnet.code.net.layers.structure.ReshapeLayer;
import rawnet.code.net.core.loss.SoftmaxCrossEntropy;
import rawnet.code.net.core.metric.Accuracy;
import rawnet.code.net.model.ModelTraining;
import rawnet.code.net.model.functional.Functional;
import rawnet.code.net.model.functional.spec.InputSpec;
import rawnet.code.net.model.functional.spec.NamedSpec;
import rawnet.code.net.model.functional.spec.RefSpec;
import rawnet.code.net.model.functional.spec.Specs;
import rawnet.code.net.model.sequence.Sequence;
import rawnet.code.net.core.optimizator.AdamW;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Задача распознавания рукописных цифр - mnist
 */
@Slf4j
public class Mnist {
    /**
     * Преобразовать набор данных MNIST в используемый формат
     *
     * @param path - путь к файлу с датасетом для задачи
     * @return - вернёт готовый датасет
     */
    @SneakyThrows
    public static List<Sample> prepareDataForMnist(Path path) {
        List<Sample> result = new ArrayList<>();

        List<String> list = Files.readAllLines(path);
        for (String s : list) {
            String[] sarr = s.split(",");

            float[] target = new float[10];
            Arrays.fill(target, 0);
            target[Integer.valueOf(sarr[0])] = 1;

            float[] data = new float[sarr.length - 1];
            for (int j = 1; j < sarr.length; j++)
                data[j - 1] = (float) ((Double.valueOf(sarr[j]) / 255.0));

            result.add(new Sample(data, target));
        }
        return result;
    }

    public static void mlp_seq(SampleDataset<Sample> dataset) {
        log.info("mlp_seq");

        new ModelTraining<Sequence, Sample>()
                .setModel(Sequence.of(List.of(
                        new Sequence.InputDto(784),
                        new DenseLayer.Dto(256), new ActivationLayer.Dto(new LeakyRelu()),
                        new DropoutLayer.Dto(0.2f),
                        new DenseLayer.Dto(128), new ActivationLayer.Dto(new LeakyRelu()),
                        new DropoutLayer.Dto(0.2f),
                        new DenseLayer.Dto(10)
                )))
                .setDataset(dataset)
                .setLoss(new SoftmaxCrossEntropy())
                .setOptimizer(new AdamW(0.001f))
                .setMaxEpoch(30)
                .setBatchSize(64)
                .setShowTestCount(10)
                .setMetrics(List.of(new Accuracy()))
                .setEndPredicate((model, mtr) -> mtr.get(1).result() > 0.98f)
                .setClipNorm(-1f)
                .train();
    }

    public static void mlp_fun(SampleDataset<Sample> dataset) {
   /*
    input (784)
      │
      ├──► dense(784→256) ──► h (256) ──┬───────────────────────┐
      │                                 │                       │
      │                                 ├──► LeakyReLU ──► dense(256→256) ──► h2 (256)
      │                                 │                       │
      │                                 └─────────── + ─────────┘
      │                                              │
      │                                              ▼
      │                                          r = h + h2 (256)
      │                                              │
      │                                              ▼
      │                                          dense(256→10)
      │                                              │
      │                                              ▼
      │                                          логиты (10)




x → dense(784→256) → h
h → LeakyReLU → dense(256→256) → h2
r = h + h2        // <-- остаточная связь
r → dense(256→10)
    */
        log.info("mlp_fun");


        log.info("{}", new ModelTraining<Functional, Sample>()
                .setModel(new Specs.LayerSpec(
                        // вход
                        new NamedSpec("input", new InputSpec(784)),

                        // ствол: input → dense(256) → h
                        new NamedSpec("h", new Specs.LayerSpec(
                                new RefSpec("input"),
                                new DenseLayer.Dto(256)
                        )),

                        // блок: h → LeakyReLU → dense(256) → h2
                        new NamedSpec("h2", new Specs.LayerSpec(
                                new RefSpec("h"),
                                new ActivationLayer.Dto(new LeakyRelu()),
                                new DenseLayer.Dto(256)
                        )),

                        // residual: r = h + h2
                        new NamedSpec("r", new Specs.AddSpec(
                                new RefSpec("h"),
                                new RefSpec("h2")
                        )),

                        // выход: r → dense(256→10)
                        new Specs.LayerSpec(
                                new RefSpec("r"),
                                new DenseLayer.Dto(10)
                        )
                ).build())
                .setDataset(dataset)
                .setLoss(new SoftmaxCrossEntropy())
                .setOptimizer(new AdamW(0.001f))
                .setMaxEpoch(30)
                .setBatchSize(64)
                .setShowTestCount(10)
                .setMetrics(List.of(new Accuracy()))
                .setEndPredicate((model, mtr) -> mtr.get(1).result() > 0.98f)
                .setClipNorm(-1f)
                .train()
                .toString());
    }

    public static void cnn_seq(SampleDataset<Sample> dataset) {
        log.info("cnn_seq");
        new ModelTraining<Sequence, Sample>()
                .setModel(Sequence.of(List.of(
                        // вход — картинка 28 × 28 = 784 пикселя
                        new Sequence.InputDto(TensorShape.cnn(1, 28, 28)),

                        // 1-й свёрточный блок: 1×28×28 → 32×28×28 → 32×14×14
                        new ConvLayer.Dto(32, 3, 1, 1), new ActivationLayer.Dto(new LeakyRelu()),
                        new MaxPoolLayer.Dto(2, 2),

                        // 2-й свёрточный блок: 32×14×14 → 64×14×14 → 64×7×7
                        new ConvLayer.Dto(64, 3, 1, 1), new ActivationLayer.Dto(new LeakyRelu()),
                        new MaxPoolLayer.Dto(2, 2),

                        // Голова: flatten → dense → dropout → логиты
                        new ReshapeLayer.FlattenLayer.Dto(),
                        new DenseLayer.Dto(128), new ActivationLayer.Dto(new LeakyRelu()),
                        new DropoutLayer.Dto(0.3f),
                        new DenseLayer.Dto(10)
                )))
                .setDataset(dataset)
                .setLoss(new SoftmaxCrossEntropy())
                .setOptimizer(new AdamW(0.001f))
                .setMaxEpoch(30)
                .setBatchSize(64)
                .setShowTestCount(10)
                .setMetrics(List.of(new Accuracy()))
                .setEndPredicate((model, mtr) -> mtr.get(1).result() > 0.98f)
                .setClipNorm(-1f)
                .train()
                .save(Path.of("model/mnist/mnist-cnn.bin"));
    }

    public static void main() {
        log.info("\n\n==== Классификация MNIST");

        SampleDataset<Sample> dataset = new SampleDataset<Sample>(
                Mnist.prepareDataForMnist(Path.of("d:\\Work\\Project\\0files\\mnist\\mnist_train.csv")),
                Mnist.prepareDataForMnist(Path.of("d:\\Work\\Project\\0files\\mnist\\mnist_test.csv")));

        log.info("\n ---MLP");
        mlp_seq(dataset);
        mlp_fun(dataset);

        log.info("\n ---CNN");
        cnn_seq(dataset);
    }
}
