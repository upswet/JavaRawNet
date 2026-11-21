package rawnet.docs.flows.main.mlp.tasks;

import lombok.extern.slf4j.Slf4j;
import rawnet.code.dataset.SampleDataset;
import rawnet.code.dataset.Sample;
import rawnet.code.net.core.activate.LeakyRelu;
import rawnet.code.net.core.activate.Sigmoid;
import rawnet.code.net.core.activate.Tanh;
import rawnet.code.net.layers.core.ActivationLayer;
import rawnet.code.net.layers.core.DenseLayer;
import rawnet.code.net.core.loss.BinaryCrossEntropy;
import rawnet.code.net.core.loss.Mse;
import rawnet.code.net.core.metric.Mae;
import rawnet.code.net.core.metric.R2;
import rawnet.code.net.core.metric.Rmse;
import rawnet.code.net.model.ModelTraining;
import rawnet.code.net.model.functional.Functional;
import rawnet.code.net.model.functional.spec.InputSpec;
import rawnet.code.net.model.functional.spec.Specs;
import rawnet.code.net.model.sequence.Sequence;
import rawnet.code.net.core.optimizator.AdamW;
import rawnet.code.net.core.optimizator.SGD;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * Задача апроксимации функции
 */
@Slf4j
public class FunctionApproximation {
    /**
     * Сгенерировать набор данных для задачи апроксимакции функции
     *
     * @param fun         - сама функция
     * @param sampleCount - количество генерируемых примеров
     * @param xCount      - кол-во входных аргументов функции
     * @param inputRange  - промежуток в котором будут меняться элементы входного вектора для фун
     * @return - список примеров
     */
    public static List<Sample> generateDatasetForTaskFunctionApproximation(Function<float[], float[]> fun, int xCount, int sampleCount, float inputRange) {
        List<Sample> result = new ArrayList<>(sampleCount);
        for (int i = 0; i < sampleCount; i++) {
            float[] input = new float[xCount];
            for (int j = 0; j < xCount; j++)
                input[j] = (float) (Math.random() * inputRange - inputRange / 2);
            float[] target = fun.apply(input);
            result.add(new Sample(input, target));
        }
        return result;
    }

    public static void seq() {
        log.info("---------------seq1");
        SampleDataset<Sample> dataset = SampleDataset.split(
                FunctionApproximation.generateDatasetForTaskFunctionApproximation(
                        x -> {
                            float sum = (float) (Math.sin(x[0]) + Math.cos(x[0]) + 0.5 * x[2]);
                            return new float[]{1.0f / (1.0f + (float) Math.exp(-sum))};
                        },
                        3,
                        1000,
                        4.0f
                ),
                0.9f
        );

        log.info("seq1");
        new ModelTraining<Sequence, Sample>()
                .setModel(Sequence.of(
                        List.of(
                                new Sequence.InputDto(3),
                                new DenseLayer.Dto(64), new ActivationLayer.Dto(new Tanh()),
                                new DenseLayer.Dto(1), new ActivationLayer.Dto(new Tanh())
                        )))
                .setDataset(dataset)
                .setLoss(new Mse())
                .setOptimizer(new SGD(0.1f))
                .setMaxEpoch(200)
                .setBatchSize(32)
                .setShowTestCount(5)
                .setMetrics(List.of(new R2(), new Rmse(), new Mae()))
                .setEndPredicate((model,metric) -> metric.get(0).result() > 0.99f && metric.get(1).result() < 0.01f)
                .setClipNorm(1f)
                .train();


        log.info("\nseq2");
        new ModelTraining<Sequence, Sample>()
                .setModel(Sequence.of(
                        List.of(
                                new Sequence.InputDto(3),
                                new DenseLayer.Dto(64), new ActivationLayer.Dto(new Tanh()),
                                new DenseLayer.Dto(1),
                                new ActivationLayer.Dto(new Sigmoid())   // сигмоида для BCE
                        )))
                .setDataset(dataset)
                .setLoss(new BinaryCrossEntropy())
                .setOptimizer(new AdamW(0.001f))
                .setMaxEpoch(100)
                .setBatchSize(32)
                .setShowTestCount(5)
                .setMetrics(List.of(new R2(), new Rmse(), new Mae()))
                .setEndPredicate((model,metric) -> metric.get(0).result() > 0.99f && metric.get(1).result() < 0.01f)
                .setClipNorm(1f)
                .train();

        log.info("seq3");
        new ModelTraining<Sequence, Sample>()
                .setModel(Sequence.of(
                        List.of(
                                new Sequence.InputDto(3),
                                new DenseLayer.Dto(64), new ActivationLayer.Dto(new Tanh()),
                                new DenseLayer.Dto(1)
                        )))
                .setDataset(dataset)
                .setLoss(new Mse())
                .setOptimizer(new SGD(0.1f))
                .setMaxEpoch(200)
                .setBatchSize(32)
                .setShowTestCount(5)
                .setMetrics(List.of(new R2(), new Rmse(), new Mae()))
                .setEndPredicate((model,metric) -> metric.get(0).result() > 0.99f && metric.get(1).result() < 0.01f)
                .setClipNorm(1f)
                .train();
    }

    public static void fun() {
        log.info("---------------fun");
        SampleDataset<Sample> dataset = SampleDataset.split(
                FunctionApproximation.generateDatasetForTaskFunctionApproximation(
                        x -> new float[]{x[0] + x[1] * x[1] - x[2] * x[2] * x[2]},
                        3,
                        1000,
                        4.0f
                ),
                0.9f
        );


        new ModelTraining<Functional, Sample>()
                .setModel(new Specs.LayerSpec(
                        new InputSpec(3),
                        new DenseLayer.Dto(64), new ActivationLayer.Dto(new LeakyRelu()),
                        new DenseLayer.Dto(32), new ActivationLayer.Dto(new LeakyRelu()),
                        new DenseLayer.Dto(1)
                ).build())
                .setDataset(dataset)
                .setLoss(new Mse())
                .setOptimizer(new AdamW(0.001f))
                .setMaxEpoch(200)
                .setBatchSize(32)
                .setShowTestCount(5)
                .setMetrics(List.of(new R2(), new Rmse(), new Mae()))
                .setEndPredicate((model,metric) -> metric.get(0).result() > 0.99f && metric.get(1).result() < 0.01f)
                .setClipNorm(-1f)
                .train();
    }


    public static void main() {
        log.info("\n\n==== Апроксимация функции");

        seq();
        fun();
    }
}
