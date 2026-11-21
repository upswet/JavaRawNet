package rawnet.docs.flows.main.mlp.tasks;

import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import rawnet.code.dataset.SampleDataset;
import rawnet.code.dataset.Sample;
import rawnet.code.net.core.activate.Tanh;
import rawnet.code.net.layers.core.ActivationLayer;
import rawnet.code.net.layers.core.DenseLayer;
import rawnet.code.net.core.loss.SoftmaxCrossEntropy;
import rawnet.code.net.core.metric.Accuracy;
import rawnet.code.net.model.ModelTraining;
import rawnet.code.net.model.sequence.Sequence;
import rawnet.code.net.core.optimizator.Momentum;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**Задача классификации ирисов*/
@Slf4j
public class Iris {
    /**
     * Преобразовать набор данных "Ирисы" в используемый формат
     * @param path - путь к файлу с датасетом для задачи классификации ирисов
     * @return - вернёт готовый датасет
     */
    @SneakyThrows
    public static List<Sample> prepareDataForIris(Path path) {
        List<Sample> result = new ArrayList<>();

        List<String> list = Files.readAllLines(path);
        for (String s : list) {
            if (s.isBlank()) continue;

            String[] sarr = s.split(",");

            float[] target = new float[3];
            Arrays.fill(target, 0);
            switch (sarr[sarr.length-1]){
                case "Iris-setosa" -> target[0] = 1;
                case "Iris-versicolor" -> target[1] = 1;
                case "Iris-virginica" -> target[2] = 1;
            }

            float[] input = new float[sarr.length - 1];
            for (int j = 0; j < sarr.length-1; j++)
                input[j] = (Float.valueOf(sarr[j]) / 10);

            result.add(new Sample(input, target));
        }
        return result;
    }

    public static void main(){
        log.info("\n\n==== Классификация Ирисов");

        SampleDataset<Sample> dataset = SampleDataset.split(
                Iris.prepareDataForIris(Path.of("d:\\Work\\Project\\0files\\iris\\iris.data.csv")),
                0.9f
        );

        new ModelTraining<Sequence, Sample>()
                .setModel(Sequence.of(List.of(
                        new Sequence.InputDto(4),
                        new DenseLayer.Dto(8), new ActivationLayer.Dto(new Tanh()),  // скрытый
                        new DenseLayer.Dto(3)                                        // логиты
                )))
                .setDataset(dataset)
                .setLoss(new SoftmaxCrossEntropy())
                .setOptimizer(new Momentum(0.01f, 0.9f))
                .setMaxEpoch(150)
                .setBatchSize(32)
                .setShowTestCount(5)
                .setMetrics(List.of(new Accuracy()))
                .setEndPredicate((model, mtr) -> mtr.get(1).result() > 0.95f)
                .setClipNorm(1f)
                .train();
    }
}
