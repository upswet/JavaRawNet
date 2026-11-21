package rawnet.docs.flows.main.mlp.glava2.code;

import lombok.AllArgsConstructor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import rawnet.code.dataset.SampleDataset;
import rawnet.code.dataset.Sample;
import rawnet.code.net.core.activate.*;
import rawnet.docs.flows.main.mlp.tasks.Iris;
import rawnet.docs.flows.main.mlp.tasks.Mnist;
import rawnet.docs.flows.main.mlp.tasks.FunctionApproximation;

import rawnet.code.net.core.loss.Loss;
import rawnet.code.net.core.loss.Mse;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import static rawnet.code.net.model.ModelPrintUtil.floatArrayToString;
import static rawnet.code.net.model.ModelPrintUtil.floatToString;

/**
 * Это простой ученический пример нейронной сети построенной на матрицах
 */
@Slf4j
@RequiredArgsConstructor
public class MlpMatrix {
    /**
     * Нейросеть
     */
    @Slf4j
    public static class Net {
        //слои
        final Layer[] layers;

        final Loss loss; //функция потери

        final float lr; //коэффициент обучения сети

        private final float[] outputGradient;  // переиспользуемый буфер для градиента функции потерь

        //ДТО-шка для создания слоёв
        @AllArgsConstructor
        public static class LayerDto {
            final int size;
            final Activate activate;
        }

        public Net(float lr, Loss loss, List<LayerDto> layerDtoList) {
            this.lr = lr;
            this.loss = loss;

            if (layerDtoList.size() < 2)
                throw new IllegalArgumentException("Нейросеть не может содержать меньше двух слоёв");
            if (layerDtoList.getFirst().activate != null)
                throw new IllegalArgumentException("Во входном слое функция активации не нужна");

            int inputSize = layerDtoList.getFirst().size;
            layers = new Layer[layerDtoList.size() - 1]; // Так как входной слой — только размер, весов у него нет
            for (int i = 1; i < layerDtoList.size(); i++) {
                LayerDto dto = layerDtoList.get(i);
                layers[i - 1] = new Layer(inputSize, dto.size, dto.activate);
                inputSize = dto.size;
            }

            // буфер для градиента равен размеру выхода последнего слоя
            this.outputGradient = new float[layers[layers.length - 1].outputSize];
        }

        /** Прямой проход всей сети
         * @param input - входной вектор
         * @return - выходной вектор*/
        public float[] forward(float[] input) {
            float[] current = input;
            for (Layer layer : layers) {
                current = layer.forward(current);
            }
            return current;
        }

        /**Обратный проход - корректировка весов (запускать только после прямого прохода)
         * @param outputs - результат прямого прохода
         * @param targets - вектор правильных значений
         */
        private void backward(float[] outputs, float[] targets) {
            // Градиент функции потерь по выходу сети
            loss.derivative(outputs, targets, outputGradient);

            // Обратный проход – переиспользуем один буфер для передачи градиентов
            float[] delta = outputGradient;
            for (int l = layers.length - 1; l >= 0; l--) {
                delta = layers[l].backward(delta);
            }

            // Обновление весов
            for (Layer layer : layers) {
                layer.update(lr);
            }
        }

        /**
         * Прогнать тестовый набор и вернуть среднюю потерю с него
         */
        public float test(List<Sample> samples, boolean isShow) {
            float lossAvg = 0;
            int show = 5;
            for (Sample sample : samples) {
                float[] predicted = this.forward(sample.getInputs());
                lossAvg += this.loss.loss(predicted, sample.getTargets());
                if (isShow && show > 0) {
                    show--;
                    log.info("ответ сети {}  (правильный ответ = {})", floatArrayToString(predicted, "0.00"), floatArrayToString(sample.getTargets(), "0.00"));
                }
            }
            lossAvg = lossAvg / samples.size();
            if (isShow) log.info("Потеря на тесте: {}", floatToString(lossAvg, "0.000000"));
            return lossAvg;
        }

        /**
         * Тренировать нейросеть
         *
         * @param maxEpoch   - максимальное кол-во эпох обучения после которых останавливаем обучение
         * @param minLossAvg - минимальная средняя потеря на тестовом наборе после которой останавливаем обучение
         * @param ds         - набор примеров
         * @return средняя потеря на тестовом наборе
         */
        public float trains(int maxEpoch, float minLossAvg, SampleDataset<Sample> ds) {
            float lossAvgTest = 0;

            int epochCountLog = maxEpoch / 10;
            if (epochCountLog == 0) epochCountLog = 1;

            for (int epoch = 0; epoch < maxEpoch; epoch++) {
                //тренируем
                for (Sample sample : ds.getTrainSamples()) {
                    float[] outputs = this.forward(sample.getInputs());
                    this.backward(outputs, sample.getTargets());
                }

                //проверяем на тестовой выборке
                lossAvgTest = test(ds.getTestSamples(), false);

                //логируем
                if (epoch % epochCountLog == 0 || epoch == maxEpoch || lossAvgTest <= minLossAvg) {
                    log.info("Эпоха обучения {}, потеря на тестовой выборке {}", epoch, floatToString(lossAvgTest, "0.0000000"));
                    if (lossAvgTest <= minLossAvg) break;
                }
            }
            return lossAvgTest;
        }
    }

    /**
     * Слой
     */
    static class Layer {
        final float[][] weights;   // [outputSize][inputSize]
        final float[] biases;      // [outputSize]
        final Activate activate;

        // Буферы для прямого/обратного прохода (выделяются один раз)
        private float[] input;   // вход слоя (ссылка, не копия) размера inputSuze
        private float[] z;           // взвешенная сумма до активации
        private float[] output;      // выход после активации размера outputSize
        private float[] delta;   // градиент по выходам слоя для обновления.
        private float[] prevDelta;   // градиент по входам (возвращается предыдущему слою)
        private float[] dActBuffer;  // буфер для производной активации

        private int inputSize, outputSize;

        public Layer(int inputSize, int outputSize, Activate activate) {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            
            this.activate = activate;
            weights = new float[outputSize][inputSize];
            biases = new float[outputSize];

            this.z = new float[outputSize];
            this.output = new float[outputSize];
            this.delta = new float[outputSize];
            this.prevDelta = new float[inputSize];
            this.dActBuffer = new float[outputSize];

            //выберем инициацию для весов связей в зависимости от функции активации
            float limit;
            if (activate instanceof Sigmoid || activate instanceof Tanh || activate instanceof Linear)
                limit = (float) Math.sqrt(6.0 / (inputSize + outputSize)); // Инициализация Xavier/Glorot
            else if (activate instanceof LeakyRelu)
                limit = (float) Math.sqrt(6.0 / inputSize); // Инициализация He/Kaiming
            else
                limit = 0.1f; //просто маленькое число

            Random rnd = ThreadLocalRandom.current();
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < inputSize; j++) {

                    weights[i][j] = rnd.nextFloat() * 2 * limit - limit;
                }
                biases[i] = rnd.nextFloat() * 0.2f - 0.1f;
            }
        }

        /**
         * Прямой проход: принимает входной вектор, возвращает выходной
         */
        public float[] forward(float[] input) {
            this.input = input;

            for (int i = 0; i < outputSize; i++) {
                float sum = biases[i];
                for (int j = 0; j < inputSize; j++) {
                    sum += weights[i][j] * input[j];
                }
                z[i] = sum;
            }
            activate.apply(z, output);
            return output;
        }

        /**
         * Обратный проход: принимает градиент по выходу слоя (deltaNext),
         * запоминает градиент для последующего обновления весов в delta
         * возвращает градиент по входу(prevDelta).
         */
        public float[] backward(float[] deltaNext) {
            // 1. Вычисляем производную активации
            activate.derivative(z, output, dActBuffer);

            Arrays.fill(prevDelta, 0f);

            for (int i = 0; i < outputSize; i++) {
                // 2. Поэлементно умножаем входящий градиент на производную и сохраняем в delta для последующего обновления весов
                float di = deltaNext[i] * dActBuffer[i]; //dL/dz = dL/dy * f'(z)
                delta[i] = di;

                float[] w = weights[i];          // строка весов
                for (int j = 0; j < inputSize; j++) {
                    // 3. Вычисляем градиент по входу: prevDelta = W^T * delta где W^T - транспонированная матрица весов
                    prevDelta[j] += w[j] * di;   // накопление по строке
                }
            }
            return prevDelta;
        }

        /**
         * Обновление весов и смещений с использованием сохранённого градиента
         */
        public void update(float learningRate) {
            for (int i = 0; i < outputSize; i++) {

                float step = learningRate * delta[i];

                float[] w = weights[i];
                for (int j = 0; j < inputSize; j++) {
                    w[j] -= step * input[j];
                }
                biases[i] -= step;
            }
        }
    }

    /**
     * Тестируем нашу нейросеть
     */
    public static void main(String[] args) {
        log.info("\n\n==== Апроксимация значений функции");
        SampleDataset<Sample> dataset = SampleDataset.split(
                FunctionApproximation.generateDatasetForTaskFunctionApproximation(
                        x -> new float[]{x[0] + x[1] * x[1] - x[2] * x[2] * x[2]},
                        //x->{float sum = (float) (Math.sin(x[0]) + Math.cos(x[0]) + 0.5 * x[2]);return new float[]{1.0f / (1.0f + (float) Math.exp(-sum))};},
                        3,
                        1000,
                        4.0f
                ),
                0.9f
        );

        Net nn = new Net(
                0.01f,
                new Mse(),
                List.of(
                        new Net.LayerDto(3, null),
                        new Net.LayerDto(32, new Sigmoid()),
                        new Net.LayerDto(1, new Linear())
                )
        );

        nn.trains(5000, 0.001f, dataset);
        nn.test(dataset.getTestSamples(), true);

        log.info("\n\n==== Классификация ирисов");
        dataset = SampleDataset.split(
                Iris.prepareDataForIris(Path.of("d:\\Work\\Project\\0files\\iris\\iris.data.csv")),
                0.9f
        );
        nn = new Net(
                0.01f,
                new Mse(),
                List.of(
                        new Net.LayerDto(4, null),
                        new Net.LayerDto(64, new Sigmoid()),
                        new Net.LayerDto(3, new Linear())
                )
        );
        nn.trains(5000, 0.001f, dataset);
        nn.test(dataset.getTestSamples(), true);

        log.info("\n\n==== Классификация MNIST");
        dataset = new SampleDataset<Sample>(
                Mnist.prepareDataForMnist(Path.of("d:\\Work\\Project\\0files\\mnist\\mnist_train.csv")),
                Mnist.prepareDataForMnist(Path.of("d:\\Work\\Project\\0files\\mnist\\mnist_test.csv"))
        );
        nn = new Net(
                0.01f,
                new Mse(),
                List.of(
                        new Net.LayerDto(784, null),
                        new Net.LayerDto(80, new Sigmoid()),
                        new Net.LayerDto(40, new Sigmoid()),
                        new Net.LayerDto(10, new Sigmoid())
                )
        );
        nn.trains(5, 0.001f, dataset);
        nn.test(dataset.getTestSamples(), true);
    }
}
