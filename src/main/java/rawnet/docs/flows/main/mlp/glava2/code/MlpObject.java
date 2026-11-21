package rawnet.docs.flows.main.mlp.glava2.code;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import rawnet.code.dataset.SampleDataset;
import rawnet.code.dataset.Sample;
import rawnet.docs.flows.main.mlp.tasks.FunctionApproximation;
import rawnet.docs.flows.main.mlp.tasks.Iris;
import rawnet.docs.flows.main.mlp.tasks.Mnist;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static rawnet.code.net.model.ModelPrintUtil.floatArrayToString;
import static rawnet.code.net.model.ModelPrintUtil.floatToString;

/**Это простой ученический пример нейронной сети построенной на нейронах как отдельных объектах*/
@Slf4j
public class MlpObject {
    /**
     * Интерфейс функции активации нейрона.
     */
    interface ActivationFunction {
        /**
         * Вычисляет значение функции активации.
         *
         * @param x взвешенная сумма входов нейрона
         * @return результат активации
         */
        float activate(float x);

        /**
         * Вычисляет производную функции активации.
         *
         * @param x взвешенная сумма (до активации)
         * @param y выход нейрона (после активации), чтобы использовать более эффективные формулы
         * @return значение производной в точке x (или y)
         */
        float derivative(float x, float y);
    }

    /**
     * Интерфейс функции потерь
     */
    interface LossFunction {
        /**
         * Значение функции потерь для одного примера.
         *
         * @param predicted выход сети
         * @param target    целевое значение
         * @return loss
         */
        float loss(float[] predicted, float[] target);

        /**
         * Производная функции потерь по каждому выходу сети.
         * Используется для вычисления delta выходного слоя.
         *
         * @param predicted выход нейрона
         * @param target    целевое значение выхода нейрона
         * @return производная dL/d(predicted) - то есть дельта ошибки
         */
        float derivative(float predicted, float target);
    }

    /**
     * Нейросеть
     */
    @Slf4j
    public static class Net {
        //слои
        final List<Layer> layers = new ArrayList<>();

        final LossFunction loss; //функция потери

        final float lr; //коэффициент обучения сети

        //ДТО-шка для создания слоёв
        @AllArgsConstructor
        public static class LayerDto {
            final int size;
            final ActivationFunction activationFunction;
        }

        public Net(float lr, LossFunction loss, List<LayerDto> layerDtoList) {
            this.lr = lr;
            this.loss = loss;

            if (layerDtoList.size() < 2) throw new RuntimeException("Нейросеть не может содержать меньше двух слоёв");
            if (layerDtoList.getFirst().activationFunction != null)
                throw new RuntimeException("Во входном слое функция активации не нужна");

            Layer prevoisLayer = null;
            for (LayerDto dto : layerDtoList) {
                layers.add(new Layer(dto.size, dto.activationFunction, prevoisLayer));
                prevoisLayer = layers.getLast();
            }
        }

        /**
         * Прямой проход - вычисление
         *
         * @param inputs - входной вектор
         * @return выходной вектор
         */
        public float[] forward(float[] inputs) {
            Layer layerInput = layers.getFirst();

            //вставим входной вектор в нейроны входного слоя
            for (int i = 0; i < inputs.length; i++)
                layerInput.neirons.get(i).output = inputs[i];

            //прямой проход по нейронам промежуточных слоёв и выходного слоя
            for (Layer layer : layers.subList(1, layers.size()))
                for (Neiron neiron : layer.neirons) {
                    float sum = neiron.bias;
                    for (Link inLink : neiron.inLinks)
                        sum += inLink.from.output * inLink.weight;
                    neiron.input = sum;
                    neiron.output = layer.activationFunction.activate(neiron.input);
                }

            //возьмём выходной вектор из нейронов выходного слоя
            Layer layerOutput = layers.getLast();
            float[] outputs = new float[layerOutput.neirons.size()];
            for (int i = 0; i < outputs.length; i++)
                outputs[i] = layerOutput.neirons.get(i).output;
            return outputs;
        }

        /**
         * Обратный проход - корректировка весов (запускать только после прямого прохода)
         *
         * @param targets - вектор правильных значений
         */
        private void backward(float[] targets) {
            //вычислим дельту ошибку для нейронов выходного слоя
            Layer layerOutput = layers.getLast();
            for (int i = 0; i < targets.length; i++) {
                Neiron neiron = layerOutput.neirons.get(i);
                neiron.delta = loss.derivative(neiron.output, targets[i]) * layerOutput.activationFunction.derivative(neiron.input, neiron.output); //dL/dz = dL/dy * f'(z)
            }

            //вычислим дельту ошибки на нейронах скрытых слоёв как взвешенную сумму дельт ошибок нейронов следующего слоя с которыми связан данный нейрон
            Layer nextLayer = layerOutput;
            for (Layer layer : layers.reversed().subList(1, layers.size() - 1)) {
                for (Neiron neiron : layer.neirons) {
                    float sum = 0;
                    for (Link outLink : neiron.outLinks)
                        sum += outLink.to.delta * outLink.weight;
                    neiron.delta = sum * layer.activationFunction.derivative(neiron.input, neiron.output); //dL/dz = dL/dy * f'(z)
                }
            }

            //обновим веса связей и смещения нейронов
            for (Layer layer : layers.subList(1, layers.size())) {
                for (Neiron neiron : layer.neirons) {
                    for (Link inLink : neiron.inLinks) {
                        float gradient = neiron.delta * inLink.from.output;
                        inLink.weight = inLink.weight - lr * gradient;
                    }
                    neiron.bias = neiron.bias - lr * neiron.delta;
                }
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
                    this.forward(sample.getInputs());
                    this.backward(sample.getTargets());
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
        List<Neiron> neirons = new ArrayList<>();
        final ActivationFunction activationFunction;

        public Layer(int size, ActivationFunction activationFunction, Layer prevoisLayer) {
            this.activationFunction = activationFunction;

            for (int i = 0; i < size; i++) {
                Neiron newNeiron = new Neiron();
                neirons.add(newNeiron);

                //пропишем связь всех нейронов предыдущего слоя с новым нейроном данного слоя
                if (prevoisLayer != null) {
                    for (Neiron n : prevoisLayer.neirons)
                        new Link(n, newNeiron);
                }
            }
        }
    }

    /**
     * Нейрон
     */
    static class Neiron {
        List<Link> inLinks = new ArrayList<>(); //входящие в нейрон связи
        List<Link> outLinks = new ArrayList<>(); //исходящие из нейрона связи
        float bias = (float) (Math.random() * 0.2f - 0.1f); //смещение
        float output; //выход нейрона
        float delta; //дельта ошибки данного нейрона
        float input; //Взвешенная сумма входов до применения функции активации
    }


    /**
     * Связь между двумя нейронами
     */
    static class Link {
        final Neiron from; //из какого нейрона
        final Neiron to; //в какой нейрон
        float weight = (float) (Math.random() * 0.2f - 0.1f); //вес связи

        public Link(Neiron from, Neiron to) {
            this.from = from;
            from.outLinks.add(this);
            this.to = to;
            to.inLinks.add(this);
        }
    }

    //sigmoid - activate
    static class Sigmoid implements ActivationFunction {
        @Override
        public float activate(float x) {
            return 1.0f / (1.0f + (float) Math.exp(-x));
        }

        @Override
        public float derivative(float x, float y) {
            //Производная м/б вычислена как y*(1-y) что быстрее чем если вычислять через x
            return y * (1.0f - y);
        }
    }

    //leaky relu - activate
    static class LeakyReluActivation implements ActivationFunction {
        private final float alpha;

        /**
         * @param alpha коэффициент наклона для отрицательных значений (обычно 0.01)
         */
        public LeakyReluActivation(float alpha) {
            this.alpha = alpha;
        }

        @Override
        public float activate(float x) {
            return x > 0 ? x : alpha * x;
        }

        @Override
        public float derivative(float x, float y) {
            // Производная зависит только от знака x (y здесь не используется)
            return x > 0 ? 1.0f : alpha;
        }
    }

    //identy - activate
    static class Identity implements ActivationFunction {
        @Override
        public float activate(float x) {
            return x;
        }

        @Override
        public float derivative(float x, float y) {
            return 1.0f;
        }
    }

    //mse - loss
    static class Mse implements LossFunction {
        @Override
        public float loss(float[] predicted, float[] target) {
            float sum = 0;
            for (int i = 0; i < predicted.length; i++) {
                float diff = predicted[i] - target[i];
                sum += diff * diff;
            }
            return sum / predicted.length;
        }

        @Override
        public float derivative(float predicted, float target) {
            return predicted - target;
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
                        new Net.LayerDto(1, new Identity())
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
                        new Net.LayerDto(3, new Sigmoid())
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
