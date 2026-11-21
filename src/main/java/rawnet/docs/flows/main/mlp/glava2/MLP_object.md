# Нейросеть где нейрон как объект

[Ссылка на полный код](code/MlpObject.java)

## Код

Функции активации: сигмоида и линейное преобразование (когда выход равен входу, то есть f(x) = x)
```java
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
```

Функция потерь: стандартная среднеквадратичная (mse)

```java
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
```

Наши объекты

Связь между нейронами
```java
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
```
Обрати внимание как мы задаём случайный начальный вес связи. Он должен быть небольшим, в пределах от -0.1 до 0.1

Нейрон:
```java
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
```

Слой
```java
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
```
Обрати внимание: при создании слоя создаём связи между каждым нейроном текущего слоя (при его создании) и всеми нейронами прошлого слоя.


Нейросеть как совокупность слоёв:
```java
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
    //....
}        
/*Пример создания нейросети
         nn = new Net(
                0.01f,
                new Mse(),
                List.of(
                        new Net.LayerDto(784, null),
                        new Net.LayerDto(80, new Sigmoid()),
                        new Net.LayerDto(40, new Sigmoid()),
                        new Net.LayerDto(10, new Sigmoid())
                )
        ); */        
```

Прямой проход. Тут всё просто:

```java
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
```

И обратный проход. Чуть посложнее
```java
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
            
            //дельту ошибки на нейронах входного слоя вычислять не надо (хотя можно, если мы хотим заняться анализом данных)

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
 * Тренировать нейросеть
 *
 * @param maxEpoch   - максимальное кол-во эпох обучения после которых останавливаем обучение
 * @param minLossAvg - минимальная средняя потеря на тестовом наборе после которой останавливаем обучение
 * @param ds         - набор примеров
 * @return средняя потеря на тестовом наборе
 */
public float trains(int maxEpoch, float minLossAvg, SampleSimple.Dataset ds) {
    float lossAvgTest = 0;

    int epochCountLog = maxEpoch / 10;
    if (epochCountLog == 0) epochCountLog = 1;

    for (int epoch = 0; epoch < maxEpoch; epoch++) {
        //тренируем
        for (SampleSimple sample : ds.getTrainSamples()) {
            this.forward(sample.getInputs());
            this.backward(sample.getTargets());
        }

        //проверяем на тестовой выборке
        lossAvgTest = test(ds.getTestSamples(), false);

        //логируем
        if (epoch % epochCountLog == 0 || epoch == maxEpoch || lossAvgTest <= minLossAvg) {
            log.info("Эпоха обучения {}, потеря на тестовой выборке {}", epoch, Utils.floatToString(lossAvgTest, "0.0000000"));
            if (lossAvgTest <= minLossAvg) break;
        }
    }
    return lossAvgTest;
}        
```

## Результаты

```java
   /**
     * Тестируем нашу нейросеть
     */
    public static void main(String[] args) {
        log.info("\n\n==== Апроксимация значений функции");
        SampleSimple.Dataset dataset = SampleSimple.split(
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
        dataset = SampleSimple.split(
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
        dataset = new SampleSimple.Dataset(
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
```

```text
18:29:59.703 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject -- 

==== Апроксимация значений функции
18:29:59.766 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 0, потеря на тестовой выборке 3.6477985
18:30:00.271 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 500, потеря на тестовой выборке 0.0070302
18:30:00.675 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 1000, потеря на тестовой выборке 0.0023987
18:30:01.080 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 1500, потеря на тестовой выборке 0.0012103
18:30:01.318 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 1793, потеря на тестовой выборке 0.0009999
18:30:01.318 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- ответ сети [8.52]  (правильный ответ = [8.43])
18:30:01.319 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- ответ сети [0.62]  (правильный ответ = [0.64])
18:30:01.319 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- ответ сети [1.85]  (правильный ответ = [1.86])
18:30:01.319 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- ответ сети [3.49]  (правильный ответ = [3.51])
18:30:01.319 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- ответ сети [-3.30]  (правильный ответ = [-3.29])
18:30:01.319 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- MSE на тесте: 0.001000
18:30:01.319 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject -- 

==== Классификация ирисов
18:30:01.334 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 0, потеря на тестовой выборке 0.2278058
18:30:01.536 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 500, потеря на тестовой выборке 0.1059319
18:30:01.698 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 1000, потеря на тестовой выборке 0.0824907
18:30:01.857 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 1500, потеря на тестовой выборке 0.0758947
18:30:02.015 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 2000, потеря на тестовой выборке 0.0719501
18:30:02.177 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 2500, потеря на тестовой выборке 0.0681612
18:30:02.341 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 3000, потеря на тестовой выборке 0.0625301
18:30:02.508 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 3500, потеря на тестовой выборке 0.0526782
18:30:02.665 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 4000, потеря на тестовой выборке 0.0404854
18:30:02.827 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 4500, потеря на тестовой выборке 0.0312832
18:30:02.986 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- ответ сети [0.12, 0.79, 0.00]  (правильный ответ = [0.00, 1.00, 0.00])
18:30:02.986 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- ответ сети [0.98, 0.16, 0.00]  (правильный ответ = [1.00, 0.00, 0.00])
18:30:02.986 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- ответ сети [0.00, 0.33, 0.90]  (правильный ответ = [0.00, 0.00, 1.00])
18:30:02.987 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- ответ сети [0.00, 0.58, 0.61]  (правильный ответ = [0.00, 0.00, 1.00])
18:30:02.987 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- ответ сети [0.00, 0.33, 0.86]  (правильный ответ = [0.00, 0.00, 1.00])
18:30:02.987 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- MSE на тесте: 0.025464
18:30:02.987 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject -- 

==== Классификация MNIST
18:30:16.482 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 0, потеря на тестовой выборке 0.0858965
18:30:29.819 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 1, потеря на тестовой выборке 0.0460392
18:30:43.557 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 2, потеря на тестовой выборке 0.0270413
18:30:57.437 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 3, потеря на тестовой выборке 0.0191251
18:31:11.276 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- Эпоха обучения 4, потеря на тестовой выборке 0.0159300
18:31:11.278 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- ответ сети [0.01, 0.00, 0.00, 0.04, 0.00, 0.00, 0.00, 0.97, 0.00, 0.04]  (правильный ответ = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00])
18:31:11.279 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- ответ сети [0.04, 0.00, 0.74, 0.05, 0.00, 0.02, 0.06, 0.00, 0.01, 0.00]  (правильный ответ = [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
18:31:11.280 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- ответ сети [0.00, 0.97, 0.01, 0.01, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00]  (правильный ответ = [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
18:31:11.280 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- ответ сети [0.98, 0.00, 0.01, 0.00, 0.00, 0.04, 0.01, 0.02, 0.00, 0.00]  (правильный ответ = [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
18:31:11.281 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- ответ сети [0.00, 0.00, 0.00, 0.00, 0.85, 0.00, 0.01, 0.01, 0.01, 0.17]  (правильный ответ = [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00])
18:31:12.188 [main] INFO ru.neiro.simple.study.mlp.object.MlpObject$Net -- MSE на тесте: 0.015930

Process finished with exit code 0

```