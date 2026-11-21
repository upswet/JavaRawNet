# Нейросеть на матрицах

[Ссылка на полный код](../../src/main/java/ru/neiro/simple/study/mlp/matrix/MlpMatrix.java)

## Код

Функции активации и потерь опустим, также как train и test - методы они практически без изменений относительно [нейросети на объектах](MLP_object.md)

```java
   /**
     * Нейросеть
     */
    @Slf4j
    public static class Net {
    final Layer[] layers; //слои
    final Loss loss; //функция потери
    final float lr; //коэффициент обучения сети

    private final float[] outputGradient;  // переиспользуемый буфер для градиента функции потерь

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
                float[] outputs = this.forward(sample.getInputs());
                this.backward(outputs, sample.getTargets());
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
}
```

```java
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
}
```