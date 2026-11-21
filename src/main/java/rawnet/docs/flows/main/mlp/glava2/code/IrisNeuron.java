package rawnet.docs.flows.main.mlp.glava2.code;

import rawnet.code.dataset.SampleDataset;
import rawnet.code.dataset.Sample;
import rawnet.docs.flows.main.mlp.tasks.Iris;

import java.nio.file.Path;
import java.util.Random;

/**
 * Решение задачи классификации ирисов в концепции «нейрон как объект».
 * Каждый нейрон хранит собственные веса, смещение, выход и дельту ошибки.
 */
public class IrisNeuron {

    static final Random RND = new Random(42);
    static final int EPOCHS   = 2000;
    static final int LOG_EVERY = 200;

    static final int IN = 4, HID = 10, OUT = 3; //Три слоя. 4 нейрона на входном, 10 на скрытом, 3 на выходном
    static final float LR = 0.01f; //коэффициент обучения

    /**Один нейрон*/
    static class Neuron {
        final float[] w;  //веса на входе
        float b; //смещение нейрона
        float in; //взвешенная сумма на входе нейрона
        float out; //выход нейрона = фун-активации от взвешенной сумме на входе
        float delta; //дельта ошибки

        Neuron(int inputSize) { //где inputSize - количество входных связей для данного нейрона
            w = new float[inputSize];
            b = rnd();
            for (int i = 0; i < inputSize; i++) w[i] = rnd();
        }

        //Первоначальная инициализация мелким весом
        static float rnd() { return RND.nextFloat() * 0.2f - 0.1f; }

        //сигмоидная функция актпивации
        static float sigmoid(float x) { return 1f / (1f + (float) Math.exp(-x)); }

        //производная сигмоидной функции активации
        static float sigmoidDerv(float out) { return out * (1 - out); }

        /** Прямой проход: взвешенная сумма + сигмоида. */
        float forward(float[] x) {
            in = b;
            for (int i = 0; i < x.length; i++) in += x[i] * w[i];
            return out = sigmoid(in);
        }

        /** Вычисление delta = dL/dz по переданному dL/da. */
        void backward(float dL_da) {
            delta = dL_da * sigmoidDerv(out);
        }

        /** Шаг градиентного спуска по весам и смещению. */
        void update(float[] x) {
            for (int i = 0; i < w.length; i++) w[i] -= LR * delta * x[i];
            b -= LR * delta;
        }
    }

    /** Слой — просто массив нейронов. */
    static class Layer {
        final Neuron[] neurons;

        Layer(int size, int inputSize) {
            neurons = new Neuron[size];
            for (int i = 0; i < size; i++) neurons[i] = new Neuron(inputSize);
        }

        float[] forward(float[] x) {
            float[] y = new float[neurons.length];
            for (int i = 0; i < neurons.length; i++) y[i] = neurons[i].forward(x);
            return y;
        }
    }

    /**
     * Тренировка и проверка точности сети.
     */
    public static void main(String[] args) {
        SampleDataset<Sample> ds = SampleDataset.split(Iris.prepareDataForIris(Path.of("d:\\Work\\Project\\0files\\iris\\iris.data.csv")), 0.9f);

        Layer hiddenLayer = new Layer(HID, IN);
        Layer outLayer = new Layer(OUT, HID);

        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            for (Sample s : ds.getTrainSamples()) {
                float[] inputs = s.getInputs();
                float[] targets = s.getTargets();

                /**прямой проход*/
                float[] h_outs = hiddenLayer.forward(inputs); //выход скрытого слоя
                float[] outs = outLayer.forward(h_outs); //выход выходного слоя

                /**Обратный проход*/

                // сначала считаем дельту-ошибки на нейрона выходного слоя через функцию потери.
                // Здесь используется стандартная MSE = 1/2 * (out - target)^2 а производная у неё: out - target
                for (int k = 0; k < OUT; k++) {
                    Neuron n = outLayer.neurons[k];
                    n.backward(n.out - targets[k]);
                }
                // потом считаем дельты скрытого слоя как сумму дельт всех нейронов выходного слоя умножимые на веса связей с ними от текущего нейрона
                for (int j = 0; j < HID; j++) {
                    float delta = 0;
                    for (int k = 0; k < OUT; k++) delta += outLayer.neurons[k].delta * outLayer.neurons[k].w[j];
                    hiddenLayer.neurons[j].backward(delta);
                }
                // наконец обновим веса для нейроной выходного и скрытого слоя
                for (Neuron n : outLayer.neurons) n.update(h_outs);
                for (Neuron n : hiddenLayer.neurons) n.update(inputs);

            }
            if (epoch % LOG_EVERY == 0 || epoch == EPOCHS) {
                test(epoch, ds, outLayer, hiddenLayer);
            }
        }
    }

    // Считаем точность на тестовом наборе
    private static float test(int epoch, SampleDataset<Sample> ds, Layer outLayer, Layer hiddenLayer) {
        float lossSum = 0;
        int correct = 0, total = 0;
        for (Sample s : ds.getTestSamples()) {
            float[] o = outLayer.forward(hiddenLayer.forward(s.getInputs()));
            lossSum += mseLoss(o, s.getTargets());
            int pred = argmax(o), truth = argmax(s.getTargets());
            if (pred == truth) correct++;
            total++;
        }
        float averageLoss = lossSum / total;
        float accuracy = (float) correct / total;
        System.out.printf("Epoch %4d: loss=%.6f, accuracy=%.1f%%%n",
                epoch, averageLoss, 100.0 * accuracy);
        return accuracy;
    }

    // Вернуть номер максимального элемента в массиве чтобы считать его за правильный ответ
    static int argmax(float[] v) {
        int m = 0;
        for (int i = 1; i < v.length; i++) if (v[i] > v[m]) m = i;
        return m;
    }

    //считать потерю
    static float mseLoss(float[] predicted, float[] target) {
        float sum = 0;
        for (int i = 0; i < predicted.length; i++) {
            float diff = predicted[i] - target[i];
            sum += diff * diff;
        }
        return sum / predicted.length;
    }
}
