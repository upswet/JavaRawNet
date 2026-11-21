package rawnet.code.net.core.metric;

import lombok.NoArgsConstructor;
import rawnet.code.net.core.Persistables;
import rawnet.code.net.core.loss.Loss;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Objects;

/**
 * Обёртка над {@link Loss}, представляющая её как {@link Metric}.
 *
 * <h2>Зачем</h2>
 * {@link Metric} считают метрики (RMSE, accuracy, F1, ...), а {@link Loss} —
 * функцию, которую <i>оптимизирует</i> сеть. Это разные вещи, и полезно
 * видеть обе на одной выборке одновременно:
 * <ul>
 *   <li>loss падает, а метрика нет → метрика плохо согласована с задачей
 *       (например, RMSE усредняет поля с разной семантикой);</li>
 *   <li>loss не падает → проблема в модели/данных/гиперпараметрах;</li>
 *   <li>loss и метрика падают согласованно → всё в порядке.</li>
 * </ul>
 *
 * <h2>Использование</h2>
 * <pre>{@code
 * Loss loss = new DetectionLoss(maxDet, numClasses);
 * model.train(epochs, ds, loss, opt, batchSize, 3,
 *         List.of(new Rmse(), new LossMetric(loss)), null);
 * // В логе: "RMSE=0.14  Loss(DetectionLoss)=0.031"
 * }</pre>
 *
 * <h2>Переиспользование</h2>
 * Работает с <b>любой</b> реализацией {@link Loss}: MSE, BCE, MAE,
 * SoftmaxCrossEntropy, DetectionLoss — без изменений. Не знает о структуре
 * выхода (плоский вектор / слоты / что угодно), просто вызывает
 * {@code loss.loss(p, t)}. Никакого состояния между вызовами, кроме счётчика.
 */
@NoArgsConstructor
public class LossMetric implements Metric {
    private Loss loss;
    private double sum;
    private int n;

    public LossMetric(Loss loss) {
        this.loss = Objects.requireNonNull(loss, "loss");
    }

    @Override
    public void update(float[] prediction, float[] target) {
        sum += loss.loss(prediction, target);
        n++;
    }

    @Override
    public float result() { return n == 0 ? 0f : (float) (sum / n); }

    @Override
    public void reset() { sum = 0; n = 0; }

    @Override
    public String name() { return "Loss(" + loss.getClass().getSimpleName() + ")"; }

    @Override
    public void save(DataOutputStream out) throws IOException {
        Persistables.save(out, loss);
    }

    @Override
    public void load(DataInputStream in) throws IOException {
        this.loss = Persistables.load(in, Loss.class);
    }
}