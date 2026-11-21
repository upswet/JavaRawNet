package rawnet.code.rl.training;

/**
 * Общий контракт RL-тренера.
 *
 * <p>Одна итерация — это либо <b>эпизод</b> (REINFORCE), либо
 * <b>rollout</b> (PPO). Метрики итерации доступны через
 * {@link #lastMetrics()} <b>после</b> {@link #trainIteration()}.
 *
 * <p><b>Зачем.</b> Позволяет использовать {@link RlTraining} (early
 * stopping, логи) для любого алгоритма: REINFORCE, PPO, будущих
 * (A2C, SAC, …) без дублирования.
 */
public interface RlTrainer {

    /**
     * Одна итерация: собрать данные, посчитать advantage/returns,
     * обновить политику (и critic, если есть).
     */
    void trainIteration();

    /**
     * Метрики последней итерации. Вызывается <b>после</b>
     * {@link #trainIteration()}.
     */
    RlMetrics lastMetrics();
}