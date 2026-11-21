package rawnet.code.rl.training;

/**
 * Метрики одной итерации RL-тренера.
 *
 * <p>Используются для логов и early stopping. Не все поля заполняются
 * всеми алгоритмами: у REINFORCE нет {@code valueLoss} (нет critic),
 * у PPO есть. {@code NaN} — «не применимо».
 *
 * @param avgReward        средняя награда за итерацию (эпизод или rollout)
 * @param avgEpisodeLength средняя длина эпизода (для CartPole — шаги)
 * @param policyLoss       loss политики (для логов)
 * @param valueLoss        loss critic (NaN для REINFORCE)
 * @param entropy          средняя энтропия политики
 * @param episodes         число эпизодов в итерации
 */
public record RlMetrics(
        float avgReward,
        float avgEpisodeLength,
        float policyLoss,
        float valueLoss,
        float entropy,
        int episodes
) {
    /** Удобный конструктор для REINFORCE (без critic). */
    public static RlMetrics reinforce(float avgReward, float avgEpisodeLength,
                                      float policyLoss, float entropy, int episodes) {
        return new RlMetrics(avgReward, avgEpisodeLength, policyLoss,
                Float.NaN, entropy, episodes);
    }
}