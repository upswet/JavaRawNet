package rawnet.docs.flows.main.rl.tasks.cart_pole;

import rawnet.code.rl.Environment;

/**
 * CartPole — тележка с шестом. Классическая RL-среда.
 *
 * <p>Наблюдение: {@code [x, x', θ, θ']} — позиция, скорость тележки,
 * угол и угловая скорость шеста. Действие: 0 — влево, 1 — вправо.
 * Награда: {@code +1} за каждый шаг, пока шест не упал
 * ({@code |θ| > 12°} или {@code |x| > 2.4}). Эпизод — до падения или
 * 500 шагов.
 *
 * <p><b>Почему CartPole хорош для policy gradient.</b> Награда
 * <b>частая</b> ({@code +1} каждый шаг), credit assignment
 * <b>простой</b> (каждый шаг даёт награду), локального оптимума
 * <b>нет</b> (стоять нельзя — шест упадёт). REINFORCE, Actor-Critic
 * и PPO <b>все</b> сходятся, разница — скорость и стабильность.
 * В GridWorld 5×5 с редкой наградой REINFORCE/Actor-Critic
 * застревают, PPO справляется.
 */
public final class CartPole implements Environment {

    private static final float GRAVITY = 9.8f;
    private static final float MASSCART = 1.0f;
    private static final float MASSPOLE = 0.1f;
    private static final float TOTAL_MASS = MASSCART + MASSPOLE;
    private static final float LENGTH = 0.5f;
    private static final float POLEMASS_LENGTH = MASSPOLE * LENGTH;
    private static final float FORCE_MAG = 10.0f;
    private static final float TAU = 0.02f;
    private static final float THETA_THRESHOLD = 12f * 2f * (float) Math.PI / 360f;
    private static final float X_THRESHOLD = 2.4f;
    private static final int MAX_STEPS = 500;

    private final java.util.Random rnd = new java.util.Random(42L);
    private float x, xDot, theta, thetaDot;
    private int steps;

    @Override
    public float[] reset() {
        x = uniform(-0.05f, 0.05f);
        xDot = uniform(-0.05f, 0.05f);
        theta = uniform(-0.05f, 0.05f);
        thetaDot = uniform(-0.05f, 0.05f);
        steps = 0;
        return observation();
    }

    @Override
    public StepResult step(int action) {
        if (action != 0 && action != 1)
            throw new IllegalArgumentException("action in {0,1}, получено " + action);

        float force = (action == 1) ? FORCE_MAG : -FORCE_MAG;
        float cosTheta = (float) Math.cos(theta);
        float sinTheta = (float) Math.sin(theta);

        float temp = (force + POLEMASS_LENGTH * thetaDot * thetaDot * sinTheta) / TOTAL_MASS;
        float thetaAcc = (GRAVITY * sinTheta - cosTheta * temp) /
                (LENGTH * (4f / 3f - MASSPOLE * cosTheta * cosTheta / TOTAL_MASS));
        float xAcc = temp - POLEMASS_LENGTH * thetaAcc * cosTheta / TOTAL_MASS;

        x += TAU * xDot;
        xDot += TAU * xAcc;
        theta += TAU * thetaDot;
        thetaDot += TAU * thetaAcc;

        steps++;
        boolean done = x < -X_THRESHOLD || x > X_THRESHOLD
                || theta < -THETA_THRESHOLD || theta > THETA_THRESHOLD
                || steps >= MAX_STEPS;

        return new StepResult(observation(), done ? 0f : 1f, done);
    }

    @Override public int numActions() { return 2; }
    @Override public int observationSize() { return 4; }

    @Override
    public void render() {
        System.out.printf(java.util.Locale.US,
                "x=%+.3f  θ=%+.3f rad  steps=%d%n", x, theta, steps);
    }

    private float[] observation() {
        return new float[]{x, xDot, theta, thetaDot};
    }

    private float uniform(float lo, float hi) {
        return lo + rnd.nextFloat() * (hi - lo);
    }
}