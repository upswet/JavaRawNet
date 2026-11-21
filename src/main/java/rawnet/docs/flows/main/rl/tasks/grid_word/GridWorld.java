package rawnet.docs.flows.main.rl.tasks.grid_word;

import rawnet.code.rl.Environment;

import java.util.Random;

/**
 * GridWorld 5×5 — классическая среда (Sutton & Barto).
 *
 * <h2>Схема награды</h2>
 * <ul>
 *   <li><b>обычный шаг</b>: {@code reward = 0};</li>
 *   <li><b>яма</b>: {@code reward = -1}, эпизод закончен;</li>
 *   <li><b>цель</b>: {@code reward = +1}, эпизод закончен;</li>
 *   <li><b>лимит шагов</b> ({@link #MAX_STEPS}): эпизод обрывается,
 *       {@code reward = 0} за последний шаг.</li>
 * </ul>
 *
 * <h2>Почему без штрафа за шаг</h2>
 * Штраф за шаг {@code -0.01} + штраф за яму {@code -1} создают
 * <b>локальный оптимум</b>: «не падать в яму любой ценой» выгоднее,
 * чем «дойти до цели». Агент ходит кругами 50 шагов ({@code -0.5})
 * вместо падения ({@code -1}) — формально оптимально локально, но
 * глобальный оптимум ({@code +1}) недостижим. Классическая схема
 * ({@code 0} за шаг) убирает этот локальный оптимум: «ходить кругами»
 * даёт {@code 0}, «дойти до цели» — {@code +1}. Разница очевидна.
 *
 * <h2>Наблюдение</h2>
 * One-hot позиции агента: {@code 25} чисел, {@code 1} на позиции
 * агента, {@code 0} остальные. Для Dense-сети это работает лучше,
 * чем {@code (row, col)} — нет ложной близости чисел.
 *
 * <h2>Действия</h2>
 * {@code 0} — вверх, {@code 1} — вниз, {@code 2} — влево,
 * {@code 3} — вправо. Стены: выход за границу = остаться на месте
 * (без штрафа).
 */
public final class GridWorld implements Environment {

    public static final int SIZE = 5;
    public static final int NUM_ACTIONS = 4;
    public static final int OBS_SIZE = SIZE * SIZE;
    public static final int MAX_STEPS = 50;

    /** Ямы: (row, col). */
    private static final int[][] PITS = {
            {1, 1}, {1, 3},
            {2, 2},
            {3, 1}, {3, 3}
    };

    /** Цель: (row, col). */
    private static final int GOAL_ROW = 4;
    private static final int GOAL_COL = 4;

    /** Старт: (row, col). */
    private static final int START_ROW = 0;
    private static final int START_COL = 0;

    private final Random rnd;
    private int row, col;
    private int steps;

    public GridWorld() {
        this(42L);
    }

    public GridWorld(long seed) {
        this.rnd = new Random(seed);
    }

    @Override
    public float[] reset() {
        // Случайный старт (не яма, не цель).
        do {
            row = rnd.nextInt(SIZE);
            col = rnd.nextInt(SIZE);
        } while (isPit(row, col) || isGoal(row, col));
        steps = 0;
        return observation();
    }

    @Override
    public StepResult step(int action) {
        if (action < 0 || action >= NUM_ACTIONS) {
            throw new IllegalArgumentException("action out of range: " + action);
        }
        steps++;

        int nr = row, nc = col;
        switch (action) {
            case 0 -> nr = row - 1;  // вверх
            case 1 -> nr = row + 1;  // вниз
            case 2 -> nc = col - 1;  // влево
            case 3 -> nc = col + 1;  // вправо
        }

        // Стены: остаться на месте.
        if (nr < 0 || nr >= SIZE || nc < 0 || nc >= SIZE) {
            nr = row;
            nc = col;
        }
        row = nr;
        col = nc;

        // Яма: -1, done.
        if (isPit(row, col)) {
            return new StepResult(observation(), -1f, true);
        }
        // Цель: +1, done.
        if (isGoal(row, col)) {
            return new StepResult(observation(), +1f, true);
        }
        // Лимит шагов: 0, done.
        if (steps >= MAX_STEPS) {
            return new StepResult(observation(), 0f, true);
        }
        // Обычный шаг: 0.
        return new StepResult(observation(), 0f, false);
    }

    @Override
    public int numActions() { return NUM_ACTIONS; }

    @Override
    public int observationSize() { return OBS_SIZE; }

    @Override
    public void render() {
        for (int r = 0; r < SIZE; r++) {
            StringBuilder sb = new StringBuilder();
            for (int c = 0; c < SIZE; c++) {
                if (r == row && c == col) sb.append(" A ");
                else if (isPit(r, c))     sb.append(" X ");
                else if (isGoal(r, c))    sb.append(" G ");
                else                      sb.append(" . ");
            }
            System.out.println(sb);
        }
    }

    private float[] observation() {
        float[] obs = new float[OBS_SIZE];
        obs[row * SIZE + col] = 1f;
        return obs;
    }

    private static boolean isPit(int r, int c) {
        for (int[] p : PITS) if (p[0] == r && p[1] == c) return true;
        return false;
    }

    private static boolean isGoal(int r, int c) {
        return r == GOAL_ROW && c == GOAL_COL;
    }
}